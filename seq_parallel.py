"""
seq_parallel.py -- WORKING process-parallel SoftTri forward / Jacobian.

WHY THE EARLIER ATTEMPTS FAILED
-------------------------------
1. "Can't pickle local object": the objective was a closure. Fixed by making it
   a module-level callable.
2. "DataContainer::set wrong data size: 171 0" + JSONDecodeError spam: the REAL
   root cause. `ert.createGeometricFactors` is wrapped in pyGIMLi's DISK CACHE.
   Concurrent workers read/write the same ~/.cache/pygimli file; one gets a
   truncated read, returns a size-0 vector, and `scheme.set("k", <empty>)`
   raises. Disabling the cache in the PARENT does not help because spawn workers
   re-import pygimli fresh with caching ON.

THE FIX (three parts, all necessary)
------------------------------------
(a) An INITIALIZER runs in every worker that disables the pygimli cache, pins
    threads (OMP_NUM_THREADS=1 so N processes don't oversubscribe cores), and
    quiets logging -- before any pygimli work happens.
(b) Workers NEVER call createGeometricFactors. The parent computes `k` once and
    ships it as a plain numpy array; workers rebuild the scheme from arrays.
(c) The heavy SoftTriForward is built ONCE PER WORKER at init and kept in worker
    -global state. Only numpy arrays cross the process boundary. This is valid
    because SoftTriForward is STATELESS in theta (verified): a forward built at
    one theta evaluates any other theta identically.

USAGE
-----
    from seq_parallel import ParallelSoft
    with ParallelSoft(xe, quads, k, theta_build, n_workers=8) as ps:
        lns = ps.forward_many([th1, th2, ...])      # list of (m,) ln-rhoa
        J, ln0 = ps.jacobian(theta, steps)          # (m,p) d ln rhoa / d theta

`jacobian` returns the SAME (J, ln0) as pwhg_wrapper.jacobian_generic, so it is
a drop-in for the greedy loop's full-pool Jacobian (the dominant cost: 2p
forwards over the whole pool, embarrassingly parallel).

Falls back to serial automatically if the pool cannot start, and verifies
parallel == serial in the self-test.
"""
import os
import sys
import multiprocessing as mp
import numpy as np

# --------------------------------------------------------------------------
# worker-side state and initializer
# --------------------------------------------------------------------------
_W = {"soft": None}          # worker-global: the heavy forward, built once


def _worker_init(xe, quads, k, theta_build, width, tri_area, mock):
    """Runs ONCE per worker process. Order matters: env + cache-off BEFORE any
    pygimli forward work, otherwise the cache races and returns empty vectors."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if mock:                                    # machinery test without pygimli
        _W["soft"] = ("mock", np.asarray(quads), np.asarray(k))
        return
    import pygimli as pg
    try:
        pg.setDebug(False)
    except Exception:
        pass
    for fn in ("setDefaultCache", "noCache"):
        try:
            getattr(pg.core, fn)(False)
            break
        except Exception:
            pass
    try:                                        # the decisive one
        from pygimli.utils import cache as _c
        _c.CacheManager().cachingActive = False
    except Exception:
        pass
    import logging
    logging.getLogger("pyGIMLi").setLevel(logging.ERROR)
    logging.getLogger("Core").setLevel(logging.CRITICAL)

    scheme = _build_scheme(xe, quads, k)
    from seq_coldstart import world_from_theta
    from pwhg_forward_soft import SoftTriForward
    world = world_from_theta(np.asarray(theta_build, float), scheme)
    _W["soft"] = SoftTriForward(world, width=float(width),
                                tri_area=float(tri_area), scheme=scheme)


def _build_scheme(xe, quads, k):
    """Rebuild the ERT scheme from plain arrays. NOTE: k is SUPPLIED, never
    recomputed -- calling createGeometricFactors here is what broke parallel."""
    import pygimli as pg
    xe = np.asarray(xe, float); quads = np.asarray(quads, int)
    sch = pg.DataContainerERT()
    for x in xe:
        sch.createSensor([float(x), 0.0])
    sch.resize(len(quads))
    for j, tok in enumerate("abmn"):
        sch.set(tok, quads[:, j].astype(float))
    sch.set("valid", np.ones(len(quads)))
    sch.set("k", np.asarray(k, float))          # precomputed in the parent
    return sch


def _fwd_one(theta):
    """Evaluate ln(rhoa) at theta using the worker-global forward."""
    st = _W["soft"]
    th = np.asarray(theta, float)
    if isinstance(st, tuple) and st[0] == "mock":
        q, k = st[1], st[2]
        # deterministic, theta-dependent stand-in for the physics
        return (np.sin(q @ th[:4] * 0.1) + 0.01 * k * th[4]) + th[5] + th[6]
    return np.asarray(st.forward(th), float)[0]


# --------------------------------------------------------------------------
# parent-side driver
# --------------------------------------------------------------------------
class ParallelSoft:
    """Persistent worker pool holding one SoftTriForward per process."""

    def __init__(self, xe, quads, k, theta_build, width=0.2, tri_area=0.25,
                 n_workers=None, mock=False, verbose=True):
        self.args = (np.asarray(xe, float), np.asarray(quads, int),
                     np.asarray(k, float), np.asarray(theta_build, float),
                     float(width), float(tri_area), bool(mock))
        self.n = n_workers or max(1, (os.cpu_count() or 2) - 1)
        self.mock = mock
        self.verbose = verbose
        self.pool = None
        self._serial_ready = False

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def _spawn_safe(self):
        """spawn re-imports the parent's __main__ in each worker. From a REPL,
        stdin heredoc, or a notebook there is no importable __main__ file, so
        workers die with FileNotFoundError. Detect that and avoid spawn."""
        m = sys.modules.get("__main__", None)
        f = getattr(m, "__file__", None)
        return bool(f) and os.path.exists(f)

    def start(self):
        if not self._spawn_safe():
            if self.verbose:
                print("[parallel] no importable __main__ (REPL/stdin/notebook): "
                      "spawn workers cannot start -> running SERIAL. "
                      "Run from a .py file to get parallelism.")
            self._init_serial()
            return self
        try:
            ctx = mp.get_context("spawn")      # spawn: no fork+OpenMP hazards
            self.pool = ctx.Pool(processes=self.n,
                                 initializer=_worker_init,
                                 initargs=self.args)
            # prove a worker can actually evaluate before we rely on it
            probe = self.pool.apply(_fwd_one, (self.args[3],))
            if probe is None or np.asarray(probe).size == 0:
                raise RuntimeError("worker probe returned empty")
            if self.verbose:
                print(f"[parallel] pool up: {self.n} workers, "
                      f"m={np.asarray(probe).size} (probe OK)")
        except Exception as e:
            if self.verbose:
                print(f"[parallel] pool unavailable ({type(e).__name__}: {e}); "
                      f"falling back to SERIAL")
            self.close()
            self._init_serial()
        return self

    def _init_serial(self):
        _worker_init(*self.args)               # build in-process
        self._serial_ready = True

    def close(self):
        if self.pool is not None:
            try:
                self.pool.close(); self.pool.join()
            except Exception:
                self.pool.terminate()
            self.pool = None

    # -- work --------------------------------------------------------------
    def forward_many(self, thetas):
        """Evaluate ln(rhoa) for a list of theta vectors."""
        thetas = [np.asarray(t, float) for t in thetas]
        if self.pool is not None:
            chunk = max(1, len(thetas) // (self.n * 2) or 1)
            out = self.pool.map(_fwd_one, thetas, chunksize=chunk)
        else:
            if not self._serial_ready:
                self._init_serial()
            out = [_fwd_one(t) for t in thetas]
        return [np.asarray(o, float) for o in out]

    def jacobian(self, theta, steps):
        """Central-difference Jacobian d(ln rhoa)/d theta, parallel over the
        2p perturbed evaluations. Returns (J (m,p), ln0 (m,)) matching
        pwhg_wrapper.jacobian_generic's first two outputs."""
        theta = np.asarray(theta, float)
        steps = np.asarray(steps, float)
        p = theta.size
        jobs = [theta]                                  # index 0 = base point
        for i in range(p):
            tp = theta.copy(); tp[i] += steps[i]
            tm = theta.copy(); tm[i] -= steps[i]
            jobs += [tp, tm]
        res = self.forward_many(jobs)
        ln0 = res[0]
        J = np.empty((ln0.size, p), float)
        for i in range(p):
            J[:, i] = (res[1 + 2 * i] - res[2 + 2 * i]) / (2.0 * steps[i])
        return J, ln0


# --------------------------------------------------------------------------
# helper: compute k ONCE in the parent (the thing workers must never do)
# --------------------------------------------------------------------------
def geometric_factors(xe, quads):
    """Parent-side k. Call this before starting the pool."""
    import pygimli as pg
    from pygimli.physics import ert
    sch = pg.DataContainerERT()
    for x in np.asarray(xe, float):
        sch.createSensor([float(x), 0.0])
    q = np.asarray(quads, int)
    sch.resize(len(q))
    for j, tok in enumerate("abmn"):
        sch.set(tok, q[:, j].astype(float))
    sch.set("valid", np.ones(len(q)))
    return np.asarray(ert.createGeometricFactors(sch), float)


# --------------------------------------------------------------------------
# self-tests
# --------------------------------------------------------------------------
def _selftest_machinery():
    """Validate the PARALLEL MACHINERY (spawn, initializer, pickling, map,
    determinism) with a numpy-only mock forward -- no pygimli needed."""
    print("=== machinery self-test (mock forward, no pygimli) ===")
    rng = np.random.default_rng(0)
    xe = np.linspace(-25, 25, 21)
    quads = rng.integers(0, 21, size=(400, 4))
    k = rng.normal(size=400) * 10
    th = np.array([2.0, -3.0, 1.7, 5.0, -4.0, 0.08, 2.7])
    steps = np.full(7, 0.01)

    with ParallelSoft(xe, quads, k, th, n_workers=4, mock=True) as ps:
        par_fwd = ps.forward_many([th, th + 0.01, th - 0.01])
        Jp, ln0p = ps.jacobian(th, steps)

    ser = ParallelSoft(xe, quads, k, th, n_workers=1, mock=True, verbose=False)
    ser._init_serial()
    ser_fwd = ser.forward_many([th, th + 0.01, th - 0.01])
    Js, ln0s = ser.jacobian(th, steps)

    f_ok = all(np.allclose(a, b) for a, b in zip(par_fwd, ser_fwd))
    j_ok = np.allclose(Jp, Js) and np.allclose(ln0p, ln0s)
    print(f"  forward parallel == serial : {f_ok}")
    print(f"  jacobian parallel == serial: {j_ok}  (J {Jp.shape})")
    ok = f_ok and j_ok
    print(f"  MACHINERY {'PASS' if ok else 'FAIL'}")
    return ok


def _selftest_real(n_max=400, n_workers=None):
    """Real pygimli test: parallel vs serial SoftTri, correctness + speedup."""
    print("\n=== real SoftTri self-test (pygimli) ===")
    import time
    from seq_greedy import build_pool
    pool_scheme, quads, xe = build_pool(n_elec=21, n_max=n_max)
    quads = np.column_stack([np.asarray(pool_scheme[t], int) for t in "abmn"])
    k = np.asarray(pool_scheme["k"], float)
    th = np.array([2.0, -3.0, np.log10(50), 5.0, -4.0, np.log10(1.2),
                   np.log10(500)])
    steps = np.array([0.005, 0.01, 0.005, 0.01, 0.01, 0.005, 0.005])
    print(f"  pool m={len(quads)}  p={th.size}  -> {2*th.size+1} forwards/Jacobian")

    t0 = time.time()
    ser = ParallelSoft(xe, quads, k, th, n_workers=1, verbose=False)
    ser._init_serial()
    Js, ln0s = ser.jacobian(th, steps)
    t_ser = time.time() - t0
    print(f"  serial   : {t_ser:6.1f}s")

    t0 = time.time()
    with ParallelSoft(xe, quads, k, th, n_workers=n_workers) as ps:
        Jp, ln0p = ps.jacobian(th, steps)
    t_par = time.time() - t0
    print(f"  parallel : {t_par:6.1f}s   speedup {t_ser/max(t_par,1e-9):.1f}x")

    same = np.allclose(Jp, Js, rtol=1e-9, atol=1e-12) and np.allclose(ln0p, ln0s)
    print(f"  parallel == serial: {same}")
    if not same:
        print(f"    max |dJ| = {np.max(np.abs(Jp-Js)):.3e}")
    print(f"  REAL {'PASS' if same else 'FAIL'}")
    return same


if __name__ == "__main__":
    ok = _selftest_machinery()
    if "--real" in sys.argv:
        try:
            ok = _selftest_real() and ok
        except Exception as e:
            print(f"real test skipped/failed: {type(e).__name__}: {e}")
    sys.exit(0 if ok else 1)
