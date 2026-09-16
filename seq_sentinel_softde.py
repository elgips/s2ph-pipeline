"""
seq_sentinel_softde.py -- can a FAST SoftTri-DE do the sentinel's job?

The sentinel's job is basin identification / escape detection, NOT final
precision. So this does NOT run any remesh forward. It runs differential
evolution on the SoftTri surrogate alone (stateless -> process-safe ->
workers=-1 parallel) and asks:

  * does SoftTri-DE land in the SAME basin as the handoff estimate theta_hat
    (position x, y, layer depth, and roughly radius)?
  * how fast is it (wall-time, parallel)?

Metric is basin agreement, not ||theta - theta_true||: a SoftTri optimum is
bias-limited to ~theta_hat's accuracy by construction. If the basin matches and
it's fast, the sentinel is viable: SoftTri-DE finds/verifies the basin, and a
short remesh step (elsewhere) de-biases the mean only when precision is needed.

Objective uses the model-discrepancy term (sigma_model) because the SoftTri
forward differs from the remesh-generated data; without it the whitened misfit
floors well above 1 (see the handoff finding).

Run in pygimli_env. Needs seq_handoff, seq_coldstart, pwhg_forward_soft,
pwhg_wrapper, Anandlyn_log, config. scipy>=1.9 for x0/workers.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import pygimli as pg
try:
    pg.setDebug(False)
except Exception:
    pass
for _fn in ("setDefaultCache", "noCache"):
    try:
        getattr(pg.core, _fn)(False); break
    except Exception:
        pass
try:
    from pygimli.utils import cache as _c
    _c.CacheManager().cachingActive = False
except Exception:
    pass
import logging
logging.getLogger("pyGIMLi").setLevel(logging.ERROR)

import time
import numpy as np
from scipy.optimize import differential_evolution
import pwhg_wrapper as W
from pwhg_forward_soft import SoftTriForward
from seq_coldstart import make_scheme, world_from_theta
from seq_handoff import read_cv_seed, load_dat, compute_handoff, THETA_NAMES

I_R, I_RHO = 5, 6
# WIDE search box (article-1-style global), theta units
HALF_WIDE = np.array([0.50, 3.00, 0.05, 8.00, 4.00, 0.90, 2.50])
LO = np.array([0.0, -10.0, 0.0, -25.0, -10.0, np.log10(0.5), 0.0])
HI = np.array([5.0,  -1.0, 5.0,  25.0,   0.0, np.log10(5.0), 5.0])

# basin-agreement tolerances (position/structure only; rho excluded -- degenerate)
BASIN_TOL = dict(x=0.5, y=0.5, layer_y=0.3, log10r=0.15)  # meters / decades

DE_KW = dict(popsize=20, maxiter=40, tol=1e-4, mutation=(0.5, 1.0),
             recombination=0.7, polish=False, seed=0, init="sobol")


class _SoftObj:
    """Picklable weighted-SSE objective on the SoftTri surrogate.

    Holds only arrays/floats so it pickles for multiprocessing. Each worker
    rebuilds SoftTriForward lazily on first call (SoftTri + pygimli scheme are
    not picklable, so they are reconstructed per process, not shipped)."""
    def __init__(self, theta_center, d_ln, Wt, width=0.2, tri_area=0.25,
                 n_elec=21, x0=-25.0, x1=25.0):
        self.tc = np.asarray(theta_center, float)
        self.d_ln = np.asarray(d_ln, float)
        self.Wt = np.asarray(Wt, float)
        self.width = float(width); self.tri_area = float(tri_area)
        self.n_elec = int(n_elec); self.x0 = float(x0); self.x1 = float(x1)
        self._soft = None                      # built lazily, per process

    def _ensure(self):
        if self._soft is None:
            sch = make_scheme(self.n_elec, self.x0, self.x1)
            world = world_from_theta(self.tc, sch)
            self._soft = SoftTriForward(world, width=self.width,
                                        tri_area=self.tri_area, scheme=sch)

    def __getstate__(self):                    # never pickle the pygimli object
        d = self.__dict__.copy(); d["_soft"] = None; return d

    def __call__(self, theta):
        self._ensure()
        ln = np.asarray(self._soft.forward(np.asarray(theta, float)))[0]
        r = self.d_ln - ln
        return float(np.sum(self.Wt * r * r))


def soft_de(theta_center, d_ln, noise_var, scheme, half=HALF_WIDE,
            x0=None, workers=1, **de_kw):
    """Global DE on the SoftTri surrogate. Returns (theta, phi, nfev, seconds).

    Falls back to serial automatically if parallel evaluation fails."""
    obj = _SoftObj(theta_center, d_ln, 1.0 / noise_var)
    bounds = list(zip(np.maximum(theta_center - half, LO),
                      np.minimum(theta_center + half, HI)))
    kw = dict(de_kw)
    kw["workers"] = workers
    kw["updating"] = "deferred" if workers != 1 else "immediate"
    if x0 is not None:
        kw["x0"] = x0
    t0 = time.time()
    try:
        res = differential_evolution(obj, bounds, **kw)
    except Exception as e:
        if workers == 1:
            raise
        print(f"  [parallel failed: {type(e).__name__}: {e}; falling back to serial]")
        kw["workers"] = 1; kw["updating"] = "immediate"
        t0 = time.time()
        res = differential_evolution(obj, bounds, **kw)
    return res.x, float(res.fun), int(res.nfev), time.time() - t0


def basin_agreement(theta_de, theta_ref):
    """Do position/structure params agree within BASIN_TOL? rho excluded."""
    d = np.asarray(theta_de, float) - np.asarray(theta_ref, float)
    checks = {
        "C0_x":     (abs(d[3]), BASIN_TOL["x"]),
        "C0_y":     (abs(d[4]), BASIN_TOL["y"]),
        "L0_y":     (abs(d[1]), BASIN_TOL["layer_y"]),
        "C0_log10r":(abs(d[I_R]), BASIN_TOL["log10r"]),
    }
    ok = all(v <= t for v, t in checks.values())
    return ok, checks


if __name__ == "__main__":
    import config as C
    scene = "c11"
    scheme = make_scheme()

    # handoff estimate theta_hat (SoftTri GN) as the basin reference
    seed = read_cv_seed(C.DIR_CV / "parameters.xlsx", scene)
    rhoa_obs, err = load_dat(C.DIR_FWD / f"{scene.upper()}_noisy.dat")
    hand = compute_handoff(seed, rhoa_obs, err, scheme, n_gn=8, verbose=False)
    theta_hat = hand["theta0"]

    theta_true = np.array([2.0, -3.0, np.log10(50), 5.0, -4.0,
                           np.log10(1.2), np.log10(500)])

    d_ln = np.log(rhoa_obs)
    sigma_model = 0.02
    noise_var = np.asarray(err, float) ** 2 + sigma_model ** 2   # surrogate: keep sigma_model

    print(f"theta_hat (SoftTri GN): {np.round(theta_hat,3)}")
    print(f"theta_true            : {np.round(theta_true,3)}")
    print(f"DE: popsize={DE_KW['popsize']} maxiter={DE_KW['maxiter']} "
          f"box=WIDE  sigma_model={sigma_model}")

    # --- SoftTri-DE, serial (cache disabled; parallel corrupts pygimli cache) ---
    print("\nrunning SoftTri-DE (serial)...")
    th_p, phi_p, nfev_p, t_p = soft_de(theta_hat, d_ln, noise_var, scheme,
                                       workers=1, **DE_KW)
    th_s, phi_s, nfev_s, t_s = th_p, phi_p, nfev_p, t_p  # no separate baseline

    print(f"\n{'':<16}{'x':>9}{'y':>9}{'layer_y':>10}{'log10r':>9}"
          f"{'log10rho':>10}{'Phi':>11}{'nfev':>8}{'t[s]':>8}")
    def row(nm, th, phi, nfev, t):
        print(f"{nm:<16}{th[3]:>9.3f}{th[4]:>9.3f}{th[1]:>10.3f}{th[I_R]:>9.3f}"
              f"{th[I_RHO]:>10.3f}{phi:>11.1f}{nfev:>8}{t:>8.1f}")
    row("theta_hat", theta_hat, float('nan'), 0, 0.0)
    row("SoftTri-DE //", th_p, phi_p, nfev_p, t_p)
    row("SoftTri-DE serial", th_s, phi_s, nfev_s, t_s)
    row("theta_true", theta_true, float('nan'), 0, 0.0)

    ok, checks = basin_agreement(th_p, theta_hat)
    print("\nbasin agreement (SoftTri-DE // vs handoff theta_hat):")
    for k, (v, tol) in checks.items():
        print(f"  {k:<10} |delta|={v:.3f}  tol={tol:.2f}  {'OK' if v<=tol else 'MISS'}")
    print(f"  -> {'SAME BASIN' if ok else 'DIFFERENT BASIN'}")

    print(f"\nserial nfev={nfev_p} in {t_p:.1f}s")
    print("\nread: if SAME BASIN and fast, the SoftTri-DE sentinel is viable "
          "(remesh only to de-bias the mean when precision is needed).")
