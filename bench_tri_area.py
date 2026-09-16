"""
bench_tri_area.py -- can SoftTri be made cheap without losing its only advantage?

MEASURED FACT (bench_forward_cost.py): soft.forward costs 366 ms/eval against
157 ms for the remesh forward -- SoftTri is 2.3x SLOWER, not cheaper. The cause
is structural: SoftTri solves on a UNIFORM triangulation over the whole domain,
while pygimli remeshes ADAPTIVELY (fine near electrodes and the anomaly, coarse
elsewhere), so SoftTri spends most of its cells where nothing is happening.

SoftTri exists for ONE reason: smooth geometric derivatives (finite-difference
convergence in x, y, r), which the staircase remesh forward cannot provide.
Cost is therefore only worth paying where derivatives are needed.

This sweeps `tri_area` and reports the three things that decide the trade:
  1. mesh size and per-evaluation cost
  2. forward accuracy against the remesh reference (the surrogate discrepancy)
  3. DERIVATIVE SMOOTHNESS -- the direction-consistency of the C0_y Jacobian
     column across finite-difference steps. This is the property SoftTri is for;
     a coarser mesh is only acceptable if this survives.

Run:  python bench_tri_area.py
"""
import time
import numpy as np
import pygimli as pg
try:
    pg.setDebug(False)
except Exception:
    pass
try:
    from pygimli.utils import cache as _c
    _c.CacheManager().cachingActive = False
except Exception:
    pass
import logging
logging.getLogger("pyGIMLi").setLevel(logging.ERROR)
logging.getLogger("Core").setLevel(logging.CRITICAL)

import pwhg_wrapper as W
from pwhg_forward_soft import SoftTriForward
from seq_coldstart import make_scheme, world_from_theta

I_Y = 4
AREAS = [0.25, 0.5, 1.0, 2.0, 4.0]
FD_STEPS = [0.01, 0.02, 0.05, 0.1, 0.2]
N_TIME = 6


def fd_col(fwd, th, k, h):
    tp = th.copy(); tp[k] += h
    tm = th.copy(); tm[k] -= h
    return (np.asarray(fwd(tp))[0] - np.asarray(fwd(tm))[0]) / (2 * h)


def main():
    scheme = make_scheme()
    th = np.array([2.0, -3.0, np.log10(50.0), 5.0, -4.0,
                   np.log10(1.2), np.log10(500.0)])
    world = world_from_theta(th, scheme)

    # remesh reference: cost and the ln-rhoa the surrogate is compared against
    W.forward(world, th)
    t0 = time.time()
    for _ in range(N_TIME):
        ln_ref, _ = W.forward(world, th)
    t_remesh = (time.time() - t0) / N_TIME
    ln_ref = np.asarray(ln_ref, float)
    try:
        n_ref = world.mesh.cellCount()
    except Exception:
        n_ref = -1
    print(f"remesh reference: {t_remesh*1000:.0f} ms/eval, {n_ref} cells\n")

    print(f"{'tri_area':>9}{'cells':>8}{'ms/eval':>9}{'vs remesh':>11}"
          f"{'discrep':>10}{'FD cos':>9}{'|col| var':>11}")
    rows = []
    for a in AREAS:
        try:
            soft = SoftTriForward(world, width=0.2, tri_area=a, scheme=scheme)
        except Exception as e:
            print(f"{a:>9.2f}  build failed: {type(e).__name__}: {e}")
            continue
        try:
            ncell = soft.mesh.cellCount()
        except Exception:
            ncell = -1

        soft.forward(th)                                    # warm-up
        t0 = time.time()
        for _ in range(N_TIME):
            out = soft.forward(th)
        t = (time.time() - t0) / N_TIME
        ln_s = np.asarray(out, float)[0]

        # surrogate discrepancy vs the faithful forward (rms in ln rhoa)
        disc = float(np.sqrt(np.mean((ln_s - ln_ref) ** 2)))

        # derivative smoothness: is the C0_y column direction stable across
        # FD steps? cos ~ 1 means a well-defined derivative; < 1 means the
        # column is FD noise and the whole point of SoftTri is lost.
        cols = [fd_col(soft.forward, th, I_Y, h) for h in FD_STEPS]
        ref = cols[2] / (np.linalg.norm(cols[2]) + 1e-30)
        coss = [float(ref @ (c / (np.linalg.norm(c) + 1e-30))) for c in cols]
        norms = [float(np.linalg.norm(c)) for c in cols]
        cos_min = min(coss)
        nvar = (max(norms) - min(norms)) / (np.mean(norms) + 1e-30)

        print(f"{a:>9.2f}{ncell:>8}{t*1000:>9.0f}"
              f"{t/max(t_remesh,1e-12):>10.2f}x{disc:>10.4f}"
              f"{cos_min:>9.4f}{nvar:>11.3f}")
        rows.append((a, ncell, t, disc, cos_min, nvar))

    print("\ncolumns: 'discrep' = rms ln-rhoa difference from the remesh forward")
    print("         'FD cos'  = worst direction-consistency of the C0_y column")
    print("                     across FD steps (1.0 = clean derivative)")
    print("         '|col| var' = relative spread of that column's magnitude")

    ok = [r for r in rows if r[4] > 0.999 and r[5] < 0.05]
    print("\nverdict:")
    if ok:
        best = min(ok, key=lambda r: r[2])
        print(f"  coarsest mesh that KEEPS a clean derivative: tri_area={best[0]}"
              f"  ({best[2]*1000:.0f} ms/eval, {best[2]/t_remesh:.2f}x remesh,"
              f" discrepancy {best[3]:.4f} ln)")
        if best[2] < t_remesh:
            print("  -> at that setting SoftTri is CHEAPER than remesh; use it for")
            print("     Jacobians and reconsider it for the sentinel.")
        else:
            print("  -> still more expensive than remesh. Use SoftTri ONLY where")
            print("     derivatives are needed (greedy Jacobian, GN update), and")
            print("     run the derivative-free sentinel DE on the REMESH forward.")
    else:
        print("  no tested tri_area keeps a clean C0_y derivative -- do not coarsen;")
        print("  keep tri_area=0.25 for Jacobians and use remesh for the sentinel.")


if __name__ == "__main__":
    main()
