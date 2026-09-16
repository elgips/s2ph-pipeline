"""
bench_forward_cost.py -- per-EVALUATION cost of the two forwards.

WHY: the ls_opti head-to-head reported wall times of 1064s (remesh) vs 2889s
(soft), but that comparison is not interpretable:
  * res3.nfev counts only the stage-3 L-BFGS evaluations (240 / 168), not the
    DE evaluations, so the two runs did unknown and DIFFERENT total work;
  * DE stops on its own tol, so a run that converges more slowly does more
    evaluations regardless of per-eval cost.

This measures the thing that actually matters: SECONDS PER FORWARD CALL for
each forward, plus the effect of OpenMP threading (the seq_* headers pin
OMP_NUM_THREADS=1, which throttles pygimli's solver and penalises both paths).

Run:  python bench_forward_cost.py            (uses ambient thread settings)
      OMP_NUM_THREADS=1 python bench_forward_cost.py
      OMP_NUM_THREADS=8 python bench_forward_cost.py
"""
import os
import time
import numpy as np

# NOTE: deliberately does NOT pin OMP_NUM_THREADS -- we want to measure it.
print(f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', '(unset)')}")

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

N = 12          # evaluations per arm


def main():
    scheme = make_scheme()
    th0 = np.array([2.0, -3.0, np.log10(50.0), 5.0, -4.0,
                    np.log10(1.2), np.log10(500.0)])
    rng = np.random.default_rng(0)
    # perturbed thetas so each call is a genuinely new geometry
    thetas = [th0 + rng.normal(scale=[0.05, 0.2, 0.05, 0.5, 0.4, 0.08, 0.15])
              for _ in range(N)]

    world = world_from_theta(th0, scheme)

    # ---- remesh forward (AnomalyWorld / pwhg_wrapper) ----
    W.forward(world, th0)                       # warm-up
    t0 = time.time()
    for th in thetas:
        W.forward(world, th)
    t_remesh = (time.time() - t0) / N

    # ---- soft forward (SoftTriForward) ----
    soft = SoftTriForward(world, width=0.2, tri_area=0.25, scheme=scheme)
    soft.forward(th0)                           # warm-up
    t0 = time.time()
    for th in thetas:
        soft.forward(th)
    t_soft = (time.time() - t0) / N

    # ---- soft via the ls_opti path (update + overridden get_forward_solution)
    # This is what ls_opti actually pays: male/msle call update(x) FIRST, which
    # may rebuild geometry even when the forward is soft.
    from seq_sentinel_ls import build_world, attach_soft_forward
    w2 = build_world(scheme, theta_init=th0)
    attach_soft_forward(w2, scheme)
    w2.update(list(th0)); w2.get_forward_solution(_noise=False)   # warm-up
    t0 = time.time()
    for th in thetas:
        w2.update(list(th))
        w2.get_forward_solution(_noise=False)
    t_soft_path = (time.time() - t0) / N

    # ---- remesh via the same ls_opti path, for a like-for-like number ----
    w3 = build_world(scheme, theta_init=th0)
    w3.update(list(th0)); w3.get_forward_solution(_noise=False)
    t0 = time.time()
    for th in thetas:
        w3.update(list(th))
        w3.get_forward_solution(_noise=False)
    t_remesh_path = (time.time() - t0) / N

    print(f"\nper-evaluation cost ({N} calls each):")
    print(f"  remesh  W.forward                 {t_remesh*1000:9.1f} ms")
    print(f"  soft    soft.forward              {t_soft*1000:9.1f} ms"
          f"   ({t_remesh/max(t_soft,1e-12):.2f}x vs remesh)")
    print(f"  remesh  update+get_forward (ls)   {t_remesh_path*1000:9.1f} ms")
    print(f"  soft    update+get_forward (ls)   {t_soft_path*1000:9.1f} ms"
          f"   ({t_remesh_path/max(t_soft_path,1e-12):.2f}x vs remesh)")

    overhead = t_soft_path - t_soft
    print(f"\n  update() overhead on the soft path: {overhead*1000:.1f} ms/eval "
          f"({100*overhead/max(t_soft_path,1e-12):.0f}% of it)")
    if overhead > 0.5 * t_soft_path:
        print("  -> update() dominates: ls_opti pays the GEOMETRY REBUILD even")
        print("     when the forward is soft. That, not the soft solve, is the")
        print("     cost. Fixing it means bypassing update() for the soft path.")

    print("\nnote: multiply by the DE budget to project a run. Stage-1 alone is")
    print("      pop 210 x generations; total evals are tracked by world.n_fev,")
    print("      NOT by res3.nfev (which counts only the stage-3 L-BFGS calls).")


if __name__ == "__main__":
    main()
