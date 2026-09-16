"""
seq_radius_sweep.py -- pin the SIZE threshold with a controlled 1-D experiment.

WHY
The 25-draw prior sweep showed radius, not depth, is the dominant axis. Among
non-straddling draws:

        r <  1.0 :  GEOM ok 2/6    median |dr|/r = 0.34
        r >= 1.0 :  GEOM ok 6/8    median |dr|/r = 0.08

while the depth trend among successes was flat (slope +0.024 pos/r per metre,
corr +0.24, n=8). But those draws varied depth, radius, contrast and seed all at
once, so the size effect is inferred rather than measured.

This isolates it: FIXED depth, FIXED contrast, FIXED seed policy, radius swept.
Repeats per radius give a spread, so the threshold is a measured transition
rather than a single-draw impression.

Scored with article-1's metric (IoU) alongside the geometric errors.

Run:  python seq_radius_sweep.py                  (depth -4.5, 3 repeats)
      python seq_radius_sweep.py -5.5 5           (depth, repeats)
"""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
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
logging.getLogger("Core").setLevel(logging.CRITICAL)

import pwhg_wrapper as W
from seq_coldstart import make_scheme, world_from_theta
from seq_handoff import compute_handoff, DEFAULT_PRIOR_STDS
from seq_greedy import build_pool, run_greedy
from seq_local_update import project_theta
import seq_scene_score as ss
import seq_log

seq_log.VERBOSE = False           # keep the table readable

I_X, I_Y, I_R, I_RHO = 3, 4, 5, 6
RADII = [0.60, 0.80, 1.00, 1.20, 1.60, 2.00]
RHO_TRUE = 500.0                  # fixed contrast: 1.0 dec above the layer
N_STEPS = 50


def truth(depth, r):
    return project_theta(np.array([2.0, -3.0, np.log10(50.0), 5.0, float(depth),
                                   np.log10(float(r)), np.log10(RHO_TRUE)]))


def cv_like_seed(th, rng):
    """Same smeared seed character the CV phase produces."""
    s = th.copy()
    s[I_X] += 0.3 * rng.standard_normal()
    s[I_Y] += 0.4 * rng.standard_normal()
    s[I_R] += 0.35                      # CV inflates the radius
    s[I_RHO] -= 0.55                    # and weakens the contrast
    s[1] += -1.2
    return project_theta(s)


def one(depth, r, rep, pool, dd):
    rng = np.random.default_rng(1000 * rep + int(100 * r))
    th_t = truth(depth, r)
    world = world_from_theta(th_t, dd)
    _, rhoa = W.forward(world, th_t)
    rhoa = np.asarray(rhoa, float)
    eps = np.asarray(W.noise_std(world, rhoa, dU=1e-6, floor=5e-3), float)
    obs = rhoa * np.exp(eps * rng.standard_normal(eps.size))

    hand = compute_handoff(cv_like_seed(th_t, rng), obs, eps, dd,
                           n_gn=8, verbose=False)
    res = run_greedy(hand['theta0'], hand['Sigma0'], th_t, pool,
                     DEFAULT_PRIOR_STDS, data_forward="soft",
                     n_max_steps=N_STEPS, gamma_target=0.10,
                     rng=np.random.default_rng(7), verbose=False)
    th = res['theta']
    err = th - th_t
    r_est = 10 ** th[I_R]
    Q, det = ss.score_pair(th_t, th)
    return dict(r=r, rep=rep, nq=len(res['acquired']),
                truncated=len(res['acquired']) >= N_STEPS,
                pos=float(np.hypot(err[I_X], err[I_Y])),
                pos_rel=float(np.hypot(err[I_X], err[I_Y])) / r,
                r_rel=abs(r_est - r) / r, dy=float(err[I_Y]),
                drho=float(err[I_RHO]), iou=det['q_circle'], Q=Q)


def main():
    depth = float(sys.argv[1]) if len(sys.argv) > 1 else -4.5
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    dd = make_scheme()
    pool, quads, xe = build_pool(n_elec=21, n_max=1200)

    print(f"radius sweep at FIXED depth {depth:+.2f} m, rho={RHO_TRUE:.0f} "
          f"(1.0 dec vs layer), {reps} repeats, budget {N_STEPS} quads")
    print(f"{'r':>6}{'rep':>5}{'nq':>5}{'pos/r':>8}{'dr/r':>8}"
          f"{'|dy|':>8}{'drho':>8}{'IoU':>7}{'flag':>10}")
    rows = []
    for r in RADII:
        for rep in range(reps):
            try:
                o = one(depth, r, rep, pool, dd)
            except Exception as e:
                print(f"{r:>6.2f}{rep:>5}   failed: {type(e).__name__}: {e}")
                continue
            flag = "BUDGET" if o['truncated'] else ""
            print(f"{r:>6.2f}{rep:>5}{o['nq']:>5}{o['pos_rel']:>8.2f}"
                  f"{o['r_rel']:>8.2f}{abs(o['dy']):>8.3f}{o['drho']:>8.3f}"
                  f"{o['iou']:>7.3f}{flag:>10}")
            rows.append(o)
        print()

    print(f"{'r':>6}{'n':>4}{'GEOM ok':>9}{'med dr/r':>10}{'med pos/r':>11}"
          f"{'med IoU':>9}{'med nq':>8}{'budget':>8}")
    for r in RADII:
        g = [o for o in rows if o['r'] == r]
        if not g:
            continue
        ok = sum(1 for o in g if o['pos_rel'] < 0.5 and o['r_rel'] < 0.25)
        print(f"{r:>6.2f}{len(g):>4}{ok:>5}/{len(g):<3}"
              f"{np.median([o['r_rel'] for o in g]):>10.2f}"
              f"{np.median([o['pos_rel'] for o in g]):>11.2f}"
              f"{np.median([o['iou'] for o in g]):>9.3f}"
              f"{np.median([o['nq'] for o in g]):>8.0f}"
              f"{sum(o['truncated'] for o in g):>8}")

    print("\nthreshold:")
    passed = [r for r in RADII
              if (lambda g: g and sum(1 for o in g if o['pos_rel'] < 0.5
                                      and o['r_rel'] < 0.25) > len(g) / 2)
              ([o for o in rows if o['r'] == r])]
    if passed:
        print(f"  radii with a majority of runs recovering geometry: "
              f"{sorted(passed)}")
        print(f"  smallest reliably recovered radius at depth {depth:+.2f}: "
              f"{min(passed):.2f} m")
        below = [r for r in RADII if r < min(passed)]
        if below:
            print(f"  below it ({below}) the radius is NOT recovered -- this is")
            print(f"  the size limit, measured at fixed depth and contrast.")
    else:
        print("  no radius met the criterion -- widen RADII or check the setup.")
    print("\nNOTE: depth and contrast are FIXED here, so any trend is size alone.")
    print("Repeat at another depth (arg 1) to test whether the threshold moves.")


if __name__ == "__main__":
    main()
