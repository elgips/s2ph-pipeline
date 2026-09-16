"""
check_fix_rho.py -- does FIXING rho actually improve (r, y) recovery?

CONTEXT
check_saturation.py refuted the tidy 'saturation decouples rho' story: at high
contrast rho becomes UNIDENTIFIABLE (std -> prior) yet stays ~0.997 correlated
with r. So the argument for fixing rho is NOT that it decouples -- it is that
estimating an unidentifiable parameter which is locked to r lets r drift along
the ridge. Fixing rho breaks that correlation by fiat.

That is a CLAIM, and this script tests it instead of assuming it. For several
contrasts it compares, on the SAME data:

  arm A  free-rho   : estimate (w_rho, L0_y, L0_rho, x, y, r, rho)   [7 param]
  arm B  fixed-rho  : same but rho PINNED at an assumed value (tight prior),
                      so only (x, y, r) + nuisances are estimated       [6 free]
  arm C  fixed-WRONG: rho pinned at a value that is WRONG by 1 decade --
                      the honest risk of fixing it, since in the field you do
                      not know the true contrast.

Scored on what matters: |dr| and |dy| (geometry), NOT on loss.

If B beats A on |dr|/|dy| at high contrast, fixing rho helps. If C is much
worse than B, the benefit depends on guessing rho well -- which is the caveat
that decides whether this is usable in practice.

Run:  python check_fix_rho.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
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
from pwhg_forward_soft import SoftTriForward
from seq_coldstart import make_scheme, world_from_theta
from seq_local_update import gn_laplace_update

I_X, I_Y, I_R, I_RHO = 3, 4, 5, 6
RHO_BG = 100.0
BASE_PRIOR = np.array([0.5, 5.0, 0.02, 15.0, 8.0, 0.7, 1.2])


def make_truth(log_u, r_true=1.2, y_true=-4.0, x_true=5.0):
    return np.array([np.log10(RHO_BG), -3.0, np.log10(50.0),
                     x_true, y_true, np.log10(r_true),
                     np.log10(RHO_BG * 10.0 ** log_u)])


def gen_data(theta_true, scheme, rng, dU=1e-6, floor=5e-3):
    world = world_from_theta(theta_true, scheme)
    soft = SoftTriForward(world, width=0.2, tri_area=0.25, scheme=scheme)
    ln = np.asarray(soft.forward(theta_true))[0]
    rhoa = np.exp(ln)
    eps = np.asarray(W.noise_std(world, rhoa, dU=dU, floor=floor), float)
    d = ln + eps * rng.standard_normal(eps.size)
    return d, eps


def perturbed_seed(theta_true, rng):
    """A CV-like handoff seed: position roughly right, shape smeared."""
    s = theta_true.copy()
    s[I_X] += 0.3 * rng.standard_normal()
    s[I_Y] += 0.4 * rng.standard_normal()
    s[I_R] += 0.30                      # radius inflated
    s[I_RHO] -= 0.50                     # contrast weakened
    s[1] += -0.8                         # layer depth offset
    return s


def fit(theta_seed, d, eps, scheme, prior_stds, sigma_model=0.01, n_gn=10):
    world = world_from_theta(theta_seed, scheme)
    soft = SoftTriForward(world, width=0.2, tri_area=0.25, scheme=scheme)
    steps = np.asarray(W.default_steps(world), float)
    f_ln = lambda th: np.asarray(soft.forward(th))[0]
    jac = lambda th: W.jacobian_generic(soft.forward, th, steps)[0]
    nv = np.asarray(eps, float) ** 2 + sigma_model ** 2
    P0 = np.diag(1.0 / np.asarray(prior_stds, float) ** 2)
    out = gn_laplace_update(theta_seed, d, nv, theta_seed.copy(), P0,
                            f_ln, jac, n_gn=n_gn, damping=1e-3, cond_cap=1e8)
    return out


def main():
    scheme = make_scheme()
    rng = np.random.default_rng(0)
    contrasts = [0.7, 1.0, 1.5, 2.0, 2.5, 3.0]      # log10 u; 0.7 = C11 (u=5)
    n_rep = 3                                        # seeds per contrast

    print("arms:  A free-rho   B rho fixed at TRUTH   C rho fixed 1 decade WRONG")
    print("scored on |dr| (log10 radius err) and |dy| (depth err, m)\n")
    print(f"{'log10u':>7}{'|dr| A':>9}{'|dr| B':>9}{'|dr| C':>9}"
          f"{'|dy| A':>9}{'|dy| B':>9}{'|dy| C':>9}{'B wins?':>9}")

    summary = []
    for lu in contrasts:
        th_t = make_truth(lu)
        eA = {'r': [], 'y': []}; eB = {'r': [], 'y': []}; eC = {'r': [], 'y': []}
        for rep in range(n_rep):
            d, eps = gen_data(th_t, scheme, rng)
            seed = perturbed_seed(th_t, rng)

            # A: rho free
            pA = BASE_PRIOR.copy()
            oA = fit(seed, d, eps, scheme, pA)

            # B: rho pinned at the TRUE value (tight prior), seed set there
            pB = BASE_PRIOR.copy(); pB[I_RHO] = 0.01
            sB = seed.copy(); sB[I_RHO] = th_t[I_RHO]
            oB = fit(sB, d, eps, scheme, pB)

            # C: rho pinned at a WRONG value (1 decade off)
            pC = BASE_PRIOR.copy(); pC[I_RHO] = 0.01
            sC = seed.copy(); sC[I_RHO] = th_t[I_RHO] - 1.0
            oC = fit(sC, d, eps, scheme, pC)

            for o, e in ((oA, eA), (oB, eB), (oC, eC)):
                e['r'].append(abs(o['theta'][I_R] - th_t[I_R]))
                e['y'].append(abs(o['theta'][I_Y] - th_t[I_Y]))

        mA_r, mB_r, mC_r = (np.median(eA['r']), np.median(eB['r']),
                            np.median(eC['r']))
        mA_y, mB_y, mC_y = (np.median(eA['y']), np.median(eB['y']),
                            np.median(eC['y']))
        win = "YES" if (mB_r < mA_r and mB_y <= mA_y * 1.1) else \
              ("part" if mB_r < mA_r or mB_y < mA_y else "no")
        print(f"{lu:>7.1f}{mA_r:>9.3f}{mB_r:>9.3f}{mC_r:>9.3f}"
              f"{mA_y:>9.3f}{mB_y:>9.3f}{mC_y:>9.3f}{win:>9}")
        summary.append((lu, mA_r, mB_r, mC_r, mA_y, mB_y, mC_y))

    print("\n--- verdict ---")
    hi = [s for s in summary if s[0] >= 1.5]
    if hi:
        aR = np.mean([s[1] for s in hi]); bR = np.mean([s[2] for s in hi])
        cR = np.mean([s[3] for s in hi])
        aY = np.mean([s[4] for s in hi]); bY = np.mean([s[5] for s in hi])
        cY = np.mean([s[6] for s in hi])
        print(f"high contrast (u >= 10^1.5):")
        print(f"  |dr|:  free {aR:.3f}   fixed-true {bR:.3f}   fixed-wrong {cR:.3f}")
        print(f"  |dy|:  free {aY:.3f}   fixed-true {bY:.3f}   fixed-wrong {cY:.3f}")
        if bR < aR * 0.8:
            print("  => FIXING rho materially improves radius recovery.")
        elif bR < aR:
            print("  => fixing rho helps radius modestly.")
        else:
            print("  => fixing rho does NOT help; the y-r coupling dominates.")
        if cR > bR * 1.5:
            print("  => but the benefit DEPENDS on guessing rho well: a 1-decade")
            print("     error costs more than freeing rho would have.")
        else:
            print("  => and it is robust to a 1-decade error in the assumed rho.")
    print("\nreminder: corr(y,r)~0.87 is untouched by any of this; the depth-size")
    print("trade-off is the residual obstacle regardless of how rho is handled.")


if __name__ == "__main__":
    main()
