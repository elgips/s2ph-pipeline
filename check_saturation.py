"""
check_saturation.py -- does high contrast collapse the (y, r, rho) degeneracy
into a clean two-parameter (y, r) problem?

PHYSICS BEING TESTED
For a compact body in a locally uniform field the secondary response scales as
(size)^d * K, where the polarization factor saturates:
      K = (rho1 - rho0) / (rho1 + rho0)  ->  +1 (perfect insulator: tunnel/air)
                                         ->  -1 (perfect conductor: metal)
and dK/d(ln rho1) -> 0. So at strong contrast the response depends on the body's
SIZE and DEPTH but not on HOW resistive/conductive it is: rho drops out.

PREDICTIONS (all measured below, on the REAL forward, not the analytic formula):
  1. ||d ln(rhoa) / d log10 rho||  falls as ~1/u  (u = rho1/rho0)
  2. corr(C0_r, C0_rho) -> 0        (no coupling if rho has no data effect)
  3. VIF(C0_r) collapses from ~230 to the value set by y-r coupling alone
  4. posterior std of rho -> its PRIOR std (honest 'unidentifiable')
  5. corr(C0_y, C0_r) SURVIVES -- depth-size trade-off is not removed by
     saturation, so the residual problem is genuinely 2-parameter.

Also prints the CONTRAST THRESHOLD: the |log10 u| beyond which rho sensitivity
is below a chosen fraction of the geometric sensitivities -- i.e. where you may
safely FIX rho and invert only (x, y, r).

Run:  python check_saturation.py
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

SHORT = ['w_rho', 'L0_y', 'L0_rho', 'C0_x', 'C0_y', 'C0_r', 'C0_rho']
I_X, I_Y, I_R, I_RHO = 3, 4, 5, 6
RHO_BG = 100.0          # background/host resistivity (world rho = 10^2)


def analytic_K(u):
    """2-D cylinder polarization factor, for reference vs the measured trend."""
    return (u - 1.0) / (u + 1.0)


def fisher_pack(J, eps, prior_stds):
    """Posterior covariance, correlations and VIFs from a Jacobian."""
    Wd = 1.0 / np.asarray(eps, float) ** 2
    P0 = np.diag(1.0 / np.asarray(prior_stds, float) ** 2)
    H = P0 + (np.asarray(J).T * Wd) @ np.asarray(J)
    S = np.linalg.inv(H)
    d = np.sqrt(np.clip(np.diag(S), 1e-300, None))
    C = S / np.outer(d, d)
    vif = np.diag(S) * np.diag(np.linalg.inv(S))
    return S, C, vif, d


def main():
    scheme = make_scheme()
    # contrast grid: conductive (metal) .. host .. resistive (tunnel/air)
    log_u = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0.7, 1, 1.5, 2, 2.5, 3])
    prior_stds = np.array([0.5, 5.0, 0.02, 15.0, 8.0, 0.7, 1.2])

    print(f"host rho0 = {RHO_BG:.0f} ohm.m   circle at (x=5, y=-4), r=1.2 m")
    print(f"{'log10 u':>8}{'rho1':>10}{'K_analytic':>12}"
          f"{'|dJ/dlog10rho|':>16}{'rel_to_r':>10}"
          f"{'corr(r,rho)':>13}{'corr(y,r)':>11}"
          f"{'VIF(r)':>9}{'VIF(rho)':>10}{'std_rho':>9}")

    rows = []
    for lu in log_u:
        u = 10.0 ** lu
        rho1 = RHO_BG * u
        th = np.array([np.log10(RHO_BG), -3.0, np.log10(50.0),
                       5.0, -4.0, np.log10(1.2), np.log10(rho1)])
        world = world_from_theta(th, scheme)
        soft = SoftTriForward(world, width=0.2, tri_area=0.25, scheme=scheme)
        steps = np.asarray(W.default_steps(world), float)
        J, ln0, rhoa0 = W.jacobian_generic(soft.forward, th, steps)
        J = np.asarray(J, float)
        eps = np.asarray(W.noise_std(world, np.asarray(rhoa0, float),
                                     dU=1e-6, floor=5e-3), float)
        S, C, vif, sd = fisher_pack(J, eps, prior_stds)

        n_rho = np.linalg.norm(J[:, I_RHO])
        n_r = np.linalg.norm(J[:, I_R])
        rows.append(dict(lu=lu, u=u, n_rho=n_rho, n_r=n_r,
                         rel=n_rho / max(n_r, 1e-30),
                         c_r_rho=C[I_R, I_RHO], c_y_r=C[I_Y, I_R],
                         vif_r=vif[I_R], vif_rho=vif[I_RHO],
                         std_rho=sd[I_RHO], std_r=sd[I_R], std_y=sd[I_Y]))
        print(f"{lu:>8.1f}{rho1:>10.3g}{analytic_K(u):>12.3f}"
              f"{n_rho:>16.4f}{n_rho/max(n_r,1e-30):>10.3f}"
              f"{C[I_R, I_RHO]:>13.3f}{C[I_Y, I_R]:>11.3f}"
              f"{vif[I_R]:>9.1f}{vif[I_RHO]:>10.1f}{sd[I_RHO]:>9.3f}")

    print("\n--- interpretation ---")
    base = [r for r in rows if abs(r['lu'] - 0.7) < 1e-9]
    if base:
        b = base[0]
        print(f"reference (your C11, u=5): |dJ/dlog10rho|={b['n_rho']:.4f}  "
              f"corr(r,rho)={b['c_r_rho']:+.3f}  VIF(r)={b['vif_r']:.1f}")

    # 1/u decay check on the resistive branch
    res = [r for r in rows if r['lu'] >= 1]
    if len(res) >= 2:
        x = np.log10([r['u'] for r in res])
        y = np.log10([max(r['n_rho'], 1e-30) for r in res])
        slope = np.polyfit(x, y, 1)[0]
        print(f"resistive branch: d log||dJ/drho|| / d log u = {slope:+.2f}"
              f"   (physics predicts about -1)")

    # contrast threshold: where rho sensitivity drops below 10% of radius's
    print("\ncontrast thresholds (rho sensitivity relative to radius):")
    for frac in (0.20, 0.10, 0.05):
        ok_res = [r for r in rows if r['lu'] > 0 and r['rel'] < frac]
        ok_con = [r for r in rows if r['lu'] < 0 and r['rel'] < frac]
        sr = f"u > 10^{min(r['lu'] for r in ok_res):.1f}" if ok_res else "not reached"
        sc = f"u < 10^{max(r['lu'] for r in ok_con):.1f}" if ok_con else "not reached"
        print(f"  rel < {frac:.2f}:  resistive {sr:<16} conductive {sc}")

    print("\nVERDICT")
    hi = [r for r in rows if abs(r['lu']) >= 2]
    if hi:
        print(f"  at |log10 u| >= 2:  mean |corr(r,rho)| = "
              f"{np.mean([abs(r['c_r_rho']) for r in hi]):.3f}   "
              f"mean VIF(r) = {np.mean([r['vif_r'] for r in hi]):.1f}")
        print(f"                      mean |corr(y,r)|   = "
              f"{np.mean([abs(r['c_y_r']) for r in hi]):.3f}  "
              f"<- SHOULD SURVIVE (depth-size trade-off)")
        print(f"                      std_rho -> {np.mean([r['std_rho'] for r in hi]):.3f} "
              f"(prior was {prior_stds[I_RHO]:.2f}; approaching it = unidentifiable)")
    print("  If corr(r,rho)->0 and VIF(r) collapses while corr(y,r) persists,")
    print("  the saturated problem is genuinely 2-parameter (y, r): FIX rho and")
    print("  invert only (x, y, r). Do NOT report a rho estimate there.")
    print("  NOTE: saturation removes the DEGENERACY, not the signal strength --")
    print("  small/deep targets stay weak, so the depth limit persists.")


if __name__ == "__main__":
    main()
