"""
check_cov.py -- inspect the posterior covariance structure.

QUESTION (El'ad): are the circle's geometric params strongly coupled to the
layer depth L0_y (and to the rho's)? If so, excluding L0_y from the greedy
target block may be wrong.

TECHNICAL NOTE: the {x,y,r,rho} sub-block of Sigma IS the marginal covariance
over those params, so it already ACCOUNTS for uncertainty in the excluded ones
(greedy will pin L0_y if that reduces the target's marginal variance). What the
sub-block does NOT do is value resolving L0_y for its own sake -- which matters
if the layer interface is itself a sharp edge you care about.

WHAT THIS PRINTS
  1. correlation matrix at the handoff and after greedy (labelled, flagged)
  2. strong pairs |corr| > 0.6
  3. VARIANCE INFLATION FACTOR per param:
        VIF_i = Sigma_ii * (Sigma^-1)_ii  =  marginal var / conditional var
     VIF=1 -> that param's uncertainty is its own.
     VIF>>1 -> its uncertainty is dominated by COUPLING to other params, i.e.
               the data constrains only a combination, not the param alone.
  4. the effect of the tight L0_log10rho prior: re-solves with the layer rho
     FREED, to see whether pinning it is what pushes error into the geometry.

Run:  python check_cov.py
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

from seq_coldstart import make_scheme
from seq_handoff import (read_cv_seed, load_dat, compute_handoff,
                         THETA_NAMES, DEFAULT_PRIOR_STDS)

SHORT = ['w_rho', 'L0_y', 'L0_rho', 'C0_x', 'C0_y', 'C0_r', 'C0_rho']


def corr_from_cov(S):
    d = np.sqrt(np.clip(np.diag(S), 1e-300, None))
    return S / np.outer(d, d)


def show_corr(S, title):
    C = corr_from_cov(S)
    print(f"\n--- correlation matrix: {title} ---")
    print("        " + "".join(f"{n:>8}" for n in SHORT))
    for i, n in enumerate(SHORT):
        row = "".join(f"{C[i, j]:>8.2f}" for j in range(len(SHORT)))
        print(f"{n:>8}{row}")
    print("\nstrong pairs |corr| > 0.6:")
    found = False
    for i in range(len(SHORT)):
        for j in range(i + 1, len(SHORT)):
            if abs(C[i, j]) > 0.6:
                found = True
                print(f"    {SHORT[i]:>7} ~ {SHORT[j]:<7} corr = {C[i, j]:+.3f}")
    if not found:
        print("    (none)")
    return C


def show_vif(S, title):
    """VIF_i = Sigma_ii * (Sigma^-1)_ii = marginal var / conditional var."""
    try:
        P = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(S)
    vif = np.diag(S) * np.diag(P)
    print(f"\n--- variance inflation ({title}) ---")
    print("  VIF=1: uncertainty is the param's own.  VIF>>1: dominated by")
    print("  coupling -- the data constrains a COMBINATION, not the param.")
    for i, n in enumerate(SHORT):
        bar = "#" * int(min(40, np.log10(max(vif[i], 1.0)) * 12))
        print(f"    {n:>7}  std={np.sqrt(S[i,i]):>8.4f}  VIF={vif[i]:>10.1f}  {bar}")
    return vif


def main():
    import config as C
    scene = "c11"
    dd = make_scheme()
    seed = read_cv_seed(C.DIR_CV / "parameters.xlsx", scene)
    rhoa_obs, err = load_dat(C.DIR_FWD / f"{scene.upper()}_noisy.dat")

    print("=" * 68)
    print("A) HANDOFF with the pipeline's tight layer-rho prior (as we run it)")
    print("=" * 68)
    h1 = compute_handoff(seed, rhoa_obs, err, dd, n_gn=8, verbose=False)
    print("prior stds:", dict(zip(SHORT, np.round(h1['prior_stds'], 3))))
    print("theta0    :", dict(zip(SHORT, np.round(h1['theta0'], 3))))
    C1 = show_corr(h1['Sigma0'], "handoff, L0_rho pinned (std 0.02)")
    v1 = show_vif(h1['Sigma0'], "L0_rho pinned")

    print("\n" + "=" * 68)
    print("B) SAME but with the LAYER RHO FREED (prior std 0.02 -> 0.5)")
    print("   If pinning L0_rho is pushing misfit into the geometry, freeing it")
    print("   should CHANGE the geometric correlations / VIFs materially.")
    print("=" * 68)
    ps2 = np.array(DEFAULT_PRIOR_STDS, float).copy()
    ps2[2] = 0.5                     # free the layer resistivity
    h2 = compute_handoff(seed, rhoa_obs, err, dd, prior_stds=ps2,
                         n_gn=8, verbose=False)
    print("theta0    :", dict(zip(SHORT, np.round(h2['theta0'], 3))))
    C2 = show_corr(h2['Sigma0'], "handoff, L0_rho FREE (std 0.5)")
    v2 = show_vif(h2['Sigma0'], "L0_rho free")

    print("\n" + "=" * 68)
    print("C) VERDICT on the target-block question")
    print("=" * 68)
    pairs = [(4, 1, "C0_y ~ L0_y   (circle depth vs layer depth)"),
             (4, 2, "C0_y ~ L0_rho (circle depth vs layer resistivity)"),
             (5, 6, "C0_r ~ C0_rho (the known radius-contrast degeneracy)"),
             (4, 5, "C0_y ~ C0_r   (depth vs size)"),
             (6, 0, "C0_rho ~ w_rho(contrast vs background)")]
    print(f"{'pair':<46}{'pinned':>10}{'freed':>10}")
    for i, j, lab in pairs:
        print(f"  {lab:<44}{C1[i, j]:>10.3f}{C2[i, j]:>10.3f}")

    print("\n  geometric VIFs (pinned -> freed):")
    for i in [3, 4, 5, 6]:
        print(f"    {SHORT[i]:>7}: {v1[i]:>9.1f} -> {v2[i]:>9.1f}")

    print("\n  READ:")
    print("   * |corr(C0_y, L0_y)| > 0.6  => circle depth and layer depth are")
    print("     confounded; L0_y SHOULD join the target block (it is also a")
    print("     sharp edge), otherwise greedy never resolves the pair.")
    print("   * VIF(C0_y) >> 1 => its uncertainty is coupling-dominated, i.e.")
    print("     the data pins a combination, not the depth itself.")
    print("   * big pinned->freed changes => the tight L0_rho prior is bending")
    print("     geometry to absorb layer-resistivity misfit.")


if __name__ == "__main__":
    main()
