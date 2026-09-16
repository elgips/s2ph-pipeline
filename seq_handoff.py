"""
seq_handoff.py -- assemble the handoff state (piece 2b).

Reads the CV seed from parameters.xlsx and the eps cold-start data from the
{scene}_noisy.dat, refines the seed with a few weighted Gauss-Newton steps on
the cold-start quads (re-linearizing J each step), and returns
(theta_hat0, Sigma0) plus a per-parameter report of what the cold-start
actually constrains -- the input the greedy loop consumes.

Design notes:
  * xlsx values are LINEAR; rho/r are converted to log10 to match the 7-param
    theta of SoftTriForward: [world_log10rho, L0_y, L0_log10rho,
    C0_x, C0_y, C0_log10r, C0_log10rho].
  * The pipeline FIXES the top-layer rho (=50), but SoftTriForward keeps it as
    the 7th free param. We bridge this by a TIGHT prior on L0_log10rho, so it
    stays put without editing SoftTriForward.
  * Prior mean = the CV seed (cold-start has no other geometric anchor). The
    prior is broad on geometric params (near-uninformative), so params the
    data can't see stay at the seed with large posterior std -> greedy targets.
  * n_gn>=1 by default: the CV seed is NOT the weighted-PWHG MAP, so we let GN
    pull it toward the MAP on the cold-start data. How far each param moves,
    and its posterior std, is the depth-limit diagnostic.
"""
import numpy as np
import openpyxl
from pygimli.physics import ert
import pwhg_wrapper as W
from pwhg_forward_soft import SoftTriForward
from seq_coldstart import make_scheme, world_from_theta
from seq_local_update import gn_laplace_update, project_theta
from seq_log import Log

# xlsx column -> theta slot, with which columns are log10-converted.
PARAM_COLS = ['rho bg', 'layer1 y', 'layer1 rho', 'c1 x', 'c1 y', 'c1 r', 'c1 rho']
LOG10_MASK = [True, False, True, False, False, True, True]
THETA_NAMES = ['world_log10rho', 'L0_y', 'L0_log10rho',
               'C0_x', 'C0_y', 'C0_log10r', 'C0_log10rho']

# broad cold-start prior stds (in theta units). L0_log10rho tight -> ~fixed.
DEFAULT_PRIOR_STDS = np.array([0.50,   # world_log10rho
                               5.0,    # L0_y  (m)
                               0.02,   # L0_log10rho  (tight -> pins layer rho)
                               15.0,   # C0_x  (m)
                               8.0,    # C0_y  (m)
                               0.70,   # C0_log10r
                               1.20])  # C0_log10rho


def _clean(v):
    if v is None or (isinstance(v, str) and v.strip() == '-'):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_cv_seed(xlsx_path, scene='c11'):
    """Parse parameters.xlsx -> 7-param theta (log10 on rho/r).

    The sheet has two stacked blocks (world-rho range, then the CV estimates).
    Mirror load_cij_table: locate the block whose 2nd header cell is 'rho bg'.
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    hdr = next((i for i, r in enumerate(rows)
                if r and len(r) > 1 and r[1] == 'rho bg'), None)
    if hdr is None:
        raise ValueError(f"no 'rho bg' header row in {xlsx_path}")
    key = scene.strip().lower()
    seed = None
    for r in rows[hdr + 1:]:
        if r and r[0] is not None and str(r[0]).strip().lower() == key:
            seed = r
            break
    if seed is None:
        raise ValueError(f"scene {scene!r} not found under the CV block")
    vals = [_clean(v) for v in seed[1:14]]
    rho_bg, l1_y, l1_rho = vals[0], vals[1], vals[2]
    c1_x, c1_y, c1_r, c1_rho = vals[5], vals[6], vals[7], vals[8]
    for nm, v in [('rho_bg', rho_bg), ('layer1_y', l1_y), ('layer1_rho', l1_rho),
                  ('c1_x', c1_x), ('c1_y', c1_y), ('c1_r', c1_r), ('c1_rho', c1_rho)]:
        if v is None:
            raise ValueError(f"scene {scene!r}: missing '{nm}' in xlsx")
    return np.array([np.log10(rho_bg), l1_y, np.log10(l1_rho),
                     c1_x, c1_y, np.log10(c1_r), np.log10(c1_rho)], float)


def load_dat(path):
    """Return (rhoa_obs, err) from a pipeline .dat."""
    d = ert.load(str(path))
    return np.asarray(d['rhoa'], float), np.asarray(d['err'], float)


def compute_handoff(theta_seed, rhoa_obs, err, scheme,
                    prior_stds=DEFAULT_PRIOR_STDS, n_gn=8,
                    sigma_model=0.02, cond_cap=1e8, damping=1e-3,
                    verbose=False):
    """Refine the seed on the cold-start data and return the handoff state.

    sigma_model : ln-space model-discrepancy std (SoftTri surrogate vs remesh
                  physics). Combined with the datum noise as
                  eps_eff^2 = err^2 + sigma_model^2. REQUIRED at low noise,
                  where the ~2% blur exceeds the ~0.5% measurement noise; set 0
                  only if inference and data share the same forward.
    cond_cap    : eigenvalue-floor ratio on the Laplace Hessian. Large here
                  (1e8) so post_std reflects TRUE conditioning -- weak
                  directions (r, rho for a deep/small anomaly) show large
                  post_std instead of being clamped. The loop can use a tighter
                  cap for stability.

    Returns dict: theta0 (refined MAP), Sigma0, seed, prior_stds, post_stds,
    report (list of per-param dicts), and the gn_laplace_update diagnostics.
    """
    lg = Log("handoff")
    lg.stage("build soft + jacobian on cold-start data")
    lg.info("CV seed theta", theta_seed)
    lg.info(f"m={np.asarray(rhoa_obs).size}  sigma_model={sigma_model}  n_gn={n_gn}")
    world = world_from_theta(theta_seed, scheme)
    soft = SoftTriForward(world, width=0.2, tri_area=0.25, scheme=scheme)
    f_ln = lambda th: np.asarray(soft.forward(th))[0]     # (m,) natural-log rhoa
    steps = W.default_steps(world)
    # jacobian_generic returns (J, ln0, rhoa0); J is (m,7) = d(ln rhoa)/d theta.
    jac = lambda th: W.jacobian_generic(soft.forward, th, steps)[0]

    d_ln = np.log(rhoa_obs)
    noise_var = np.asarray(err, float) ** 2 + float(sigma_model) ** 2
    m0 = theta_seed.copy()
    P0 = np.diag(1.0 / np.asarray(prior_stds, float) ** 2)

    lg.stage("warm GN/Laplace refine")
    out = gn_laplace_update(theta_seed, d_ln, noise_var, m0, P0,
                            f_ln, jac, n_gn=n_gn, damping=damping,
                            cond_cap=cond_cap, verbose=verbose,
                            project=project_theta)
    post_stds = np.sqrt(np.clip(np.diag(out['Sigma']), 0, None))
    lg.check("whitened resid ~ 1", out['resid_rms'], lo=0.3, hi=3.0)
    lg.info("theta0 (refined)", out['theta'])
    lg.info("post_stds", post_stds)
    lg.done("handoff (theta0, Sigma0) ready")

    report = []
    for i, nm in enumerate(THETA_NAMES):
        report.append(dict(param=nm, seed=theta_seed[i], refined=out['theta'][i],
                           delta=out['theta'][i] - theta_seed[i],
                           prior_std=prior_stds[i], post_std=post_stds[i],
                           shrink=post_stds[i] / prior_stds[i]))
    return dict(theta0=out['theta'], Sigma0=out['Sigma'], seed=theta_seed,
                prior_stds=np.asarray(prior_stds, float), post_stds=post_stds,
                report=report, diag=out, world=world, soft=soft)


def print_report(res, true_theta=None):
    print(f"\nhandoff: gn_iters={res['diag']['gn_iters']} "
          f"converged={res['diag']['converged']} "
          f"whit_rms={res['diag']['resid_rms']:.3f} cond={res['diag']['cond']:.1e}")
    hdr = f"{'param':<14}{'seed':>9}{'refined':>9}"
    if true_theta is not None:
        hdr += f"{'true':>9}"
    hdr += f"{'prior_sd':>9}{'post_sd':>9}{'shrink':>8}"
    print(hdr)
    for i, r in enumerate(res['report']):
        line = f"{r['param']:<14}{r['seed']:>9.3f}{r['refined']:>9.3f}"
        if true_theta is not None:
            line += f"{true_theta[i]:>9.3f}"
        line += f"{r['prior_std']:>9.3f}{r['post_std']:>9.3f}{r['shrink']:>8.2f}"
        print(line)
    print("shrink = post_sd/prior_sd; near 1 => cold-start did NOT constrain "
          "this param (greedy's job); << 1 => localized by cold-start")


if __name__ == "__main__":
    import config as C
    scene = "c11"
    scheme = make_scheme()

    xlsx = C.DIR_CV / "parameters.xlsx"
    dat = C.DIR_FWD / f"{scene.upper()}_noisy.dat"
    print(f"reading seed  : {xlsx}")
    print(f"reading data  : {dat}")

    theta_seed = read_cv_seed(xlsx, scene)
    rhoa_obs, err = load_dat(dat)
    print(f"seed theta    : {np.round(theta_seed, 3)}")
    print(f"m={rhoa_obs.size}  err range [{err.min():.4f},{err.max():.4f}]")

    # true theta for the nominal C11 (for the report only)
    true_theta = np.array([2.0, -3.0, np.log10(50), 5.0, -4.0,
                           np.log10(1.2), np.log10(500)])

    res = compute_handoff(theta_seed, rhoa_obs, err, scheme, n_gn=8, verbose=True)
    print_report(res, true_theta=true_theta)

    # Sigma0 is the object the greedy loop consumes, alongside theta0.
    print("\nSigma0 diag:", np.round(np.diag(res['Sigma0']), 4))
