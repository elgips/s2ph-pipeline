"""
seq_local_update.py -- warm-started Gauss-Newton / Laplace local update.

This is the load-bearing primitive of the sequential adaptive loop. Given the
current geometric estimate theta_hat, a Gaussian prior N(m0, P0^{-1}), and the
set of quadrupoles acquired so far with their observed data, it does a few
Gauss-Newton iterations to the local MAP -- RE-LINEARIZING the Jacobian at each
iteration -- and returns the Laplace covariance at the optimum.

Two roles, one function:
  * per-step refinement in the loop (n_gn >= 1, warm-started from previous theta)
  * handoff covariance Sigma_0 (n_gn = 0 evaluates the Laplace cov at the seed
    WITHOUT moving it, or n_gn >= 1 to also refine the CV/DE seed on the
    cold-start data). This replaces the placeholder prior-std diagonal.

Design decisions, stated explicitly:
  * All residuals are in NATURAL-LOG resistivity space. A relative datum error
    eps_i (= 0.005 + |dU * k_i / rhoa_i|) maps to a ln-space std of eps_i,
    because d ln(rhoa) = d(rhoa)/rhoa = relative error. Pass noise as ln-space
    variances s2 (length m). If you work in log10 elsewhere, convert:
    s2_ln = (ln10 * s2_log10). Keep forward() and jacobian() in the SAME base.
  * forward(theta) -> y  (length m, ln rhoa) and jacobian(theta) -> J (m x p,
    d ln rhoa / d theta) are PASSED IN. Wire SoftTriForward + jacobian_generic
    here; the primitive itself is physics-agnostic and unit-testable.
  * The data-generating forward (reality) and the inference forward may differ
    (remesh vs SoftTriForward). That model mismatch (~1.9%, below the 0.5%+
    noise floor) enters only through the residual and is handled like noise.
    This primitive does NOT assume they are the same.
  * Non-PD GN Hessian in soft directions: eigenvalue floor at H_max / COND_CAP
    (COND_CAP = 1e3), so Sigma variance in near-null directions is capped rather
    than infinite. This is the practical form of "fall back to conditional
    precision, cap at 1e3".

The primitive is deterministic given its inputs and holds no state.
"""
import numpy as np

LN10 = np.log(10.0)
COND_CAP = 1.0e3          # max condition number allowed on the GN Hessian
DEFAULT_MAX_GN = 5
DEFAULT_STEP_CAP = None   # optional per-parameter absolute step cap (array or None)



# ---------------------------------------------------------------------------
# Article-1 containment constraints (varlims + circle fully inside the domain):
#     y + r <= YTOP,  y - r >= YMIN,  x + r <= XMAX,  x - r >= XMIN
# These are NONLINEAR in theta (r = 10**log10r), so a box clamp cannot express
# them; we project instead. Lives here because both the handoff and the greedy
# loop need it and this module imports neither of them.
# ---------------------------------------------------------------------------
XMIN, XMAX, YTOP, YMIN = -25.0, 25.0, 0.0, -20.0
RMIN, RMAX = 0.5, 5.0


def project_theta(th, lo=None, hi=None):
    """Box-clamp then enforce circle containment. Returns a valid theta."""
    t = np.array(th, float)
    if lo is not None and hi is not None:
        t = np.clip(t, lo, hi)
    t[3] = np.clip(t[3], XMIN + RMIN, XMAX - RMIN)          # C0_x
    t[4] = np.clip(t[4], YMIN + RMIN, YTOP - RMIN)          # C0_y
    r = 10.0 ** t[5]
    r_max = min(YTOP - t[4], t[4] - YMIN, XMAX - t[3], t[3] - XMIN, RMAX)
    r = float(np.clip(r, RMIN, max(r_max, RMIN)))
    t[5] = np.log10(r)
    return t


def _psd_inverse(H, cond_cap=COND_CAP):
    """Symmetric inverse with an eigenvalue floor.

    Floors eigenvalues at lam_max / cond_cap so soft directions get large-but-
    finite variance instead of blowing up. Returns (Sigma, H_reg, cond_used).
    """
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    lam_max = float(w[-1])
    if not np.isfinite(lam_max) or lam_max <= 0:
        # degenerate: fall back to a scaled identity precision
        lam_max = 1.0
    floor = lam_max / cond_cap
    w_reg = np.clip(w, floor, None)
    H_reg = (V * w_reg) @ V.T
    Sigma = (V * (1.0 / w_reg)) @ V.T
    cond_used = float(w_reg[-1] / w_reg[0])
    return 0.5 * (Sigma + Sigma.T), 0.5 * (H_reg + H_reg.T), cond_used


def gn_laplace_update(theta_init, data, noise_var,
                      prior_mean, prior_prec,
                      forward, jacobian,
                      n_gn=DEFAULT_MAX_GN, step_cap=DEFAULT_STEP_CAP,
                      damping=0.0, tol=1e-8, cond_cap=COND_CAP, verbose=False,
                      project=None):
    """One local update: warm-started GN to the MAP + Laplace covariance.

    Parameters
    ----------
    theta_init  : (p,) starting point (previous theta_hat, or the CV/DE seed).
    data        : (m,) observed ln rhoa for the acquired quads, from the TRUE
                  field / faithful forward + noise. Order matches forward()'s
                  output and noise_var.
    noise_var   : (m,) ln-space measurement variances s_i^2. If noise depends
                  on predicted rhoa, evaluate it outside and pass the frozen
                  vector -- kept fixed within a call for a well-defined
                  quadratic subproblem.
    prior_mean  : (p,) prior mean m0, in the SAME coordinates as theta (circle
                  x,y linear; log10 r, log10 rho; layer/world per your
                  convention -- must match forward/jacobian).
    prior_prec  : (p,p) prior precision P0 = Sigma_prior^{-1}. Small diagonal
                  for a near-uninformative cold-start handoff.
    forward     : callable theta -> y (m,)   predicted ln rhoa for the acquired
                  quads.
    jacobian    : callable theta -> J (m,p)  d(ln rhoa)/d theta at theta.
    n_gn        : GN iterations. 0 = evaluate the Laplace cov at the seed
                  WITHOUT moving it (handoff Sigma_0 use).
    project     : optional callable theta -> theta enforcing domain/containment
                  constraints after each step. WITHOUT it the handoff is
                  unconstrained and can leave the domain entirely (the 25-draw
                  sweep produced x = -29.7 m and +16.2 m on a +/-25 m electrode
                  line). Pass seq_greedy.project_theta.

    Returns
    -------
    dict: theta (MAP), Sigma (Laplace cov, eigen-floored), precision (the
    regularized GN Hessian inverted), resid_rms (whitened), gn_iters,
    converged, cond, step_last.
    """
    theta = np.asarray(theta_init, float).copy()
    m0 = np.asarray(prior_mean, float)
    P0 = np.asarray(prior_prec, float)
    d = np.asarray(data, float)
    s2 = np.asarray(noise_var, float)
    p = theta.size
    W = 1.0 / s2                       # (m,) diagonal data precision

    converged = False
    step_last = np.nan
    J = np.asarray(jacobian(theta), float)   # linearize at the start point

    for it in range(max(int(n_gn), 0)):
        g = np.asarray(forward(theta), float)
        J = np.asarray(jacobian(theta), float)          # RE-LINEARIZE here
        r = d - g                                        # ln-space residual
        # GN gradient of the negative log-posterior:
        grad = P0 @ (theta - m0) - J.T @ (W * r)
        # GN Hessian (+ optional Levenberg damping):
        H = P0 + (J.T * W) @ J
        if damping > 0:
            H = H + damping * np.diag(np.diag(H))
        Sigma, H_reg, cond = _psd_inverse(H, cond_cap)
        delta = -Sigma @ grad
        if step_cap is not None:
            cap = np.asarray(step_cap, float)
            delta = np.clip(delta, -cap, cap)
        theta = theta + delta
        if project is not None:
            theta = np.asarray(project(theta), float)
        step_last = float(np.linalg.norm(delta))
        if verbose:
            wr = (d - np.asarray(forward(theta), float)) * np.sqrt(W)
            print(f"  gn {it+1}: |step|={step_last:.3e} "
                  f"whit_rms={np.sqrt(np.mean(wr**2)):.3e} cond={cond:.2e}")
        if step_last < tol:
            converged = True
            break

    # Final Laplace at the (possibly unmoved, if n_gn=0) point.
    g = np.asarray(forward(theta), float)
    J = np.asarray(jacobian(theta), float)
    H = P0 + (J.T * W) @ J
    if damping > 0:
        H = H + damping * np.diag(np.diag(H))
    Sigma, H_reg, cond = _psd_inverse(H, cond_cap)
    r = d - g
    whit = r * np.sqrt(W)
    return dict(
        theta=theta, Sigma=Sigma, precision=H_reg,
        resid_rms=float(np.sqrt(np.mean(whit**2))) if whit.size else 0.0,
        gn_iters=int(it + 1) if n_gn > 0 else 0,
        converged=bool(converged), cond=float(cond),
        step_last=float(step_last),
    )


# ----------------------------------------------------------------------
# Standalone self-test: toy forward, no pygimli. Verifies the mechanics
# independent of the physics. El'ad can then swap in SoftTriForward +
# jacobian_generic and rerun the same checks against known synthetic truth.
# ----------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    p, m = 4, 40
    theta_true = np.array([1.0, -0.5, 0.3, -0.2])
    # A mildly nonlinear forward so GN re-linearization actually matters:
    #   y_i = A_i . theta + 0.15 * (b_i . theta)^2
    A = rng.normal(size=(m, p))
    b = rng.normal(size=(m, p)) * 0.3

    def forward(th):
        lin = A @ th
        return lin + 0.15 * (b @ th) ** 2

    def jacobian(th):
        # d/dth [ A th + 0.15 (b th)^2 ] = A + 0.3 (b th) b
        return A + 0.3 * (b @ th)[:, None] * b

    s = 0.02
    noise_var = np.full(m, s ** 2)
    d = forward(theta_true) + rng.normal(scale=s, size=m)

    # near-uninformative prior, warm start away from truth
    m0 = np.zeros(p)
    P0 = np.eye(p) * 1e-3
    theta0 = theta_true + rng.normal(scale=0.5, size=p)

    out = gn_laplace_update(theta0, d, noise_var, m0, P0,
                                 forward, jacobian, n_gn=8, verbose=True)
    err = np.linalg.norm(out["theta"] - theta_true)
    # theta should land near truth; whitened residual ~1; cov should be small.
    print("\nself-test results")
    print(f"  ||theta_hat - theta_true|| = {err:.4e}")
    print(f"  whitened resid rms         = {out['resid_rms']:.4f} (expect ~1)")
    print(f"  Sigma trace                = {np.trace(out['Sigma']):.3e}")
    print(f"  cond(H)                    = {out['cond']:.2e}")
    print(f"  gn_iters / converged       = {out['gn_iters']} / {out['converged']}")

    # covariance sanity: adding a redundant measurement must not INCREASE
    # any posterior variance (information is monotone).
    var_before = np.diag(out["Sigma"]).copy()
    A2 = np.vstack([A, A[:5]]); b2 = np.vstack([b, b[:5]])

    def forward2(th):
        lin = A2 @ th
        return lin + 0.15 * (b2 @ th) ** 2

    def jacobian2(th):
        return A2 + 0.3 * (b2 @ th)[:, None] * b2

    d2 = np.concatenate([d, d[:5]])
    nv2 = np.concatenate([noise_var, noise_var[:5]])
    out2 = gn_laplace_update(out["theta"], d2, nv2, m0, P0,
                                  forward2, jacobian2, n_gn=8)
    var_after = np.diag(out2["Sigma"])
    monotone = np.all(var_after <= var_before + 1e-12)
    print(f"  variance monotone under added data: {monotone}")

    # n_gn=0 must return the seed unmoved but a finite Laplace cov (handoff use)
    out0 = gn_laplace_update(theta0, d, noise_var, m0, P0,
                                  forward, jacobian, n_gn=0)
    unmoved = np.allclose(out0["theta"], theta0)
    finite_cov = np.all(np.isfinite(out0["Sigma"]))
    print(f"  n_gn=0 leaves seed unmoved / finite cov: {unmoved} / {finite_cov}")

    ok = (err < 0.05) and monotone and unmoved and finite_cov
    print(f"\nSELF-TEST {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
