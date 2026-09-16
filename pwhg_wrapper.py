# -*- coding: utf-8 -*-
"""
pwhg_wrapper.py
===============
Computational primitive for the PWHG OED phase (Priority 1 remainder).

Wraps an existing AnomalyWorld (from Anandlyn_log) and exposes:

    forward(world, theta)            -> (ln_rhoa, rhoa)
    jacobian(world, theta, steps)    -> J = d ln(rhoa) / d theta   (n_data, n_param)
    noise_std(world, rhoa)           -> eps  (per-datum std in ln-rhoa space)
    fisher(J, eps)                   -> J^T C_d^-1 J
    posterior_precision(J, eps, Pi0) -> J^T C_d^-1 J + prior_precision
    gamma_infnorm(post_cov, var0)    -> prior-normalized inf-norm  Gamma

Conventions (fixed and documented so nothing downstream has to guess):

  * DATA space is d = ln(rhoa). The error model eps_i = 0.005 + |dU/u_i| is a
    RELATIVE error on rhoa, hence an ADDITIVE std on ln(rhoa) to first order
    (d ln rhoa = d rhoa / rhoa). So C_d = diag(eps_i^2) with no state coupling.

  * PARAMETER space theta is exactly what AnomalyWorld.get_x0()/update() use:
    LINEAR for anomaly/layer position (x, y in metres),
    LOG10   for radius r and every resistivity rho.
    The returned Fisher/precision are therefore in this mixed geometric-parameter
    space directly (which is also the space you want normalized for Gamma).

  * The Jacobian is central finite difference with a FRESH base-point copy per
    perturbation and full state restoration afterwards. This is the fix for the
    reset bug in AnomalyWorld.sensitivity_analysis (earlier columns were being
    evaluated at a base point drifted by -eps in every prior coordinate).

Nothing here is executed on your behalf — run it in ERT_GUI.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Parameter bookkeeping (mirrors get_x0 / update ordering exactly)
# ---------------------------------------------------------------------------
def param_labels(world):
    """
    Return a list of (name, kind) for the varying parameters, in the SAME order
    as world.get_x0(). kind in {'x','y','r','rho'}. Used for per-kind FD steps
    and for readable reporting. If this list ever desyncs from get_x0(), the
    Jacobian columns are mislabeled -- so this is deliberately a literal copy of
    the get_x0 walk, not an independent re-derivation.
    """
    labels = []
    if world.rho_world_varflag:
        labels.append(("world", "rho"))
    for i, layer in enumerate(world.layers):
        if layer.varflag[0]:
            labels.append((f"L{i}", "y"))
        if layer.varflag[1]:
            labels.append((f"L{i}", "rho"))
    for i, circle in enumerate(world.Circles):
        if circle.varflag[0]:
            labels.append((f"C{i}", "x"))
        if circle.varflag[1]:
            labels.append((f"C{i}", "y"))
        if circle.varflag[2]:
            labels.append((f"C{i}", "r"))
        if circle.varflag[3]:
            labels.append((f"C{i}", "rho"))
    return labels


def default_steps(world, pos_step=1e-2, log_step=5e-3):
    """
    Per-parameter absolute FD step, keyed by kind. These are DEFAULTS ONLY and
    are unvalidated until check_step_convergence has been run on this world.

      * position (x, y): absolute metres. Default 1e-2 m. Beware: this triggers
        a remesh, so too-small a step is dominated by discretization jitter.
      * r, rho (log10): step in decades. Default 5e-3 decade (~1.2% linear).

    Returns an array aligned with get_x0().
    """
    steps = []
    for _, kind in param_labels(world):
        steps.append(pos_step if kind in ("x", "y") else log_step)
    return np.asarray(steps, float)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
def forward(world, theta):
    """
    Evaluate the PWHG forward model at theta (in get_x0 units).
    Returns (ln_rhoa, rhoa) as float arrays of length n_data.
    Noise is OFF (deterministic response), as required for a Jacobian.
    """
    world.update(list(np.asarray(theta, float)))
    data = world.get_forward_solution(_noise=False)
    rhoa = np.asarray(data["rhoa"], float)
    if np.any(rhoa <= 0):
        raise ValueError("Non-positive rhoa in forward response; cannot take ln. "
                         "Check the model / mesh at this theta.")
    return np.log(rhoa), rhoa


# ---------------------------------------------------------------------------
# Jacobian  (d ln rhoa / d theta), central FD, base point reset per column
# ---------------------------------------------------------------------------
def jacobian(world, theta=None, steps=None, scalar_step=None, restore=True):
    """
    Central finite-difference Jacobian of ln(rhoa) w.r.t. theta.

    Parameters
    ----------
    theta : array or None
        Base point. If None, uses world.get_x0().
    steps : array or None
        Per-parameter absolute step (aligned with get_x0). If None, uses
        default_steps(world). Ignored if scalar_step is given.
    scalar_step : float or None
        If set, use this single step for every parameter. Only for reproducing /
        comparing against the old scalar-eps behaviour; not recommended for real
        use because the parameters are dimensionally heterogeneous.
    restore : bool
        Restore world to `theta` after differencing (default True).

    Returns
    -------
    J    : (n_data, n_param)  ndarray,  d ln(rhoa) / d theta
    ln0  : (n_data,)          ln(rhoa) at the base point
    rhoa0: (n_data,)          rhoa at the base point
    steps: (n_param,)         the steps actually used
    """
    theta0 = np.asarray(world.get_x0() if theta is None else theta, float).copy()
    n = theta0.size

    if scalar_step is not None:
        steps = np.full(n, float(scalar_step))
    elif steps is None:
        steps = default_steps(world)
    steps = np.asarray(steps, float)
    if steps.size != n:
        raise ValueError(f"steps length {steps.size} != n_param {n}")

    ln0, rhoa0 = forward(world, theta0)
    m = ln0.size
    J = np.zeros((m, n))

    for i in range(n):
        # FRESH copies -- this is the reset-bug fix. Every column is evaluated
        # at the true base point theta0, not a drifted one.
        tp = theta0.copy(); tp[i] += steps[i]
        tm = theta0.copy(); tm[i] -= steps[i]
        lnp, _ = forward(world, tp)
        lnm, _ = forward(world, tm)
        J[:, i] = (lnp - lnm) / (2.0 * steps[i])

    if restore:
        world.update(list(theta0))
    return J, ln0, rhoa0, steps


def jacobian_generic(fwd, theta0, steps):
    """
    Backend-agnostic central-FD Jacobian. `fwd(theta) -> (ln_rhoa, rhoa)` is any
    forward callable: the remesh world (`lambda th: forward(world, th)`) or the
    soft-tri forward (`SoftTriForward.forward`). The soft forward is stateless
    (rebuilds resistivity on a fixed mesh each call), so no restore is needed.

    Returns J (n_data, n_param) = d ln(rhoa)/d theta, plus ln0, rhoa0.
    """
    theta0 = np.asarray(theta0, float).copy()
    n = theta0.size
    steps = np.asarray(steps, float)
    if steps.size != n:
        raise ValueError(f"steps length {steps.size} != n_param {n}")
    ln0, rhoa0 = fwd(theta0)
    J = np.zeros((ln0.size, n))
    for i in range(n):
        tp = theta0.copy(); tp[i] += steps[i]
        tm = theta0.copy(); tm[i] -= steps[i]
        lnp, _ = fwd(tp)
        lnm, _ = fwd(tm)
        J[:, i] = (lnp - lnm) / (2.0 * steps[i])
    return J, ln0, rhoa0


def directional_derivative(world, theta, v, step):
    """
    Central FD of ln(rhoa) along an arbitrary unit-ish direction v in theta space.
    Used by the harness for a single global consistency check (J @ v vs this),
    which catches column ordering / indexing errors that per-column checks miss.
    """
    theta0 = np.asarray(theta, float)
    v = np.asarray(v, float)
    lnp, _ = forward(world, theta0 + step * v)
    lnm, _ = forward(world, theta0 - step * v)
    out = (lnp - lnm) / (2.0 * step)
    world.update(list(theta0))
    return out


# ---------------------------------------------------------------------------
# Noise model  ->  per-datum std in ln-rhoa space
# ---------------------------------------------------------------------------
def geometric_factors(scheme):
    """
    Return the geometric factors k for the scheme (needed to turn rhoa into the
    measured potential u = R = rhoa / k at I = 1 A). Tries the cached 'k' first,
    then pygimli's constructors across versions.
    """
    try:
        k = np.asarray(scheme["k"], float)
        if k.size == scheme.size() and np.all(np.isfinite(k)) and np.any(k != 0):
            return k
    except Exception:
        pass
    import pygimli.physics.ert as ert
    for fn in ("createGeometricFactors", "geometricFactors"):
        f = getattr(ert, fn, None)
        if f is not None:
            try:
                return np.asarray(f(scheme), float)
            except Exception:
                continue
    raise RuntimeError("Could not obtain geometric factors k from the scheme.")


def noise_std(world, rhoa, dU=1e-3, floor=0.005):
    """
    eps_i = floor + |dU * k_i / rhoa_i|,  interpreted as the std of ln(rhoa_i).

    With I = 1 A the measured potential is u_i = R_i = rhoa_i / k_i, so
    |dU / u_i| = |dU * k_i / rhoa_i|. This is the ln-space C_d^{1/2} diagonal.

    dU defaults to 1e-3 V (primary case); pass 1e-4 for the secondary comparison.
    """
    k = geometric_factors(world.scheme)
    rhoa = np.asarray(rhoa, float)
    return floor + np.abs(dU * k / rhoa)


# ---------------------------------------------------------------------------
# Fisher / posterior precision / Gamma
# ---------------------------------------------------------------------------
def fisher(J, eps):
    """Weighted Fisher information  J^T C_d^{-1} J,  C_d = diag(eps^2)."""
    J = np.asarray(J, float)
    w = 1.0 / np.asarray(eps, float) ** 2          # (n_data,)
    return (J * w[:, None]).T @ J                   # (n_param, n_param)


def prior_precision(stds):
    """Diagonal prior precision from per-parameter prior stds (in theta units)."""
    return np.diag(1.0 / np.asarray(stds, float) ** 2)


def posterior_precision(J, eps, prior_prec):
    """Laplace / linear-Gaussian posterior precision  F + Pi0."""
    return fisher(J, eps) + np.asarray(prior_prec, float)


def posterior_cov(J, eps, prior_prec):
    return np.linalg.inv(posterior_precision(J, eps, prior_prec))


def gamma_infnorm(post_cov, prior_var):
    """
    Gamma = max_i Var(theta_i | D) / Var(theta_i | prior).
    prior_var is the vector of prior variances (diag of prior covariance).
    """
    return float(np.max(np.diag(post_cov) / np.asarray(prior_var, float)))
