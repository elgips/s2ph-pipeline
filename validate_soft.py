# -*- coding: utf-8 -*-
"""
validate_soft.py
================
Confirm the soft-tri forward fixed the geometric sensitivities, then produce the
first Fisher / posterior-Gamma readout on C_1,1 with a trustworthy Jacobian.

Steps:
  1. Per-parameter FD step-convergence on the soft forward -> every geometric
     column should now show an interior valley (contrast the remesh path, which
     never did). Report the chosen step per parameter.
  2. Directional consistency J@v vs FD along v -> catches indexing errors.
  3. Weighted Fisher F = J^T C_d^-1 J with the heteroscedastic ln-space noise;
     report eigenvalues / conditioning and which parameters are (un)informable.
  4. Posterior precision F + Pi0 with a (placeholder) prior; report Gamma and the
     per-parameter posterior/prior std ratios -- the convergence-criterion inputs.

The PRIOR here is a placeholder (problem statement Open Q2) -- replace with the
real prior before reading Gamma as anything but a smoke test.

Run in ERT_GUI. Needs pwhg_forward_soft.py, pwhg_wrapper.py,
compare_forward_meshes.py, validate_sensitivity.py, Anandlyn_log.py.
"""

import numpy as np

import pwhg_wrapper as W
import compare_forward_meshes as cfm
from pwhg_forward_soft import SoftTriForward


# placeholder prior stds in theta units (linear m for x/y; log10 decades for r/rho)
PRIOR_STD = {"x": 5.0, "y": 2.0, "r": 0.3, "rho": 0.7}
# geometric FD step grids per kind (same as the sweep)
STEP_GRID = {"x": np.logspace(-3, -0.3, 9), "y": np.logspace(-3, -0.3, 9),
             "r": np.logspace(-3.5, -0.7, 9), "rho": np.logspace(-3.5, -1, 9)}


def choose_steps(soft, theta0):
    """Per-parameter FD step from step-convergence valleys on the soft forward."""
    rhoa_fwd = lambda th: soft.forward(th)[1]
    labels = soft.param_labels()
    steps = np.zeros(len(labels))
    print("\n[1] FD step convergence on soft forward:")
    for j, (nm, kind) in enumerate(labels):
        sc = cfm.step_convergence(rhoa_fwd, theta0, j, STEP_GRID[kind])
        steps[j] = sc["best_step"]
        print(f"    {nm:<5} {kind:<4} valley={str(sc['has_valley']):<5} "
              f"step={sc['best_step']:.3e}  min_diff={sc['min_diff']:.2e}")
    return steps


def check_directional(soft, theta0, J, steps, n_dirs=5, seed=0):
    rng = np.random.default_rng(seed)
    fwd = soft.forward
    errs = []
    for _ in range(n_dirs):
        v = rng.standard_normal(theta0.size); v /= np.linalg.norm(v)
        st = float(np.median(steps))
        lnp, _ = fwd(theta0 + st * v)
        lnm, _ = fwd(theta0 - st * v)
        fd = (lnp - lnm) / (2 * st)
        errs.append(np.linalg.norm(fd - J @ v) / (np.linalg.norm(fd) + 1e-300))
    errs = np.array(errs)
    print(f"\n[2] directional J@v vs FD: median {np.median(errs):.2e}, "
          f"max {errs.max():.2e}")
    return errs


def report_fisher_gamma(soft, J, rhoa0):
    labels = soft.param_labels()
    eps = W.noise_std(soft, rhoa0)                 # uses soft.scheme
    F = W.fisher(J, eps)
    evals = np.linalg.eigvalsh(0.5 * (F + F.T))
    print("\n[3] Fisher (weighted J^T C_d^-1 J):")
    print(f"    eigenvalues: min {evals.min():.3e}  max {evals.max():.3e}  "
          f"cond {evals.max()/max(evals.min(),1e-300):.2e}")

    prior_std = np.array([PRIOR_STD[k] for _, k in labels])
    Pi0 = W.prior_precision(prior_std)
    post_cov = np.linalg.inv(F + Pi0)
    post_std = np.sqrt(np.clip(np.diag(post_cov), 0, None))
    ratio = post_std / prior_std
    gamma = float(np.max(ratio ** 2))

    print("\n[4] posterior on C_1,1 (placeholder prior):")
    print(f"    {'param':<8} {'prior_std':>10} {'post_std':>10} {'var_ratio':>10}")
    for (nm, kind), ps, qs, r in zip(labels, prior_std, post_std, ratio):
        print(f"    {nm+'/'+kind:<8} {ps:>10.3f} {qs:>10.3f} {r**2:>10.3e}")
    print(f"    Gamma = max var ratio = {gamma:.3e}")
    print("    (Gamma near 1 = parameter barely informed; near 0 = well resolved.)")
    return dict(F=F, post_cov=post_cov, gamma=gamma, eps=eps)


def main():
    from validate_sensitivity import build_c11_world
    world = build_c11_world()
    soft = SoftTriForward(world, width=0.2, tri_area=0.25)
    theta0 = np.asarray(soft.get_x0(), float)
    print("param order:", soft.param_labels())

    steps = choose_steps(soft, theta0)
    J, ln0, rhoa0 = W.jacobian_generic(soft.forward, theta0, steps)
    check_directional(soft, theta0, J, steps)
    report_fisher_gamma(soft, J, rhoa0)


if __name__ == "__main__":
    main()
