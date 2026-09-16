# -*- coding: utf-8 -*-
"""
oed_c1.py
=========
Gate C1, corrected. Two fixes over oed_linearized.main:

  (1) Step selection decoupled from pool composition. The FD step is a property
      of f(theta) smoothness (validated at width 0.2), NOT of which configs we
      read out. On the full comprehensive pool, choose_steps was aggregating the
      FD successive-difference over ~18k mostly-low-SNR configs, reporting false
      valley=False and picking steps ~4x too small -> a noisy J that greedy-D
      then games (it maximizes j^T Sigma j, so it selects FD noise). Here steps
      are chosen on the HIGH-SNR subset (eps < thresh), where valleys are clean,
      and reused for the whole pool.

  (2) Honest fixed baseline. "fixed dd" now = the actual 171-quad dipole-dipole
      protocol scored in its own order, not the first n of the enumeration order
      of the comprehensive pool (which is pathologically clustered).

Comparison: greedy-D on the comprehensive pool vs standard-dd protocol vs
random-from-pool. Believe greedy-vs-random and the per-parameter story; this
run is what makes the greedy-vs-dd magnitude quotable.

Run in ERT_GUI. Needs oed_linearized.py, build_pool.py, pwhg_forward_soft.py,
pwhg_wrapper.py, validate_soft.py, validate_sensitivity.py.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pwhg_wrapper as W
import compare_forward_meshes as cfm
from pwhg_forward_soft import SoftTriForward
from validate_soft import PRIOR_STD, STEP_GRID
from oed_linearized import _sequential, _greedy_D, _fixed, _random_order_fn, n_to_target


def choose_steps_hisnr(soft, theta0, eps, thresh=0.05):
    """FD steps chosen using only high-SNR configs (clean valleys)."""
    mask = eps < thresh
    if mask.sum() < 20:
        mask = eps <= np.quantile(eps, 0.1)
    fwd_sub = lambda th: soft.forward(th)[1][mask]
    labels = soft.param_labels()
    steps = np.zeros(len(labels))
    print(f"\nstep selection on {int(mask.sum())} high-SNR configs (eps<{thresh}):")
    for j, (nm, kind) in enumerate(labels):
        sc = cfm.step_convergence(fwd_sub, theta0, j, STEP_GRID[kind])
        steps[j] = sc["best_step"]
        print(f"    {nm:<5} {kind:<4} valley={str(sc['has_valley']):<5} "
              f"step={sc['best_step']:.3e}")
    return steps


def run(world, pool, budget=120, target=0.1, n_random=10, out="oed_c1.png"):
    labels = W.param_labels(world)
    theta0 = np.asarray(world.get_x0(), float)
    prior_std = np.array([PRIOR_STD[k] for _, k in labels])
    prior_var = prior_std ** 2
    Pi0 = W.prior_precision(prior_std)

    # --- comprehensive pool: base forward, noise, clean steps, J ---
    soft = SoftTriForward(world, width=0.2, scheme=pool)
    _, rhoa0 = soft.forward(theta0)
    eps = W.noise_std(soft, rhoa0)
    steps = choose_steps_hisnr(soft, theta0, eps)
    J, _, _ = W.jacobian_generic(soft.forward, theta0, steps)
    print(f"pool size {J.shape[0]}; steps reused for full pool.")

    # --- standard dd protocol: its own J, scored in dd order ---
    soft_dd = SoftTriForward(world, width=0.2, scheme=None)   # dd
    _, rhoa_dd = soft_dd.forward(theta0)
    eps_dd = W.noise_std(soft_dd, rhoa_dd)
    J_dd, _, _ = W.jacobian_generic(soft_dd.forward, theta0, steps)

    gd = _sequential(J, eps, Pi0, prior_var, _greedy_D, budget)
    rnd = [_sequential(J, eps, Pi0, prior_var,
                       _random_order_fn(np.random.default_rng(s)), budget)
           for s in range(n_random)]
    rnd_gamma = np.mean([r["gamma"] for r in rnd], axis=0)
    fx = _sequential(J_dd, eps_dd, Pi0, prior_var, _fixed, J_dd.shape[0])

    ng = n_to_target(gd["gamma"], target)
    nf = n_to_target(fx["gamma"], target)
    nr = n_to_target(rnd_gamma, target)
    print(f"\nGate C1 (target Gamma={target}):")
    print(f"  greedy-D (comprehensive)  n = {ng}")
    print(f"  standard dd protocol      n = {nf}  (pool {J_dd.shape[0]})")
    print(f"  random from pool          n = {nr}  (mean {n_random})")
    if ng and nf:
        print(f"  saving greedy vs dd: {nf}/{ng} = {nf/ng:.2f}x")
    lim = labels[int(np.argmax(gd["ratios"][ng or -1]))]
    print(f"  limiting parameter at target (greedy): {lim}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(gd["gamma"], label="greedy-D (comprehensive)")
    ax[0].semilogy(fx["gamma"], label="standard dd")
    ax[0].semilogy(rnd_gamma, label=f"random from pool (mean {n_random})")
    ax[0].axhline(target, color="gray", ls="--", lw=1)
    ax[0].set_xlabel("n measurements"); ax[0].set_ylabel("Gamma (max var ratio)")
    ax[0].set_title("Gate C1: Gamma(n)"); ax[0].legend(fontsize=8)
    for j, (nm, kind) in enumerate(labels):
        ax[1].semilogy(gd["ratios"][:, j], label=f"{nm}/{kind}")
    ax[1].set_xlabel("n measurements"); ax[1].set_ylabel("per-param var ratio")
    ax[1].set_title("greedy-D: which parameter limits Gamma"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  figure: {out}")
    return dict(greedy=gd, fixed_dd=fx, random_gamma=rnd_gamma)


def main():
    from validate_sensitivity import build_c11_world
    from build_pool import build_comprehensive_pool
    world = build_c11_world()
    pool, _ = build_comprehensive_pool(world, u_floor=5e-3)
    run(world, pool)


if __name__ == "__main__":
    main()
