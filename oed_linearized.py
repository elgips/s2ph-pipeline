# -*- coding: utf-8 -*-
"""
oed_linearized.py
=================
Gate C1, the cheap way. In the linear-Gaussian regime the posterior covariance is
DATA-INDEPENDENT: given the Jacobian J (at theta0), the per-quadrupole noise, and
the prior, the entire Gamma(n) sequence for any selection order is knowable before
a single measurement. So we can answer "does adaptive quadrupole selection beat a
fixed protocol on the geometric parameters?" now -- no MCMC, no simulated data.

Selection is greedy D-optimal (the linearized EIG for a Gaussian: each step adds
the quadrupole maximizing log det gain = log(1 + j^T Sigma j / eps^2)). We compare:
  greedy-D : adaptive optimal design
  fixed-dd : the standard dipole-dipole ordering (pool order from createData)
  random   : random subset (seed-averaged)
and report Gamma(n) = max_i Var(theta_i|D)/Var(theta_i|prior), plus which
parameter limits Gamma and which quadrupoles greedy picks first.

Caveats carried from validate_soft: Jacobian is at the TRUE theta0 (optimistic;
in practice use the prior-mean Jacobian), the soft blur may slightly deflate
anomaly r/rho information, and the prior is a placeholder. This is the linearized
gate; if it shows an advantage, confirm with the nonlinear posterior later.

Run in ERT_GUI. Needs pwhg_forward_soft.py, pwhg_wrapper.py,
compare_forward_meshes.py, validate_sensitivity.py, validate_soft.py.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pwhg_wrapper as W
from pwhg_forward_soft import SoftTriForward
from validate_soft import choose_steps, PRIOR_STD


def _sequential(J, eps, Pi0, prior_var, order_fn, budget):
    """
    Accumulate the posterior one quadrupole at a time (Sherman-Morrison downdate
    of Sigma). order_fn(Sigma, avail) -> next quadrupole index. Returns Gamma(n),
    logdet-precision(n), per-parameter var-ratio(n), and the selection order.
    """
    nq, p = J.shape
    Sigma = np.linalg.inv(Pi0)
    avail = list(range(nq))
    order = []
    gam = [float(np.max(np.diag(Sigma) / prior_var))]
    ldet = [float(np.linalg.slogdet(np.linalg.inv(Sigma))[1])]
    ratios = [np.diag(Sigma) / prior_var]

    for _ in range(min(budget, nq)):
        V = Sigma @ J.T                              # (p, nq); V[:,q] = Sigma j_q
        jSj = np.einsum("qi,iq->q", J, V)            # (nq,)
        q = order_fn(jSj, eps, avail)
        v = V[:, q]
        Sigma = Sigma - np.outer(v, v) / (eps[q] ** 2 + jSj[q])
        order.append(q); avail.remove(q)
        gam.append(float(np.max(np.diag(Sigma) / prior_var)))
        ldet.append(float(np.linalg.slogdet(np.linalg.inv(Sigma))[1]))
        ratios.append(np.diag(Sigma) / prior_var)

    return dict(gamma=np.array(gam), logdet=np.array(ldet),
                ratios=np.array(ratios), order=order)


def _greedy_D(jSj, eps, avail):
    a = np.array(avail)
    return int(a[np.argmax(jSj[a] / eps[a] ** 2)])


def _fixed(jSj, eps, avail):
    return avail[0]                                  # pool order = standard dd


def _random_order_fn(rng):
    def f(jSj, eps, avail):
        return int(rng.choice(avail))
    return f


def n_to_target(gamma, target):
    hit = np.where(gamma <= target)[0]
    return int(hit[0]) if hit.size else None


def run(world, budget=120, target=0.1, n_random=10, out="oed_c11.png", scheme=None):
    soft = SoftTriForward(world, width=0.2, tri_area=0.25, scheme=scheme)
    theta0 = np.asarray(soft.get_x0(), float)
    labels = soft.param_labels()
    print(f"candidate pool size: {soft.scheme.size()}")

    steps = choose_steps(soft, theta0)
    J, ln0, rhoa0 = W.jacobian_generic(soft.forward, theta0, steps)
    eps = W.noise_std(soft, rhoa0)
    prior_std = np.array([PRIOR_STD[k] for _, k in labels])
    prior_var = prior_std ** 2
    Pi0 = W.prior_precision(prior_std)

    gd = _sequential(J, eps, Pi0, prior_var, _greedy_D, budget)
    fx = _sequential(J, eps, Pi0, prior_var, _fixed, budget)
    rnd = [ _sequential(J, eps, Pi0, prior_var, _random_order_fn(np.random.default_rng(s)),
                        budget) for s in range(n_random) ]
    rnd_gamma = np.mean([r["gamma"] for r in rnd], axis=0)

    ng, nf = n_to_target(gd["gamma"], target), n_to_target(fx["gamma"], target)
    nr = n_to_target(rnd_gamma, target)
    print(f"\nGate C1 on C_1,1 (target Gamma={target}):")
    print(f"  greedy-D reaches target at n = {ng}")
    print(f"  fixed-dd reaches target at n = {nf}")
    print(f"  random   reaches target at n = {nr} (mean of {n_random})")
    if ng and nf:
        print(f"  measurement saving greedy vs fixed: {nf}/{ng} = {nf/ng:.2f}x")
    print(f"  limiting parameter at target (greedy): "
          f"{labels[int(np.argmax(gd['ratios'][ng or -1]))]}")
    print(f"  first 8 greedy quadrupoles (pool idx): {gd['order'][:8]}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(gd["gamma"], label="greedy-D (adaptive)")
    ax[0].semilogy(fx["gamma"], label="fixed dd")
    ax[0].semilogy(rnd_gamma, label=f"random (mean {n_random})")
    ax[0].axhline(target, color="gray", ls="--", lw=1)
    ax[0].set_xlabel("n measurements"); ax[0].set_ylabel("Gamma (max var ratio)")
    ax[0].set_title("Gate C1: Gamma(n)"); ax[0].legend(fontsize=8)

    for j, (nm, kind) in enumerate(labels):
        ax[1].semilogy(gd["ratios"][:, j], label=f"{nm}/{kind}")
    ax[1].set_xlabel("n measurements"); ax[1].set_ylabel("per-param var ratio")
    ax[1].set_title("greedy-D: which parameter limits Gamma"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  figure: {out}")
    return dict(greedy=gd, fixed=fx, random_gamma=rnd_gamma, J=J, eps=eps)


def main():
    from validate_sensitivity import build_c11_world
    from build_pool import build_comprehensive_pool
    world = build_c11_world()
    pool, _ = build_comprehensive_pool(world, u_floor=5e-3)
    run(world, scheme=pool)          # pass scheme=None to compare against dd only


if __name__ == "__main__":
    main()
