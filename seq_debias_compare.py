"""
seq_debias_compare.py -- does polishing the SoftTri estimate on the REMESH
forward remove the surrogate bias, and which polish?

Takes the SoftTri-loop estimate theta_hat (from the handoff) and runs four arms,
all against the faithful remesh forward (W.forward), then compares PARAMETER
error in the weak (r, rho) directions -- NOT the loss.

  (c)  none            theta_hat as-is (baseline)
  (a1) weighted DE     short elite-exploit, eps-whitened objective, tight box
  (a2) unweighted DE   same DE, plain SSE (article-1 msle style)
  (b)  remesh-GN       few Gauss-Newton steps, remesh FD Jacobian (staircase)

Why no sigma_model here: the remesh forward IS the data-generating forward, so
there is no surrogate discrepancy to absorb -- the polish can reach theta_true if
the data constrains it. (The SoftTri handoff needed sigma_model; the polish does
not.)

Hypotheses tested:
  H1 (DE-polish helps):    a1 weak-dir error  <  c
  H2 (article-1 reproduced): b  weak-dir error >= c   (GN on staircase fails)
  H3 (weighting matters):  a1 weak-dir error  <  a2

Cost warning: every DE eval is one remesh solve. Defaults are modest
(popsize=8, maxiter=10 -> up to ~560 solves per DE arm). Bump for a finer polish.

Run in pygimli_env. Needs seq_handoff, seq_coldstart, seq_local_update,
pwhg_wrapper, pwhg_forward_soft, Anandlyn_log, config.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
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

import time
import numpy as np
from scipy.optimize import differential_evolution
import pwhg_wrapper as W
from seq_coldstart import make_scheme, world_from_theta
from seq_local_update import gn_laplace_update
from seq_handoff import read_cv_seed, load_dat, compute_handoff, THETA_NAMES

# indices of the weak directions in the 7-param theta
I_R, I_RHO = 5, 6            # C0_log10r, C0_log10rho
# DE search half-widths per param (theta units); clipped to varlims below
HALF = np.array([0.20, 0.30, 0.05, 0.50, 0.50, 0.50, 1.00])
# WIDE box for an article-1-style global DE (search from CV-scale bounds)
HALF_WIDE = np.array([0.50, 3.00, 0.05, 8.00, 4.00, 0.90, 2.50])
# hard varlim clips (log10 r in [log10 0.5, log10 5]; log10 rho in [0,5])
LO = np.array([0.0, -10.0, 0.0, -25.0, -10.0, np.log10(0.5), 0.0])
HI = np.array([5.0,  -1.0, 5.0,  25.0,   0.0, np.log10(5.0), 5.0])

# scaled DE (was popsize=8,maxiter=10 -- too small; article-1 uses hundreds)
DE_KW = dict(popsize=20, maxiter=40, tol=1e-4, mutation=(0.5, 1.0),
             recombination=0.7, polish=False, seed=0, init="sobol")


def remesh_forward_factory(world):
    """f_ln(theta)->(m,) natural-log rhoa via the faithful remesh forward.
    NOTE mutates `world` in place; call serially."""
    def f_ln(theta):
        ln, _ = W.forward(world, np.asarray(theta, float))
        return np.asarray(ln, float)
    return f_ln


def de_polish(theta_hat, d_ln, noise_var, world, weighted, half=HALF,
              x0=None, **de_kw):
    f_ln = remesh_forward_factory(world)
    W_ = 1.0 / noise_var if weighted else np.ones_like(noise_var)

    def obj(theta):
        r = d_ln - f_ln(theta)
        return float(np.sum(W_ * r * r))

    bounds = list(zip(np.maximum(theta_hat - half, LO),
                      np.minimum(theta_hat + half, HI)))
    t0 = time.time()
    kw = dict(de_kw)
    if x0 is not None:
        kw["x0"] = x0            # scipy>=1.9 injects x0 into initial population
    res = differential_evolution(obj, bounds, **kw)
    return res.x, float(res.fun), int(res.nfev), time.time() - t0


def gn_polish(theta_hat, d_ln, noise_var, world, n_gn=4):
    f_ln = remesh_forward_factory(world)
    steps = W.default_steps(world)
    jac = lambda th: W.jacobian_generic(lambda t: W.forward(world, t), th, steps)[0]
    P0 = np.diag(1.0 / (HALF ** 2))          # broad-ish, centered at theta_hat
    t0 = time.time()
    out = gn_laplace_update(theta_hat, d_ln, noise_var, theta_hat, P0,
                            f_ln, jac, n_gn=n_gn, damping=1e-3, cond_cap=1e8)
    return out["theta"], out, time.time() - t0


def weighted_sse(theta, d_ln, noise_var, world):
    ln, _ = W.forward(world, np.asarray(theta, float))
    r = d_ln - np.asarray(ln, float)
    return float(np.sum(r * r / noise_var))


def errs(theta, theta_true):
    e = np.asarray(theta, float) - np.asarray(theta_true, float)
    return dict(full=float(np.linalg.norm(e)),
                r=float(abs(e[I_R])), rho=float(abs(e[I_RHO])))


def run(theta_hat, theta_true, d_ln, noise_var, scheme, de_kw=DE_KW, gn_iters=4):
    # each arm gets its OWN world (mutation isolation)
    w_c = world_from_theta(theta_hat, scheme)   # only for scoring / obj
    arms = {}

    # (c) none
    arms["(c) none"] = dict(theta=theta_hat.copy(),
                            phi=weighted_sse(theta_hat, d_ln, noise_var, w_c),
                            nfev=0, t=0.0)

    # (a1) weighted DE
    w_a1 = world_from_theta(theta_hat, scheme)
    th, phi, nfev, t = de_polish(theta_hat, d_ln, noise_var, w_a1, True, **de_kw)
    arms["(a1) weighted DE"] = dict(theta=th,
        phi=weighted_sse(th, d_ln, noise_var, w_a1), nfev=nfev, t=t)

    # (a2) unweighted DE
    w_a2 = world_from_theta(theta_hat, scheme)
    th, phi, nfev, t = de_polish(theta_hat, d_ln, noise_var, w_a2, False, **de_kw)
    arms["(a2) unweighted DE"] = dict(theta=th,
        phi=weighted_sse(th, d_ln, noise_var, w_a2), nfev=nfev, t=t)

    # (a3) WIDE weighted DE -- article-1-style global search from CV-scale bounds
    w_a3 = world_from_theta(theta_hat, scheme)
    th, phi, nfev, t = de_polish(theta_hat, d_ln, noise_var, w_a3, True,
                                 half=HALF_WIDE, **de_kw)
    arms["(a3) wide weighted DE"] = dict(theta=th,
        phi=weighted_sse(th, d_ln, noise_var, w_a3), nfev=nfev, t=t)

    # (d) DE seeded AT theta_true, narrow box -> does it STAY at truth (Phi~151)
    #     or drift up to ~547? distinguishes "valley too flat to descend" from
    #     "truth is not actually the minimum".
    w_d = world_from_theta(theta_hat, scheme)
    th, phi, nfev, t = de_polish(theta_true, d_ln, noise_var, w_d, True,
                                 half=HALF * 0.2, x0=theta_true, **de_kw)
    arms["(d) DE from truth"] = dict(theta=th,
        phi=weighted_sse(th, d_ln, noise_var, w_d), nfev=nfev, t=t)

    # (b) remesh-GN
    w_b = world_from_theta(theta_hat, scheme)
    th, out, t = gn_polish(theta_hat, d_ln, noise_var, w_b, n_gn=gn_iters)
    arms["(b) remesh-GN"] = dict(theta=th,
        phi=weighted_sse(th, d_ln, noise_var, w_b), nfev=gn_iters, t=t,
        gn_whit=out["resid_rms"])

    return arms


def report(arms, theta_true, theta_hat, d_ln=None, noise_var=None, scheme=None):
    # sanity: is truth the loss minimum? score Phi_w at theta_true and theta_hat
    if d_ln is not None:
        w_t = world_from_theta(theta_true, scheme)
        phi_true = weighted_sse(theta_true, d_ln, noise_var, w_t)
        w_h = world_from_theta(theta_hat, scheme)
        phi_hat = weighted_sse(theta_hat, d_ln, noise_var, w_h)
        print(f"\nSANITY  Phi_w(theta_true) = {phi_true:.2f}   "
              f"Phi_w(theta_hat) = {phi_hat:.2f}")
        print("  if a polish reaches Phi_w BELOW Phi_w(theta_true) with WORSE "
              "params -> genuine degeneracy; if truth is lowest -> search was the issue")

    print(f"\n{'arm':<22}{'||dtheta||':>11}{'|dr| (r)':>11}{'|drho|':>11}"
          f"{'Phi_w':>12}{'time[s]':>9}")
    base = errs(theta_hat, theta_true)
    for name, a in arms.items():
        e = errs(a["theta"], theta_true)
        tag = ""
        if name != "(c) none":
            dr = "v" if e["r"] < base["r"] - 1e-6 else ("^" if e["r"] > base["r"]+1e-6 else "=")
            dp = "v" if e["rho"] < base["rho"] - 1e-6 else ("^" if e["rho"] > base["rho"]+1e-6 else "=")
            tag = f"  r{dr} rho{dp}"
        print(f"{name:<22}{e['full']:>11.4f}{e['r']:>11.4f}{e['rho']:>11.4f}"
              f"{a['phi']:>12.2f}{a['t']:>9.1f}{tag}")

    a1, a2, b, c = (arms["(a1) weighted DE"], arms["(a2) unweighted DE"],
                    arms["(b) remesh-GN"], arms["(c) none"])
    e1, e2, eb, ec = (errs(a1["theta"], theta_true), errs(a2["theta"], theta_true),
                      errs(b["theta"], theta_true), errs(c["theta"], theta_true))
    wk = lambda e: e["r"] + e["rho"]     # combined weak-direction error
    print("\nverdicts (weak-direction error = |dr| + |drho|):")
    print(f"  H1 DE-polish helps      : a1 {wk(e1):.3f} {'<' if wk(e1)<wk(ec) else '>='} none {wk(ec):.3f}"
          f"   -> {'SUPPORTED' if wk(e1) < wk(ec) else 'not supported'}")
    print(f"  H2 article-1 reproduced : b  {wk(eb):.3f} {'>=' if wk(eb)>=wk(ec)-1e-6 else '<'} none {wk(ec):.3f}"
          f"   -> {'SUPPORTED (GN polish fails)' if wk(eb) >= wk(ec)-1e-6 else 'not supported (GN helped)'}")
    print(f"  H3 weighting matters    : a1 {wk(e1):.3f} {'<' if wk(e1)<wk(e2) else '>='} a2 {wk(e2):.3f}"
          f"   -> {'SUPPORTED' if wk(e1) < wk(e2) else 'not supported'}")
    if "gn_whit" in b:
        print(f"  (remesh-GN whitened rms = {b['gn_whit']:.2f}; large/erratic => staircase Jacobian)")
    # loss-vs-parameter divergence check (the article-1 point)
    print("\nloss vs parameter error (does lower Phi mean better theta?):")
    for name, a in arms.items():
        e = errs(a["theta"], theta_true)
        print(f"  {name:<20} Phi_w={a['phi']:8.2f}   weak-err={wk(e):.3f}")


if __name__ == "__main__":
    import config as C
    scene = "c11"
    scheme = make_scheme()

    # 1) get the SoftTri handoff estimate theta_hat
    seed = read_cv_seed(C.DIR_CV / "parameters.xlsx", scene)
    rhoa_obs, err = load_dat(C.DIR_FWD / f"{scene.upper()}_noisy.dat")
    hand = compute_handoff(seed, rhoa_obs, err, scheme, n_gn=8, verbose=False)
    theta_hat = hand["theta0"]

    # 2) ground truth (nominal C11); pass the actual true theta for prior draws
    theta_true = np.array([2.0, -3.0, np.log10(50), 5.0, -4.0,
                           np.log10(1.2), np.log10(500)])

    d_ln = np.log(rhoa_obs)
    noise_var = np.asarray(err, float) ** 2        # NO sigma_model: remesh polish

    print(f"theta_hat (SoftTri): {np.round(theta_hat,3)}")
    print(f"theta_true         : {np.round(theta_true,3)}")
    print(f"m={rhoa_obs.size}  DE: popsize={DE_KW['popsize']} maxiter={DE_KW['maxiter']}")

    arms = run(theta_hat, theta_true, d_ln, noise_var, scheme)
    report(arms, theta_true, theta_hat, d_ln, noise_var, scheme)
