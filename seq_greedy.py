"""
seq_greedy.py -- greedy EIG acquisition loop (piece 3).

Consumes the handoff (theta_hat0, Sigma0) and repeatedly:
  1. re-linearizes J at the current theta_hat over a candidate pool,
  2. selects the quadrupole with the largest D-optimal EIG,
     score_i = 0.5*log(1 + j_i Sigma j_i^T / eps_i^2),
  3. "acquires" it (data-gen forward at theta_true + eps noise),
  4. does one warm Gauss-Newton step over ALL acquired data (re-linearized J,
     reused from selection -> no extra solves) -> new (theta_hat, Sigma),
  5. tracks Gamma = max_i Var_i / prior_var_i; stops at target.

Why D-optimal targets the right thing: maximizing log-det gain most reduces the
LARGEST posterior-variance direction, which for this problem is rho (then r) --
the radius-contrast degeneracy that the handoff left wide and that no polish on
the fixed cold-start could break. Greedy breaks it by acquiring NEW informative
quads.

Data model: SELF-CONSISTENT by default (data generated with the same SoftTri
forward used for inference), so there is no surrogate bias -- a clean test of
whether adaptive selection collapses the rho variance. Set data_forward="remesh"
to add faithful physics (and the ~2% discrepancy) later.

Caveats carried from earlier findings:
  * Winner's curse over a large pool with FD Jacobians: greedy can pick noise
    spikes. Mitigation hook `sel_eval_split` uses a second FD step for the
    Fisher used in the UPDATE vs the one used to SELECT. Off by default; turn on
    for large pools.
  * Covariance in linear-Gaussian inference is data-independent GIVEN the
    linearization point; the value comes from theta_hat MOVING and J being
    re-linearized -- which this loop does every step.

Run in pygimli_env. Needs seq_coldstart, seq_handoff, seq_local_update,
pwhg_wrapper, pwhg_forward_soft, Anandlyn_log, config.
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
from seq_coldstart import world_from_theta
from seq_handoff import (read_cv_seed, load_dat, compute_handoff,
                         THETA_NAMES, DEFAULT_PRIOR_STDS)
from seq_local_update import _psd_inverse, project_theta
from seq_log import Log
from seq_parallel import ParallelSoft, geometric_factors

I_R, I_RHO = 5, 6
# geometric (shape) parameters: circle x, y, log10 r, log10 rho.
# world/layer rho are nuisances excluded from selection and Gamma.
GEOM = [3, 4, 5, 6]


# ----------------------------------------------------------------------
# candidate pool: disjoint dipole-dipole quads over the electrode line
# ----------------------------------------------------------------------
def build_pool(n_elec=21, x0=-25.0, x1=25.0, n_max=1500, seed=0):
    """Enumerate all disjoint dipole-dipole quads; optional random subsample.
    Full enumeration for 21 electrodes is ~18k after the disjointness filter
    (matches the comprehensive pool). n_max caps it for a fast first run;
    set n_max=None for the full pool."""
    xe = np.linspace(x0, x1, n_elec)
    quads = []
    for a in range(n_elec):
        for b in range(a + 1, n_elec):
            for m in range(n_elec):
                for n in range(m + 1, n_elec):
                    if len({a, b, m, n}) == 4:
                        quads.append((a, b, m, n))
    quads = np.asarray(quads, int)
    if n_max and len(quads) > n_max:
        rng = np.random.default_rng(seed)
        quads = quads[rng.choice(len(quads), n_max, replace=False)]
    scheme = pg.DataContainerERT()
    for x in xe:
        scheme.createSensor([float(x), 0.0])
    scheme.resize(len(quads))
    for k, tok in enumerate("abmn"):
        scheme.set(tok, quads[:, k].astype(float))
    scheme.set("valid", np.ones(len(quads)))
    from pygimli.physics import ert
    k = np.asarray(ert.createGeometricFactors(scheme), float)
    # prune geometrically bad quads: non-finite or extreme geometric factor
    good = np.isfinite(k) & (np.abs(k) > 1e-6) & (np.abs(k) < 1e5)
    if not np.all(good):
        quads = quads[good]
        # rebuild a clean scheme with only the good quads
        scheme = pg.DataContainerERT()
        for x in xe:
            scheme.createSensor([float(x), 0.0])
        scheme.resize(len(quads))
        for j, tok in enumerate("abmn"):
            scheme.set(tok, quads[:, j].astype(float))
        scheme.set("valid", np.ones(len(quads)))
        k = np.asarray(ert.createGeometricFactors(scheme), float)
    scheme.set("k", k)
    return scheme, quads, xe






# containment constants/projection live in seq_local_update (shared)


def _acquired_soft(idx, quads_all, xe, theta):
    """Build a SoftTriForward over only the acquired quads (sub-scheme).
    Cheap forward for the inner GN iterations."""
    q = quads_all[idx]
    scheme = pg.DataContainerERT()
    for x in xe:
        scheme.createSensor([float(x), 0.0])
    scheme.resize(len(q))
    for j, tok in enumerate("abmn"):
        scheme.set(tok, q[:, j].astype(float))
    scheme.set("valid", np.ones(len(q)))
    from pygimli.physics import ert as _ert
    scheme.set("k", _ert.createGeometricFactors(scheme))
    w = world_from_theta(theta, scheme)
    return SoftTriForward(w, width=0.2, tri_area=0.25, scheme=scheme)


def pool_forward_jac(soft, theta, steps):
    """Return (J, ln, rhoa) over the whole pool at theta. One FD sweep."""
    J, ln0, rhoa0 = W.jacobian_generic(soft.forward, theta, steps)
    return np.asarray(J, float), np.asarray(ln0, float), np.asarray(rhoa0, float)


def eig_scores(J, Sigma, noise_var, acquired_mask, geom=GEOM):
    """Geometry-targeted EIG: the reduction in log-det of the GEOM sub-block
    from a rank-1 update, per candidate.

    Full D-optimal gain of adding row j is 0.5 log(1 + jSj^T/sigma^2) on ALL
    params. To target the shape block G, we score the drop in log-det of the
    marginal covariance over G:  delta = 0.5 [logdet Sigma_GG - logdet Sigma'_GG],
    where Sigma' = Sigma - Sigma j j^T Sigma / (sigma^2 + jSj^T) (Sherman-Morrison).
    A quad that mostly informs nuisances (world/layer rho) scores ~0 here even if
    its total information is large -- so greedy spends budget on the boundary."""
    G = np.asarray(geom)
    m = J.shape[0]
    jsj = np.einsum("ij,jk,ik->i", J, Sigma, J)          # (m,) j_i Sigma j_i^T
    denom = noise_var + jsj                               # (m,)
    Sj = J @ Sigma                                        # (m, p): rows j_i^T Sigma
    Sj_G = Sj[:, G]                                       # (m, |G|)
    # base marginal cov over G and its inverse
    S_GG = Sigma[np.ix_(G, G)]
    sign, logdet0 = np.linalg.slogdet(S_GG)
    S_GG_inv = np.linalg.inv(S_GG)
    # rank-1 downdate of the G-block: Sigma'_GG = S_GG - (Sj_G outer Sj_G)/denom
    # logdet drop = -log(1 - (Sj_G^T S_GG_inv Sj_G)/denom)   (matrix determinant lemma)
    quad = np.einsum("ij,jk,ik->i", Sj_G, S_GG_inv, Sj_G)  # (m,)
    frac = np.clip(quad / denom, 0.0, 1.0 - 1e-12)
    s = -0.5 * np.log1p(-frac)                             # >=0, gain in logdet_G
    s[acquired_mask] = -np.inf
    return s


def _obj(theta, m0, P0, forward_ln, d, W, idx):
    """Negative log-posterior (up to const) over the acquired set."""
    r = d[idx] - np.asarray(forward_ln(theta))[idx]
    dp = theta - m0
    return 0.5 * dp @ P0 @ dp + 0.5 * np.sum(W[idx] * r * r)


def gn_step(theta, m0, P0, J_acq, r_acq, W_acq, forward_ln, d, W, idx,
            lo, hi, cond_cap=1e6, lam0=1e-2, step_cap=0.5):
    """Damped, line-searched, box-clamped Gauss-Newton step.

    Levenberg damping in the flat (rho) valley prevents the huge overshoot;
    a backtracking line search rejects steps that raise the objective; theta is
    clamped to the valid box so the forward stays physical. Returns
    (theta_new, Sigma_new, H_reg, accepted)."""
    grad = P0 @ (theta - m0) - J_acq.T @ (W_acq * r_acq)
    JTJ = (J_acq.T * W_acq) @ J_acq
    f0 = _obj(theta, m0, P0, forward_ln, d, W, idx)
    lam = lam0
    for _ in range(8):                      # increase damping until improvement
        H = P0 + JTJ + lam * np.diag(np.diag(P0 + JTJ))
        Sigma, H_reg, _ = _psd_inverse(H, cond_cap)
        delta = -Sigma @ grad
        n = np.linalg.norm(delta)
        if n > step_cap:                    # hard cap on step length
            delta = delta * (step_cap / n)
        cand = project_theta(theta + delta, lo, hi)
        if _obj(cand, m0, P0, forward_ln, d, W, idx) <= f0 + 1e-9:
            # Laplace covariance at the accepted point (GN Hessian, no damping)
            Sig, Hr, _ = _psd_inverse(P0 + JTJ, cond_cap)
            return cand, Sig, Hr, True
        lam *= 4.0
    # no improvement: stay put, return current Laplace cov
    Sig, Hr, _ = _psd_inverse(P0 + JTJ, cond_cap)
    return theta, Sig, Hr, False


def run_greedy(theta_hat0, Sigma0, theta_true, scheme, prior_stds,
               data_forward="soft", n_max_steps=40, gamma_target=0.10,
               dU=1e-6, floor=5e-3, sigma_model=0.01, rng=None, verbose=True,
               min_steps=1, n_workers=1, par=None, max_inner=4,
               drift_tol=0.005):
    if rng is None:
        rng = np.random.default_rng(0)
    p = theta_hat0.size
    lg = Log("greedy", quiet=not verbose)
    lg.stage(f"init: data_forward={data_forward}  pool={scheme.size()}")
    lg.info("theta_hat0", theta_hat0)

    # inference forward on the pool
    world = world_from_theta(theta_hat0, scheme)
    soft = SoftTriForward(world, width=0.2, tri_area=0.25, scheme=scheme)
    steps = W.default_steps(world)

    # prune candidates that give non-positive rhoa at the seed (unphysical quads)
    ln0 = np.asarray(soft.forward(theta_hat0))[0]
    rhoa0 = np.exp(ln0)
    valid = np.isfinite(rhoa0) & (rhoa0 > 1e-6)
    if not np.all(valid):
        keep = np.where(valid)[0]
        print(f"pruned {np.sum(~valid)} unphysical quads; {len(keep)} remain")
        # restrict scheme to valid quads and rebuild soft
        import pygimli as _pg
        xe = np.linspace(-25, 25, 21)
        q_all = np.column_stack([np.asarray(scheme[t], int) for t in "abmn"])
        q_keep = q_all[keep]
        scheme = _pg.DataContainerERT()
        for x in xe:
            scheme.createSensor([float(x), 0.0])
        scheme.resize(len(q_keep))
        for j, tok in enumerate("abmn"):
            scheme.set(tok, q_keep[:, j].astype(float))
        scheme.set("valid", np.ones(len(q_keep)))
        from pygimli.physics import ert as _ert
        scheme.set("k", _ert.createGeometricFactors(scheme))
        world = world_from_theta(theta_hat0, scheme)
        soft = SoftTriForward(world, width=0.2, tri_area=0.25, scheme=scheme)
        steps = W.default_steps(world)

    # data-generating forward at theta_true (self-consistent = soft)
    world_t = world_from_theta(theta_true, scheme)
    if data_forward == "soft":
        soft_t = SoftTriForward(world_t, width=0.2, tri_area=0.25, scheme=scheme)
        ln_true = np.asarray(soft_t.forward(theta_true))[0]
        rhoa_true = np.exp(ln_true)
    else:  # remesh (faithful physics + ~2% discrepancy)
        ln_true, rhoa_true = W.forward(world_t, theta_true)
        ln_true = np.asarray(ln_true, float); rhoa_true = np.asarray(rhoa_true, float)
    eps_true = np.asarray(W.noise_std(world_t, rhoa_true, dU=dU, floor=floor), float)
    # inference noise: always inflate by a model-discrepancy term. Even in
    # self-consistent mode the INFERENCE forward is the FD-linearized soft
    # re-evaluated at a moving theta, which differs from the true-soft data
    # forward by more than the ~0.5% measurement noise -> whitened residual
    # floors well above 1 without it (same effect as the handoff blur).
    # sigma_model keeps the Laplace covariance HONEST: with eps~0.5% the
    # Fisher weight 1/eps^2 ~ 4e4 makes a single quad collapse Sigma (Gamma->0)
    # -- overconfident. A small floor prevents that without biasing the mean.
    infl = max(sigma_model, 0.01)
    noise_var = eps_true ** 2 + infl ** 2
    d_all = ln_true + eps_true * rng.standard_normal(eps_true.size)  # noisy data

    prior_var = np.asarray(prior_stds, float) ** 2
    # parameters with a very tight prior are FIXED (e.g. layer rho pinned
    # at the handoff); exclude them from Gamma so the metric reflects only
    # the directions greedy actually estimates.
    free = np.asarray(prior_stds, float) > 0.05
    m0 = theta_hat0.copy()
    P0 = np.diag(1.0 / prior_var)

    # valid box for theta (clamp so soft forward stays physical); generous
    lo = theta_hat0 - np.array([0.5, 6.0, 0.5, 20.0, 8.0, 0.6, 2.5])
    hi = theta_hat0 + np.array([0.5, 6.0, 0.5, 20.0, 8.0, 0.6, 2.5])
    # clip radius/rho to sane physical ranges
    lo[5], hi[5] = max(lo[5], np.log10(0.5)), min(hi[5], np.log10(5.0))
    lo[6], hi[6] = max(lo[6], 0.0), min(hi[6], 5.0)

    theta = project_theta(theta_hat0.copy())
    Sigma = Sigma0.copy()
    n_pool = scheme.size()
    quads_all = np.column_stack([np.asarray(scheme[t], int) for t in 'abmn'])
    xe = np.asarray([scheme.sensor(k)[0] for k in range(scheme.sensorCount())], float)
    acquired = np.zeros(n_pool, bool)
    acq_idx = []

    def gamma_of(S):
        ratios = np.diag(S) / prior_var
        # Gamma over the geometric (shape) block only
        mask = np.full(ratios.size, -np.inf)
        mask[GEOM] = ratios[GEOM]
        i = int(np.argmax(mask))
        return float(ratios[i]), THETA_NAMES[i]

    # Persistent parallel pool for the FULL-POOL Jacobian (the dominant cost:
    # 2p+1 forwards over every candidate, recomputed at each acquisition).
    # Startup ~2s amortizes over the whole loop; warm speedup ~4x at 8 workers.
    _par = par                 # externally-owned pool (sweep reuses one)
    _own_par = False           # do we close it at the end?
    if _par is None and n_workers and n_workers > 1:
        try:
            _k = np.asarray(scheme["k"], float)
            _par = ParallelSoft(xe, quads_all, _k, theta, n_workers=n_workers,
                                verbose=verbose)
            _par.start()
            _own_par = True
        except Exception as _e:
            lg.warn(f"parallel pool failed ({type(_e).__name__}); serial")
            _par = None

    _sub_cache = {"n": -1, "obj": None}   # acquired-set forward, rebuilt on growth
    lg.stage("greedy acquisition loop")
    hist = []
    g0, g0p = gamma_of(Sigma)
    if verbose:
        print(f"start: Gamma={g0:.3f} (by {g0p})  "
              f"rho_std={np.sqrt(Sigma[I_RHO,I_RHO]):.3f} "
              f"r_std={np.sqrt(Sigma[I_R,I_R]):.3f}  pool={n_pool}")

    for step in range(n_max_steps):
        # re-linearize at current theta over the pool (parallel when available)
        if _par is not None:
            J, ln = _par.jacobian(theta, steps)
        else:
            J, ln, _ = pool_forward_jac(soft, theta, steps)
        # select
        scores = eig_scores(J, Sigma, noise_var, acquired)
        i = int(np.argmax(scores))
        acquired[i] = True; acq_idx.append(i)
        # update: ITERATE damped GN over the ACQUIRED set, re-linearizing each
        # inner iteration on a SMALL soft built over just the acquired quads --
        # NOT the full pool (that was the slowdown: full-pool Jacobian is 1500
        # quads x 14 solves per inner step; acquired is ~10). Full-pool Jacobian
        # is computed once above, for selection only.
        idx = np.asarray(acq_idx)
        Wv = 1.0 / noise_var
        d_acq = d_all[idx]
        W_acq_v = Wv[idx]
        # Build the acquired-set forward ONCE PER NEW QUAD, not per inner
        # iteration, and cache it. Rebuilding it (mesh + geometric factors)
        # every acquisition was a regression introduced with the pool change.
        if _sub_cache["n"] != idx.size:
            _sub_cache["obj"] = _acquired_soft(idx, quads_all, xe, theta)
            _sub_cache["n"] = idx.size
        sub = _sub_cache["obj"]
        f_sub = lambda th: np.asarray(sub.forward(th))[0]  # (len idx,) ln rhoa
        # full-theta forward for the objective/whit uses the SAME sub-scheme rows
        # Inner-iteration cap. Cost analysis: the full-pool Jacobian for
        # SELECTION is 2p+1 = 15 forwards, but this inner loop was up to
        # 12 x 15 = 180 forwards -- ~92% of the per-acquisition runtime.
        # The sub-scheme forward is NOT cheaper per call (FEM cost is set by
        # the mesh and the current injections, not the number of quadrupoles
        # read out), so only the NUMBER OF SOLVES matters. The outer loop
        # re-linearizes at every acquisition anyway, so the inner solve need
        # not fully converge -- it only needs to make progress.
        prev_obj = np.inf
        ok = True
        n_inner = 0
        for _inner in range(max_inner):
            n_inner += 1
            Js, lns, _ = W.jacobian_generic(sub.forward, theta, steps)
            r_acq = d_acq - np.asarray(lns, float)
            theta, Sigma, _, ok = gn_step(theta, m0, P0,
                                          np.asarray(Js, float), r_acq, W_acq_v,
                                          f_sub, d_acq, W_acq_v,
                                          np.arange(idx.size), lo, hi)
            obj = _obj(theta, m0, P0, f_sub, d_acq, W_acq_v, np.arange(idx.size))
            if not ok or abs(prev_obj - obj) < 1e-6 * (abs(prev_obj) + 1e-9):
                break
            prev_obj = obj
        g, gp = gamma_of(Sigma)
        # whitened residual over acquired set at the updated theta
        # use the acquired-set forward (len(idx) quads), NOT the full pool --
        # re-solving 1500 candidates to read ~10 rows was pure waste.
        g_now = np.asarray(sub.forward(theta))[0]
        wr = (d_acq - g_now) * np.sqrt(W_acq_v)
        whit = float(np.sqrt(np.mean(wr**2))) if idx.size else 0.0
        ey = float(theta[4] - theta_true[4]); ex = float(theta[3] - theta_true[3])
        hist.append(dict(step=step + 1, quad=i, gamma=g, gpar=gp, whit=whit,
                         ex=ex, ey=ey,
                         rho_std=float(np.sqrt(Sigma[I_RHO, I_RHO])),
                         r_std=float(np.sqrt(Sigma[I_R, I_R])),
                         theta=theta.copy()))
        if verbose and (step < 8 or (step + 1) % 5 == 0 or g <= gamma_target):
            print(f"  step {step+1:3d}: quad={i:6d}  Gamma={g:.1e} (by {gp:<12s})  "
                  f"whit={whit:.2f}  dy={hist[-1]['ey']:+.3f} dx={hist[-1]['ex']:+.3f}  "
                  f"y_std={np.sqrt(Sigma[4,4]):.3f} rho_std={hist[-1]['rho_std']:.3f}  "
                  f"in={n_inner}  ok={ok}")
        # STOP on estimate stability + valid fit. Gamma is NOT used to stop:
        # at low noise the Laplace covariance collapses on the first quad
        # (overconfident), so Gamma is only a diagnostic here.
        stable = False
        if len(hist) >= 5:
            recent = np.array([h['theta'][GEOM] for h in hist[-5:]])
            drift = np.max(np.linalg.norm(np.diff(recent, axis=0), axis=1))
            # drift_tol was 0.01, which fired while DEPTH was still descending
            # ~0.002-0.003 per acquisition: the reported |dy| was then set by
            # the stopping rule, not by the physics. Tightened to 0.005 so the
            # depth floor is the method's, not the criterion's.
            stable = drift < drift_tol
        if (step + 1) >= min_steps and whit < 1.5 and stable:
            if verbose:
                print(f"  -> converged (stable, whit={whit:.2f}) in "
                      f"{step+1} acquisitions")
            break

    err = theta - theta_true
    print(f"\nfinal: Gamma {g0:.1e} -> {hist[-1]['gamma']:.1e} (pinned by {hist[-1]['gpar']}) in {len(acq_idx)} quads")
    print(f"  theta_err: " + "  ".join(
        f"{THETA_NAMES[k].split('_')[-1]}={err[k]:+.3f}" for k in range(p)))
    print(f"  rho: seed_err={theta_hat0[I_RHO]-theta_true[I_RHO]:+.3f} -> "
          f"final_err={err[I_RHO]:+.3f}   r: {theta_hat0[I_R]-theta_true[I_R]:+.3f} -> "
          f"{err[I_R]:+.3f}")
    lg.check("final whit ~ 1", hist[-1].get("whit", np.nan), lo=0.3, hi=1.6)
    lg.check("|dy| depth error", abs(err[4]), hi=0.15)
    lg.check("|d log10rho| contrast error", abs(err[I_RHO]), hi=0.15)
    lg.done(f"loop finished in {len(acq_idx)} acquisitions")
    if _par is not None and _own_par:
        _par.close()
    return dict(theta=theta, Sigma=Sigma, acquired=acq_idx, hist=hist,
                gamma0=g0, gamma=hist[-1]["gamma"])


if __name__ == "__main__":
    import config as C
    scene = "c11"

    # 1) handoff on the standard dd cold-start (as before) -> theta_hat0, Sigma0
    from seq_coldstart import make_scheme
    dd = make_scheme()
    seed = read_cv_seed(C.DIR_CV / "parameters.xlsx", scene)
    rhoa_obs, err = load_dat(C.DIR_FWD / f"{scene.upper()}_noisy.dat")
    hand = compute_handoff(seed, rhoa_obs, err, dd, n_gn=8, verbose=False)
    theta_hat0, Sigma0 = hand["theta0"], hand["Sigma0"]
    prior_stds = hand["prior_stds"]

    theta_true = np.array([2.0, -3.0, np.log10(50), 5.0, -4.0,
                           np.log10(1.2), np.log10(500)])

    # 2) greedy on the comprehensive pool (capped for a fast first run)
    pool, quads, xe = build_pool(n_elec=21, n_max=1500)
    print(f"theta_hat0: {np.round(theta_hat0,3)}")
    print(f"pool size : {pool.size()}  (set n_max=None for full ~18k)\n")

    run_greedy(theta_hat0, Sigma0, theta_true, pool, prior_stds,
               data_forward="soft", n_max_steps=30, gamma_target=0.10,
               min_steps=6)
