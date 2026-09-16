"""
seq_sweep.py -- characterize the greedy loop across prior-drawn true theta.

For each drawn scene it runs the full Stage-3 spine:
    cold-start data -> S2PH handoff (theta_hat0, Sigma0) -> greedy EIG refinement
and records: did Gamma reach target, #quads used, final rho/r error, and the
anomaly depth/size. Runs under BOTH data models:
    data_forward="soft"   (self-consistent; clean mechanism)
    data_forward="remesh" (faithful physics + ~2% discrepancy; robustness)

The output table, sorted by circle DEPTH, shows where the loop stops working --
the depth limit that bounds adaptive tracking (Stage-3b input). Deep/small
anomalies the cold-start cannot localize should fail to reach target or need
many quads; shallow-central ones should converge in a few.

NOTE: this reuses the ACTUAL pipeline handoff only if you have run stage1-3 for
each drawn scene. That is expensive. For the sweep we instead build the handoff
seed synthetically per draw by inverting the cold-start with the SAME GN/Laplace
machinery from a perturbed CV-like seed -- i.e. we skip re-running smooth+CV per
draw and start the handoff GN from a smeared version of the true theta (bigger
r, weaker rho, offset depth), mimicking what CV produces. This isolates the
GREEDY question (does adaptive selection break the degeneracy across scenes)
from the smooth/CV localization question (studied separately).

Run in pygimli_env. Slow with data_forward="remesh"; start with "soft".
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

from seq_coldstart import make_scheme, world_from_theta
from seq_handoff import compute_handoff, DEFAULT_PRIOR_STDS
from seq_greedy import build_pool, run_greedy
from seq_parallel import ParallelSoft
import pwhg_wrapper as W
from pwhg_forward_soft import SoftTriForward

I_R, I_RHO = 5, 6

# prior over the 7-param theta: [world_log10rho, L0_y, L0_log10rho,
# C0_x, C0_y, C0_log10r, C0_log10rho]. Draw circle depth/size/contrast broadly.
PRIOR = dict(
    world_log10rho=(2.0, 0.0),           # fixed background ~100
    L0_y=(-3.0, 0.0), L0_log10rho=(np.log10(50), 0.0),   # fixed layer
    C0_x=(0.0, 6.0),                     # lateral position
    C0_y=(-4.5, 2.0),                    # DEPTH (mean -4.5, spread 2)  <- key axis
    C0_log10r=(np.log10(1.2), 0.18),     # radius
    C0_log10rho=(np.log10(500), 0.30),   # contrast
)


def draw_theta(rng):
    keys = ['world_log10rho', 'L0_y', 'L0_log10rho', 'C0_x', 'C0_y',
            'C0_log10r', 'C0_log10rho']
    th = np.array([PRIOR[k][0] + PRIOR[k][1] * rng.standard_normal()
                   for k in keys])
    # keep circle inside the domain and above the bottom
    th[I_R] = np.clip(th[I_R], np.log10(0.6), np.log10(3.0))
    th[4] = np.clip(th[4], -8.0, -1.5)          # C0_y depth
    th[3] = np.clip(th[3], -18.0, 18.0)         # C0_x
    return th


def cv_like_seed(theta_true, rng):
    """Mimic what smooth->CV produces: localized position, smeared shape
    (bigger radius, weaker contrast), offset layer depth."""
    s = theta_true.copy()
    s[3] += 0.3 * rng.standard_normal()          # x ~localized
    s[4] += 0.4 * rng.standard_normal()          # y ~localized
    s[I_R] += 0.35                               # radius inflated (smearing)
    s[I_RHO] -= 0.55                             # contrast weakened
    s[1] += -1.2                                 # layer depth offset
    return s


def run_one(theta_true, pool, scheme, data_forward, rng,
            prior_stds=DEFAULT_PRIOR_STDS, gamma_target=0.10, n_steps=50,
            par=None):
    # cold-start data on the standard dd scheme -> handoff seed
    dd = make_scheme()
    seed = cv_like_seed(theta_true, rng)
    # generate cold-start observation at truth for the handoff GN
    world_t = world_from_theta(theta_true, dd)
    ln_t, rhoa_t = W.forward(world_t, theta_true)
    eps = np.asarray(W.noise_std(world_t, np.asarray(rhoa_t, float),
                                 dU=1e-6, floor=5e-3), float)
    rhoa_obs = np.asarray(rhoa_t, float) * np.exp(eps * rng.standard_normal(eps.size))
    hand = compute_handoff(seed, rhoa_obs, eps, dd, n_gn=8, verbose=True)

    res = run_greedy(hand['theta0'], hand['Sigma0'], theta_true, pool,
                     prior_stds, data_forward=data_forward,
                     n_max_steps=n_steps, gamma_target=gamma_target,
                     rng=rng, verbose=True, par=par)
    err = res['theta'] - theta_true
    truncated = len(res['acquired']) >= n_steps   # hit budget, did not settle
    r_true = 10 ** theta_true[I_R]
    # ERROR-BASED success, scaled by the anomaly's own size (a 0.2 m position
    # error means something different for r=0.8 than r=1.9). Gamma is recorded
    # but NOT used: at low noise the Laplace covariance collapses regardless of
    # whether the estimate is right, so 'Gamma reached' tracks nothing.
    pos_err = float(np.hypot(err[3], err[4]))          # |d(x,y)| in metres
    pos_rel = pos_err / r_true                          # fraction of a radius
    r_rel = float(abs(10 ** res['theta'][I_R] - r_true) / r_true)  # |dr|/r
    geom_ok = (pos_rel < 0.5) and (r_rel < 0.25)        # boundary overlaps truth
    rho_ok = abs(err[I_RHO]) < 0.30                     # contrast within 2x
    return dict(depth=theta_true[4], r=r_true,
                rho=10 ** theta_true[I_RHO],
                geom_ok=geom_ok, rho_ok=rho_ok,
                pos_err=pos_err, pos_rel=pos_rel, r_rel=r_rel,
                nquad=len(res['acquired']), truncated=truncated,
                gamma=res['gamma'], drho=err[I_RHO], dr=err[I_R],
                dx=err[3], dy=err[4])


def sweep(n=12, data_forward="soft", n_max=1200, seed=0, n_workers=1):
    rng = np.random.default_rng(seed)
    pool, quads, xe = build_pool(n_elec=21, n_max=n_max)
    scheme = pool
    # ONE parallel pool shared by every draw: the candidate set is identical
    # across draws and SoftTriForward is stateless in theta, so per-draw
    # rebuild (~2s each) is pure waste.
    par = None
    if n_workers and n_workers > 1:
        try:
            q = np.column_stack([np.asarray(pool[t], int) for t in 'abmn'])
            xs = np.asarray([pool.sensor(i)[0]
                             for i in range(pool.sensorCount())], float)
            kk = np.asarray(pool['k'], float)
            th0 = np.array([2.0, -3.0, np.log10(50), 5.0, -4.0,
                            np.log10(1.2), np.log10(500)])
            par = ParallelSoft(xs, q, kk, th0, n_workers=n_workers,
                               verbose=True)
            par.start()
        except Exception as e:
            print(f"  [sweep] parallel unavailable ({type(e).__name__}); serial")
            par = None
    rows = []
    for j in range(n):
        th = draw_theta(rng)
        try:
            r = run_one(th, pool, scheme, data_forward, rng, par=par)
        except Exception as e:
            r = dict(depth=th[4], r=10**th[I_R], rho=10**th[I_RHO],
                     geom_ok=False, rho_ok=False, pos_err=np.nan,
                     pos_rel=np.nan, r_rel=np.nan,
                     nquad=-1, truncated=True, gamma=np.nan,
                     drho=np.nan, dr=np.nan, dx=np.nan, dy=np.nan,
                     err=f"{type(e).__name__}")
        rows.append(r)
        print(f"  draw {j+1:2d}: depth={r['depth']:+.2f} r={r['r']:.2f} "
              f"rho={r['rho']:.0f}  GEOM={'OK ' if r['geom_ok'] else 'BAD'} "
              f"RHO={'OK ' if r['rho_ok'] else 'BAD'}  nq={r['nquad']}  "
              f"pos={r['pos_err']:.2f}m ({r['pos_rel']:.2f}r)  "
              f"dr/r={r['r_rel']:.2f}  drho={r['drho']:+.3f}"
              f"{'  [BUDGET]' if r.get('truncated') else ''}")
    if par is not None:
        par.close()
    return rows


def report(rows, data_forward):
    print(f"\n=== SWEEP ({data_forward} data), sorted by depth ===")
    print("success = GEOM (|d(x,y)| < r/2 AND |dr|/r < 0.25)  and  RHO (|dlog10rho| < 0.30)")
    print("Gamma is shown as a DIAGNOSTIC ONLY -- at low noise the Laplace")
    print("covariance collapses regardless of accuracy, so it is not a success test.\n")
    print(f"{'depth':>7}{'r':>6}{'rho':>7}{'GEOM':>6}{'RHO':>5}{'nq':>4}"
          f"{'pos_m':>7}{'pos/r':>7}{'dr/r':>7}{'dx':>7}{'dy':>7}{'drho':>8}{'Gamma':>10}")
    for r in sorted(rows, key=lambda z: z['depth']):
        print(f"{r['depth']:>7.2f}{r['r']:>6.2f}{r['rho']:>7.0f}"
              f"{'OK' if r['geom_ok'] else 'BAD':>6}{'OK' if r['rho_ok'] else 'BAD':>5}"
              f"{r['nquad']:>4}{r['pos_err']:>7.2f}{r['pos_rel']:>7.2f}"
              f"{r['r_rel']:>7.2f}{r['dx']:>7.3f}{r['dy']:>7.3f}"
              f"{r['drho']:>8.3f}{r['gamma']:>10.1e}")

    g = [r for r in rows if r['geom_ok']]
    rh = [r for r in rows if r['rho_ok']]
    print(f"\ngeometry recovered: {len(g)}/{len(rows)}    "
          f"contrast recovered: {len(rh)}/{len(rows)}")
    if g:
        d = np.array([r['depth'] for r in g])
        n = np.array([r['nquad'] for r in g])
        print(f"  GEOM-OK depth range: {d.min():.2f} .. {d.max():.2f} m   "
              f"median quads {np.median(n):.0f}")
    bad = [r for r in rows if not r['geom_ok']]
    if bad:
        print("  GEOM failures:")
        for r in sorted(bad, key=lambda z: z['depth']):
            # flag the shallow circle-layer confound: circle top vs layer at -3
            top = r['depth'] + r['r']
            note = "  <- circle straddles layer(-3)" if top > -3.0 else ""
            print(f"    depth={r['depth']:+.2f} r={r['r']:.2f} "
                  f"top={top:+.2f} pos/r={r['pos_rel']:.2f} "
                  f"dr/r={r['r_rel']:.2f}{note}")
    # depth trend among geometry-successful draws
    if len(g) >= 3:
        d = np.array([abs(r['depth']) for r in g])
        e = np.array([r['pos_rel'] for r in g])
        if d.std() > 0:
            slope = np.polyfit(d, e, 1)[0]
            print(f"  position error vs depth (GEOM-OK only): "
                  f"slope {slope:+.3f} (pos/r per metre)")


if __name__ == "__main__":
    import sys
    df = sys.argv[1] if len(sys.argv) > 1 else "soft"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(f"sweep: n={n} data_forward={df}")
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 1   # serial default;
    # pass a 3rd arg to enable the pool deliberately, e.g. 'soft 25 8'
    rows = sweep(n=n, data_forward=df, n_workers=nw)
    report(rows, df)
