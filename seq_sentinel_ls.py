"""
seq_sentinel_ls.py -- sentinel re-globalization using ARTICLE 1's ls_opti.

WHY THIS REPLACES THE HAND-ROLLED DE
The scipy DE written for seq_sentinel_softde.py was under-powered and used a
greedier search than the optimizer that actually worked in article 1:

                       article-1 ls_opti        earlier soft_de
    population         popsize=30 x ndim=210    15-20 x 7 = 105-140
    strategy           rand1bin (explorative)   best1bin (greedy)
    init               latinhypercube           sobol
    mutation           (0.7, 1.0)               (0.5, 1.0)
    recombination      0.9                      0.7
    objective          male = mean|log10(rhoa/meas)|   weighted SSE
    stages             3, with elite re-seeding 1

`male` is a log-domain mean-absolute misfit: far more robust to the handful of
quadrupoles with extreme geometric factors than a squared-error objective, which
those configurations otherwise dominate. Stage 2 re-seeds around the top 10% of
the Stage-1 population (the "elite exploitation" step). Together these are why
article 1 solved scenes -- including anomalies on the layer transition -- that
the local loop fails on.

So: do not re-implement DE. Call ls_opti.

USAGE
    from seq_sentinel_ls import ls_globalize
    theta, info = ls_globalize(meas_rhoa, scheme, seed=0)          # global
    theta, info = ls_globalize(meas_rhoa, scheme, center=theta_hat,
                               half=np.array([...]))               # narrowed

`meas_rhoa` is LINEAR apparent resistivity (that is what male/msle compare
against, via aworld.meas).

NOTE ON THE FORWARD: ls_opti uses the AnomalyWorld's own (remesh) forward. That
is correct here -- DE is derivative-free, so the non-smooth remesh forward is
not a problem, and it is the faithful physics. SoftTri exists for smooth
JACOBIANS in the greedy loop, which is a different need.
"""
import os
import tempfile
import numpy as np

from Anandlyn_log import AnomalyWorld, Layer, CircleAnom

# ---------------------------------------------------------------------------
# article-1 style budgets and bounds (cfgA-like: 7 free params, wide box)
# ---------------------------------------------------------------------------
MAX_ITER = 26          # BUDGETS['it26'][0]; 26 generations chosen by isocost sweep
POPSIZE = 30           # scipy MULTIPLIER -> 30 * ndim(7) = 210 for stage 1
N_STAGES = 3
ATOL = 0.0

DOMAIN = dict(start=[-25, 0], end=[25, -20])

# wide, global box -- the varlims are LINEAR; ls_opti log10's the rho/r entries
LAYER_VARLIMS = ((-10, -1), (1, 1e5))          # (y, rho)
CIRCLE_VARLIMS = ((-25, 25), (-10, 0), (0.5, 3), (1, 1e5))   # (x, y, r, rho)
WORLD_VARLIM = [1, 1e5]

I_W, I_LY, I_LR, I_X, I_Y, I_R, I_RHO = range(7)


def build_world(scheme, theta_init=None, layer_varlims=LAYER_VARLIMS,
                circle_varlims=CIRCLE_VARLIMS, world_varlim=WORLD_VARLIM):
    """AnomalyWorld for the 7-param C_1,1 case with article-1 style varlims."""
    if theta_init is None:
        theta_init = np.array([2.0, -3.0, np.log10(50.0), 5.0, -4.0,
                               np.log10(1.2), np.log10(500.0)])
    t = np.asarray(theta_init, float)
    layers = [Layer(name='layer1', y=float(t[I_LY]), rho=float(10 ** t[I_LR]),
                    _cnum=1, _varflag=[True, True], _varlims=layer_varlims)]
    circles = [CircleAnom(name='C1', x=float(t[I_X]), y=float(t[I_Y]),
                          r=float(10 ** t[I_R]), rho=float(10 ** t[I_RHO]),
                          _c_num=1, _varflag=[True] * 4,
                          _varlims=circle_varlims)]
    w = AnomalyWorld(_start=DOMAIN['start'], _end=DOMAIN['end'], _scheme=scheme,
                     _layers=layers, _circles=circles,
                     _rho_world=float(10 ** t[I_W]),
                     _rho_world_varflag=True, _rho_world_varlim=world_varlim)
    # REQUIRED (run_article.py): constraints frozen at __init__ are not updated
    # by later add_constraint calls; refresh so any relational constraint reaches
    # the optimizer. Harmless when there are none.
    try:
        w.refresh_constraints()
    except AttributeError:
        pass
    return w


def attach_soft_forward(world, scheme, width=0.2, tri_area=0.25):
    """Swap the world onto the SoftTri forward for ls_opti.

    male() and msle() both do  self.update(x) -> self.get_forward_solution()
    -> data['rhoa'] (LINEAR). So overriding that ONE method on the instance
    routes the whole 3-stage optimizer through SoftTriForward.

    Valid because SoftTriForward is stateless in theta (verified): built once,
    it evaluates any theta. Row 1 of its output is linear rhoa, row 0 is ln.
    """
    from pwhg_forward_soft import SoftTriForward
    soft = SoftTriForward(world, width=width, tri_area=tri_area, scheme=scheme)

    def _soft_forward_solution(_noise=False, **kw):
        th = np.asarray(world.get_x0(), float)
        out = np.asarray(soft.forward(th), float)
        return {'rhoa': out[1]}            # linear apparent resistivity

    world.get_forward_solution = _soft_forward_solution
    world._soft = soft
    return world


def narrowed_varlims(center, half):
    """Box around `center` (7-param theta, log10 for rho/r) intersected with the
    global box -- for a LOCAL re-globalization rather than a full global search.
    Returns (layer_varlims, circle_varlims, world_varlim) in LINEAR units."""
    c = np.asarray(center, float)
    h = np.asarray(half, float)
    lin = lambda lo, hi: (float(10 ** lo), float(10 ** hi))
    wl = lin(max(c[I_W] - h[I_W], 0.0), min(c[I_W] + h[I_W], 5.0))
    lay = ((float(max(c[I_LY] - h[I_LY], -10)), float(min(c[I_LY] + h[I_LY], -1))),
           lin(max(c[I_LR] - h[I_LR], 0.0), min(c[I_LR] + h[I_LR], 5.0)))
    cir = ((float(max(c[I_X] - h[I_X], -25)), float(min(c[I_X] + h[I_X], 25))),
           (float(max(c[I_Y] - h[I_Y], -10)), float(min(c[I_Y] + h[I_Y], 0))),
           lin(max(c[I_R] - h[I_R], np.log10(0.5)),
               min(c[I_R] + h[I_R], np.log10(3.0))),
           lin(max(c[I_RHO] - h[I_RHO], 0.0), min(c[I_RHO] + h[I_RHO], 5.0)))
    return lay, cir, list(wl)


def ls_globalize(meas_rhoa, scheme, center=None, half=None, seed=0,
                 max_iter=MAX_ITER, popsize=POPSIZE, n_stages=N_STAGES,
                 run_dir=None, verbose=False, forward="remesh"):
    """Run article-1's ls_opti. Returns (theta, info).

    meas_rhoa : LINEAR apparent resistivity (aworld.meas is compared in male/msle)
    center/half : if both given, search a narrowed box around `center`
                  (sentinel re-globalization); otherwise the full global box.
    forward   : "remesh" (AnomalyWorld, faithful physics, STAIRCASE) or
                "soft"   (SoftTriForward, smooth).

    ON STAGE 3. ls_opti runs three stages: 1 = DE on `male` (log10 MAE,
    exploration), 2 = DE on `msle` re-seeded from the stage-1 elite, 3 =
    L-BFGS-B on `msle` (LOCAL GRADIENT POLISH). Stage 3 is only meaningful on a
    SMOOTH forward: on the remesh staircase its gradients are garbage, which is
    the polish failure reported in article 1. Because ls_opti always runs all
    three, this returns BOTH the post-polish point and the best-ever msle point
    seen during the run, so the polish's effect is measurable rather than
    assumed:
        info['theta_final']     = res3.x            (after stage 3)
        info['theta_best_msle'] = world.best_x_msle (best over all evals)
    For the remesh forward, prefer theta_best_msle.
    """
    if center is not None and half is not None:
        lay, cir, wl = narrowed_varlims(center, half)
        world = build_world(scheme, theta_init=center, layer_varlims=lay,
                            circle_varlims=cir, world_varlim=wl)
    else:
        world = build_world(scheme, theta_init=center)

    if forward == "soft":
        attach_soft_forward(world, scheme)
    elif forward != "remesh":
        raise ValueError("forward must be 'remesh' or 'soft'")

    world.meas = np.asarray(meas_rhoa, float)      # male/msle compare to this

    tmp = run_dir or tempfile.mkdtemp(prefix="ls_opti_")
    run_name = os.path.join(tmp, "")

    kw = dict(max_iter=max_iter, _n=n_stages, atol=ATOL, _popsize=popsize,
              _run_name=run_name)
    # the patched ls_opti in run_article takes _seed; the project copy may not
    try:
        res = world.ls_opti(_seed=seed, **kw)
    except TypeError:
        np.random.seed(seed)
        res = world.ls_opti(**kw)

    theta_final = np.asarray(res.x, float)
    best_msle = getattr(world, "best_x_msle", None)
    theta_best = (np.asarray(best_msle, float)
                  if best_msle is not None and np.size(best_msle) == theta_final.size
                  else theta_final)
    # on the staircase remesh forward the stage-3 gradient polish is unreliable
    theta = theta_best if forward == "remesh" else theta_final
    info = dict(fun=float(res.fun), nfev=int(getattr(res, "nfev", -1)),
                success=bool(getattr(res, "success", True)),
                popsize_realised=popsize * theta_final.size, run_dir=tmp,
                forward=forward, theta_final=theta_final,
                theta_best_msle=theta_best,
                polish_shift=float(np.linalg.norm(theta_final - theta_best)),
                best_msle=float(getattr(world, "best_msle", np.nan)),
                best_male=float(getattr(world, "best_male", np.nan)))
    if verbose:
        print(f"[ls_opti/{forward}] fun={info['fun']:.5g} nfev={info['nfev']} "
              f"pop1={info['popsize_realised']} "
              f"|stage3 shift|={info['polish_shift']:.4f}")
    return theta, info


def _demo():
    """Head-to-head: ls_opti on the REMESH forward vs the SOFT forward.

    Same data, same budget, same seed. Also reports how far stage 3 (L-BFGS on
    msle) moved the estimate -- expected to be unreliable on remesh (staircase)
    and meaningful on soft (smooth).
    """
    import time
    import pwhg_wrapper as W
    from seq_coldstart import make_scheme, world_from_theta

    scheme = make_scheme()
    th_true = np.array([2.0, -3.0, np.log10(50.0), 5.0, -4.0,
                        np.log10(1.2), np.log10(500.0)])
    wt = world_from_theta(th_true, scheme)
    _, rhoa = W.forward(wt, th_true)
    meas = np.asarray(rhoa, float)          # clean data from the REMESH physics

    names = ['w_rho', 'L0_y', 'L0_rho', 'C0_x', 'C0_y', 'C0_r', 'C0_rho']
    print(f"truth : {np.round(th_true, 3)}")
    print(f"m={meas.size}  budget: pop {POPSIZE}x7={POPSIZE*7}, "
          f"{MAX_ITER} gens, 3 stages (1=male, 2=msle DE, 3=msle L-BFGS)")
    print("data generated with the REMESH forward, so 'soft' also carries the "
          "~2% surrogate discrepancy.\n")

    out = {}
    for fwd in ("remesh", "soft"):
        t0 = time.time()
        th, info = ls_globalize(meas, scheme, seed=0, forward=fwd, verbose=True)
        dt = time.time() - t0
        out[fwd] = (th, info, dt)
        err = th - th_true
        print(f"  {fwd:>6}: " + "  ".join(f"{n}={e:+.3f}"
                                          for n, e in zip(names, err)))
        print(f"          |dx|={abs(err[I_X]):.3f} |dy|={abs(err[I_Y]):.3f} "
              f"|dlog10r|={abs(err[I_R]):.3f} |dlog10rho|={abs(err[I_RHO]):.3f}"
              f"   {dt:.0f}s\n")

    print(f"{'':>8}{'|dr|':>8}{'|dy|':>8}{'|dx|':>8}{'stage3 shift':>14}{'time[s]':>9}")
    for fwd in ("remesh", "soft"):
        th, info, dt = out[fwd]
        e = th - th_true
        print(f"{fwd:>8}{abs(e[I_R]):>8.3f}{abs(e[I_Y]):>8.3f}{abs(e[I_X]):>8.3f}"
              f"{info['polish_shift']:>14.4f}{dt:>9.0f}")

    print("\nreading:")
    print("  * stage-3 shift LARGE on remesh with a worse estimate -> the")
    print("    staircase gradient polish is harmful (article-1 finding).")
    print("  * soft comparable or better despite its ~2% discrepancy -> the")
    print("    smooth surrogate is a usable sentinel forward, and much cheaper.")
    print("  * remesh better -> keep the faithful forward for the sentinel and")
    print("    use theta_best_msle (pre-polish) rather than the stage-3 point.")


if __name__ == "__main__":
    _demo()
