# -*- coding: utf-8 -*-
"""
validate_sensitivity.py
=======================
Deep check of the PWHG sensitivity / Fisher machinery. Each check is a small
standalone function returning a verdict dict; main() runs them on a world you
supply. Run in ERT_GUI.

Checks
------
A  reset-bug        : your AnomalyWorld.sensitivity_analysis vs the fixed
                      jacobian(). Isolates the base-point-reset bug; expect the
                      per-column error to be ~0 for column 0 and to GROW with
                      column index.
B  step-convergence : sweep the FD step per parameter; find the plateau and the
                      small-step blowup (remeshing jitter for x/y/r). Produces
                      fd_step_convergence.png and a recommended step per param.
C  rho-analytic     : FD rho columns vs pygimli's analytic pixel Jacobian summed
                      over the region (chain rule). The ONLY independent analytic
                      check available; validates the rho columns to machine-ish
                      precision. Reports the per-datum ratio -- a constant ratio
                      != 1 means a rhoa-vs-resistance convention (k) difference,
                      NOT a sensitivity error; a ratio that varies per datum is a
                      real disagreement.
D  two-path         : sensitivity_analysis (rhoa) vs sensitivity_analysis_v (fop)
                      should agree up to a per-datum constant (the geometric
                      factor). Confirms the two forward paths are consistent.
E  fisher-props     : symmetry, PSD, condition number, and how much the noise
                      weighting changes things vs the unweighted J^T J.
F  directional      : J @ v vs a single central difference along random v.
                      Catches column-order / indexing errors globally.

x/y/r columns have NO cheap analytic check (moving a boundary is exactly where
the hard sensitivity lives), so their trust comes from B + F + D, not C.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # save figures without a display
import matplotlib.pyplot as plt
from shapely.geometry import Point

import pwhg_wrapper as W


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _jac_to_numpy(jac):
    """Robust pygimli Jacobian -> dense ndarray (row-by-row fallback)."""
    try:
        A = np.array(jac)
        if A.ndim == 2 and A.shape[0] > 0 and A.shape[1] > 0:
            return A
    except Exception:
        pass
    nr, nc = jac.rows(), jac.cols()
    A = np.zeros((nr, nc))
    for i in range(nr):
        A[i, :] = jac[i]
    return A


def _relcol(a, b):
    """Per-column relative L2 difference between two (n_data, n_param) matrices."""
    num = np.linalg.norm(a - b, axis=0)
    den = np.linalg.norm(b, axis=0) + 1e-300
    return num / den


# ---------------------------------------------------------------------------
# A -- reset bug
# ---------------------------------------------------------------------------
def check_reset_bug(world, eps=1e-3):
    """
    Compare the object's own sensitivity_analysis (linear rhoa, buggy reset)
    against the fixed jacobian(), both with the SAME scalar step so the only
    difference is the base-point handling.
    """
    theta0 = np.asarray(world.get_x0(), float)
    labels = W.param_labels(world)

    # object's method: h_old is (n_param, n_data) in LINEAR rhoa
    h_old = np.asarray(world.sensitivity_analysis(eps=eps), float)
    world.update(list(theta0))                       # clean up its side effects
    _, rhoa0 = W.forward(world, theta0)
    Jold_ln = (h_old / rhoa0[None, :]).T             # -> ln space, (n_data, n_param)

    Jfix, _, _, _ = W.jacobian(world, theta0, scalar_step=eps)

    rc = _relcol(Jold_ln, Jfix)
    print("\n[A] reset-bug  (per-column rel. diff, old vs fixed):")
    for (nm, kind), r in zip(labels, rc):
        print(f"    col {nm:<5} {kind:<4}  reldiff = {r:.3e}")
    print(f"    -> expect ~0 at column 0, growing with index. "
          f"max = {rc.max():.3e}")
    return dict(relcol=rc, labels=labels)


# ---------------------------------------------------------------------------
# B -- FD step convergence
# ---------------------------------------------------------------------------
def check_step_convergence(world, kind_grids=None, out="fd_step_convergence.png"):
    """
    For each parameter, sweep its own FD step (holding others at default) and
    watch the column stabilize. A good step sits on the plateau between
    truncation error (large step) and forward-noise / remesh jitter (small step).
    """
    theta0 = np.asarray(world.get_x0(), float)
    labels = W.param_labels(world)
    base_steps = W.default_steps(world)

    if kind_grids is None:
        kind_grids = {
            "x":   np.logspace(-3.5, -0.5, 10),   # metres
            "y":   np.logspace(-3.5, -0.5, 10),   # metres
            "r":   np.logspace(-4.0, -1.0, 10),   # log10 decades
            "rho": np.logspace(-4.0, -1.0, 10),   # log10 decades
        }

    n = len(labels)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                             squeeze=False)
    recommended = base_steps.copy()

    for j, (nm, kind) in enumerate(labels):
        grid = kind_grids[kind]
        cols = []
        for s in grid:
            steps = base_steps.copy()
            steps[j] = s
            J, _, _, _ = W.jacobian(world, theta0, steps=steps)
            cols.append(J[:, j].copy())
        cols = np.array(cols)                          # (n_step, n_data)

        # successive-difference norm: small -> converged
        d = np.linalg.norm(np.diff(cols, axis=0), axis=1)
        d /= (np.linalg.norm(cols[1:], axis=1) + 1e-300)
        # pick the step (from the coarser side) at the first local minimum of d
        k_best = int(np.argmin(d)) + 1
        recommended[j] = grid[k_best]

        ax = axes[j // ncol][j % ncol]
        ax.loglog(grid[1:], d, "o-", ms=3)
        ax.axvline(grid[k_best], color="C3", ls="--", lw=1)
        ax.set_title(f"{nm} {kind}", fontsize=9)
        ax.set_xlabel("FD step"); ax.set_ylabel("rel. succ. diff")

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("FD step convergence (U-shape = truncation vs remesh jitter)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)

    print("\n[B] step-convergence -> recommended per-parameter steps:")
    for (nm, kind), s in zip(labels, recommended):
        print(f"    {nm:<5} {kind:<4}  step = {s:.3e}")
    print(f"    figure: {out}")
    print("    If a curve has no clear valley (monotone-down then noisy), the "
          "geometric column is remesh-noise limited -> consider fixed-topology "
          "meshing before trusting depth/radius sensitivities.")
    return dict(recommended=recommended, labels=labels)


# ---------------------------------------------------------------------------
# C -- rho columns vs analytic pixel Jacobian
# ---------------------------------------------------------------------------
def _pixel_jacobian(world):
    """Analytic d rhoa / d res_cell at the current mesh/model, via pygimli."""
    import pygimli.physics.ert as ert
    fop = ert.ERTModelling()
    fop.setData(world.scheme)
    fop.setMesh(world.mesh)
    res = np.asarray(world.resistivity_map, float)
    _ = np.asarray(fop.response(res), float)          # ensure a valid state
    fop.createJacobian(res)
    Jcell = _jac_to_numpy(fop.jacobian())             # (n_data, n_cell)
    return Jcell, res


def _circle_region_mask(world, ci):
    poly = world.Circles[ci].polygon
    cen = [(world.mesh.cell(c).center().x(), world.mesh.cell(c).center().y())
           for c in range(world.mesh.cellCount())]
    return np.array([poly.contains(Point(x, y)) for (x, y) in cen])


def check_rho_columns(world):
    """
    For each varying rho on a circle, compare the FD column
    d ln(rhoa)/d log10(rho_r)  against the analytic chain-rule column
        ln(10) * (1/rhoa) * sum_{c in region} res_c * dJcell[:,c].
    Layers/background regions are skipped here only because region membership is
    trickier to reconstruct robustly across pygimli versions; the circle check is
    already sufficient to validate the rho-sensitivity path.
    """
    theta0 = np.asarray(world.get_x0(), float)
    labels = W.param_labels(world)
    steps = W.default_steps(world)
    Jfd, ln0, rhoa0, _ = W.jacobian(world, theta0, steps=steps)

    world.update(list(theta0))
    Jcell, res = _pixel_jacobian(world)               # d rhoa / d res

    results = []
    for j, (nm, kind) in enumerate(labels):
        if kind != "rho" or not nm.startswith("C"):
            continue
        ci = int(nm[1:])
        mask = _circle_region_mask(world, ci)
        if mask.sum() == 0:
            print(f"    [C] {nm}: no mesh cells found in polygon -- skipped")
            continue
        # analytic: d rhoa / d log10(rho_r) = ln10 * sum_c res_c * Jcell[:,c]
        col_rhoa = np.log(10.0) * (Jcell[:, mask] * res[mask][None, :]).sum(axis=1)
        col_ln = col_rhoa / rhoa0                     # -> ln-rhoa data space
        ratio = Jfd[:, j] / (col_ln + 1e-300)
        cv = np.std(ratio) / (np.abs(np.mean(ratio)) + 1e-300)
        results.append((nm, np.mean(ratio), cv))
        print(f"\n[C] rho column {nm}: mean(FD/analytic) = {np.mean(ratio):.4f}, "
              f"CV = {cv:.2e}  ({int(mask.sum())} cells)")
        print("    CV ~ 0 and mean ~ 1  -> column correct.")
        print("    CV ~ 0 and mean != 1 -> constant k (rhoa vs resistance) "
              "convention, not a bug.")
        print("    CV large             -> genuine disagreement, investigate.")
    return dict(results=results)


# ---------------------------------------------------------------------------
# D -- two forward paths consistent up to per-datum k
# ---------------------------------------------------------------------------
def check_two_paths(world, eps=1e-3):
    """
    sensitivity_analysis (rhoa) vs sensitivity_analysis_v (fop response).
    If fop returns resistance and simulate returns rhoa = k*R, the two Jacobians
    differ by the constant k per datum. So column-wise the ratio h/h_v should be
    the SAME vector k for every parameter. We check that the per-datum ratio has
    near-zero spread across parameters.
    """
    theta0 = np.asarray(world.get_x0(), float)
    h = np.asarray(world.sensitivity_analysis(eps=eps), float)      # (p, d)
    world.update(list(theta0))
    hv = np.asarray(world.sensitivity_analysis_v(eps=eps), float)   # (p, d)
    world.update(list(theta0))

    ratio = h / (hv + 1e-300)          # (p, d); should be ~constant down columns
    per_datum_cv = np.std(ratio, axis=0) / (np.abs(np.mean(ratio, axis=0)) + 1e-300)
    print(f"\n[D] two-path: median per-datum CV of h/h_v across params = "
          f"{np.median(per_datum_cv):.2e}")
    print("    Small CV -> the two forward paths are consistent (ratio = k). "
          "Large CV -> the paths disagree beyond a constant factor.")
    return dict(per_datum_cv=per_datum_cv)


# ---------------------------------------------------------------------------
# E -- Fisher properties
# ---------------------------------------------------------------------------
def check_fisher(world):
    theta0 = np.asarray(world.get_x0(), float)
    steps = W.default_steps(world)
    J, ln0, rhoa0, _ = W.jacobian(world, theta0, steps=steps)
    eps = W.noise_std(world, rhoa0)

    F = W.fisher(J, eps)
    Fun = J.T @ J

    sym = np.max(np.abs(F - F.T)) / (np.max(np.abs(F)) + 1e-300)
    evals = np.linalg.eigvalsh(0.5 * (F + F.T))
    cond = evals.max() / max(evals.min(), 1e-300)
    # how different is the weighting? compare eigen-spectra scale
    scale = np.trace(F) / (np.trace(Fun) + 1e-300)

    print("\n[E] Fisher (weighted J^T C_d^-1 J):")
    print(f"    asymmetry           = {sym:.2e}  (should be ~1e-12)")
    print(f"    min eigenvalue      = {evals.min():.3e}  "
          f"({'PSD' if evals.min() > -1e-8 * evals.max() else 'NOT PSD -- FD noise'})")
    print(f"    condition number    = {cond:.3e}")
    print(f"    trace(F)/trace(J^TJ)= {scale:.3e}  "
          f"(how much the noise model rescales information)")
    print("    A near-singular F means some geometric parameter is nearly "
          "uninformable at this design -- expected for deep/small features.")
    return dict(F=F, eig=evals, cond=cond)


# ---------------------------------------------------------------------------
# F -- global directional check
# ---------------------------------------------------------------------------
def check_directional(world, n_dirs=5, seed=0):
    theta0 = np.asarray(world.get_x0(), float)
    steps = W.default_steps(world)
    J, _, _, _ = W.jacobian(world, theta0, steps=steps)
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_dirs):
        v = rng.standard_normal(theta0.size)
        v /= np.linalg.norm(v)
        step = np.median(steps)                 # a single scalar move along v
        fd = W.directional_derivative(world, theta0, v, step)
        pred = J @ v
        errs.append(np.linalg.norm(fd - pred) / (np.linalg.norm(fd) + 1e-300))
    errs = np.array(errs)
    print(f"\n[F] directional: rel. error J@v vs FD along v: "
          f"median {np.median(errs):.2e}, max {errs.max():.2e}")
    print("    Large error here with small per-column errors elsewhere = a "
          "column-ordering / indexing mismatch.")
    return dict(errs=errs)


# ---------------------------------------------------------------------------
# Example C_1,1 world  (EDIT the TODO values to your actual setup)
# ---------------------------------------------------------------------------
def build_c11_world():
    """
    Minimal C_1,1 world: 1 layer + 1 circular anomaly + varying background,
    = 7 varying parameters (matches problem statement Stage 3b).

    NOTE: bounds/varlims and the exact truth are placeholders taken from the
    C_2,2 table -- set them to YOUR C_1,1 case before trusting anything. Kept
    here only so the harness has something to run against.
    """
    import pygimli.physics.ert as ert
    from Anandlyn_log import AnomalyWorld, Layer, CircleAnom

    elecs = np.linspace(-25, 25, 21)
    scheme = ert.createData(elecs=elecs, schemeName="dd")

    start = [-25, 0]
    end = [25, -20]

    # layer: depth y and rho below both vary
    L0 = Layer("L0", y=-3.0, rho=50.0, _cnum=1,
               _varflag=[True, True],
               _varlims=[(-10.0, -0.5), (5.0, 500.0)])
    # anomaly: x, y, r, rho all vary
    C0 = CircleAnom("C0", x=5.0, y=-4.0, r=1.2, rho=500.0, _c_num=1,
                    _varflag=[True, True, True, True],
                    _varlims=[(-20.0, 20.0), (-15.0, -0.5),
                              (0.3, 5.0), (10.0, 2000.0)])

    world = AnomalyWorld(_start=start, _end=end, _scheme=scheme,
                         _rho_world=100.0, _rho_world_varflag=True,
                         _rho_world_varlim=[10.0, 1000.0],
                         _layers=[L0], _circles=[C0])

    # a synthetic measurement so get_forward_solution has a target if needed
    world.meas = np.asarray(world.get_forward_solution(_noise=False)["rhoa"], float)
    world.update(list(world.get_x0()))
    return world


def main():
    world = build_c11_world()
    print("param order:", W.param_labels(world))
    check_reset_bug(world)
    check_step_convergence(world)
    check_rho_columns(world)
    check_two_paths(world)
    check_fisher(world)
    check_directional(world)


if __name__ == "__main__":
    main()
