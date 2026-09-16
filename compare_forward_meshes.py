# -*- coding: utf-8 -*-
"""
compare_forward_meshes.py
=========================
Compare discretization strategies for GEOMETRIC-parameter sensitivity, to decide
how to build the fixed-topology forward. Candidates:

  remesh-tri   : the current AnomalyWorld path (retriangulates per theta).
                 Accurate per-configuration, but non-smooth in theta -> the bug.
  fixed-tri    : one uniform triangular mesh, hard resistivity reassignment.
  fixed-sq     : one uniform square grid, hard reassignment (circle pixelated).
  fixed-sq-soft: same square grid, SOFT (sigmoid) boundary of width w.
  fixed-sq-rect: same square grid, hard, but the anomaly is an AXIS-ALIGNED
                 SQUARE of half-width r (no boundary pixelation). This is the
                 "square anomaly" case -- included ONLY to quantify how much of
                 the artifact is circle-on-squares pixelation vs discretization
                 in general. It is a validation geometry, not the research model.

Metrics, per candidate, for one geometric parameter at a time:
  roughness      : total |2nd difference| / peak-to-peak of the response over a
                   small theta sweep. Dimensionless jaggedness; lower = smoother.
  jump-free win. : largest sub-window with no detected jump = usable FD step ceiling.
  has-valley     : does an FD step-convergence sweep show an interior minimum
                   (good) or bottom out at the grid edge (bad)?
  accuracy       : relative L2 misfit of base-point rhoa vs the remesh-tri
                   reference (fixed meshes trade accuracy for smoothness).
  cost           : parameter-mesh cell count + wall-clock per forward.

The decision: pick the candidate with the lowest roughness / clearest valley
whose accuracy is still acceptable. Square vs triangular is settled empirically
here, not by argument.

Specialized to the C_1,1 scene (1 layer + 1 circle + background, 7 params).
Run in ERT_GUI. Needs pwhg_wrapper.py and Anandlyn_log.py on the path.

pygimli spots that may need tweaking on 1.6.0 are flagged with [PYGIMLI?].
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point

import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert

import pwhg_wrapper as W


# ---------------------------------------------------------------------------
# scene decode (theta -> physical values) without touching world geometry
# ---------------------------------------------------------------------------
def decode_theta(world, theta):
    """theta (get_x0 order) -> scene dict. C_1,1 only. 10** for log params."""
    labels = W.param_labels(world)
    if len(labels) != 7:
        raise ValueError("decode_theta is specialized to the 7-param C_1,1 case.")
    L, C = world.layers[0], world.Circles[0]
    scene = dict(bg=world.rho_world, ly=L.y, lrho=L.rho,
                 cx=C.x, cy=C.y, cr=C.r, crho=C.rho)
    for val, (nm, kind) in zip(np.asarray(theta, float), labels):
        if nm == "world" and kind == "rho": scene["bg"] = 10.0 ** val
        elif nm == "L0" and kind == "y":    scene["ly"] = val
        elif nm == "L0" and kind == "rho":  scene["lrho"] = 10.0 ** val
        elif nm == "C0" and kind == "x":    scene["cx"] = val
        elif nm == "C0" and kind == "y":    scene["cy"] = val
        elif nm == "C0" and kind == "r":    scene["cr"] = 10.0 ** val
        elif nm == "C0" and kind == "rho":  scene["crho"] = 10.0 ** val
    return scene


# ---------------------------------------------------------------------------
# fixed meshes
# ---------------------------------------------------------------------------
def build_square_grid(h, start, end, elec_x):
    """
    Uniform square grid, cell size h. h must divide the electrode spacing and the
    top edge must be y=0 so electrodes land on nodes. [PYGIMLI?] simulate needs
    electrodes as nodes; grid nodes coincide when h | 2.5.
    """
    x = np.arange(start[0], end[0] + 1e-9, h)
    y = np.arange(end[1], 0 + 1e-9, h)          # end[1] negative (depth) -> 0
    mesh = pg.createGrid(x=x, y=y)
    on_node = np.all([np.any(np.isclose(x, ex)) for ex in elec_x])
    if not on_node:
        print(f"[warn] electrodes not on grid nodes at h={h}; pick h that divides "
              f"the electrode spacing.")
    return mesh


def build_uniform_tri(area, start, end, elec_x):
    """Uniform-area triangular mesh with electrode nodes and quality constraint."""
    world = mt.createWorld(start=start, end=end, worldMarker=True)
    for ex in elec_x:
        world.createNode([ex, 0.0])
        world.createNode([ex, -0.1])            # buried node aids the solve
    mesh = mt.createMesh(world, quality=34.0, area=area)
    return mesh


# ---------------------------------------------------------------------------
# resistivity assignment on a fixed mesh
# ---------------------------------------------------------------------------
def _centers(mesh):
    return np.array([[mesh.cell(i).center().x(), mesh.cell(i).center().y()]
                     for i in range(mesh.cellCount())])


def assign_hard(mesh, scene, rect=False):
    """Piecewise-constant res: background, top layer band, then anomaly overwrite."""
    cen = _centers(mesh)
    res = np.full(mesh.cellCount(), scene["bg"], float)
    inlayer = (cen[:, 1] <= 0.0) & (cen[:, 1] >= scene["ly"])   # band [ly, 0]
    res[inlayer] = scene["lrho"]
    if rect:
        inside = (np.abs(cen[:, 0] - scene["cx"]) <= scene["cr"]) & \
                 (np.abs(cen[:, 1] - scene["cy"]) <= scene["cr"])
    else:
        d = np.hypot(cen[:, 0] - scene["cx"], cen[:, 1] - scene["cy"])
        inside = d <= scene["cr"]
    res[inside] = scene["crho"]
    return res


def assign_soft(mesh, scene, width):
    """
    Smooth res(theta): log-space blend via sigmoids of signed distance. width sets
    the transition thickness (in metres). Differentiable in cx, cy, cr, ly.
    """
    cen = _centers(mesh)
    sig = lambda t: 1.0 / (1.0 + np.exp(-t / width))
    lbg, llay = np.log(scene["bg"]), np.log(scene["lrho"])
    lcir = np.log(scene["crho"])
    s_lay = sig(0.0 - cen[:, 1]) * sig(cen[:, 1] - scene["ly"])      # band [ly,0]
    d = np.hypot(cen[:, 0] - scene["cx"], cen[:, 1] - scene["cy"])
    s_cir = sig(scene["cr"] - d)
    lrho = (1 - s_cir) * ((1 - s_lay) * lbg + s_lay * llay) + s_cir * lcir
    return np.exp(lrho)


# ---------------------------------------------------------------------------
# forward callables theta -> rhoa   (all via ert.simulate for consistent units)
# ---------------------------------------------------------------------------
def make_fixed_forward(world, mesh, scheme, assign_fn):
    def f(theta):
        scene = decode_theta(world, theta)
        res = assign_fn(mesh, scene)
        data = ert.simulate(mesh, scheme=scheme, res=res, verbose=False)
        return np.asarray(data["rhoa"], float)
    return f


def make_baseline_forward(world):
    def f(theta):
        _, rhoa = W.forward(world, theta)       # remesh path
        return rhoa
    return f


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _rep_datum(resp):
    p2p = resp.max(0) - resp.min(0)
    return int(np.argmax(p2p)), p2p


def smoothness(fwd, theta0, j, hw, n=31, jump_k=6.0):
    """Sweep param j over +-hw; return roughness and largest jump-free window."""
    s = np.linspace(-hw, hw, n)
    resp = []
    for ds in s:
        th = theta0.copy(); th[j] += ds
        resp.append(fwd(th))
    resp = np.array(resp)
    d, p2p = _rep_datum(resp)
    r = resp[:, d]
    d2 = np.abs(np.diff(r, 2))
    roughness = float(d2.sum() / (p2p[d] + 1e-300))
    thr = jump_k * (np.median(d2) + 1e-300)
    events = np.where(d2 > thr)[0] + 1
    epos = s[events] if events.size else np.array([])
    bounds = np.concatenate(([s[0]], epos, [s[-1]]))
    max_gap = float(np.max(np.diff(bounds))) if bounds.size > 1 else 2 * hw
    return dict(s=s, r=r, roughness=roughness, n_jumps=int(events.size),
                max_window=max_gap)


def step_convergence(fwd, theta0, j, grid):
    """FD column of the rep-datum vs step; interior valley (good) or edge (bad)?"""
    cols = []
    for st in grid:
        thp = theta0.copy(); thp[j] += st
        thm = theta0.copy(); thm[j] -= st
        cols.append((fwd(thp) - fwd(thm)) / (2 * st))
    cols = np.array(cols)
    diff = np.linalg.norm(np.diff(cols, axis=0), axis=1)
    diff /= (np.linalg.norm(cols[1:], axis=1) + 1e-300)
    kmin = int(np.argmin(diff))
    interior = 0 < kmin < len(diff) - 1
    return dict(grid=grid, diff=diff, has_valley=bool(interior),
                best_step=float(grid[kmin + 1]), min_diff=float(diff.min()))


def accuracy(fwd, theta0, ref_rhoa):
    rhoa = fwd(theta0)
    return float(np.linalg.norm(rhoa - ref_rhoa) / (np.linalg.norm(ref_rhoa) + 1e-300))


def cost(fwd, theta0, mesh, reps=3):
    t0 = time.time()
    for _ in range(reps):
        fwd(theta0)
    dt = (time.time() - t0) / reps
    ncell = mesh.cellCount() if mesh is not None else None
    return dict(t_ms=1e3 * dt, ncells=ncell)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def compare(world, param="C0/r", h=0.5, tri_area=0.25, soft_width=None,
            hw=None, out_prefix="mesh_cmp"):
    """
    Build all candidates and compare on one geometric parameter.
    param in {'C0/x','C0/y','C0/r','L0/y'}.
    """
    labels = W.param_labels(world)
    names = [f"{nm}/{kind}" for nm, kind in labels]
    if param not in names:
        raise ValueError(f"{param} not in {names}")
    j = names.index(param)
    kind = labels[j][1]
    theta0 = np.asarray(world.get_x0(), float)

    start, end = list(world.start), list(world.end)
    elec_x = np.array(world.scheme.sensorPositions())[:, 0]
    scheme = world.scheme
    if soft_width is None:
        soft_width = h
    if hw is None:
        hw = {"x": 0.6, "y": 0.6, "r": 0.2, "rho": 0.05}[kind]

    sq = build_square_grid(h, start, end, elec_x)
    tri = build_uniform_tri(tri_area, start, end, elec_x)

    candidates = {
        "remesh-tri":    (make_baseline_forward(world), None),
        "fixed-tri":     (make_fixed_forward(world, tri, scheme, assign_hard), tri),
        "fixed-sq":      (make_fixed_forward(world, sq, scheme, assign_hard), sq),
        "fixed-sq-soft": (make_fixed_forward(world, sq, scheme,
                                             lambda m, s: assign_soft(m, s, soft_width)), sq),
        "fixed-sq-rect": (make_fixed_forward(world, sq, scheme,
                                             lambda m, s: assign_hard(m, s, rect=True)), sq),
    }

    ref_rhoa = candidates["remesh-tri"][0](theta0)     # accuracy reference

    # geometric grids for step convergence, by kind
    step_grid = {"x": np.logspace(-3, -0.3, 9), "y": np.logspace(-3, -0.3, 9),
                 "r": np.logspace(-3.5, -0.7, 9), "rho": np.logspace(-3.5, -1, 9)}[kind]

    results = {}
    fig_s, ax_s = plt.subplots(figsize=(7, 4))
    fig_c, ax_c = plt.subplots(figsize=(7, 4))

    print(f"\n=== comparison on {param}  (h={h}, tri_area={tri_area}, "
          f"soft_width={soft_width}) ===")
    print(f"{'candidate':<14} {'rough':>9} {'jumpwin':>9} {'valley':>7} "
          f"{'acc':>8} {'ncell':>7} {'t_ms':>7}")
    for name, (fwd, mesh) in candidates.items():
        sm = smoothness(fwd, theta0, j, hw)
        sc = step_convergence(fwd, theta0, j, step_grid)
        ac = accuracy(fwd, theta0, ref_rhoa)
        co = cost(fwd, theta0, mesh)
        results[name] = dict(sm=sm, sc=sc, acc=ac, cost=co)

        rr = sm["r"] - sm["r"].mean()
        ax_s.plot(sm["s"], rr, ".-", ms=3, label=name)
        ax_c.loglog(sc["grid"][1:], sc["diff"], ".-", ms=3, label=name)

        print(f"{name:<14} {sm['roughness']:>9.2e} {sm['max_window']:>9.2e} "
              f"{str(sm['n_jumps']):>7} {ac:>8.2e} "
              f"{str(co['ncells']):>7} {co['t_ms']:>7.1f}")

    ax_s.set_title(f"response vs {param} (centered)"); ax_s.set_xlabel(f"delta {kind}")
    ax_s.set_ylabel("rhoa (rep. datum)"); ax_s.legend(fontsize=7)
    fig_s.tight_layout(); fig_s.savefig(f"{out_prefix}_{kind}_smooth.png", dpi=140)
    ax_c.set_title(f"FD step convergence, {param}"); ax_c.set_xlabel("step")
    ax_c.set_ylabel("rel. successive diff"); ax_c.legend(fontsize=7)
    fig_c.tight_layout(); fig_c.savefig(f"{out_prefix}_{kind}_stepconv.png", dpi=140)
    plt.close(fig_s); plt.close(fig_c)

    print(f"figures: {out_prefix}_{kind}_smooth.png , {out_prefix}_{kind}_stepconv.png")
    print("read: lowest roughness + smallest acc + a valley = best. Compare "
          "fixed-sq vs fixed-sq-rect to isolate circle-pixelation cost.")
    return results


def main():
    from validate_sensitivity import build_c11_world
    world = build_c11_world()
    for p in ("C0/r", "C0/x", "L0/y"):
        compare(world, param=p)


if __name__ == "__main__":
    main()
