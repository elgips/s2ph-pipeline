#!/usr/bin/env python3
"""
Real-pyGIMLi smoke test for the full 3D forward path: a single survey line
over a 3D box, one flat Layer (horizontal slab), and one CylinderAnom
(tunnel/shaft). Run test_cylinder3d_pygimli.py FIRST -- it's much cheaper
and isolates the cylinder rotate/translate call on its own; only move to
this once that passes, since this one does real 3D meshing (slower, and
errors here are harder to localize).

Run from the repo root, inside ERT_GUI / pygimli_env:

    python test_world3d_pygimli.py

Builds: a single line of electrodes along x at y=0, z=0 (surface); a world
box [-25,25] x [-8,8] x [-15,0]; one layer at z=-4 (rho=80); one cylinder
("shaft") from (-10,0,-2) to (10,2,-6), r=1.0, rho=2000 -- an oblique shaft
that drifts in y and z along its length, the whole point of a 3D primitive
instead of a 2D circle/ellipse cross-section.

Checks:
  1. AnomalyWorld construction succeeds with is_3d=True (mesh + resistivity
     map built without error).
  2. A real 3D forward solve returns finite, positive rhoa.
  3. update() moving the shaft (including its y-drift and z-drift) triggers
     a real mesh rebuild and produces a different, still-finite/positive
     forward response.

NOTE: pyGIMLi's 3D ERT forward solve is much slower than 2D and the mesh
is coarser here (quality=1.2, a guess -- tune per your pyGIMLi/TetGen
version) specifically to keep this test's runtime reasonable; treat the
absolute rhoa values as sanity checks, not validated field-realistic
numbers.
"""
import os
import sys
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe: always saves a PNG, never blocks on a window
import matplotlib.pyplot as plt

from pygimli.physics import ert
from Anandlyn_log import AnomalyWorld, Layer, CylinderAnom

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def _cell_centers_and_rho(world):
    centers = np.array([[c.center().x(), c.center().y(), c.center().z()]
                        for c in world.mesh.cells()])
    rho = np.asarray(world.resistivity_map, dtype=float)
    return centers, rho


def plot_cross_section(world, title, out_path, y_slab_halfwidth=1.0):
    """
    x-z slice through the mesh near y=0 (the survey line's plane), colored
    by log10(resistivity) -- the 3D counterpart of the 2D pipeline's
    _mesh.png. Cells are picked by |y - 0| < y_slab_halfwidth rather than
    resliced, so this is a scatter of actual cell centers, not an
    interpolated image.
    """
    centers, rho = _cell_centers_and_rho(world)
    m = np.abs(centers[:, 1]) < y_slab_halfwidth
    fig, ax = plt.subplots(figsize=(9, 4))
    sc = ax.scatter(centers[m, 0], centers[m, 2], c=np.log10(rho[m]),
                    cmap="viridis", s=8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10(resistivity) [ohm.m]")

    # Cylinder axis, projected onto this slice (only meaningful near y=0,
    # which is where this slab is centered).
    for cyl in world.Cylinders:
        ax.plot([cyl.x1, cyl.x2], [cyl.z1, cyl.z2], color="red", linewidth=1.5,
                label=f"{cyl.name} axis")

    # Electrodes at the surface.
    pos = np.array(world.scheme.sensorPositions())
    ax.scatter(pos[:, 0], np.zeros(len(pos)), marker="v", color="black", s=25,
              label="electrodes", zorder=5)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m] (depth, negative down)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved plot: {out_path}")


def plot_rhoa_comparison(rhoa1, rhoa2, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    idx = np.arange(len(rhoa1))
    axes[0].plot(idx, rhoa1, label="initial geometry", marker=".", linewidth=0.8)
    axes[0].plot(idx, rhoa2, label="perturbed geometry", marker=".", linewidth=0.8)
    axes[0].set_xlabel("quadrupole index")
    axes[0].set_ylabel("apparent resistivity [ohm.m]")
    axes[0].set_title("Forward response, before vs after update()")
    axes[0].legend(fontsize=8)

    axes[1].scatter(rhoa1, rhoa2, s=10)
    lims = [min(rhoa1.min(), rhoa2.min()), max(rhoa1.max(), rhoa2.max())]
    axes[1].plot(lims, lims, color="gray", linestyle="--", linewidth=1, label="y=x")
    axes[1].set_xlabel("rhoa1 (initial)")
    axes[1].set_ylabel("rhoa2 (perturbed)")
    axes[1].set_title("1:1 comparison (off the line = geometry change took effect)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved plot: {out_path}")


def main():
    x_min, x_max = -25.0, 25.0
    y_min, y_max = -8.0, 8.0
    z_top, z_bot = 0.0, -15.0
    n_elec = 15

    scheme = ert.createData(elecs=np.linspace(x_min, x_max, n_elec), schemeName="dd")

    L1 = Layer(name="layer1", y=-4.0, rho=80.0, _cnum=1,
               _varflag=[True, False], _varlims=(z_bot, z_top))

    shaft = CylinderAnom(
        name="shaft", x1=-10.0, y1=0.0, z1=-2.0, x2=10.0, y2=2.0, z2=-6.0,
        r=1.0, rho=2000.0, _c_num=2,
        _varflag=[True, True, True, True, True, True, True, True],
        _varlims=[(x_min, x_max), (y_min, y_max), (z_bot, z_top),
                  (x_min, x_max), (y_min, y_max), (z_bot, z_top),
                  (0.2, 3.0), (10.0, 20000.0)])

    print("Building 3D world (mesh + resistivity map)...")
    t0 = time.time()
    world = AnomalyWorld(
        _start=[x_min, y_min, z_top], _end=[x_max, y_max, z_bot],
        _scheme=scheme, _layers=[L1], _cylinders=[shaft],
        _rho_world=200.0, _rho_world_varflag=False)
    print(f"  built in {time.time() - t0:.1f}s, is_3d={world.is_3d}, "
          f"mesh cells={world.mesh.cellCount()}")

    x0 = world.get_x0()
    print("get_x0:", x0, f"({len(x0)} free vars, expect 9: 1 layer + 8 cylinder)")
    if len(x0) != 9:
        fail(f"expected 9 free vars, got {len(x0)}")

    if world.constraints is None:
        fail("expected non-None constraints")
    print(f"constraint rows: {world.constraints.A.shape[0]} (expect 13)")

    mesh_nodes_before = world.mesh.nodeCount()

    plot_cross_section(world, "3D world cross-section (y~0), initial geometry",
                        os.path.join(OUT_DIR, "world3d_cross_section_initial.png"))

    print("\nForward solve #1 (initial geometry)...")
    t0 = time.time()
    data1 = world.get_forward_solution(_noise=False)
    print(f"  solved in {time.time() - t0:.1f}s")
    rhoa1 = np.asarray(data1["rhoa"])
    if not np.all(np.isfinite(rhoa1)) or not np.all(rhoa1 > 0):
        fail("forward solve #1 produced non-finite/non-positive rhoa")
    print(f"rhoa1: n={len(rhoa1)} min={rhoa1.min():.3f} max={rhoa1.max():.3f}")

    # A uniform "+0.5 to every free variable" perturbation is too blunt here:
    # two of the free vars are log10(r)/log10(rho), so +0.5 roughly TRIPLES
    # the shaft radius (1.0 -> 10**0.5 ~= 3.16). For this shaft's tilt, that
    # enlarged end-cap disc pokes up to z~=+1.6 -- above the surface (z=0),
    # i.e. outside the world box entirely. That's exactly the containment
    # violation generate_auto_constraints() exists to prevent during a real
    # optimization run (differential_evolution respects the constraint);
    # update() itself does not validate its input, so it crashed TetGen
    # instead of failing cleanly. Use a smaller, per-variable-aware delta
    # instead: +0.3 for the six linear position vars, only +0.05 for the two
    # log-scaled vars (log10(r), log10(rho)), which stays comfortably inside
    # get_bounds() for this geometry.
    is_log_idx = set()
    for (ent_name, pname), idx in world.variable_map.items():
        entity = next((e for e in ([L1] + [shaft]) if e.name == ent_name), None)
        if entity is not None and getattr(entity, 'log_flags', {}).get(pname, False):
            is_log_idx.add(idx)
    x_perturbed = [v + (0.05 if i in is_log_idx else 0.3) for i, v in enumerate(x0)]

    lb, ub = zip(*world.get_bounds())
    if not (np.all(np.asarray(x_perturbed) >= np.asarray(lb) - 1e-9)
            and np.all(np.asarray(x_perturbed) <= np.asarray(ub) + 1e-9)):
        fail(f"test perturbation itself violates get_bounds() -- fix the "
             f"deltas above rather than calling update() with this x:\n"
             f"  x_perturbed={x_perturbed}\n  lb={lb}\n  ub={ub}")

    world.update(x_perturbed)
    if not np.allclose(world.get_x0(), x_perturbed):
        fail(f"update() did not apply: get_x0()={world.get_x0()} vs {x_perturbed}")
    print("\nupdate() applied (incl. shaft y/z drift) -- OK")

    mesh_nodes_after = world.mesh.nodeCount()
    if mesh_nodes_after == mesh_nodes_before:
        print("WARNING: mesh node count unchanged -- 3D rebuild may not have triggered.")
    else:
        print(f"mesh rebuilt: {mesh_nodes_before} -> {mesh_nodes_after} nodes -- OK")

    plot_cross_section(world, "3D world cross-section (y~0), perturbed geometry",
                        os.path.join(OUT_DIR, "world3d_cross_section_perturbed.png"))

    print("\nForward solve #2 (perturbed geometry)...")
    t0 = time.time()
    data2 = world.get_forward_solution(_noise=False)
    print(f"  solved in {time.time() - t0:.1f}s")
    rhoa2 = np.asarray(data2["rhoa"])
    if not np.all(np.isfinite(rhoa2)) or not np.all(rhoa2 > 0):
        fail("forward solve #2 produced non-finite/non-positive rhoa")
    if np.allclose(rhoa1, rhoa2):
        fail("forward response identical before/after perturbing geometry")
    print(f"rhoa2: n={len(rhoa2)} min={rhoa2.min():.3f} max={rhoa2.max():.3f}")
    print(f"mean |log10 ratio| change vs rhoa1: "
          f"{np.mean(np.abs(np.log10(rhoa2 / rhoa1))):.4f}")

    plot_rhoa_comparison(rhoa1, rhoa2, os.path.join(OUT_DIR, "world3d_rhoa_comparison.png"))

    print("\nALL 3D WORLD CHECKS PASSED.")
    print("Plots saved next to this script: world3d_cross_section_initial.png, "
          "world3d_cross_section_perturbed.png, world3d_rhoa_comparison.png")

    print("\nOpening interactive pyvista window (final/perturbed geometry) -- "
          "drag the plane widget to slice through the model, use the "
          "checkboxes (top-left) to show/hide each region, close the "
          "window to end the script...")
    world.show_mesh_interactive(cMap="Spectral_r", logScale=True)


if __name__ == "__main__":
    main()
