#!/usr/bin/env python3
"""
Geometry-ONLY smoke test for CylinderAnom's pyGIMLi 3D block construction
(mt.createCylinder + rotate()/translate()) -- the one piece of the 3D
extension that was written without a local pyGIMLi install and so is the
most likely thing to need a small API fix on your machine (the exact
rotate()/translate() call signature can differ across pyGIMLi versions).

Run this BEFORE test_world3d_pygimli.py / a full forward solve: it's much
cheaper (no meshing, no physics) and isolates exactly where a signature
mismatch would show up.

Run from the repo root, inside ERT_GUI / pygimli_env:

    python test_cylinder3d_pygimli.py

Checks, for a cylinder from (0,0,0) to (10,0,-10) (a 45-degree shaft) and
one straight down from (0,0,0) to (0,0,-5):
  1. CylinderAnom(...) construction doesn't raise (the rotate/translate
     calls succeed against your installed pyGIMLi).
  2. The resulting PLC's bounding box is centered near the segment
     midpoint and sized roughly consistent with the segment length + 2r
     (a rough sanity check that the rotation actually pointed the cylinder
     along the segment, rather than leaving it along z or collapsing it).

If step 1 raises, the traceback will point at exactly which pyGIMLi call
needs adjusting in CylinderAnom._build_block() in Anandlyn_log.py.
"""
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe: always saves a PNG, never blocks on a window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the '3d' projection)

from Anandlyn_log import CylinderAnom

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def _block_node_coords(block):
    """(N,3) array of every node position in a pyGIMLi PLC/mesh."""
    return np.array([[n.pos().x(), n.pos().y(), n.pos().z()] for n in block.nodes()])


def _plot_cylinder(name, a, b, r, nodes, out_path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=6, alpha=0.5, label="block nodes")
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="red", linewidth=2, label="axis (a->b)")
    ax.scatter(*a, color="green", s=60, label="start (a)")
    ax.scatter(*b, color="orange", s=60, label="end (b)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"CylinderAnom '{name}'  r={r}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved plot: {out_path}")


def check_cylinder(name, x1, y1, z1, x2, y2, z2, r):
    print(f"--- {name}: ({x1},{y1},{z1}) -> ({x2},{y2},{z2}), r={r} ---")
    try:
        cyl = CylinderAnom(name, x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2, r=r, rho=1000.0, _c_num=1)
    except Exception as e:
        fail(f"CylinderAnom construction raised: {type(e).__name__}: {e}")

    bb = cyl.block.boundingBox()
    # .x()/.y()/.z() are used elsewhere in Anandlyn_log.py for RVector3-like
    # objects (e.g. cell.center().x()), so used here too rather than
    # indexing, which some pyGIMLi point/vector types don't support.
    bb_min = np.array([bb.min().x(), bb.min().y(), bb.min().z()])
    bb_max = np.array([bb.max().x(), bb.max().y(), bb.max().z()])
    center = (bb_min + bb_max) / 2.0
    expected_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, (z1 + z2) / 2.0])
    print(f"bbox min={bb_min} max={bb_max}")
    print(f"bbox center={center}  expected~{expected_center}")

    # Plot BEFORE asserting, so a failing case still leaves a picture behind
    # to look at (the fastest way to see *how* rotate/translate went wrong).
    out_path = os.path.join(OUT_DIR, f"cylinder_{name}.png")
    _plot_cylinder(name, np.array([x1, y1, z1]), np.array([x2, y2, z2]), r,
                  _block_node_coords(cyl.block), out_path)

    if not np.allclose(center, expected_center, atol=0.5):
        fail(f"{name}: bbox center {center} far from expected segment midpoint {expected_center} "
             f"-- rotate()/translate() likely didn't apply as intended (see {out_path})")

    length = float(np.linalg.norm(np.array([x2, y2, z2]) - np.array([x1, y1, z1])))
    diag = float(np.linalg.norm(bb_max - bb_min))
    print(f"segment length={length:.2f}  bbox diagonal={diag:.2f} (expect >= length, roughly length+2r)")
    if diag < length * 0.9:
        fail(f"{name}: bbox diagonal ({diag:.2f}) is smaller than the segment length ({length:.2f}) -- "
             f"the cylinder is probably still oriented along z instead of the segment (see {out_path})")
    print(f"{name}: OK\n")


check_cylinder("diag_shaft", 0.0, 0.0, 0.0, 10.0, 0.0, -10.0, 1.0)
check_cylinder("straight_down", 0.0, 0.0, 0.0, 0.0, 0.0, -5.0, 0.5)
check_cylinder("along_y", -3.0, -5.0, -2.0, -3.0, 5.0, -2.0, 0.8)

print("ALL CYLINDER GEOMETRY CHECKS PASSED.")
print(f"Plots saved next to this script: cylinder_diag_shaft.png, "
      f"cylinder_straight_down.png, cylinder_along_y.png")
