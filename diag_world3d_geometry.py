#!/usr/bin/env python3
"""
Diagnostic for the TetGen crash ("terminate called after throwing an
instance of 'int'" / Aborted) seen when meshing the full 3D world.

TetGen's abort is a C++-level `terminate`/core-dump: it kills the whole
process, so Python can't catch it and move on to the next stage in the same
run. Each stage below is therefore run in its own subprocess (this script
re-invokes itself with --stage N) so a crash in one stage doesn't prevent
the others from running and reporting their result.

Stages:
  1: single box, no layers, no cylinder, no electrodes
  2: box built as two stacked slabs (layer at z=-4), merged -- the OLD
     per-layer-slab construction
  3: stage 2 + electrode nodes, no cylinder
  4: stage 3 + the oblique CylinderAnom that crosses the z=-4 layer
     boundary -- this is the OLD code path and is EXPECTED to crash
  5: the FIX under test: a single full-depth box (no per-layer slabs) +
     electrodes + the same oblique cylinder -- should NOT crash

Each stage exports its PLC to a .poly file next to this script before
meshing, so a crash still leaves the geometry file to inspect.

Run from the repo root, inside ERT_GUI:

    python diag_world3d_geometry.py
"""
import os
import sys
import subprocess
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

x_min, x_max = -25.0, 25.0
y_min, y_max = -8.0, 8.0
z_top, z_bot = 0.0, -15.0
size_x, size_y = x_max - x_min, y_max - y_min
cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
layer_z = -4.0  # same single layer as test_world3d_pygimli.py


def _run_stage(stage):
    import pygimli as pg
    import pygimli.meshtools as mt

    def try_mesh(name, geom, quality=1.2):
        poly_path = os.path.join(OUT_DIR, f"world_{name}.poly")
        try:
            mt.exportPLC(geom, poly_path)
            print(f"  exported PLC: {poly_path}")
        except Exception as e:
            print(f"  (could not export PLC: {type(e).__name__}: {e})")
        print(f"  meshing '{name}' (quality={quality}) ...")
        sys.stdout.flush()
        mesh = mt.createMesh(geom, quality=quality)
        print(f"  OK: {name} -> {mesh.nodeCount()} nodes, {mesh.cellCount()} cells")
        return mesh

    if stage == 1:
        print("=== stage 1: single box, no layers, no cylinder, no electrodes ===")
        box_only = mt.createCube(size=[size_x, size_y, abs(z_top - z_bot)],
                                  pos=[cx, cy, (z_top + z_bot) / 2.0],
                                  marker=1, boundaryMarker=10)
        try_mesh("stage1_box_only", box_only)

    elif stage == 2:
        print("=== stage 2: box built as TWO stacked slabs (layer at z=-4), merged (OLD) ===")
        slabs = [(z_top, layer_z), (layer_z, z_bot)]
        geom2 = None
        for i, (zt, zb) in enumerate(slabs, start=1):
            depth = abs(zt - zb)
            slab = mt.createCube(size=[size_x, size_y, depth],
                                  pos=[cx, cy, (zt + zb) / 2.0],
                                  marker=i, boundaryMarker=10)
            geom2 = slab if geom2 is None else mt.mergePLC([geom2, slab])
        try_mesh("stage2_layered_box", geom2)

    elif stage == 3:
        print("=== stage 3: stage 2 + electrode nodes (15 electrodes along x, y=0) (OLD) ===")
        slabs = [(z_top, layer_z), (layer_z, z_bot)]
        geom2 = None
        for i, (zt, zb) in enumerate(slabs, start=1):
            depth = abs(zt - zb)
            slab = mt.createCube(size=[size_x, size_y, depth],
                                  pos=[cx, cy, (zt + zb) / 2.0],
                                  marker=i, boundaryMarker=10)
            geom2 = slab if geom2 is None else mt.mergePLC([geom2, slab])
        elecs_x = np.linspace(x_min, x_max, 15)
        for ex in elecs_x:
            p = [ex, 0.0, 0.0]
            geom2.createNode(p)
            geom2.createNode([p[0], p[1], p[2] - 0.1])
        try_mesh("stage3_box_electrodes", geom2)

    elif stage == 4:
        print("=== stage 4: stage 3 + oblique cylinder crossing z=-4 (OLD, EXPECTED TO CRASH) ===")
        sys.path.insert(0, OUT_DIR)
        from Anandlyn_log import CylinderAnom
        slabs = [(z_top, layer_z), (layer_z, z_bot)]
        geom2 = None
        for i, (zt, zb) in enumerate(slabs, start=1):
            depth = abs(zt - zb)
            slab = mt.createCube(size=[size_x, size_y, depth],
                                  pos=[cx, cy, (zt + zb) / 2.0],
                                  marker=i, boundaryMarker=10)
            geom2 = slab if geom2 is None else mt.mergePLC([geom2, slab])
        elecs_x = np.linspace(x_min, x_max, 15)
        for ex in elecs_x:
            p = [ex, 0.0, 0.0]
            geom2.createNode(p)
            geom2.createNode([p[0], p[1], p[2] - 0.1])
        shaft = CylinderAnom("shaft", x1=-10.0, y1=0.0, z1=-2.0, x2=10.0, y2=2.0, z2=-6.0,
                              r=1.0, rho=2000.0, _c_num=2)
        geom4 = mt.mergePLC([geom2, shaft.block])
        try_mesh("stage4_full_world", geom4)

    elif stage == 5:
        print("=== stage 5: SINGLE full-depth box (FIX) + electrodes + same oblique cylinder ===")
        sys.path.insert(0, OUT_DIR)
        from Anandlyn_log import CylinderAnom
        box5 = mt.createCube(size=[size_x, size_y, abs(z_top - z_bot)],
                              pos=[cx, cy, (z_top + z_bot) / 2.0],
                              marker=1, boundaryMarker=10)
        elecs_x = np.linspace(x_min, x_max, 15)
        for ex in elecs_x:
            p = [ex, 0.0, 0.0]
            box5.createNode(p)
            box5.createNode([p[0], p[1], p[2] - 0.1])
        shaft5 = CylinderAnom("shaft", x1=-10.0, y1=0.0, z1=-2.0, x2=10.0, y2=2.0, z2=-6.0,
                               r=1.0, rho=2000.0, _c_num=2)
        geom5 = mt.mergePLC([box5, shaft5.block])
        try_mesh("stage5_single_box_fix", geom5)

    elif stage == 6:
        print("=== stage 6: FIX geometry, but electrodes added AFTER merging the "
              "cylinder (exact order _build_geometry_3d uses) ===")
        sys.path.insert(0, OUT_DIR)
        from Anandlyn_log import CylinderAnom
        box6 = mt.createCube(size=[size_x, size_y, abs(z_top - z_bot)],
                              pos=[cx, cy, (z_top + z_bot) / 2.0],
                              marker=1, boundaryMarker=10)
        shaft6 = CylinderAnom("shaft", x1=-10.0, y1=0.0, z1=-2.0, x2=10.0, y2=2.0, z2=-6.0,
                               r=1.0, rho=2000.0, _c_num=2)
        geom6 = mt.mergePLC([box6, shaft6.block])
        elecs_x = np.linspace(x_min, x_max, 15)
        for ex in elecs_x:
            p = [ex, 0.0, 0.0]
            geom6.createNode(p)
            geom6.createNode([p[0], p[1], p[2] - 0.1])
        try_mesh("stage6_electrodes_after_merge", geom6)

    elif stage == 7:
        print("=== stage 7: FIX geometry with the PERTURBED cylinder from "
              "test_world3d_pygimli.py's update() call (x1=-9.5,y1=0.5,z1=-1.5, "
              "x2=10.5,y2=2.5,z2=-5.5, r=10**0.5~3.162) -- reproduces the "
              "rebuild-after-update() crash ===")
        sys.path.insert(0, OUT_DIR)
        from Anandlyn_log import CylinderAnom
        box7 = mt.createCube(size=[size_x, size_y, abs(z_top - z_bot)],
                              pos=[cx, cy, (z_top + z_bot) / 2.0],
                              marker=1, boundaryMarker=10)
        elecs_x = np.linspace(x_min, x_max, 15)
        for ex in elecs_x:
            p = [ex, 0.0, 0.0]
            box7.createNode(p)
            box7.createNode([p[0], p[1], p[2] - 0.1])
        shaft7 = CylinderAnom("shaft", x1=-9.5, y1=0.5, z1=-1.5, x2=10.5, y2=2.5, z2=-5.5,
                               r=10.0 ** 0.5, rho=6309.57, _c_num=2)
        geom7 = mt.mergePLC([box7, shaft7.block])
        try_mesh("stage7_perturbed_radius", geom7)

    elif stage == 8:
        print("=== stage 8: same as stage 7 but r left at the ORIGINAL 1.0 "
              "(only x1,y1,z1,x2,y2,z2 perturbed by +0.5, r unperturbed) -- "
              "isolates whether the radius jump specifically is the cause ===")
        sys.path.insert(0, OUT_DIR)
        from Anandlyn_log import CylinderAnom
        box8 = mt.createCube(size=[size_x, size_y, abs(z_top - z_bot)],
                              pos=[cx, cy, (z_top + z_bot) / 2.0],
                              marker=1, boundaryMarker=10)
        elecs_x = np.linspace(x_min, x_max, 15)
        for ex in elecs_x:
            p = [ex, 0.0, 0.0]
            box8.createNode(p)
            box8.createNode([p[0], p[1], p[2] - 0.1])
        shaft8 = CylinderAnom("shaft", x1=-9.5, y1=0.5, z1=-1.5, x2=10.5, y2=2.5, z2=-5.5,
                               r=1.0, rho=6309.57, _c_num=2)
        geom8 = mt.mergePLC([box8, shaft8.block])
        try_mesh("stage8_perturbed_position_only", geom8)

    else:
        raise ValueError(f"unknown stage {stage}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--stage":
        _run_stage(int(sys.argv[2]))
        sys.exit(0)

    results = {}
    for stage in [7, 8]:
        print(f"\n########## running stage {stage} in a subprocess ##########")
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--stage", str(stage)],
                               cwd=OUT_DIR)
        ok = (proc.returncode == 0)
        results[stage] = ok
        status = "OK" if ok else f"CRASHED (exit code {proc.returncode})"
        print(f"---- stage {stage}: {status} ----")

    print("\n================ SUMMARY ================")
    for stage, ok in results.items():
        print(f"  stage {stage}: {'OK' if ok else 'CRASHED'}")
    print("\nStages 1-6 already confirmed (electrodes-before-cylinder-merge "
          "fix works for the INITIAL build). Stage 7 reproduces the exact "
          "perturbed cylinder from update() (r roughly tripled, 1.0 -> "
          "10**0.5). Stage 8 is the same position perturbation with r left "
          "at 1.0. If stage 7 crashes but stage 8 doesn't, the radius jump "
          "itself is the trigger (e.g. the enlarged cylinder now grazes the "
          "box boundary or an electrode node); if both crash, it's the "
          "position perturbation instead.")
