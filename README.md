# s2ph_3D update bundle

Drop these files into your local `s2ph_3D/` folder (overwriting the existing
copies) and commit. Suggested commit message below.

## What changed

**`Anandlyn_log.py`**

- `_build_geometry_3d` / `_parse_all_3d` now delegate to a new
  `_build_mesh_3d_gmsh`, which replaces the old pyGIMLi `mergePLC`-based 3D
  mesh construction with an exact boolean CSG build via gmsh's OCC kernel
  (`occ.addBox` / `occ.addCylinder` / `occ.fragment`). This gives exact,
  conformal boundaries between layer slabs and `CylinderAnom` bodies, even
  when a cylinder crosses a layer boundary at a shallow angle — the old
  approach could crash TetGet or silently miscount there.
  - Electrode positions are fragmented into the CSG *together* with the
    box/cylinder solids (as 0D tool entities in the same `occ.fragment`
    call), not embedded afterwards via a separate `mesh.embed()` call. The
    latter was tried first and turned out to make gmsh's 3D tet
    reconstruction silently drop an entire volume near the electrodes (no
    exception — just a swallowed "No elements in volume N" warning),
    which showed up downstream as pyGIMLi's
    `"There is a requested electrode that does not match the given mesh."`
    Fragmenting the points in from the start avoids it entirely (verified:
    0 warnings, every electrode lands on an exact mesh node, real forward
    solve finite/positive, before *and* after `update()` moves the
    geometry).
  - Region markers are tracked in `self._region_markers`, mapping
    `(layer_idx, cyl_idx)` → integer marker (`cyl_idx=None` for
    background-only regions). These survive into the pyGIMLi mesh as
    `mesh.cellMarkers()`.
  - Requires `gmsh` (`pip install gmsh`); on Linux you may also need
    `libglu1-mesa libgl1 libxft2 libxinerama1 libxcursor1 libxrandr2 libxi6`
    at the OS level if `import gmsh` fails with a missing `.so`.

- New `show_mesh_interactive(cMap="Spectral_r", logScale=True)` on
  `AnomalyWorld`: opens an interactive pyvista window for a 3D world with:
  - a draggable clip-plane widget (grab/rotate it to slice into the model)
  - one checkbox per region (left column) to show/hide each
    `(layer, cylinder)` region independently — e.g. hide the layers to
    isolate a shaft, or the reverse
  - display-mode checkboxes (right column): **Mesh only** (wireframe),
    **Transparent** (35% opacity surface), **Opaque** (default), plus an
    independent **Show mesh** toggle that overlays cell edges on any mode
  - falls back to the plain `show_mesh()` (`pg.show`) for a 2D world.

**`test_world3d_pygimli.py`**

- Calls `world.show_mesh_interactive()` at the end of the run (after both
  forward-solve checks pass) instead of just saving static PNGs, so you get
  a live look at the final/perturbed geometry.

**`test_cylinder3d_pygimli.py`, `diag_world3d_geometry.py`,
`visualize_world3d.py`** — unchanged, included only so the folder is
self-consistent; no need to overwrite if you haven't touched them locally.

## Suggested commit message

```
3D: exact gmsh CSG meshing + interactive pyvista viewer

- Replace mergePLC-based 3D mesh build with gmsh OCC boolean CSG
  (exact layer/cylinder boundaries, fixes TetGen crash on shallow
  layer-crossing cylinders)
- Fragment electrodes into the CSG directly instead of a post-hoc
  mesh.embed() call (fixes silently-dropped near-surface volume /
  "electrode does not match mesh" on real 3D worlds)
- Add AnomalyWorld.show_mesh_interactive(): clip-plane, per-region
  show/hide, mesh/transparent/opaque display modes
- Wire the new viewer into test_world3d_pygimli.py
```

## Before running on this machine

1. `pip install gmsh pyvista` if not already present in this env.
2. Run `test_cylinder3d_pygimli.py` first (cheap, isolates the
   `CylinderAnom` block math).
3. Then `test_world3d_pygimli.py` for the full check + interactive viewer.
