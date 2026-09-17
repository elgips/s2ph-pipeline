# s2ph_3D update package

Drop these files into your `s2ph_3D` folder on the git machine, overwriting
the existing copies (or `git add -A` / commit as usual).

## Files in this package

- `Anandlyn_log.py` — **updated**
- `test_world3d_pygimli.py` — **updated**
- `test_cylinder3d_pygimli.py` — unchanged (included for convenience)
- `diag_world3d_geometry.py` — unchanged (included for convenience)
- `visualize_world3d.py` — unchanged (included for convenience)

## Changelog

### `Anandlyn_log.py`

1. **`AnomalyWorld.invert_2d(...)`** (new method) — runs a standard
   pyGIMLi `ERTManager` 2.5D inversion on the survey line's
   forward-simulated data. Reuses `self.scheme` directly (the same
   straight line already used for the 3D forward solve is exactly what
   `ERTManager` expects — no separate 2D-only scheme needed). Defaults to
   noisy synthetic data (3% relative error by default; inverting
   noise-free data is unrealistically easy and doesn't test anything
   meaningful about model recovery). Stores `mgr_2d` / `model_2d` /
   `chi2_2d` / `rrms_2d` on the world and also returns `(mgr, model)`.
   Lets a dataset generated on the TRUE 3D geometry (an oblique shaft
   drifting off-line, real topography, etc.) be inverted the way field
   data normally is — assuming no along-strike structure — so the
   resulting image can be compared against the world's actual resistivity
   distribution to quantify what the 2D/2.5D assumption costs when the
   true structure isn't 2D. Includes the same flat-model sanity warning
   used in the Acre field-data pipeline (fires if the inversion never
   moves off the homogeneous starting guess).

2. **`show_mesh_interactive(...)`** — added a "Show electrodes" checkbox
   (same column as the "Mesh only" / "Transparent" / "Opaque" / "Show
   mesh" controls) that toggles black sphere markers at each electrode's
   surface position. Positions are computed the same way they're placed
   as exact mesh nodes in `_build_mesh_3d_gmsh`: x, y from
   `scheme.sensorPositions()`, z pinned to the domain's top face
   (`self.start[2]`). Independent of the clip plane and region/display
   controls — electrodes sit on the survey datum, not inside any one
   clipped region.

### `test_world3d_pygimli.py`

- Added `plot_true_vs_2d_inversion(world, out_path)` — a side-by-side
  plot: (left) the true resistivity cross-section near y=0, same
  convention as the existing `plot_cross_section`; (right) the 2.5D
  `ERTManager` inversion of the same line via `pg.show(world.mgr_2d.
  paraDomain, world.model_2d, ...)`, titled with chi2/rrms.
- `main()` now calls `world.invert_2d(noise=True, noise_level=3.0,
  lam=20.0, cType=1)` after the existing forward-solve checks and saves
  the comparison plot as `world3d_2d_inversion_vs_true.png`, before
  opening the interactive pyvista viewer.

## Suggested commit message

```
Add 2.5D ERTManager inversion of the survey line + electrode display toggle

- AnomalyWorld.invert_2d(): standard pyGIMLi 2.5D inversion of the
  existing survey line's forward-simulated data, for comparing against
  the true 3D geometry (model error from the 2D/2.5D assumption).
- show_mesh_interactive(): on/off checkbox for electrode markers.
- test_world3d_pygimli.py: exercises invert_2d() and saves a
  true-vs-inverted comparison plot.
```

## Verification

Both files were sandbox-tested (syntax-checked and run against a small
synthetic 3D world) before being sent back to this machine: `invert_2d()`
runs end-to-end with no exceptions and the electrode-coordinate
computation is correct. The sandbox's pyGIMLi build has an unrelated,
environment-specific zero-Jacobian bug that flattens every inversion run
there, so inversion *quality* could only be verified on your machine, not
in the sandbox — the flat-model warning is expected to fire there and
should NOT fire on your machine if it's still working the way earlier
Acre-script runs showed.
