#!/usr/bin/env python3
"""
3D visualization of the S2PH 3D world: full mesh (wireframe/edges visible),
plus a vertical slice through the cylinder's axis and a horizontal slice at
the layer depth, both colored by resistivity -- so you can see the cylinder
and the layer boundary actually cutting through the tetrahedral mesh.

Uses pyGIMLi to build the model (same geometry as test_world3d_pygimli.py)
and pyvista for the 3D rendering (pyGIMLi's mesh export -> pyvista is more
reliable across pyGIMLi versions than relying on pg.show()'s internal 3D
backend directly).

Needs pyvista in your ERT_GUI env:  pip install pyvista

Run from the repo root, inside ERT_GUI:

    python visualize_world3d.py

Opens an interactive window with 3 panels:
  1. Full 3D mesh, semi-transparent, edges visible, cylinder visible inside
  2. Vertical slice through the cylinder's axis, colored by resistivity
  3. Horizontal slice at the layer depth, colored by resistivity

Close the window (or press 'q') to exit.
"""
import numpy as np
import pyvista as pv

from pygimli.physics import ert
from Anandlyn_log import AnomalyWorld, Layer, CylinderAnom

# ---------------------------------------------------------------------
# Build the same world as test_world3d_pygimli.py. Swap this block out
# for your own world/geometry as needed.
# ---------------------------------------------------------------------
x_min, x_max = -25.0, 25.0
y_min, y_max = -8.0, 8.0
z_top, z_bot = 0.0, -15.0
n_elec = 15

scheme = ert.createData(elecs=np.linspace(x_min, x_max, n_elec), schemeName="dd")

L1 = Layer(name="layer1", y=-4.0, rho=80.0, _cnum=1,
           _varflag=[True, False], _varlims=(z_bot, z_top))

shaft = CylinderAnom(
    name="shaft", x1=-10.0, y1=0.0, z1=-2.0, x2=10.0, y2=2.0, z2=-6.0,
    r=1.0, rho=2000.0, _c_num=2)

print("Building 3D world (mesh + resistivity map)...")
world = AnomalyWorld(
    _start=[x_min, y_min, z_top], _end=[x_max, y_max, z_bot],
    _scheme=scheme, _layers=[L1], _cylinders=[shaft],
    _rho_world=200.0, _rho_world_varflag=False)
print(f"  mesh cells={world.mesh.cellCount()}, nodes={world.mesh.nodeCount()}")

# ---------------------------------------------------------------------
# pyGIMLi mesh -> pyvista, via VTK export (most version-robust route).
# ---------------------------------------------------------------------
rho = np.asarray(world.resistivity_map, dtype=float)
try:
    world.mesh['resistivity'] = rho  # newer pyGIMLi: dict-like cell data assignment
except Exception:
    world.mesh.addData('resistivity', rho)  # older pyGIMLi fallback

vtk_path = "world3d_mesh.vtk"
world.mesh.exportVTK(vtk_path)
grid = pv.read(vtk_path)
if 'resistivity' not in grid.array_names:
    # exportVTK sometimes names the first cell array differently; grab
    # whatever single cell array is present as a fallback.
    cell_arrays = [n for n in grid.array_names if n in grid.cell_data.keys()]
    if cell_arrays:
        grid.rename_array(cell_arrays[0], 'resistivity')
    else:
        raise RuntimeError(f"no 'resistivity' array found in {vtk_path}; "
                            f"available arrays: {grid.array_names}")

log_rho = np.log10(grid['resistivity'])
grid['log10_resistivity'] = log_rho

# ---------------------------------------------------------------------
# Slice geometry: vertical plane through the cylinder axis (spanned by
# the axis direction and global z), and a horizontal plane at the layer
# depth.
# ---------------------------------------------------------------------
a = np.array([shaft.x1, shaft.y1, shaft.z1])
b = np.array([shaft.x2, shaft.y2, shaft.z2])
axis = (b - a) / np.linalg.norm(b - a)
z_hat = np.array([0.0, 0.0, 1.0])
vertical_slice_normal = np.cross(axis, z_hat)
if np.linalg.norm(vertical_slice_normal) < 1e-9:
    vertical_slice_normal = np.array([0.0, 1.0, 0.0])  # axis was vertical
else:
    vertical_slice_normal /= np.linalg.norm(vertical_slice_normal)
midpoint = (a + b) / 2.0

layer_origin = (0.0, 0.0, L1.y)

vertical_slice = grid.slice(normal=vertical_slice_normal, origin=midpoint)
horizontal_slice = grid.slice(normal=(0.0, 0.0, 1.0), origin=layer_origin)

# ---------------------------------------------------------------------
# Plot: 3 panels.
# ---------------------------------------------------------------------
pl = pv.Plotter(shape=(1, 3), window_size=(1800, 650))

pl.subplot(0, 0)
pl.add_text("Full 3D mesh (edges visible)", font_size=10)
pl.add_mesh(grid, scalars='log10_resistivity', cmap='viridis',
            show_edges=True, edge_color='gray', opacity=0.35,
            scalar_bar_args={'title': 'log10(rho)'})
pl.add_mesh(pv.Line(a, b), color='red', line_width=4, label='shaft axis')
pl.add_axes()

pl.subplot(0, 1)
pl.add_text("Vertical slice through shaft axis", font_size=10)
pl.add_mesh(vertical_slice, scalars='log10_resistivity', cmap='viridis',
            show_edges=True, edge_color='black', line_width=1,
            scalar_bar_args={'title': 'log10(rho)'})
pl.add_mesh(pv.Line(a, b), color='red', line_width=3)
pl.add_axes()

pl.subplot(0, 2)
pl.add_text(f"Horizontal slice at layer depth (z={L1.y})", font_size=10)
pl.add_mesh(horizontal_slice, scalars='log10_resistivity', cmap='viridis',
            show_edges=True, edge_color='black', line_width=1,
            scalar_bar_args={'title': 'log10(rho)'})
pl.add_axes()

pl.link_views()
pl.show()
