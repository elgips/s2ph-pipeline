# -*- coding: utf-8 -*-
"""
simple straight layers ERT optimal inversion
@author: Elad Gips
"""
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import scipy as sp
from scipy.optimize import minimize, LinearConstraint, differential_evolution, NonlinearConstraint, shgo
from shapely.geometry import Point, Polygon
from shapely.affinity import scale as _shapely_scale, rotate as _shapely_rotate, translate as _shapely_translate
import copy
from IPython.display import display
from scipy.stats import qmc


def generate_hybrid_population(bounds, x0, epsilon=0.1, popsize_multiplier=15, guess_ratio=0.25, relative=True):
    """
    Generates a hybrid initial population combining Latin Hypercube Sampling (LHS)
    and random variations around an initial guess.
    
    Parameters:
    -----------
    bounds : list of tuples
        The optimization bounds, e.g., [(-5, 5), (-5, 5)]
    x0 : array-like
        The initial guess vector.
    epsilon : float
        The radius of variation. If relative=True, this is a percentage (e.g., 0.1 = 10%).
    popsize_multiplier : int
        The multiplier used by SciPy to determine total population (popsize * ndim).
    guess_ratio : float
        The fraction of the total population that should cluster around x0 (0.0 to 1.0).
    relative : bool
        If True, variation is relative to the magnitude of each element in x0.
        If False, variation is absolute.
    """
    bounds = np.array(bounds)
    x0 = np.array(x0, dtype=float)
    ndim = len(bounds)
    
    # 1. Calculate precise population sizes required by SciPy
    total_pop = popsize_multiplier * ndim
    num_around_guess = int(np.round(total_pop * guess_ratio))
    num_around_guess = max(1, min(num_around_guess, total_pop - 1)) # Ensure at least 1 of each
    num_lhs = total_pop - num_around_guess
    
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    
    # 2. Generate Latin Hypercube Samples (LHS)
    sampler = qmc.LatinHypercube(d=ndim)
    sample = sampler.random(n=num_lhs)
    p_lhs = qmc.scale(sample, lower_bounds, upper_bounds)
    
    # 3. Determine the variation bounds for each element
    if relative:
        # Avoid zero-magnitude issues by adding a tiny floor value if an element is exactly 0
        magnitudes = np.abs(x0)
        magnitudes[magnitudes == 0] = 1e-5 
        delta = epsilon * magnitudes
    else:
        delta = np.full(ndim, epsilon)
        
    # 4. Generate points around the initial guess
    # Generates uniform noise between [-delta, +delta] for each dimension
    noise = np.random.uniform(-delta, delta, size=(num_around_guess, ndim))
    p_guess = x0 + noise
    
    # 5. Combine and strictly enforce system bounds
    p_combined = np.vstack([p_lhs, p_guess])
    p_combined = np.clip(p_combined, lower_bounds, upper_bounds)
    
    return p_combined

def generate_hybrid_population_mixed(bounds, x0, param_types, epsilon=0.25,
                                      popsize_multiplier=15, guess_ratio=0.25,
                                      relative=True):
    """
    Like generate_hybrid_population, but treats parameters differently by kind:
      - param_types[i] == 'rho'  -> dimension i is sampled uniformly across its
        FULL bounds range for every population member (both the LHS part and
        the "around x0" part), since resistivity contrast sign/magnitude is
        the thing under test and shouldn't be pre-seeded near the truth.
      - param_types[i] == 'pos'  -> dimension i behaves like in
        generate_hybrid_population: LHS for the exploration part, clustered
        within +/-epsilon (relative to |x0[i]| if relative=True) for the
        guess part.

    Parameters
    ----------
    bounds : list of (lb, ub) tuples, in optimizer order (matches get_x0()).
    x0 : array-like, the initial guess vector, same order as bounds.
    param_types : list of str, same length as x0, each 'pos' or 'rho'.
    epsilon, popsize_multiplier, guess_ratio, relative : see
        generate_hybrid_population.
    """
    bounds = np.array(bounds, dtype=float)
    x0 = np.array(x0, dtype=float)
    param_types = np.array(param_types)
    ndim = len(bounds)
    if len(param_types) != ndim:
        raise ValueError(f"param_types length ({len(param_types)}) must match "
                          f"bounds/x0 length ({ndim})")

    rho_mask = (param_types == 'rho')
    pos_mask = ~rho_mask

    total_pop = popsize_multiplier * ndim
    num_around_guess = int(np.round(total_pop * guess_ratio))
    num_around_guess = max(1, min(num_around_guess, total_pop - 1))
    num_lhs = total_pop - num_around_guess

    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    # 1. LHS part: full range in every dimension (unchanged from the base fn)
    sampler = qmc.LatinHypercube(d=ndim)
    sample = sampler.random(n=num_lhs)
    p_lhs = qmc.scale(sample, lower_bounds, upper_bounds)

    # 2. "Guess" part: start from x0 everywhere, then...
    if relative:
        magnitudes = np.abs(x0)
        magnitudes[magnitudes == 0] = 1e-5
        delta = epsilon * magnitudes
    else:
        delta = np.full(ndim, epsilon)

    noise = np.random.uniform(-delta, delta, size=(num_around_guess, ndim))
    p_guess = x0 + noise

    # ...then overwrite the rho-type columns with independent full-range
    # uniform draws, so rho parameters never get clustered near the guess.
    if rho_mask.any():
        rho_uniform = np.random.uniform(
            lower_bounds[rho_mask], upper_bounds[rho_mask],
            size=(num_around_guess, rho_mask.sum())
        )
        p_guess[:, rho_mask] = rho_uniform

    # 3. Combine and clip to bounds (clip is a no-op for the rho columns,
    # which are already drawn inside bounds; it only guards the pos columns).
    p_combined = np.vstack([p_lhs, p_guess])
    p_combined = np.clip(p_combined, lower_bounds, upper_bounds)

    return p_combined
# def assign_layer_resistivities(mesh, layers, resistivities, default_resistivity=1e6):
#     """
#     Assigns resistivity values to the mesh cells based on the layers.
#
#     Parameters:
#     - mesh: pg.Mesh object, the mesh to which resistivities will be assigned.
#     - layers: list of tuples, each tuple contains (top_y, bottom_y) coordinates defining a layer.
#     - resistivities: list of float, the resistivity values corresponding to each layer.
#
#     Returns:
#     - res_map: numpy array, resistivity values assigned to each cell of the mesh.
#     """
#     if len(layers) != len(resistivities):
#         raise ValueError("Number of layers must match number of resistivity values.")
#
#     # Initialize resistivity map with a default value, e.g., resistivity of air
#     res_map = [default_resistivity] * mesh.cellCount()
#
#     # Iterate over each layer and assign the corresponding resistivity to cells within it
#     for layer, res in zip(layers, resistivities):
#         top_y, bottom_y = layer
#         for cell_index, cell in enumerate(mesh.cells()):
#             cell_center_y = cell.center().y()
#             if bottom_y <= cell_center_y <= top_y:
#                 res_map[cell_index] = res
#
#     return np.array(res_map)
def get_top_solutions(res1, n_elite):
    """
    Extract top N solutions from differential evolution result.

    Parameters:
    -----------
    res1 : OptimizeResult
        Result from differential evolution Stage 1
    n_elite : int
        Number of top solutions to extract

    Returns:
    --------
    elite_solutions : ndarray
        Array of shape (n_elite, ndim) with best solutions
    """
    import numpy as np

    # Method 1: If population is accessible (scipy >= 1.9.0)
    try:
        population = res1.population
        population_energies = res1.population_energies

        # Sort by fitness (MSE)
        sorted_indices = np.argsort(population_energies)
        elite_indices = sorted_indices[:n_elite]
        elite_solutions = population[elite_indices]

        return elite_solutions

    except AttributeError:
        # Method 2: Fallback if population not saved
        print("Warning: Population not accessible, using perturbation around best solution")
        x_best = res1.x
        ndim = len(x_best)

        # Create diverse solutions around best
        elite_solutions = []
        for i in range(n_elite):
            scale = 0.1 * (1 + i / n_elite)  # Varying scales
            perturbed = x_best + scale * np.random.randn(ndim)
            elite_solutions.append(perturbed)

        return np.array(elite_solutions)

def fmt_array(arr):
    return "[" + " ".join(f"{v:.2e}" for v in arr) + "]"

def assign_layer_resistivities(mesh, layers, resistivities, default_resistivity=1e6):
    if len(layers) != len(resistivities):
        raise ValueError("Number of layers must match number of resistivity values.")

    res_map = np.full(mesh.cellCount(), default_resistivity, dtype=float)

    for layer, res in zip(layers, resistivities):
        top_y, bottom_y = layer
        for cell_index, cell in enumerate(mesh.cells()):
            cell_center_y = cell.center().y()
            if bottom_y <= cell_center_y <= top_y:
                res_map[cell_index] = res
    return res_map

# def assign_polygon_resistivities(mesh, polygons, resistivities, res_map):
#     """
#     Overwrites resistivity values in the mesh cells based on the polygons.
#
#     Parameters:
#     - mesh: pg.Mesh object, the mesh to which resistivities will be assigned.
#     - polygons: list of shapely.geometry.Polygon objects, the polygons defining the regions.
#     - resistivities: list of float, the resistivity values corresponding to each polygon.
#     - res_map: numpy array, initial resistivity values to be overwritten by polygon resistivities.
#
#     Returns:
#     - res_map: numpy array, updated resistivity values.
#     """
#     if len(polygons) != len(resistivities):
#         raise ValueError("Number of polygons must match number of resistivity values.")
#
#     # Convert mesh cells to shapely Points for easy intersection checks
#     cell_centers = [Point(cell.center().x(), cell.center().y()) for cell in mesh.cells()]
#
#     # Iterate over each polygon and assign the corresponding resistivity to cells within it
#     for poly, res in zip(polygons, resistivities):
#         for cell_index, cell_center in enumerate(cell_centers):
#             if poly.contains(cell_center):
#                 res_map[cell_index] = res
#
#     return np.array(res_map)

def assign_polygon_resistivities(mesh, polygons, resistivities, res_map):
    if len(polygons) != len(resistivities):
        raise ValueError("Number of polygons must match number of resistivity values.")

    cell_centers = [Point(cell.center().x(), cell.center().y()) for cell in mesh.cells()]

    for poly, res in zip(polygons, resistivities):
        for cell_index, cell_center in enumerate(cell_centers):
            if poly.contains(cell_center):
                res_map[cell_index] = res
    return res_map


def assign_layer_resistivities_3d(mesh, layers_z, resistivities, default_resistivity=1e6):
    """
    3D counterpart of assign_layer_resistivities: horizontal (flat) slabs,
    so a cell only needs its z coordinate checked, same as the 2D version
    checks y. layers_z is a list of (z_top, z_bottom) pairs (z=0 is the
    surface, more negative is deeper -- either order per pair is accepted),
    matched 1:1 with resistivities.
    """
    if len(layers_z) != len(resistivities):
        raise ValueError("Number of layers must match number of resistivity values.")

    res_map = np.full(mesh.cellCount(), default_resistivity, dtype=float)
    for (z_a, z_b), res in zip(layers_z, resistivities):
        lo, hi = (z_a, z_b) if z_a <= z_b else (z_b, z_a)
        for cell_index, cell in enumerate(mesh.cells()):
            z = cell.center().z()
            if lo <= z <= hi:
                res_map[cell_index] = res
    return res_map


def _point_segment_distance(p, a, b):
    """Distance from point p (shape (3,)) to the segment a-b (each (3,))."""
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def assign_cylinder_resistivities(mesh, cylinders, resistivities, res_map):
    """
    3D counterpart of assign_polygon_resistivities, for CylinderAnom
    bodies. `cylinders` is a list of (x1,y1,z1,x2,y2,z2,r) tuples. A cell
    is assigned the cylinder's rho if its center is within r of the
    START-END SEGMENT (a capsule, not an infinite cylinder) -- for a body
    much longer than it is wide (a shaft, a beam) this only rounds the two
    end caps slightly instead of leaving them flat, which is a minor and
    usually harmless over-inclusion right at the ends.
    """
    if len(cylinders) != len(resistivities):
        raise ValueError("Number of cylinders must match number of resistivity values.")

    cell_centers = [np.array([c.center().x(), c.center().y(), c.center().z()])
                    for c in mesh.cells()]

    for (x1, y1, z1, x2, y2, z2, r), res in zip(cylinders, resistivities):
        a = np.array([x1, y1, z1], dtype=float)
        b = np.array([x2, y2, z2], dtype=float)
        for cell_index, p in enumerate(cell_centers):
            if _point_segment_distance(p, a, b) <= r:
                res_map[cell_index] = res
    return res_map


# import pybert as pb
class CircleAnom:
    # Ordered decision-variable names, in the exact order varflag/varlims/get_x0/
    # get_bounds/update expect them. AnomalyWorld and Constraints read this
    # instead of hardcoding indices, so a new anomaly shape (e.g. an ellipse
    # cross-section) only needs to define its own param_names/log_flags/
    # containment_vars -- no changes to AnomalyWorld or Constraints.
    param_names = ['x', 'y', 'r', 'rho']
    # Which params act as a "half-extent" for the automatic World-containment
    # constraints (generate_auto_constraints): for a circle, r is the exact
    # half-extent in both x and y.
    containment_vars = ['r']

    @property
    def log_flags(self):
        # Per-instance (self.log_r / self.log_rho control the actual
        # transform in get_x0/update/get_bounds today), exposed by name so
        # AnomalyWorld/Constraints can look it up generically instead of
        # hardcoding "index 2 -> log_r, index 3 -> log_rho".
        return {'x': False, 'y': False, 'r': self.log_r, 'rho': self.log_rho}

    def __init__(self, name, x=0, y=0, r=1.0, rho=1.0, _c_num=1, _varflag=None, _varlims=None,_log_rho=True,_log_r=True):
        if _varflag is None:
            _varflag = [False] * 4
        else:
            if _varlims is None:
                _varlims = []
                for i, vf in enumerate(_varflag):
                    if vf:
                        _varlims.append((-np.inf * ((i == 0) + (i == 1)), np.inf))
        self.name = name
        self.x = x
        self.y = y
        self.r = r
        self.rho = rho
        self.log_rho = True
        self.log_r = True
        self.varflag = _varflag
        self.varlims = _varlims
        self.marker = _c_num + 1
        self.center = Point(self.x, self.y)
        self.polygon = self.center.buffer(self.r)
        self.block = mt.createCircle(pos=[self.x, self.y], radius=self.r, marker=self.marker,
                                     boundaryMarker=10, area=0.01, isHole=False)

    def __str__(self):
        return f"{self.name}- x: {self.x}, y: {self.y},y: {self.r}, rho: {self.rho}"

    def show(self):
        pg.show(self.block)

    def update(self, name, x, y, r, rho, _c_num, _varflag=None, _varlims=None):
        if _varflag is None:
            _varflag = [False] * 4
        else:
            if _varlims is None:
                _varlims = []
                for i, vf in enumerate(_varflag):
                    if vf:
                        _varlims.append((-np.inf * ((i == 0) + (i == 1)), np.inf))
        self.name = name
        self.x = x
        self.y = y
        self.r = r
        self.rho = rho
        self.varflag = _varflag
        self.varlims = _varlims
        self.marker = _c_num + 1
        self.center = Point(self.x, self.y)
        self.polygon = self.center.buffer(self.r)
        self.block = mt.createCircle(pos=[self.x, self.y], radius=self.r, marker=self.marker,
                                     boundaryMarker=10, area=0.1, isHole=False)
    def update_block(self):
        self.block = mt.createCircle(pos=[self.x, self.y], radius=self.r, marker=self.marker,
                                     boundaryMarker=10, area=0.1, isHole=False)
        self.center = Point(self.x, self.y)
        self.polygon = self.center.buffer(self.r)


class EllipseAnom:
    """
    Elliptical cross-section anomaly -- a sibling of CircleAnom for shapes a
    circle can't represent, e.g. an oblique mine shaft whose visible
    cross-section in the 2D measurement plane is elongated rather than
    round (Case 1 of the field-demonstrations project), or an
    archaeological wall/bridge modelled as a thin elongated body (Case 2).

    Parameters, in optimizer order: x, y (center), a, b (semi-axes in the
    ellipse's own frame, before rotation), theta (rotation, degrees,
    counter-clockwise from the x-axis), rho. A circle is the special case
    a == b (theta then has no effect on the shape, only on its varlims).
    """
    param_names = ['x', 'y', 'a', 'b', 'theta', 'rho']
    # Both semi-axes act as "half-extent" params for the automatic
    # World-containment constraint (see Constraints.generate_auto_constraints).
    # With theta free, the true worst-case footprint in x (or y) across all
    # rotations is max(a, b); enforcing x+a<=end_x together with x+b<=end_x
    # is exactly x+max(a,b)<=end_x -- sound but conservative versus the true
    # rotation-dependent footprint, since it does not use theta.
    containment_vars = ['a', 'b']

    @property
    def log_flags(self):
        return {'x': False, 'y': False, 'a': self.log_a, 'b': self.log_b,
                'theta': False, 'rho': self.log_rho}

    def __init__(self, name, x=0, y=0, a=1.0, b=0.5, theta=0.0, rho=1.0,
                 _c_num=1, _varflag=None, _varlims=None,
                 _log_rho=True, _log_a=True, _log_b=True, _resolution=64):
        if _varflag is None:
            _varflag = [False] * 6
        else:
            if _varlims is None:
                _varlims = []
                for i, vf in enumerate(_varflag):
                    if vf:
                        # x, y, theta unbounded by default (like CircleAnom's
                        # x/y); a, b default to a >=0 lower bound.
                        lower = -np.inf if i in (0, 1, 4) else 0.0
                        _varlims.append((lower, np.inf))
        self.name = name
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.theta = theta
        self.rho = rho
        self.log_rho = _log_rho
        self.log_a = _log_a
        self.log_b = _log_b
        self.varflag = _varflag
        self.varlims = _varlims
        self.marker = _c_num + 1
        self.resolution = _resolution
        self.center = Point(self.x, self.y)
        self.polygon = self._build_polygon()
        self.block = self._build_block(area=0.01)

    def _build_polygon(self):
        base = Point(0, 0).buffer(1, resolution=self.resolution)
        ell = _shapely_scale(base, xfact=self.a, yfact=self.b, origin=(0, 0))
        ell = _shapely_rotate(ell, self.theta, origin=(0, 0), use_radians=False)
        ell = _shapely_translate(ell, xoff=self.x, yoff=self.y)
        return ell

    def _build_block(self, area=0.1):
        # Vertices come straight from the same shapely polygon used for
        # resistivity assignment, so the mesh block and the "is this cell
        # inside the anomaly" test can never disagree on the shape.
        verts = list(self.polygon.exterior.coords)[:-1]  # drop closing dup
        return mt.createPolygon(verts, isClosed=True, marker=self.marker,
                                boundaryMarker=10, area=area, isHole=False)

    def __str__(self):
        return (f"{self.name}- x: {self.x}, y: {self.y}, a: {self.a}, "
                f"b: {self.b}, theta: {self.theta}, rho: {self.rho}")

    def show(self):
        pg.show(self.block)

    def update(self, name, x, y, a, b, theta, rho, _c_num, _varflag=None, _varlims=None):
        if _varflag is None:
            _varflag = [False] * 6
        else:
            if _varlims is None:
                _varlims = []
                for i, vf in enumerate(_varflag):
                    if vf:
                        lower = -np.inf if i in (0, 1, 4) else 0.0
                        _varlims.append((lower, np.inf))
        self.name = name
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        self.theta = theta
        self.rho = rho
        self.varflag = _varflag
        self.varlims = _varlims
        self.marker = _c_num + 1
        self.center = Point(self.x, self.y)
        self.polygon = self._build_polygon()
        self.block = self._build_block(area=0.1)

    def update_block(self):
        self.block = self._build_block(area=0.1)
        self.center = Point(self.x, self.y)
        self.polygon = self._build_polygon()


class CylinderAnom:
    """
    3D cylindrical anomaly for a single-survey-line 3D forward model (see
    AnomalyWorld's 3D path, active when start/end are 3-tuples): a tunnel,
    shaft, or elongated metallic body, parameterized directly by its two
    end points and a radius, as a "faithful sibling" of CircleAnom/
    EllipseAnom for the case where the target is a genuinely 3D body rather
    than a 2D cross-section (Case 1's oblique mine shaft, running out of
    the measurement plane).

    Parameters, in optimizer order: x1,y1,z1 (start point), x2,y2,z2 (end
    point), r, rho. Convention: z=0 is the surface, more negative z is
    deeper (matches Layer.y / the 2D primitives' 'y' in AnomalyWorld's
    depth axis).

    Containment: `anchor_points` lists BOTH end points, so
    Constraints.generate_auto_constraints bounds each of them (x,y,z)
    independently inside the world box, inflated by r on every axis -- an
    axis-aligned capsule bound. This is sound but conservative: it ignores
    that the cylinder's actual footprint along, say, x depends on its
    orientation, exactly like the ellipse's (a,b) bound ignores theta.

    NOTE ON pyGIMLi 3D API: `_build_block()` uses pyGIMLi's
    meshtools.createCylinder() (built along the local z-axis, centered at
    the origin) and then rotates/translates it onto (x1,y1,z1)->(x2,y2,z2).
    This was written and syntax-checked WITHOUT a local pyGIMLi install (not
    available in the environment this was developed in) -- verify it with
    the accompanying test_cylinder3d_pygimli.py before relying on it; the
    exact rotate()/translate() call signature can differ across pyGIMLi
    versions.
    """
    param_names = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'r', 'rho']
    containment_vars = ['r']
    anchor_points = [('x1', 'y1', 'z1'), ('x2', 'y2', 'z2')]

    @property
    def log_flags(self):
        return {'x1': False, 'y1': False, 'z1': False,
                'x2': False, 'y2': False, 'z2': False,
                'r': self.log_r, 'rho': self.log_rho}

    def __init__(self, name, x1=0.0, y1=0.0, z1=0.0, x2=1.0, y2=0.0, z2=-1.0,
                 r=0.5, rho=1.0, _c_num=1, _varflag=None, _varlims=None,
                 _log_rho=True, _log_r=True, _n_segments=16):
        if _varflag is None:
            _varflag = [False] * 8
        else:
            if _varlims is None:
                _varlims = []
                for i, vf in enumerate(_varflag):
                    if vf:
                        # r (index 6) is bounded below at 0; every position
                        # component (the six endpoint coordinates) and rho
                        # default to unbounded, same convention as
                        # CircleAnom/EllipseAnom.
                        lower = 0.0 if i == 6 else -np.inf
                        _varlims.append((lower, np.inf))
        self.name = name
        self.x1, self.y1, self.z1 = x1, y1, z1
        self.x2, self.y2, self.z2 = x2, y2, z2
        self.r = r
        self.rho = rho
        self.log_rho = _log_rho
        self.log_r = _log_r
        self.varflag = _varflag
        self.varlims = _varlims
        self.marker = _c_num + 1
        self.n_segments = _n_segments
        self.block = self._build_block()

    def _axis_vectors(self):
        a = np.array([self.x1, self.y1, self.z1], dtype=float)
        b = np.array([self.x2, self.y2, self.z2], dtype=float)
        return a, b

    def _build_block(self, area=0.1):
        a, b = self._axis_vectors()
        d = b - a
        height = float(np.linalg.norm(d))
        if height < 1e-9:
            raise ValueError(f"{self.name}: start and end points coincide "
                              f"(zero-length cylinder) -- a and b must differ.")
        cyl = mt.createCylinder(radius=self.r, height=height, nSegments=self.n_segments,
                                marker=self.marker, boundaryMarker=10, area=area, isHole=False)
        axis = d / height
        z_hat = np.array([0.0, 0.0, 1.0])
        dot = float(np.clip(np.dot(z_hat, axis), -1.0, 1.0))
        rot_axis = np.cross(z_hat, axis)
        rot_axis_norm = np.linalg.norm(rot_axis)
        if rot_axis_norm > 1e-9:
            angle = np.arccos(dot)  # radians
            # pg's rotate() takes a rotation VECTOR (unit axis * angle); if
            # your pygimli expects rotate(axis, angle) instead, split this
            # into that call -- see test_cylinder3d_pygimli.py.
            cyl.rotate(pg.core.RVector3(*(rot_axis / rot_axis_norm * angle)))
        elif dot < 0:
            # axis anti-parallel to z (straight down): 180 deg flip around x.
            cyl.rotate(pg.core.RVector3(np.pi, 0.0, 0.0))
        midpoint = (a + b) / 2.0
        cyl.translate(pg.core.RVector3(*midpoint))
        return cyl

    def update_block(self):
        self.block = self._build_block()

    def __str__(self):
        return (f"{self.name}- ({self.x1:.2f},{self.y1:.2f},{self.z1:.2f}) -> "
                f"({self.x2:.2f},{self.y2:.2f},{self.z2:.2f}), r: {self.r}, rho: {self.rho}")

    def show(self):
        pg.show(self.block)

    def update(self, name, x1, y1, z1, x2, y2, z2, r, rho, _c_num,
               _varflag=None, _varlims=None):
        if _varflag is None:
            _varflag = [False] * 8
        else:
            if _varlims is None:
                _varlims = []
                for i, vf in enumerate(_varflag):
                    if vf:
                        lower = 0.0 if i == 6 else -np.inf
                        _varlims.append((lower, np.inf))
        self.name = name
        self.x1, self.y1, self.z1 = x1, y1, z1
        self.x2, self.y2, self.z2 = x2, y2, z2
        self.r = r
        self.rho = rho
        self.varflag = _varflag
        self.varlims = _varlims
        self.marker = _c_num + 1
        self.block = self._build_block()


class Layer:
    param_names = ['y', 'rho']

    @property
    def log_flags(self):
        return {'y': False, 'rho': self.log_rho}

    def __init__(self, name, y=0, rho=1.0, _cnum=1, _varflag=None, _varlims=None,_log_rho=True):
        if _varflag is None:
            _varflag = [False] * 2
        else:
            if _varlims is None:
                _varlims = []
                for i, vf in enumerate(_varflag):
                    if vf:
                        if i == 0:
                            _varlims.append((-np.inf, np.inf))
                        if i == 1:
                            _varlims.append((0, np.inf))
        self.name = name
        self.y = y
        self.rho = rho
        self.log_rho=_log_rho
        self.varflag = _varflag
        self.varlims = _varlims
        self.marker = _cnum + 1

    def __str__(self):
        return f"{self.name} - y: {self.y}, rho: {self.rho}"

    def update(self, name, y, rho, cnum, _varflag=None, _varlims=None):
        if _varflag is None:
            _varflag = [False] * 2
        else:
            if _varlims is None:
                _varlims = []
                for i, vf in enumerate(_varflag):
                    if vf:
                        _varlims.append((-np.inf * (i == 0) + 0 * int(i == 1), np.inf))
        self.name = name
        self.y = y
        self.rho = rho
        self.varflag = _varflag
        self.varlims = _varlims
        self.marker = cnum + 1


class Constraints:
    def __init__(self):
        """
        Initializes the Constraints class, which stores a list of constraints.
        Each constraint is represented as a dictionary for better readability.
        """
        self.constraints = []

    def add_constraint(self, entity1, var1, constraint_type, param_value1=None, entity2=None, var2=None,
                       param_value2=None, entity3=None, var3=None):
        """
        Adds a constraint to the list.

        Parameters:
            entity1 (object): The first entity (e.g., Layer or CircleAnomaly).
            var1 (str): The variable name of the first entity (e.g., 'x', 'y', 'rho').
            constraint_type (str): The type of constraint (e.g., '<=', '>=', '==').
            param_value1 (float): Constraint value for var1.
            entity2 (object): The second entity (optional).
            var2 (str): The variable name of the second entity (optional).
            param_value2 (float): Constraint value for var2 (optional).
            :param param_value2:
            :param var2:
            :param entity2:
            :param param_value1:
            :param constraint_type:
            :param var1:
            :param entity1:
            :param var3:
            :param entity3:
        """
        # Extract entity names
        entity1_name = entity1.name
        entity2_name = entity2.name if entity2 else None
        entity3_name = entity3.name if entity3 else None

        # Store the constraint as a dictionary
        constraint = {
            "entity1": entity1_name,
            "var1": var1,
            "constraint_type": constraint_type,
            "param_value1": param_value1,
            "entity2": entity2_name,
            "var2": var2,
            "param_value2": param_value2,
            "entity3": entity3_name,
            "var3": var3
        }

        self.constraints.append(constraint)

    @staticmethod
    def generate_variable_map(layers, anomalies, var_flag_world=False, world_rho=None):
        """
        Generates two maps:
        1. variable_map: For variables with varflag=1.
        2. parameter_map: For variables with varflag=0, storing their actual values.

        Parameters:
            layers (list): A list of Layer objects.
            anomalies (list): A list of CircleAnomaly objects.

        Returns:
            tuple: A tuple containing:
                - variable_map (dict): A mapping of (entity_name, variable_name) to their indices for varflag=1.
                - parameter_map (dict): A mapping of (entity_name, variable_name) to a tuple (index, value) for varflag=
                0.
                :param anomalies:
                :param layers:
                :param world_rho:
                :param var_flag_world:
        """
        variable_map = {}
        parameter_map = {}
        variable_idx = 0
        parameter_idx = 0
        # Process world
        if var_flag_world:  # varflag = 0
            variable_map[('world', 'rho')] = variable_idx
            variable_idx += 1
        else:  # varflag = 0
            parameter_map[('world', 'rho')] = (parameter_idx, world_rho)
            parameter_idx += 1
        # Process layers
        for layer in layers:
            if hasattr(layer, 'varflag'):
                names = getattr(layer, 'param_names', ['y', 'rho'])
                for attr, flag in zip(names, layer.varflag):
                    value = getattr(layer, attr)
                    if flag:  # varflag = 1
                        variable_map[(layer.name, attr)] = variable_idx
                        variable_idx += 1
                    else:  # varflag = 0
                        parameter_map[(layer.name, attr)] = (parameter_idx, value)
                        parameter_idx += 1

        # Process anomalies. `param_names` lets each anomaly shape (circle,
        # ellipse, ...) declare its own ordered parameter list instead of the
        # loop assuming exactly ['x', 'y', 'r', 'rho'].
        for anomaly in anomalies:
            if hasattr(anomaly, 'varflag'):
                names = getattr(anomaly, 'param_names', ['x', 'y', 'r', 'rho'])
                for attr, flag in zip(names, anomaly.varflag):
                    value = getattr(anomaly, attr)
                    if flag:  # varflag = 1
                        variable_map[(anomaly.name, attr)] = variable_idx
                        variable_idx += 1
                    else:  # varflag = 0
                        parameter_map[(anomaly.name, attr)] = (parameter_idx, value)
                        parameter_idx += 1

        num_variables = variable_idx
        num_parameters = parameter_idx
        return variable_map, parameter_map, num_variables, num_parameters

    def generate_auto_constraints(self, layers, anomalies, _start, _end):
        """
        Automatically generates constraints for layers and anomalies based on their varflag and varlims.

        Parameters:
            layers (list): A list of Layer objects.
            anomalies (list): A list of CircleAnomaly objects.
            :param layers:
            :param anomalies:
            :param _end:
            :param _start:
        """
        # The LAST coordinate axis is always the depth-like one: start=top
        # (0 in this project's convention), end=bottom (more negative =
        # deeper), so its containment bound is asymmetric (swapped
        # start/end) exactly like the 2D primitives' 'y' always was. Every
        # other axis (2D's 'x'; 3D's 'x' and the new cross-line 'y') is a
        # plain symmetric [start, end] bound. This one rule is what lets the
        # same code serve the 2D world (start=[x,y]) and the 3D world
        # (start=[x,y,z]) without a separate 3D branch.
        depth_axis = len(_start) - 1

        for i, layer in enumerate(layers):
            if hasattr(layer, 'varflag') and hasattr(layer, 'varlims'):
                if i == 0:
                    self.add_constraint(entity1=layer, var1='y', constraint_type='<=',
                                        param_value1=_start[depth_axis] - 0.01)
                if i > 0:
                    self.add_constraint(entity1=layer, var1='y', entity2=layers[i - 1], var2='y',
                                        constraint_type='<=', param_value1=-0.01)
        for anomaly in anomalies:
            if hasattr(anomaly, 'varflag') and hasattr(anomaly, 'varlims'):
                # World limits.
                # `containment_vars` is the anomaly's list of "half-extent"
                # params (a circle: ['r']; an ellipse: ['a', 'b']; a
                # cylinder: ['r'] applied at both endpoints via
                # `anchor_points`). With a free rotation/orientation the
                # worst-case footprint along any one axis is
                # max(containment_vars), and enforcing e.g. x+a<=end_x
                # together with x+b<=end_x is exactly x+max(a,b)<=end_x --
                # sound but conservative, and doesn't need to know the
                # orientation (theta, or the cylinder's own axis direction).
                #
                # `anchor_points` is a list of coordinate-name tuples, one
                # per point on the anomaly that must stay in the box: a
                # circle/ellipse has one anchor [('x','y')]; a cylinder has
                # two, one per end point, e.g.
                # [('x1','y1','z1'), ('x2','y2','z2')].
                names = getattr(anomaly, 'param_names', ['x', 'y', 'r', 'rho'])
                idx = {n: k for k, n in enumerate(names)}
                containment_vars = getattr(anomaly, 'containment_vars', ['r'])
                extent_free = any(anomaly.varflag[idx[v]] for v in containment_vars if v in idx)
                anchor_points = getattr(anomaly, 'anchor_points', [('x', 'y')])
                for anchor in anchor_points:
                    for axis_i, vname in enumerate(anchor):
                        if vname is None or vname not in idx:
                            continue
                        v_free = anomaly.varflag[idx[vname]]
                        if not (v_free or extent_free):
                            continue
                        depth_like = (axis_i == depth_axis)
                        lo = _end[axis_i] if depth_like else _start[axis_i]
                        hi = _start[axis_i] if depth_like else _end[axis_i]
                        for cvar in containment_vars:
                            # lo <= v+cvar <= hi   and   v-cvar >= lo
                            self.add_constraint(anomaly, var1=vname, entity2=anomaly, var2=cvar,
                                                constraint_type='sum_bounds',
                                                param_value1=lo, param_value2=hi)
                            self.add_constraint(anomaly, var1=vname, entity2=anomaly, var2=cvar,
                                                constraint_type='>=', param_value1=lo)

    def __str__(self):
        """
        Provides a string representation of all constraints for display.
        """
        result = "Constraints:\n"
        for i, constraint in enumerate(self.constraints):
            if constraint['constraint_type'] == "bounds":
                result += f"{i + 1}.{constraint['param_value1']} {'<='}{constraint['entity1']}.{constraint['var1']}"
                if constraint['entity2']:
                    result += f"+{constraint['entity2']}.{constraint['var2']}"
                result += f"{'<='}{constraint['param_value2']}"
            else:
                result += f"{i + 1}. {constraint['entity1']}.{constraint['var1']} {constraint['constraint_type']} "
                if constraint['entity2']:
                    result += f"{constraint['entity2']}.{constraint['var2']} "
                if constraint['param_value1'] is not None:
                    result += f"({constraint['param_value1']}) "
                if constraint['param_value2'] is not None:
                    result += f"({constraint['param_value2']}) "
                if constraint['entity3']:
                    result += f"+{constraint['entity3']}.{constraint['var3']}"
            result += "\n"
        return result

    def evaluate_constraints(self, variable_map, parameter_map, num_variables):
        # num_parameters):
        a = np.empty((0, num_variables))
        lb = np.empty((0, 1))
        ub = np.empty((0, 1))
        for constraint in self.constraints:
            entity1, var1, entity2, var2, entity3, var3, constraint_type, param_value1, param_value2 = (
                constraint['entity1'],
                constraint['var1'],
                constraint['entity2'],
                constraint['var2'],
                constraint['entity3'],
                constraint['var3'],
                constraint['constraint_type'],
                constraint['param_value1'],
                constraint['param_value2'],
            )
            a_row = np.zeros(num_variables)  # Initialize a new row for matrix a
            param_l = 0
            param_u = 0
            var_stat_val2 = None
            var_stat_val3 = None
            try:
                value1 = variable_map[(entity1, var1)]  # Key does not exist
                a_row[value1] = 1
            except KeyError:
                value1 = False  # Alternative response if KeyError occurs
                if entity1 is not None:
                    param_l -= parameter_map[(entity1, var1)][1]
                    param_u -= parameter_map[(entity1, var1)][1]
            try:
                value2 = variable_map[(entity2, var2)]  # Key does not exist
                # idx2 = variable_map.get((entity2, var2))  # Compute indices of properties in the mapping
            except KeyError:
                value2 = False  # Alternative response if KeyError occurs
                if entity2 is not None:
                    var_stat_val2 = parameter_map[(entity2, var2)][1]
            try:
                value3 = variable_map[(entity3, var3)]  # Key does not exist
                # idx3 = variable_map.get((entity3, var3))  # Compute indices of properties in the mapping
            except KeyError:
                value3 = False  # Alternative response if KeyError occurs
                if entity3 is not None:
                    var_stat_val3 = parameter_map[(entity3, var3)][1]
            if all(v is None for v in [value1, value2, value3]):
                pass
            else:
                # Update coefficients based on the constraint
                if constraint_type == "sum_bounds":  # param1<=var1+var2<=param2
                    try:
                        param_l += param_value1
                        param_u += param_value2
                    except KeyError:
                        raise ValueError(
                            f"Unsupported input constraint type: {constraint['constraint_type']}, param_value1 and "
                            f"param_value2 must not be None")
                    if value2 is not None:
                        a_row[value2] = 1  # Coefficient for var2
                    elif var_stat_val2 is not None:
                        param_l -= var_stat_val2
                        param_u -= var_stat_val2
                if constraint_type == "diff_bounds":  # param1<=var1-var2<=param2
                    try:
                        param_l += param_value1
                        param_u += param_value2
                    except KeyError:
                        raise ValueError(
                            f"Unsupported input constraint type: {constraint['constraint_type']}, param_value1 and "
                            f"param_value2 must not be None")
                    if value2 is not None:
                        a_row[value2] = -1  # Coefficient for var2
                    elif var_stat_val2 is not None:
                        param_l = var_stat_val2
                        param_u = var_stat_val2
                elif constraint_type == "bounds":  # param1<=var1<=param2
                    try:
                        param_l += param_value1
                        param_u += param_value2
                    except KeyError:
                        raise ValueError(
                            f"Unsupported input constraint type: {constraint['constraint_type']}, param_value1 and "
                            f"param_value2 must not be None")

                elif constraint_type == "<=":  # -np.inf<=var1<=var2+param1
                    param_l = -np.inf
                    param_u += param_value1 if param_value1 is not None else 0
                    if value2 is not None:
                        a_row[value2] = -1  # Coefficient for var2
                    elif var_stat_val2 is not None:
                        param_u += var_stat_val2

                elif constraint_type == ">=":  # np.inf>=var1>=var2+param1
                    param_l += param_value1 if param_value1 is not None else 0
                    param_u = np.inf
                    if value2 is not None:
                        a_row[value2] = -1  # Coefficient for var2
                    elif var_stat_val2 is not None:
                        param_l += var_stat_val2

                elif constraint_type == "==":  # var2+param1>=var1>=var2+param1
                    param_l += param_value1 if param_value1 is not None else 0
                    param_u += param_value1 if param_value1 is not None else 0
                    if value2 is not None:
                        a_row[value2] = -1  # Coefficient for var2
                    elif var_stat_val2 is not None:
                        param_l += var_stat_val2
                        param_u += var_stat_val2

                # 3 variable inputs constraints
                elif constraint_type == "3sum_bounds":  # param1>=var1+var2+var3>=param2
                    try:
                        param_l += param_value1
                        param_u += param_value2
                    except KeyError:
                        raise ValueError(
                            f"Unsupported input constraint type: {constraint['constraint_type']}, param_value1 and "
                            f"param_value2 must not be None")
                    if value2 is not None:
                        a_row[value2] = 1  # Coefficient for var2
                    elif var_stat_val2 is not None:
                        param_l -= var_stat_val2
                        param_u -= var_stat_val2
                    if value3 is not None:
                        a_row[value3] = 1  # Coefficient for var3
                    elif var_stat_val3 is not None:
                        param_l -= var_stat_val3
                        param_u -= var_stat_val3

                elif constraint_type == "3<=":  # -np.inf<=var1+var2<=var3+param1
                    param_l += -np.inf
                    param_u += param_value1 if param_value1 is not None else 0
                    if value2 is not None:
                        a_row[value2] = 1  # Coefficient for var2
                    elif var_stat_val2 is not None:
                        param_u -= var_stat_val2
                    if value3 is not None:
                        a_row[value3] = -1  # Coefficient for var3
                    elif var_stat_val3 is not None:
                        param_u += var_stat_val3

                elif constraint_type == "3==":  # var3+param1>=var1+var2>=var3+param1
                    param_l += param_value1 if param_value1 is not None else 0
                    param_u += param_value1 if param_value1 is not None else 0
                    if value2 is not None:
                        a_row[value2] = 1  # Coefficient for var2
                    elif var_stat_val2 is not None:
                        param_l -= var_stat_val2
                        param_u -= var_stat_val2
                    if value3 is not None:
                        a_row[value3] = -1  # Coefficient for var3
                    elif var_stat_val3 is not None:
                        param_l += var_stat_val3
                        param_u += var_stat_val3

            a = np.append(a, [a_row], axis=0)
            lb = np.append(lb, param_l)
            ub = np.append(ub, param_u)
        #         else:
        #             raise ValueError(f"Unsupported constraint type: {constraint['constraint_type']}")
        return a, lb, ub


class World:
    def __init__(self, name='world', varflag=False, _rho_world=1e6, _rho_world_varlim=None):
        if _rho_world_varlim is None:
            _rho_world_varlim = [None, None]
        self.name = name
        self.varflag = varflag
        self.rho_world = _rho_world
        self.rho_world_varlim = _rho_world_varlim


class AnomalyWorld:
    def __init__(self, _start, _end, _scheme, _rho_world=1e6, _rho_world_varflag=False, _rho_world_varlim=None,
                 meas=None, _layers=None,
                 _circles=None, _cylinders=None):
        if _circles is None:
            _circles = []
        if _cylinders is None:
            _cylinders = []
        if _layers is None:
            _layers = []
        if _rho_world_varlim is None:
            _rho_world_varlim = [0, np.inf]
        if meas is None:
            self.meas = []
        self.background = World(name='world', varflag=_rho_world_varflag, _rho_world=_rho_world)
        self.scheme = _scheme
        self.start = _start
        self.end = _end
        self.layers = _layers
        self.Circles = _circles
        # `Cylinders` (3D anomalies) and `Circles` (2D anomalies) are
        # mutually exclusive in practice: is_3d picks the geometry-building
        # path (2D circles/ellipses in a 2D world, or 3D cylinders in a 3D
        # box), but for the decision-variable machinery (get_x0/get_bounds/
        # update/constraints) both lists are just treated as "anomalies" --
        # see self._anomalies below.
        self.Cylinders = _cylinders
        self.is_3d = len(self.start) == 3
        self.rho_world = _rho_world
        self.rho_world_varflag = _rho_world_varflag
        self.rho_world_varlim = _rho_world_varlim
        self.constraint_group = Constraints()
        self._last_circles_state = [self._geometry_key(c) for c in self._anomalies()]
        self._last_layers_state  = [(l.y) for l in self.layers]
        self.constraint_group.generate_auto_constraints(self.layers, self._anomalies(), self.start, self.end)
        (self.variable_map,
         self.parameter_map,
         self.num_variables,
         self.num_parameters) = self.constraint_group.generate_variable_map(self.layers, self._anomalies(),
                                                                            self.rho_world_varflag, self.rho_world)

        a, lb, ub = self.constraint_group.evaluate_constraints(self.variable_map, self.parameter_map,
                                                               self.num_variables)
        # print(a, lb, ub)
        if a.size > 0:
            self.constraints = LinearConstraint(a, lb, ub)
        else:
            self.constraints = None
        for i, layer in enumerate(self.layers):
            kwargs = {n: getattr(layer, n) for n in self._entity_param_names(layer, ['y', 'rho'])}
            layer.update(name=layer.name, cnum=i + 1, _varflag=layer.varflag,
                         _varlims=layer.varlims, **kwargs)

        self.meas = meas
        self.fop = ert.ERTModelling()
        if self.is_3d:
            self._build_geometry_3d()
        else:
            self._build_geometry_2d()
        self.n_fev = 0
        self.best_male = np.inf
        self.best_msle = np.inf
        self.best_x_male = None
        self.best_x_msle = None
        self.history = {
            'fevals': [],
            'best_male': [],
            'best_msle': []
        }
        
        self._fev_display = display("", display_id=True)

    def _anomalies(self):
        """
        Combined anomaly list for the decision-variable machinery
        (get_x0/get_bounds/update/get_varflag_n/constraints), which only
        cares about param_names/log_flags/varflag/varlims and doesn't need
        to know whether an entry is a 2D CircleAnom/EllipseAnom or a 3D
        CylinderAnom. Circles and Cylinders are otherwise built/meshed by
        entirely different code paths (_build_geometry_2d/_build_geometry_3d),
        selected by self.is_3d; in practice only one of the two lists is
        non-empty for a given world.
        """
        return self.Circles + self.Cylinders

    def _geometry_state(self):
        layer_state = tuple(layer.y for layer in self.layers)
        circle_state = tuple(self._geometry_key(c) for c in self._anomalies())
        return layer_state, circle_state

    def show_mesh(self):
        pg.show(self.mesh, data=self.resistivity_map)

    def show_mesh_interactive(self, cMap="Spectral_r", logScale=True):
        """
        Interactive pyvista view of the model: a draggable clip-plane
        widget to slice through it, plus one checkbox per region (each
        (layer, cylinder) combination from self._region_markers -- see
        _build_mesh_3d_gmsh) to show/hide that region independently, e.g.
        hide the layer(s) to isolate a cylinder, or the reverse. Dragging
        the plane re-clips whichever regions are currently checked on.

        3D (gmsh-built) worlds only -- falls back to the plain show_mesh()
        for a 2D world, where there's no self._region_markers/cellMarkers
        split to check-box and no volume to clip through.
        """
        if not self.is_3d or not hasattr(self, '_region_markers'):
            self.show_mesh()
            return

        import pyvista as pv
        import pygimli.viewer.pv as pgpv

        rho = np.asarray(self.resistivity_map, dtype=float)
        label = "Resistivity (Ohm.m)"
        full = pgpv.pgMesh2pvMesh(self.mesh, data=rho, label=label)
        cell_markers = np.asarray(self.mesh.cellMarkers())
        cmin, cmax = float(np.nanmin(rho)), float(np.nanmax(rho))

        inv = {mk: key for key, mk in self._region_markers.items()}

        def _region_name(mk):
            key = inv.get(int(mk))
            if key is None:
                return f"region {int(mk)}"
            layer_idx, cyl_idx = key
            name = f"layer {layer_idx}"
            if cyl_idx is not None:
                cyl = self.Cylinders[cyl_idx] if cyl_idx < len(self.Cylinders) else None
                name += f" + {cyl.name if cyl is not None else f'cyl{cyl_idx}'}"
            return name

        markers = sorted(int(m) for m in np.unique(cell_markers))
        region_grids = {mk: full.extract_cells(np.where(cell_markers == mk)[0])
                        for mk in markers}
        region_visible = {mk: True for mk in markers}
        region_actors = {}
        # style: 'surface' (opaque/transparent) or 'wireframe' ("mesh only");
        # opacity only matters for style='surface'; show_edges overlays the
        # cell edges on top of either style (harmless/no-op on wireframe).
        state = {"normal": [1.0, 0.0, 0.0], "origin": list(full.center),
                 "style": "surface", "opacity": 1.0, "show_edges": False}

        plotter = pv.Plotter()
        plotter.set_background("white")

        def _rebuild():
            for mk in markers:
                if mk in region_actors:
                    plotter.remove_actor(region_actors[mk], render=False)
                    del region_actors[mk]
                if not region_visible[mk]:
                    continue
                clipped = region_grids[mk].clip(normal=state["normal"], origin=state["origin"])
                region_actors[mk] = plotter.add_mesh(
                    clipped, scalars=label, cmap=cMap, log_scale=logScale,
                    clim=[cmin, cmax], show_scalar_bar=(mk == markers[0]),
                    name=f"region_{mk}", style=state["style"],
                    opacity=state["opacity"], show_edges=state["show_edges"])
            plotter.render()

        def _on_plane(normal, origin):
            state["normal"] = list(normal)
            state["origin"] = list(origin)
            _rebuild()

        plotter.add_plane_widget(_on_plane, normal=state["normal"], origin=state["origin"])

        y = 10
        for mk in markers:
            def _make_cb(mk=mk):
                def _cb(flag):
                    region_visible[mk] = flag
                    _rebuild()
                return _cb
            plotter.add_checkbox_button_widget(_make_cb(), value=True, position=(10, y), size=24)
            plotter.add_text(_region_name(mk), position=(42, y), font_size=10)
            y += 32

        # Display-mode controls, in a second column so they don't collide
        # with the per-region checkboxes above. "Mesh only"/"Transparent"/
        # "Opaque" act as mode-select buttons (clicking one applies that
        # mode outright, regardless of the checkbox's own new state) rather
        # than a true mutually-exclusive radio group -- pyvista has no
        # built-in radio widget, and this keeps the callback logic simple.
        # "Show mesh" is an independent on/off toggle for cell-edge overlay,
        # combinable with any of the three modes above.
        col2_x = 280
        y2 = 10

        def _set_mode(style, opacity):
            def _cb(_flag):
                state["style"] = style
                state["opacity"] = opacity
                _rebuild()
            return _cb

        plotter.add_checkbox_button_widget(_set_mode("wireframe", 1.0), value=False,
                                           position=(col2_x, y2), size=24)
        plotter.add_text("Mesh only", position=(col2_x + 32, y2), font_size=10)
        y2 += 32

        plotter.add_checkbox_button_widget(_set_mode("surface", 0.35), value=False,
                                           position=(col2_x, y2), size=24)
        plotter.add_text("Transparent", position=(col2_x + 32, y2), font_size=10)
        y2 += 32

        plotter.add_checkbox_button_widget(_set_mode("surface", 1.0), value=True,
                                           position=(col2_x, y2), size=24)
        plotter.add_text("Opaque", position=(col2_x + 32, y2), font_size=10)
        y2 += 32

        def _toggle_edges(flag):
            state["show_edges"] = flag
            _rebuild()

        plotter.add_checkbox_button_widget(_toggle_edges, value=False,
                                           position=(col2_x, y2), size=24)
        plotter.add_text("Show mesh", position=(col2_x + 32, y2), font_size=10)

        _rebuild()
        plotter.add_text("Drag the plane widget to slice through the model; "
                         "left checkboxes toggle regions, right checkboxes "
                         "set display mode / mesh overlay",
                         position="upper_edge", font_size=10)
        plotter.show()

    def show_geom(self):
        pg.show(self.geom)

    def _build_geometry_2d(self):
        """
        The original 2D path: a rectangular World with horizontal layer
        boundaries (mt.createWorld(layers=...)), CircleAnom/EllipseAnom
        blocks merged in, a 2D triangular mesh, and resistivity assigned by
        shapely polygon containment. Unchanged from the pre-3D
        implementation other than being extracted into its own method (was
        inline in __init__).
        """
        self.world = mt.createWorld(start=self.start, end=self.end, worldMarker=True,
                                    layers=[layer.y for layer in self.layers])
        self.geom = self.world
        for i, circle in enumerate(self.Circles):
            # Build the update() kwargs from the anomaly's own param_names
            # instead of hardcoding x/y/r/rho, so a shape with a different
            # parameter set (e.g. EllipseAnom's x,y,a,b,theta,rho) can reuse
            # this same loop.
            kwargs = {n: getattr(circle, n) for n in self._entity_param_names(circle, ['x', 'y', 'r', 'rho'])}
            circle.update(name=circle.name, _c_num=i + 1 + len(self.layers),
                          _varflag=circle.varflag, _varlims=circle.varlims, **kwargs)
            self.geom = mt.mergePLC([self.geom, circle.block])
        pos = np.array(self.scheme.sensorPositions())
        pos_xy = pos[:, 0:2]
        for sen in pos_xy:
            self.geom.createNode(sen)
            self.geom.createNode(sen - [0, 0.1])
        self.mesh = mt.createMesh(self.geom, quality=34)
        self.mesh = mt.appendBoundary(self.mesh)
        # Add the initial layer from 0 to the first layer's y value
        self.layer_resistivities = []
        self.layers_lim = []
        if self.layers:
            initial_layer = [(0, self.layers[0].y)]

            # Create the rest of the layers
            subsequent_layers = [(self.layers[i].y, self.layers[i + 1].y) for i in range(len(self.layers) - 1)]

            # Combine initial layer with the rest
            self.layers_lim = initial_layer + subsequent_layers

            # Extract resistivities
            self.layer_resistivities = [layer.rho for layer in self.layers]

        self.resistivity_map = assign_layer_resistivities(self.mesh, self.layers_lim, self.layer_resistivities,
                                                          default_resistivity=self.rho_world)
        # extract polygons from circles:
        self.polygons = [circle.polygon for circle in self.Circles]
        self.polygon_resistivities = [circle.rho for circle in self.Circles]
        self.resistivity_map = assign_polygon_resistivities(self.mesh, self.polygons, self.polygon_resistivities,
                                                            self.resistivity_map)

    def _layer_z_boundaries(self):
        """
        [(z_top, z_bottom), ...] slabs for the 3D layered box, plus the
        markers to give each slab (and the background below the last
        layer) a distinct region marker for mt.createCube. z=0 is the
        surface (self.start[-1]); self.end[-1] is the box floor.
        """
        z_top_of_world = self.start[-1]
        z_bottom_of_world = self.end[-1]
        boundaries = [z_top_of_world] + [layer.y for layer in self.layers] + [z_bottom_of_world]
        slabs = list(zip(boundaries[:-1], boundaries[1:]))
        markers = list(range(1, len(slabs) + 1))
        return slabs, markers

    def _build_mesh_3d_gmsh(self):
        """
        Builds self.mesh with EXACT (conformal) boundaries between layer
        slabs and CylinderAnom bodies, via gmsh's OCC CSG kernel.

        HISTORY, why this replaced pyGIMLi's own mergePLC/TetGen path: an
        earlier version built one mt.createCube() slab per layer and
        mergePLC'd them into a stack of internal horizontal facets. That
        broke in 3D whenever a CylinderAnom crosses a layer boundary: the
        cylinder's curved side surface then pierces straight through a flat
        internal facet without sharing an edge with it, leaving TetGen a
        self-intersecting PLC it cannot triangulate (a bare TetGen crash,
        not a Python exception -- terminate called after throwing an
        instance of 'int' / segfaulting). Root cause: pyGIMLi's mergePLC
        merges by distance tolerance, not exact boolean intersection, and
        its own docs say it does not handle crossing/node intersections.

        gmsh's OCC kernel computes an EXACT boolean fragment of the box
        slabs and cylinders (occ.fragment), so a cylinder crossing a layer
        boundary is cut cleanly into two conformal sub-volumes with a
        shared face -- no self-intersection, no TetGen crash. Each
        resulting fragment volume is classified by (layer_idx, cyl_idx)
        from its center of mass and tagged with a pyGIMLi-compatible
        physical-group marker; assign_layer_resistivities_3d /
        assign_cylinder_resistivities still work fine on this mesh too, and
        are what _build_geometry_3d/_parse_all_3d call afterwards -- this
        method only builds the mesh with exact geometry, it does not itself
        assign resistivities.

        Electrodes are embedded as exact mesh nodes on the domain's flat
        top face (z=z0, the survey datum) via gmsh.model.mesh.embed, found
        by bounding-box matching against that face.

        gmsh 4.15 writes MSH format 4.1 by default, which pyGIMLi's
        readGmsh() cannot parse (UnboundLocalError on 'nodes') -- forced to
        legacy ASCII v2.2 via Mesh.MshFileVersion before gmsh.write().

        Region marker scheme: self._region_markers maps (layer_idx,
        cyl_idx) -> integer marker, cyl_idx=None for background-only
        (no-cylinder) regions in that layer. Verified live: a 1-layer/
        1-cylinder world produces {(0, None): 1, (0, 0): 2, (1, None): 3,
        (1, 0): 4}.
        """
        import os
        import tempfile
        import gmsh

        x0, y0, z0 = self.start
        x1, y1, z1 = self.end
        slabs, _ = self._layer_z_boundaries()

        xlo, xhi = min(x0, x1), max(x0, x1)
        ylo, yhi = min(y0, y1), max(y0, y1)
        dx, dy = xhi - xlo, yhi - ylo

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(f"world3d_{id(self)}")
            occ = gmsh.model.occ

            box_tags = []
            for z_top, z_bot in slabs:
                lo, hi = min(z_top, z_bot), max(z_top, z_bot)
                box_tags.append(occ.addBox(xlo, ylo, lo, dx, dy, hi - lo))

            cyl_tags = []
            for cyl in self.Cylinders:
                a = np.array([cyl.x1, cyl.y1, cyl.z1], dtype=float)
                b = np.array([cyl.x2, cyl.y2, cyl.z2], dtype=float)
                axis = b - a
                cyl_tags.append(occ.addCylinder(
                    a[0], a[1], a[2], axis[0], axis[1], axis[2], cyl.r))

            # Electrode points are created here and fed into the SAME
            # occ.fragment() call below as 0D "tool" entities, not embedded
            # afterwards via a separate gmsh.model.mesh.embed() call. Both
            # approaches are gmsh-legal, but for a domain this size (tens of
            # metres) with many small-meshSize electrode points on a face
            # that's also being split by a fragmenting cylinder, a post-hoc
            # embed() call was found (empirically, via a real failing 3D
            # world test) to leave gmsh's 3D tet reconstruction unable to
            # recover some boundary faces near the electrodes -- it doesn't
            # raise an exception, just silently emits "No elements in
            # volume N" and produces a mesh missing that entire region
            # (which happened to be the whole near-surface layer, so every
            # electrode ended up off the actual meshed domain --
            # "There is a requested electrode that does not match the given
            # mesh." downstream in ert.simulate/invert). Fragmenting the
            # points in together makes them exact conformal vertices of the
            # CSG result from the start, matching the pattern in pyGIMLi's
            # own gmsh CAD-tutorial reference; verified to remove the
            # failure entirely (0 warnings, all electrodes exact mesh
            # nodes, real forward solve finite/positive) on this same test
            # geometry (a shaft that drifts across the first layer
            # boundary -- the case that triggered the bug).
            pos = np.array(self.scheme.sensorPositions())
            elec_pts = []
            for sen in pos:
                ex = sen[0]
                ey = sen[1] if len(sen) > 1 else 0.0
                elec_pts.append(occ.addPoint(ex, ey, z0, meshSize=self._gmsh_elec_size()))

            occ.synchronize()

            tool_dimtags = [(3, t) for t in cyl_tags] + [(0, p) for p in elec_pts]
            if tool_dimtags:
                out, _ = occ.fragment([(3, t) for t in box_tags], tool_dimtags)
                occ.synchronize()
            else:
                out = [(3, t) for t in box_tags]
            vols = [t for (d, t) in out if d == 3]

            marker_of = {}
            next_marker = [1]
            groups = {}
            for v in vols:
                cx, cy, cz = occ.getCenterOfMass(3, v)
                p = np.array([cx, cy, cz])

                layer_idx = len(slabs) - 1
                for i, (z_top, z_bot) in enumerate(slabs):
                    lo, hi = min(z_top, z_bot), max(z_top, z_bot)
                    if lo - 1e-6 <= cz <= hi + 1e-6:
                        layer_idx = i
                        break

                cyl_idx = None
                for i, cyl in enumerate(self.Cylinders):
                    a = np.array([cyl.x1, cyl.y1, cyl.z1], dtype=float)
                    b = np.array([cyl.x2, cyl.y2, cyl.z2], dtype=float)
                    if _point_segment_distance(p, a, b) < cyl.r * 0.9:
                        cyl_idx = i
                        break

                key = (layer_idx, cyl_idx)
                if key not in marker_of:
                    marker_of[key] = next_marker[0]
                    next_marker[0] += 1
                groups.setdefault(marker_of[key], []).append(v)

            for mk, tags in groups.items():
                pgtag = gmsh.model.addPhysicalGroup(3, tags, mk)
                gmsh.model.setPhysicalName(3, pgtag, f"region_{mk}")
            self._region_markers = marker_of

            # Electrodes are already exact conformal vertices of the CSG
            # result (fragmented in above, alongside the cylinders) -- just
            # sanity-check each one still exists as a 0D point entity at
            # its expected position; no further embed() call needed.
            _pt_tags = {t for (d, t) in gmsh.model.getEntities(0)}
            missing = [pt for pt in elec_pts if pt not in _pt_tags]
            if missing:
                print(f"WARNING: {len(missing)} electrode point(s) did not survive "
                      f"the CSG fragment as distinct vertices -- check for "
                      f"electrodes sitting exactly on a cylinder/layer edge.")

            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self._gmsh_max_size())
            gmsh.option.setNumber("Mesh.MeshSizeMin", self._gmsh_elec_size())
            gmsh.model.mesh.generate(3)

            fd, msh_path = tempfile.mkstemp(suffix=".msh")
            os.close(fd)
            gmsh.write(msh_path)
        finally:
            gmsh.finalize()

        self.mesh = mt.readGmsh(msh_path)
        os.remove(msh_path)
        self.geom = self.mesh

    def _gmsh_max_size(self):
        if hasattr(self, 'gmsh_mesh_size_max'):
            return self.gmsh_mesh_size_max
        x0, y0, _ = self.start
        x1, y1, _ = self.end
        return max(min(abs(x1 - x0), abs(y1 - y0)) / 15.0, 0.5)

    def _gmsh_elec_size(self):
        if hasattr(self, 'gmsh_mesh_size_min'):
            return self.gmsh_mesh_size_min
        if self.Cylinders:
            return max(min(cyl.r for cyl in self.Cylinders) / 3.0, 0.05)
        return self._gmsh_max_size() / 6.0

    def _build_geometry_3d(self):
        """
        3D counterpart of _build_geometry_2d. Delegates the actual mesh
        construction to _build_mesh_3d_gmsh (see its docstring for why:
        exact, conformal layer/cylinder boundaries via gmsh's OCC boolean
        CSG kernel, replacing an earlier pyGIMLi mergePLC-based approach
        that could not reliably handle a cylinder crossing a layer
        boundary). This method just keeps entity bookkeeping (cyl.update)
        consistent with the 2D path before delegating.
        """
        for i, cyl in enumerate(self.Cylinders):
            kwargs = {n: getattr(cyl, n) for n in self._entity_param_names(
                cyl, ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'r', 'rho'])}
            cyl.update(name=cyl.name, _c_num=i + 1 + len(self.layers),
                      _varflag=cyl.varflag, _varlims=cyl.varlims, **kwargs)

        self._build_mesh_3d_gmsh()

        slabs, _ = self._layer_z_boundaries()
        self.layer_resistivities = [layer.rho for layer in self.layers]
        self.layers_lim = slabs[:len(self.layers)]
        self.resistivity_map = assign_layer_resistivities_3d(
            self.mesh, self.layers_lim, self.layer_resistivities,
            default_resistivity=self.rho_world)

        cyl_specs = [(c.x1, c.y1, c.z1, c.x2, c.y2, c.z2, c.r) for c in self.Cylinders]
        cyl_rhos = [c.rho for c in self.Cylinders]
        self.resistivity_map = assign_cylinder_resistivities(
            self.mesh, cyl_specs, cyl_rhos, self.resistivity_map)

    # def parse_all(self):
    #     for i, layer in enumerate(self.layers):
    #         layer.update(name=layer.name, y=layer.y, rho=layer.rho, cnum=i + 1, _varflag=layer.varflag,
    #                      _varlims=layer.varlims)
    #     self.world = mt.createWorld(start=self.start, end=self.end, worldMarker=True,
    #                                 layers=[layer.y for layer in self.layers])
    #     self.geom = self.world
    #     for i, circle in enumerate(self.Circles):
    #         circle.update(name=circle.name, x=circle.x, y=circle.y, r=circle.r, rho=circle.rho,
    #                       _c_num=i + 1 + len(self.layers),
    #                       _varflag=circle.varflag, _varlims=circle.varlims)
    #         self.geom = mt.mergePLC([self.geom, circle.block])
    #     pos = np.array(self.scheme.sensorPositions())
    #     pos_xy = pos[:, 0:2]
    #     for sen in pos_xy:
    #         self.geom.createNode(sen)
    #         self.geom.createNode(sen - [0, 0.1])
    #     self.mesh = mt.createMesh(self.geom, quality=34)
    #     if self.layers:
    #         # Add the initial layer from 0 to the first layer's y value
    #         initial_layer = [(0, self.layers[0].y)]
    #
    #         # Create the rest of the layers
    #         subsequent_layers = [(self.layers[i].y, self.layers[i + 1].y) for i in range(len(self.layers) - 1)]
    #
    #         # Combine initial layer with the rest
    #         self.layers_lim = initial_layer + subsequent_layers
    #
    #         # Extract resistivities
    #         self.layer_resistivities = [layer.rho for layer in self.layers]
    #
    #         self.resistivity_map = assign_layer_resistivities(self.mesh, self.layers_lim, self.layer_resistivities,
    #                                                           default_resistivity=self.rho_world)
    #     else:
    #         self.resistivity_map = [self.rho_world] * self.mesh.cellCount()
    #     # extract polygons from circles:
    #     self.polygons = [circle.polygon for circle in self.Circles]
    #     self.polygon_resistivities = [circle.rho for circle in self.Circles]
    #     self.resistivity_map = assign_polygon_resistivities(self.mesh, self.polygons, self.polygon_resistivities,
    #                                                         self.resistivity_map)
    #     a, lb, ub = self.constraint_group.evaluate_constraints(self.variable_map, self.parameter_map,
    #                                                            self.num_variables)
    #     if a.size <= 0:
    #         self.constraints = None
    #     else:
    #         self.constraints = LinearConstraint(a, lb, ub)
    def parse_all(self, force_regenerate=False):
        geometry_changed = False

        # Check if layer or anomaly geometry has changed
        for i, layer in enumerate(self.layers):
            last_layer = self._last_layers_state[i]
            if layer.y != last_layer:
                geometry_changed = True
            # Update last state
            self._last_layers_state[i] = layer.y
        anomalies = self._anomalies()
        for i, anomaly in enumerate(anomalies):
            last_state = self._last_circles_state[i]
            key = self._geometry_key(anomaly)
            if key != last_state:
                geometry_changed = True
                # Update last state
                self._last_circles_state[i] = key

        if self.is_3d:
            self._parse_all_3d(force_regenerate, geometry_changed)
        else:
            self._parse_all_2d(force_regenerate, geometry_changed)

    def _parse_all_2d(self, force_regenerate, geometry_changed):
        if force_regenerate or geometry_changed or not hasattr(self, 'mesh'):
            # Regenerate geometry and mesh only if necessary
            self.world = mt.createWorld(start=self.start, end=self.end, worldMarker=True,
                                        layers=[layer.y for layer in self.layers])
            self.geom = self.world
            for circle in self.Circles:
                circle.update_block()
                self.geom = mt.mergePLC([self.geom, circle.block])

            pos = np.array(self.scheme.sensorPositions())
            pos_xy = pos[:, 0:2]
            for sen in pos_xy:
                self.geom.createNode(sen)
                self.geom.createNode(sen - [0, 0.1])
            self.mesh = mt.createMesh(self.geom, quality=34)

        # Always update resistivity map (since rho values might change)
        if self.layers:
            initial_layer = [(0, self.layers[0].y)]
            subsequent_layers = [(self.layers[i].y, self.layers[i + 1].y)
                                 for i in range(len(self.layers) - 1)]
            self.layers_lim = initial_layer + subsequent_layers
            self.layer_resistivities = [layer.rho for layer in self.layers]
            self.resistivity_map = assign_layer_resistivities(self.mesh, self.layers_lim,
                                                              self.layer_resistivities,
                                                              default_resistivity=self.rho_world)
        else:
            self.resistivity_map = [self.rho_world] * self.mesh.cellCount()

        self.polygons = [circle.polygon for circle in self.Circles]
        self.polygon_resistivities = [circle.rho for circle in self.Circles]
        self.resistivity_map = assign_polygon_resistivities(self.mesh, self.polygons,
                                                            self.polygon_resistivities,
                                                            self.resistivity_map)

    def _parse_all_3d(self, force_regenerate, geometry_changed):
        if force_regenerate or geometry_changed or not hasattr(self, 'mesh'):
            for cyl in self.Cylinders:
                cyl.update_block()
            self._build_mesh_3d_gmsh()

        slabs, _ = self._layer_z_boundaries()
        self.layers_lim = slabs[:len(self.layers)]
        self.layer_resistivities = [layer.rho for layer in self.layers]
        self.resistivity_map = assign_layer_resistivities_3d(
            self.mesh, self.layers_lim, self.layer_resistivities,
            default_resistivity=self.rho_world)

        cyl_specs = [(c.x1, c.y1, c.z1, c.x2, c.y2, c.z2, c.r) for c in self.Cylinders]
        cyl_rhos = [c.rho for c in self.Cylinders]
        self.resistivity_map = assign_cylinder_resistivities(
            self.mesh, cyl_specs, cyl_rhos, self.resistivity_map)

    def get_varflag_n(self):
        n_var = int(self.rho_world_varflag)
        for layer in self.layers:
            n_var += sum(int(vf) for vf in layer.varflag)
        for circle in self._anomalies():
            n_var += sum(int(vf) for vf in circle.varflag)
        return n_var

    @staticmethod
    def _entity_param_names(entity, default):
        return getattr(entity, 'param_names', default)

    @staticmethod
    def _geometry_key(anomaly):
        """
        Tuple of the anomaly's geometry-affecting param values (everything
        but rho, which only changes the resistivity map, not the mesh),
        used to decide whether parse_all() must rebuild the mesh. Was
        hardcoded to (x, y, r); generalized so a shape with more geometry
        params (e.g. an ellipse's x, y, a, b, theta) is tracked correctly
        instead of silently skipping a mesh rebuild when only its non-(x,y,r)
        params change.
        """
        names = getattr(anomaly, 'param_names', ['x', 'y', 'r', 'rho'])
        return tuple(getattr(anomaly, n) for n in names if n != 'rho')

    def update(self, x):
        if isinstance(x, int):
            if self.rho_world_varflag:
                if self.rho_world_varflag:
                    self.rho_world = 10 ** x
                for layer in self.layers:
                    names = self._entity_param_names(layer, ['y', 'rho'])
                    for i, name in enumerate(names):
                        if layer.varflag[i]:
                            log = layer.log_flags.get(name, False) if hasattr(layer, 'log_flags') else (name == 'rho' and layer.log_rho)
                            setattr(layer, name, 10 ** x if log else x)
                for circle in self._anomalies():
                    names = self._entity_param_names(circle, ['x', 'y', 'r', 'rho'])
                    for i, name in enumerate(names):
                        if circle.varflag[i]:
                            log = circle.log_flags.get(name, False) if hasattr(circle, 'log_flags') else False
                            setattr(circle, name, 10 ** x if log else x)
                return None
        if len(x) == self.get_varflag_n():
            idx = 0
            if self.rho_world_varflag:
                self.rho_world = 10 ** x[idx]
                idx += 1
            for layer in self.layers:
                names = self._entity_param_names(layer, ['y', 'rho'])
                log_flags = layer.log_flags if hasattr(layer, 'log_flags') else {}
                for i, name in enumerate(names):
                    if layer.varflag[i]:
                        log = log_flags.get(name, False)
                        setattr(layer, name, 10 ** x[idx] if log else x[idx])
                        idx += 1
            for circle in self._anomalies():
                names = self._entity_param_names(circle, ['x', 'y', 'r', 'rho'])
                log_flags = circle.log_flags if hasattr(circle, 'log_flags') else {}
                for i, name in enumerate(names):
                    if circle.varflag[i]:
                        log = log_flags.get(name, False)
                        setattr(circle, name, 10 ** x[idx] if log else x[idx])
                        idx += 1
        else:
            raise ValueError
        self.parse_all(force_regenerate=False)

    def get_bounds(self):
        """
        Build the (lb, ub) bounds list used by the DE/L-BFGS-B optimizers, in
        the exact same parameter order as get_x0()/update(). Kept as a single
        source of truth so external callers (e.g. run scripts building a
        custom initial population) don't have to re-derive it by hand.
        """
        lb, ub = [], []

        if self.rho_world_varflag:
            lb.append(np.log10(self.rho_world_varlim[0]))
            ub.append(np.log10(self.rho_world_varlim[1]))

        for layer in self.layers:
            names = self._entity_param_names(layer, ['y', 'rho'])
            log_flags = layer.log_flags if hasattr(layer, 'log_flags') else {}
            if sum(map(int, layer.varflag)) == 1:
                for i, v in enumerate(layer.varflag):
                    if v:
                        if log_flags.get(names[i], False):
                            lb.append(np.log10(layer.varlims[0]))
                            ub.append(np.log10(layer.varlims[1]))
                        else:
                            lb.append(layer.varlims[0])
                            ub.append(layer.varlims[1])
            else:
                k = 0
                for i, v in enumerate(layer.varflag):
                    if v:
                        if log_flags.get(names[i], False):
                            lb.append(np.log10(layer.varlims[i][0]))
                            ub.append(np.log10(layer.varlims[i][1]))
                        else:
                            lb.append(layer.varlims[k][0])
                            ub.append(layer.varlims[k][1])
                        k += 1

        for circle in self._anomalies():
            names = self._entity_param_names(circle, ['x', 'y', 'r', 'rho'])
            log_flags = circle.log_flags if hasattr(circle, 'log_flags') else {}
            if sum(map(int, circle.varflag)) == 1:
                for i, v in enumerate(circle.varflag):
                    if v:
                        if log_flags.get(names[i], False):
                            lb.append(np.log10(circle.varlims[0]))
                            ub.append(np.log10(circle.varlims[1]))
                        else:
                            lb.append(circle.varlims[0])
                            ub.append(circle.varlims[1])
            else:
                for i, v in enumerate(circle.varflag):
                    if v:
                        if log_flags.get(names[i], False):
                            lb.append(np.log10(circle.varlims[i][0]))
                            ub.append(np.log10(circle.varlims[i][1]))
                        else:
                            lb.append(circle.varlims[i][0])
                            ub.append(circle.varlims[i][1])

        return list(zip(lb, ub))

    def refresh_constraints(self):
        """
        Recompute self.constraints from the current state of constraint_group.
        Must be called after adding constraints via
        constraint_group.add_constraint(...) post-__init__, otherwise the new
        constraints silently never reach the optimizer (self.constraints was
        already frozen once during __init__).
        """
        a, lb, ub = self.constraint_group.evaluate_constraints(
            self.variable_map, self.parameter_map, self.num_variables)
        if a.size > 0:
            self.constraints = LinearConstraint(a, lb, ub)
        else:
            self.constraints = None

    def get_x0(self):
        x0 = []
        if self.rho_world_varflag:
            x0.append(np.log10(self.rho_world))  # Append since self.rho_world is a single value
        for layer in self.layers:
            names = self._entity_param_names(layer, ['y', 'rho'])
            log_flags = layer.log_flags if hasattr(layer, 'log_flags') else {}
            for i, name in enumerate(names):
                if layer.varflag[i]:
                    value = getattr(layer, name)
                    x0.append(np.log10(value) if log_flags.get(name, False) else value)
        for circle in self._anomalies():
            names = self._entity_param_names(circle, ['x', 'y', 'r', 'rho'])
            log_flags = circle.log_flags if hasattr(circle, 'log_flags') else {}
            for i, name in enumerate(names):
                if circle.varflag[i]:
                    value = getattr(circle, name)
                    x0.append(np.log10(value) if log_flags.get(name, False) else value)
        return x0  # Ensure the method returns x0

    # def get_forward_solution(self, _noise=False, _noise_abs=1e-6):
    #     self.parse_all()
    #     if _noise:
    #         data = ert.simulate(self.mesh, scheme=self.scheme, res=self.resistivity_map, noiseLevel=1,
    #                             noiseAbs=_noise_abs, seed=1337, verbose=False)
    #     else:
    #         data = ert.simulate(self.mesh, scheme=self.scheme, res=self.resistivity_map, verbose=False)
    #
    #     return data

    def get_forward_solution(self, _noise=False, _noise_level=0.5, _noiseAbs=1e-6, _seed=2351):
        self.parse_all()  # Call without forcing regeneration unless needed

        if _noise==False:
            data = ert.simulate(self.mesh, scheme=self.scheme, res=self.resistivity_map,
                                verbose=False)
        if _noise==True:
            data = ert.simulate(self.mesh, scheme=self.scheme, res=self.resistivity_map,
                                verbose=False,noiseLevel=_noise_level, noiseAbs=_noiseAbs, seed=_seed)
        return data

    def get_forward_solution_v(self):
        self.parse_all()
        self.fop.setMesh(self.mesh)
        self.fop.setData(self.scheme)
        return self.fop(self.resistivity_map)

    # A simple RMS just to begin with
    # def rms(self, x):
    #     self.n_fev += 1
    #     model = self.copy()        # deep copy of world
    #     model.update(x)
    #     data = model.get_forward_solution()
    #     err = np.array(model.meas) - np.array(data['rhoa'])
    #     self._fev_display.update(f"Function evals: {self.n_fev} | RMS: {er_rms:.3e}")
    #     return np.sqrt(np.mean(err**2))
    # def mse(self, x):
    #     self.n_fev += 1
    #
    #     x0 = self.get_x0()
    #     self.update(x)
    #     data = self.get_forward_solution()
    #     err = np.array(self.meas) - np.array(data['rhoa'])
    #     er_mse = np.sqrt(np.mean(err**2))
    #     self.update(x0)
    #
    #     # Update best
    #     if er_mse < self.best_mse:
    #         self.best_mse = er_mse
    #         self.best_x = np.array(x, copy=True)
    #
    #     # Update display every 10 evals
    #     if self.n_fev % 10 == 0:
    #         print(
    #             f"Eval: {self.n_fev} | "
    #             f"Current MSE: {er_mse:.3e} | "
    #             f"Best MSE: {self.best_mse:.3e}\n"
    #             f"Current x: {np.round(x, 4)}\n"
    #             f"Best x:    {np.round(self.best_x, 4)}"
    #         )
    #
    #     return er_mse
    def male(self, x):
        self.n_fev += 1

        # x0 = self.get_x0()
        self.update(x)
        # print(self.get_x0())
        data = self.get_forward_solution(_noise=False)
        # print(data['rhoa'])
        # log_abs_err =np.log10(np.array(data['rhoa'])) - np.log10(np.array(self.meas))
        # er_male = np.sum(np.sqrt(log_abs_err ** 2)) / len(log_abs_err)
        rhoa = np.asarray(data['rhoa'])
        meas = np.asarray(self.meas)

        male = np.mean(np.abs(np.log10(rhoa / meas)))
        # print(er_mse)
        # self.update(x0)

        # Update best
        if male < self.best_male:
            msle=self.msle(x)
            self.best_male = male
            self.best_x_male = np.array(x, copy=True)
            self.history['fevals'].append(self.n_fev)
            self.history['best_male'].append(male)
            self.history['best_msle'].append(msle)

        # Update display every 10 evals
        if self.n_fev % 100 == 0:
            print(
                f"Eval: {self.n_fev} | "
                f"Current MALE: {male:.2e} | "
                f"Best MALE: {self.best_male:.2e}\n"
                f"Current x: {fmt_array(x)}\n"
                f"Best x:    {fmt_array(self.best_x_male)}"
            )

        return male

    def msle(self, x):
        self.n_fev += 1

        # x0 = self.get_x0()
        self.update(x)
        # print(self.get_x0())
        data = self.get_forward_solution(_noise=False)
        # print(data['rhoa'])
        # log_abs_err =np.log10(np.array(data['rhoa'])) - np.log10(np.array(self.meas))
        # er_male = np.sum(np.sqrt(log_abs_err ** 2)) / len(log_abs_err)
        y_true = np.array(data['rhoa'])
        y_pred = np.array(self.meas)
        # ensure positivity
        y_true = np.maximum(y_true, 1e-12)
        y_pred = np.maximum(y_pred, 1e-12)
        log_diff = np.log1p(y_true) - np.log1p(y_pred)  # log1p = log(1+x), more stable
        msle = np.mean(log_diff**2)
        # print(er_mse)
        # self.update(x0)

        # Update best
        if msle < self.best_msle:
            self.best_msle = msle
            self.best_x_msle = np.array(x, copy=True)
            male=self.male(x)
            self.history['fevals'].append(self.n_fev)
            self.history['best_male'].append(male)
            self.history['best_msle'].append(msle)

        # Update display every 10 evals
        if self.n_fev % 100 == 0:
            print(
                f"Eval: {self.n_fev} | "
                f"Current MSLE: {msle:.2e} | "
                f"Best MSLE: {self.best_msle:.2e}\n"
                f"Current x: {fmt_array(x)}\n"
                f"Best x:    {fmt_array(self.best_x_msle)}"
            )

        return msle
        


    # only if there is a variable layer height
    # def switch_case(self, value, n_vars, idx, circle):
    #     switcher = {
    #         (1, 0, 0): [
    #             [self.create_array(n_vars, idx, 1), self.start[0] - circle.r, self.end[0] - circle.r],
    #             [self.create_array(n_vars, idx, 1), self.start[0] + circle.r, self.end[0] + circle.r]
    #         ],
    #         (0, 1, 0): [
    #             [self.create_array(n_vars, idx, 1), self.end[1] - circle.r, self.start[1] - circle.r],
    #             [self.create_array(n_vars, idx, 1), self.end[1] + circle.r, self.start[1] + circle.r]
    #         ],
    #         (1, 0, 1): [
    #             [self.create_array(n_vars, idx, 1, idx + 2, 1), self.start[0], self.end[0]],
    #             [self.create_array(n_vars, idx, 1, idx + 2, -1), self.start[0], self.end[0]],
    #             [self.create_array(n_vars, idx + 2, 1), self.end[1] - circle.y, self.start[1] - circle.y],
    #             [self.create_array(n_vars, idx + 2, -1), self.end[1] - circle.y, self.start[1] - circle.y]
    #         ],
    #         (0, 1, 1): [
    #             [self.create_array(n_vars, idx, 1, idx + 1, 1), self.end[1], self.start[1]],
    #             [self.create_array(n_vars, idx, 1, idx + 1, -1), self.end[1], self.start[1]],
    #             [self.create_array(n_vars, idx + 1, 1), self.start[0] - circle.x, self.end[0] - circle.x],
    #             [self.create_array(n_vars, idx + 1, -1), self.start[0] - circle.x, self.end[0] - circle.x]
    #         ],
    #         (1, 1, 0): [
    #             [self.create_array(n_vars, idx, 1), self.start[0] + circle.r, self.end[0] + circle.r],
    #             [self.create_array(n_vars, idx, 1), self.start[0] - circle.r, self.end[0] - circle.r],
    #             [self.create_array(n_vars, idx + 1, 1), self.end[1] + circle.r, self.start[1] + circle.r],
    #             [self.create_array(n_vars, idx + 1, -1), self.end[1] - circle.r, self.start[1] - circle.r]
    #         ],
    #         (1, 1, 1): [
    #             [self.create_array(n_vars, idx, 1, idx + 2, 1), self.start[0], self.end[0]],
    #             [self.create_array(n_vars, idx, 1, idx + 2, -1), self.start[0], self.end[0]],
    #             [self.create_array(n_vars, idx + 1, 1, idx + 2, 1), self.end[1], self.start[1]],
    #             [self.create_array(n_vars, idx + 1, 1, idx + 2, -1), self.end[1], self.start[1]]
    #         ]
    #     }
    #     return switcher.get(tuple(value), [])

    # def create_array(self, n_vars, idx, val, idx2=None, val2=None):
    #     arr = np.zeros(n_vars)
    #     arr[idx] = val
    #     if idx2 is not None:
    #         arr[idx2] = val2
    #     return arr

    # def generate_auto_constraints(self):
    #     """
    #     מייצרת אילוצים אוטומטיים עבור שכבות ואנומליות עגולות.
    #
    #     Returns:
    #     - List of constraints in the format:
    #       [entity1, variabletype1, entity2, variabletype2, constraint_type, parameter_value].
    #     """
    #
    #     # אילוצים בין גבהים של שכבות - צריך עדכון בהתאם למקור , לא לשכוח כלום
    #     for i in range(len(self.layers) - 1):
    #         layer1 = self.layers[i]
    #         layer2 = self.layers[i + 1]
    #         if layer1.varflag[0] and layer2.varflag[0]:  # רק אם הגובה ניתן לשינוי
    #             self.constraint_group.add_constraint(
    #                 [layer1, "y", layer2, "y", "<=", 0])  # גובה השכבה העליונה קטן מגובה התחתונה
    #
    #     # אילוצים בין שכבות לאנומליות
    #     for circle in self.Circles:
    #         if circle.varflag[1]:  # רק אם Y של המעגל ניתן לשינוי
    #             for layer in self.layers:
    #                 if layer.varflag[0]:  # רק אם Y של השכבה ניתן לשינוי
    #                     self.constraint_group.add_constraint(
    #                         [circle, "y", layer, "y", ">=", circle.r])  # מרכז המעגל גבוה משכבה + רדיוס
    #
    #     # אילוצים על רדיוס המעגלים
    #     for circle in self.Circles:
    #         if circle.varflag[2]:  # רק אם הרדיוס ניתן לשינוי
    #             self.constraint_group.add_constraint(entity1=circle, var1="r", entity2=None, var2=None,
    #                                                  constraint_type=">=", param_value1=0)  # רדיוס חייב להיות חיובי

    # def get_layer_constraints(self):
    #     n = 0
    #     for layer in self.layers:
    #         n += layer.varflag[0]
    #     # print("layer_n:",n)
    #     A = np.zeros((2 * n, self.get_varflag_n()))
    #     # print("A:", A)
    #     lb = np.zeros(2 * n)
    #     ub = np.zeros(2 * n)
    #     # print("ub:",len(ub))
    #     keep_feasible = True  # assuming this is a scalar
    #     idx = int(self.rho_world_varflag)
    #     # print("idx:", idx)
    #     l_A = 0
    #     for i, layer in enumerate(self.layers):
    #         if layer.varflag[0]:
    #             # print("CHECK")
    #             if i == 0:
    #                 if i + 1 < len(self.layers):
    #                     A[l_A, idx] = 1
    #                     A[l_A, idx + 1 + self.layers[i].varflag[1]] = -1 * self.layers[i + 1].varflag[0]
    #                     lb[l_A] = self.layers[i + 1].y * (1 - self.layers[i + 1].varflag[0])
    #                     ub[l_A] = np.inf
    #                 else:
    #                     A[l_A, idx] = 1
    #                     lb[l_A] = self.end[1]
    #                     ub[l_A] = np.inf
    #                 A[l_A + 1, idx] = 1
    #                 lb[l_A + 1] = -np.inf
    #                 ub[l_A + 1] = self.start[1]
    #             else:
    #                 if i + 1 == len(self.layers):
    #                     print("i_check:", i)
    #                     print("idx:", idx)
    #                     A[l_A + 1, idx] = 1
    #                     A[l_A + 1, idx - 1 - self.layers[i - 1].varflag[1]] = -1 * self.layers[
    #                         i - 1].varflag[0]
    #                     ub[l_A + 1] = self.layers[i - 1].y * (1 - self.layers[i - 1].varflag[0])
    #                     lb[l_A + 1] = -np.inf
    #                     A[l_A, idx] = 1
    #                     lb[l_A] = self.end[1]
    #                     ub[l_A] = np.inf
    #                 else:
    #                     A[l_A, idx] = 1
    #                     A[l_A, idx + 1 + self.layers[i].varflag[1]] = -1 * self.layers[i + 1].varflag[0]
    #                     lb[l_A] = self.layers[i + 1].y * (1 - self.layers[i + 1].varflag[0])
    #                     ub[l_A] = np.inf
    #                     A[l_A + 1, idx] = 1
    #                     A[l_A + 1, idx - 1 - self.layers[i - 1].varflag[1]] = -1 * self.layers[
    #                         i - 1].varflag[0]
    #                     ub[l_A + 1] = self.layers[i - 1].y * (1 - self.layers[i - 1].varflag[1])
    #                     lb[l_A + 1] = -np.inf
    #             l_A += 2
    #             idx += layer.varflag[1]
    #             idx += layer.varflag[0]
    #     n = 0
    #     for circle in self.Circles:
    #         switcher = self.switch_case((int(circle.varflag[0]), int(circle.varflag[1]), int(circle.varflag[2])),
    #                                     self.get_varflag_n(), idx, circle)
    #         A = np.vstack((A, [output[0] for output in switcher]))
    #         lb = np.hstack((lb, [output[1] for output in switcher]))
    #         ub = np.hstack((ub, [output[2] for output in switcher]))
    #         idx += int(circle.varflag[0]) + int(circle.varflag[1]) + int(circle.varflag[2]) + int(circle.varflag[3])
    #         # first_elements = [output[1] for output in switcher]
    #         # print(first_elements)
    #     # A2 = np.zeros((4 * n, self.get_varflag_n()))
    #     # lb2 = np.zeros(4 * n)
    #     # ub2 = np.zeros(4 * n)
    #     # l_A = 0
    #     # for circle in self.Circles:
    #     #     if circle.varflag[0] or circle.varflag[1] or circle.varflag[2]:
    #     #         A[l_A, idx] = 1
    #     #         A[l_A, idx + 1 + self.layers[i].varflag[1]] = -1 * self.layers[i + 1].varflag[1]
    #     # print("lb:", lb)
    #     # print("ub:", ub)
    #     return LinearConstraint(A, lb=lb, ub=ub, keep_feasible=keep_feasible)

    # def ls_opti(self, x0=None, max_iter=25, _n=3, atol=0, target_value=None,_workers=1,_updating='immediate',_popsize=1000):
    #     res = None
    #     # RelTol = 1.0e-3,
    #     if x0 is None:
    #         x0 = []
    #     if not x0:
    #         x0 = self.get_x0()
    #     print("Initial x0:", x0)
    #     print("Number of layers:", len(self.layers))
    #     print("Bounds and constraints setup...")
    #     lb = []
    #     ub = []
    #     if self.rho_world_varflag:
    #         lb.append(self.rho_world_varlim[0])
    #         ub.append(self.rho_world_varlim[1])
    #     for layer in self.layers:
    #         if int(layer.varflag[0]) + int(layer.varflag[1]) == 1:
    #             for i, var in enumerate(layer.varflag):
    #                 if var:
    #                     lb.append(layer.varlims[0])
    #                     ub.append(layer.varlims[1])
    #         else:
    #             for i, var in enumerate(layer.varflag):
    #                 if var:
    #                     lb.append(layer.varlims[i][0])
    #                     ub.append(layer.varlims[i][1])
    #     for circle in self.Circles:
    #         if int(circle.varflag[0]) + int(circle.varflag[1]) + int(circle.varflag[2]) + int(circle.varflag[3]) == 1:
    #             for i, var in enumerate(circle.varflag):
    #                 if var:
    #                     lb.append(circle.varlims[0])
    #                     ub.append(circle.varlims[1])
    #         else:
    #             for i, var in enumerate(circle.varflag):
    #                 if var:
    #                     lb.append(circle.varlims[i][0])
    #                     ub.append(circle.varlims[i][1])
    #     #     # Print bounds to ensure they match the number of parameters
    #     print("Lower bounds:", lb)
    #     print("Upper bounds:", ub)
    #
    #     bounds = sp.optimize.Bounds(lb=lb, ub=ub, keep_feasible=True)
    #     print("residual bounds:", bounds.residual(x0))
    #     constraints = self.constraints
    #
    #     # Define callback for stopping based on target value
    #     def callback(xk, convergence):
    #         current_value = self.mse(xk)
    #         # print(f"Current objective value: {current_value}")
    #         if target_value is not None and current_value <= target_value:
    #             print("Target value reached. Stopping optimization.")
    #             return True  # Stops optimization
    #         return False
    #
    #     # Debug: Check if constraints are empty or valid
    #     if not constraints:
    #         print("No constraints found.")
    #     else:
    #         print("CHECK!!!")
    #         print("Constraints:", constraints)
    #
    #     try:
    #         if len(constraints.A) == 0:
    #             print("No constraints found.")
    #             res = differential_evolution(self.mse, bounds=bounds, strategy='best1bin', maxiter=max_iter,
    #                                          popsize=_popsize, tol=1e-5, atol=atol,
    #                                          mutation=(0.5, 1), recombination=0.7, seed=None,
    #                                          disp=True, callback=callback,
    #                                          polish=True, init='latinhypercube',
    #                                          updating=_updating,
    #                                          workers=_workers)
    #         else:
    #             print("CHECK!!!")
    #             res = differential_evolution(self.mse, bounds=bounds, strategy='best1bin', maxiter=max_iter,
    #                                          popsize=_popsize, tol=0.01, atol=atol,
    #                                          mutation=(0.5, 1), recombination=0.7, seed=None,
    #                                          callback=callback,
    #                                          disp=True,
    #                                          polish=True, init='latinhypercube',
    #                                          updating=_updating,
    #                                          workers=_workers,
    #                                          constraints=constraints)
    #
    #     except IndexError as e:
    #         print("An IndexError occurred:", e)
    #         print("Please check the length of x and the indices being accessed.")
    #         return None
    #     except ValueError as e:
    #         print("A ValueError occurred:", e)
    #     print("Function evaluations:", self.n_fev)
    #     self.n_fev=0
    #
    #     return res
    def ls_opti_cs(self, x0=None, max_iter=25, _n=3, atol=0,
                target_value=None, _workers=1, _updating='immediate',
                _popsize=500,threshold=1e-3,_run_name="",_init_pop=None):

        import numpy as np
        from scipy.optimize import differential_evolution

        if x0 is None or not x0:
            x0 = self.get_x0()

        print("Initial x0:", x0)

        # --------------------------------------------------
        # Bounds construction (single source of truth: get_bounds())
        # --------------------------------------------------
        bounds = self.get_bounds()
        constraints = self.constraints

        # differential_evolution's `init` kwarg does not accept None: it must
        # be a recognized string ('latinhypercube', 'random', 'sobol',
        # 'halton') or an explicit (popsize, ndim) array. Fall back to
        # 'latinhypercube' whenever no custom initial population was supplied.
        _init_pop_arg = _init_pop if _init_pop is not None else 'latinhypercube'

        # --------------------------------------------------
        # Callback
        # --------------------------------------------------
        def callback(xk, convergence):
            if target_value is not None:
                val = self.msle(xk)
                if val <= target_value:
                    print("Target reached.")
                    return True
            return False

        # ==================================================
        # STAGE 1 — EXPLORATION
        # ==================================================
        print("Constraints:", constraints)
        print("\n=== Stage 1: Exploration ===")
        # if  not _init_pop:
            # res1 = differential_evolution(
                # self.male,
                # bounds=bounds,
                # strategy='rand1bin',
                # maxiter=max_iter,
                # popsize=_popsize,
                # mutation=(0.7, 1.0),
                # recombination=0.9,
                # tol=0.05,
                # atol=atol,
                # polish=False,
                # init='latinhypercube',
                # updating=_updating,
                # workers=_workers,
                # disp=True,
                # callback=callback,
                # constraints=() if constraints is None else constraints
            # )
            # else:
        res1 = differential_evolution(
            self.male,
            bounds=bounds,
            strategy='rand1bin',
            maxiter=max_iter,
            popsize=_popsize,
            mutation=(0.7, 1.0),
            recombination=0.9,
            tol=0.05,
            atol=atol,
            polish=False,
            init=_init_pop_arg,
            updating=_updating,
            workers=_workers,
            disp=True,
            callback=callback,
            constraints=() if constraints is None else constraints
            )

        x_best = res1.x
        print("Stage 1 best male:", res1.fun)
        # # After Stage 1
        # if res1.success and res1.fun < threshold:
        #     # Good exploration, proceed to exploitation
        #     max_iter_stage2 = max_iter
        # else:
        #     # Need more exploration
        #     max_iter_stage2 = max_iter * 2
        # ==================================================
        # STAGE 2 — EXPLOITATION
        # ==================================================
        print("\n=== Stage 2: Exploitation ===")

        # Get top N% solutions from Stage 1 population (at least 1 elite, even
        # for tiny _popsize -- int(0.1*_popsize) is 0 for _popsize < 10, which
        # would leave the population empty).
        n_elite = max(1, int(0.1 * _popsize))  # top 10%, but never 0
        elite_solutions = get_top_solutions(res1, n_elite)
        n_elite = elite_solutions.shape[0]     # in case fewer were available
        ndim = elite_solutions.shape[1]
        if elite_solutions.shape[0] > 1:
            spread = np.std(elite_solutions, axis=0)
            scale = 0.1 * np.mean(spread)
        else:
            scale = 0.1
        # Create init_pop around elite solutions
        n_samples = max(1, _popsize // n_elite)
        init_pop = []
        for elite in elite_solutions:
            init_pop.extend(elite + scale * np.random.randn(n_samples, ndim))
        init_pop = np.array(init_pop)
        # scipy's differential_evolution requires an initial population with
        # more than 4 members; top up around the best solution if a small
        # _popsize left us short.
        if init_pop.shape[0] < 5:
            extra = 5 - init_pop.shape[0]
            init_pop = np.vstack([init_pop,
                                  x_best + scale * np.random.randn(extra, ndim)])
        # clip back into the bounds so seeded members stay feasible
        _lb = np.array([b[0] for b in bounds])
        _ub = np.array([b[1] for b in bounds])
        init_pop = np.clip(init_pop, _lb, _ub)

        res2 = differential_evolution(
            self.msle,
            bounds=bounds,
            strategy='best1bin',
            maxiter=max_iter,
            popsize=_popsize,
            mutation=(0.4, 0.7),
            recombination=0.7,
            tol=1e-5,
            atol=atol,
            polish=False,
            init=init_pop,
            updating=_updating,
            workers=_workers,
            disp=True,
            callback=callback,
            constraints=() if constraints is None else constraints
        )

        print("Stage 2 best MSLE:", res2.fun)
        print("Function evaluations:", self.n_fev)

        # After Stage 2
        print("\n=== Stage 3: Local Polish ===")
        res3 = minimize(
            self.msle,
            x0=res2.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-9}
        )

        self.n_fev = 0
        # Stack as columns
        arr = np.column_stack((self.history['fevals'], self.history['best_male'], self.history['best_msle']))

        # Save to CSV
        np.savetxt(_run_name+"optimization_history.csv", arr, delimiter=',', header='fevals,best_male,best_msle', comments='',
                   fmt='%.6e')

        #Save res.x to text file

        np.savetxt(_run_name+"best_solution.txt", res3.x, fmt="%.6e")  # 6 decimals in scientific notation

        return res3

    def ls_opti(self, x0=None, max_iter=25, _n=3, atol=0,
                target_value=None, _workers=1, _updating='immediate',
                _popsize=500,threshold=1e-3,_run_name=""):

        import numpy as np
        from scipy.optimize import differential_evolution

        if x0 is None or not x0:
            x0 = self.get_x0()

        print("Initial x0:", x0)

        # --------------------------------------------------
        #         # Bounds construction (UNCHANGED)
        # --------------------------------------------------
        lb, ub = [], []

        if self.rho_world_varflag:
            lb.append(np.log10(self.rho_world_varlim[0]))
            ub.append(np.log10(self.rho_world_varlim[1]))

        for layer in self.layers:
            if sum(map(int, layer.varflag)) == 1:
                for i, v in enumerate(layer.varflag):
                    if v:
                        if i == 1 and layer.log_rho:
                            lb.append(np.log10(layer.varlims[0]))
                            ub.append(np.log10(layer.varlims[1]))
                        else:
                            lb.append(layer.varlims[0])
                            ub.append(layer.varlims[1])
            else:
                k = 0
                for i, v in enumerate(layer.varflag):
                    if v:
                        if i == 1 and layer.log_rho:
                            lb.append(np.log10(layer.varlims[i][0]))
                            ub.append(np.log10(layer.varlims[i][1]))
                        else:
                            lb.append(layer.varlims[k][0])
                            ub.append(layer.varlims[k][1])
                        k += 1
        # print("Number of circles:", len(self.Circles))
        for circle in self.Circles:
            if sum(map(int, circle.varflag)) == 1:
                # print(circle.varflag)
                for i, v in enumerate(circle.varflag):
                    if v:
                        if i == 2 and circle.log_r:
                            lb.append(np.log10(circle.varlims[0]))
                            ub.append(np.log10(circle.varlims[1]))
                        elif i == 3 and circle.log_rho:
                            lb.append(np.log10(circle.varlims[0]))
                            ub.append(np.log10(circle.varlims[1]))
                        else:
                            lb.append(circle.varlims[0])
                            ub.append(circle.varlims[1])
            else:
                # print(circle.varflag)
                for i, v in enumerate(circle.varflag):
                    if v:
                        if i == 2 and circle.log_r:
                            lb.append(np.log10(circle.varlims[i][0]))
                            ub.append(np.log10(circle.varlims[i][1]))
                        elif i == 3 and circle.log_rho:
                            lb.append(np.log10(circle.varlims[i][0]))
                            ub.append(np.log10(circle.varlims[i][1]))
                        else:
                            # print(i,v)
                            print(circle.varlims[i])
                            lb.append(circle.varlims[i][0])
                            ub.append(circle.varlims[i][1])

        bounds = list(zip(lb, ub))
        # print("lower and upper:",len(lb),len(ub))
        constraints = self.constraints

        # --------------------------------------------------
        # Callback
        # --------------------------------------------------
        def callback(xk, convergence):
            if target_value is not None:
                val = self.msle(xk)
                if val <= target_value:
                    print("Target reached.")
                    return True
            return False

        # ==================================================
        # STAGE 1 — EXPLORATION
        # ==================================================
        print("Constraints:", constraints)
        print("\n=== Stage 1: Exploration ===")

        res1 = differential_evolution(
            self.male,
            bounds=bounds,
            strategy='rand1bin',
            maxiter=max_iter,
            popsize=_popsize,
            mutation=(0.7, 1.0),
            recombination=0.9,
            tol=0.05,
            atol=atol,
            polish=False,
            init='latinhypercube',
            updating=_updating,
            workers=_workers,
            disp=True,
            callback=callback,
            constraints=() if constraints is None else constraints
        )

        x_best = res1.x
        print("Stage 1 best male:", res1.fun)
        # # After Stage 1
        # if res1.success and res1.fun < threshold:
        #     # Good exploration, proceed to exploitation
        #     max_iter_stage2 = max_iter
        # else:
        #     # Need more exploration
        #     max_iter_stage2 = max_iter * 2
        # ==================================================
        # STAGE 2 — EXPLOITATION
        # ==================================================
        print("\n=== Stage 2: Exploitation ===")

        # Get top N% solutions from Stage 1 population (at least 1 elite, even
        # for tiny _popsize -- int(0.1*_popsize) is 0 for _popsize < 10, which
        # would leave the population empty).
        n_elite = max(1, int(0.1 * _popsize))  # top 10%, but never 0
        elite_solutions = get_top_solutions(res1, n_elite)
        n_elite = elite_solutions.shape[0]     # in case fewer were available
        ndim = elite_solutions.shape[1]
        if elite_solutions.shape[0] > 1:
            spread = np.std(elite_solutions, axis=0)
            scale = 0.1 * np.mean(spread)
        else:
            scale = 0.1
        # Create init_pop around elite solutions
        n_samples = max(1, _popsize // n_elite)
        init_pop = []
        for elite in elite_solutions:
            init_pop.extend(elite + scale * np.random.randn(n_samples, ndim))
        init_pop = np.array(init_pop)
        # scipy's differential_evolution requires an initial population with
        # more than 4 members; top up around the best solution if a small
        # _popsize left us short.
        if init_pop.shape[0] < 5:
            extra = 5 - init_pop.shape[0]
            init_pop = np.vstack([init_pop,
                                  x_best + scale * np.random.randn(extra, ndim)])
        # clip back into the bounds so seeded members stay feasible
        _lb = np.array([b[0] for b in bounds])
        _ub = np.array([b[1] for b in bounds])
        init_pop = np.clip(init_pop, _lb, _ub)

        res2 = differential_evolution(
            self.msle,
            bounds=bounds,
            strategy='best1bin',
            maxiter=max_iter,
            popsize=_popsize,
            mutation=(0.4, 0.7),
            recombination=0.7,
            tol=1e-5,
            atol=atol,
            polish=False,
            init=init_pop,
            updating=_updating,
            workers=_workers,
            disp=True,
            callback=callback,
            constraints=() if constraints is None else constraints
        )

        print("Stage 2 best MSLE:", res2.fun)
        print("Function evaluations:", self.n_fev)

        # After Stage 2
        print("\n=== Stage 3: Local Polish ===")
        res3 = minimize(
            self.msle,
            x0=res2.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-9}
        )

        self.n_fev = 0
        # Stack as columns
        arr = np.column_stack((self.history['fevals'], self.history['best_male'], self.history['best_msle']))

        # Save to CSV
        np.savetxt(_run_name+"optimization_history.csv", arr, delimiter=',', header='fevals,best_male,best_msle', comments='',
                   fmt='%.6e')

        #Save res.x to text file

        np.savetxt(_run_name+"best_solution.txt", res3.x, fmt="%.6e")  # 6 decimals in scientific notation

        return res3

    def sensitivity_analysis(self, eps=0.001):
        x0 = self.get_x0()
        original_x0 = copy.deepcopy(self.get_x0())
        y0 = self.get_forward_solution()
        h = np.zeros((len(original_x0), len(y0['rhoa'])))
        for i, x in enumerate(x0):
            X = x0
            xp = x + eps
            xm = x - eps

            X[i] = xp
            self.update(X)
            yp = self.get_forward_solution()

            X[i] = xm
            self.update(X)
            ym = self.get_forward_solution()
            h[i, :] = (yp['rhoa'] - ym['rhoa']) / (xp - xm)
            self.update(original_x0)
            # print("x0:",original_x0)
            # print("get x0 output:",self.get_x0())
        return h

    def sensitivity_analysis_v(self, eps=0.001):
        X0 = self.get_x0()
        Y0 = self.get_forward_solution_v()
        h = np.zeros((len(X0), len(Y0)))
        for i, x in enumerate(X0):
            X = X0
            xp = x + eps
            xm = x - eps

            X[i] = xp
            self.update(X)
            yp = self.get_forward_solution_v()

            X[i] = xm
            self.update(X)
            ym = self.get_forward_solution_v()
            h[i, :] = (yp - ym) / (xp - xm)
            self.update(X0)
        return h

    def fisher_information_matrix(self, eps=0.01):
        h = self.sensitivity_analysis(eps)
        F = np.matmul(h, h.T)
        return F

    def meas_fisher_information_matrix(self, eps=0.01):
        h = self.sensitivity_analysis(eps)
        F = np.matmul(h.T, h)
        return F

    def fisher_information_matrix_v(self, eps=0.01):
        h = self.sensitivity_analysis_v(eps)
        F = np.matmul(h, h.T)
        return F

    def meas_fisher_information_matrix_v(self, eps=0.01):
        h = self.sensitivity_analysis_v(eps)
        F = np.matmul(h.T, h)
        return F

    def get_layers_polygons(self):
        h = self.start[1]
        layers_polygons = []
        for layer in self.layers:
            layer_polygon = Polygon(
                [(self.start[0], h), (self.end[0], h), (self.end[0], h - layer.y), (self.start[0], h - layer.y),
                 (self.start[0], h)])
            layers_polygons.append(layer_polygon)
            h = h - layer.y
        return layers_polygons

    def circles_polygons(self):
        circles_polygons = []
        for circle in self.Circles:
            # Define the center of the circle and its radius
            center = Point(circle.x, circle.y)
            radius = circle.r  # Circle radius
            # Create the circle as a polygon
            circle_polygon = center.buffer(radius, resolution=50)
            circles_polygons.append(circle_polygon)
        return circles_polygons

        # 'only for single anomaly test'
    # def spatial_sensitivity_map(self,eps=0.01,n=10):
    #     V0=self.get_x0
    #     nv=len(V0)
    #     x=np.linspace(self.start[0]*(1+0.01),self.end[0]*(1-0.01),n)
    #     y=np.linspace(self.end[1]*(1+0.01),self.start[1]*(1+0.01),n)
    #     xx, yy = np.meshgrid(x, y)
    #     XY = np.c_[xx.ravel(), yy.ravel()]
    #     for xy in XY:
    #         V=V0;
    #         V[0]=xy[0]
    #         V[1]=xy[1]
    #         self.update(V)
    #
    #     h=self.sensitivity_analysis(eps)
    #     F=np.matmul(h.T , h)
    #     return F

    # def Ls_opti(self, x0=None, RelTol=1.0e-3, MaxIt=1500, _n=6):
    #     lb = [0.25, 10] * len(self.layers)  # assuming 2 parameters (y, rho) per layer
    #     ub = [10, 1000] * len(self.layers)
    #     # print(lb)
    #     # print(ub)
    #     bounds = sp.optimize.Bounds(lb, ub, keep_feasible=False)
    #     if not x0:
    #         x0 = self.get_x0()
    #     # res=shgo(self.RMS,bounds, n=_n,iters=2, constraints=self.cons(), sampling_method='simplicial') 
    #     res = sp.optimize.differential_evolution(self.rms, bounds, strategy='best1bin', max_iter=20, popsize=20,
    #                                              tol=0.01, mutation=(0.5, 1), recombination=0.7, disp=True, polish=True,
    #                                              init='latinhypercube', atol=0, updating='immediate',
    #                                              constraints=self.cons())
    #     return res

    # def jac(x,eps):
    # for i in range(len(x)):

    # def LsOpti(self,x0=[],RelTol=1.0e-3,MaxIt=1000):
    #     if x0==[]:
    #         x0=[self.circleA.x,self.circleA.y,self.circleA.r,self.circleA.rho];
    #     res = minimize(self.RMS, x0 ,method='Nelder-Mead' ,tol=1.0e-8,options={'disp': True})
    #     return res;
