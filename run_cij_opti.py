"""
run_all_scenarios_parallel.py
==============================
Runs ls_opti_cs N times per scenario with an optional initial guess (x0) 
and hybrid population generation, each run in its own subprocess.

Usage:
    python run_all_scenarios_parallel.py
"""

import os
import sys
import time
import math
import traceback
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, Manager
from functools import partial

# ── Scenario definitions ─────────────────────────────────────────────────────
#
# All scenarios below are built programmatically from parameters.xlsx (the
# C_ij CV-estimate / world-rho-range table), rather than being typed out by
# hand, so there's a single source of truth and no risk of transcription
# errors across the 9 configs. If you move the spreadsheet, update
# PARAMETERS_XLSX below.

import openpyxl

# Read stage 3's output directly rather than a copy in the repository root:
# a stale root-level copy silently supplied bounds from an earlier
# segmentation threshold.
PARAMETERS_XLSX = 'runs/03_cv/parameters.xlsx'

# Settings shared by every C_ij scenario
from Anandlyn_log import STAGE1_POP   # absolute exploration population

GLOBAL_MAX_ITER = 26     # matches the main-text budget
# Population size is PROPORTIONAL to the number of free model parameters:
# both the DE population (scipy uses popsize * ndim) and the hybrid seed
# Populations are absolute: STAGE1_POP individuals for exploration and
# GLOBAL_POPSIZE for exploitation, identical across all C_i,j cells so that
# scene comparison is not confounded with search effort.
GLOBAL_POPSIZE  = 120    # stage-2 population, absolute
GLOBAL_N        = 3
GLOBAL_ATOL     = 1e-5
TARGET_VALUE    = None     # early stopping disabled: with the corrected forward
                           # operator a good run reaches ~2.5e-6, so a 1e-6 threshold
                           # fires in some cells and not others, making the budget
                           # non-uniform and the cross-scene comparison invalid.
# Initial-population spread around the CV guess x0, for the positional
# parameters (x, y, r) only -- NOT the hard search bounds. The hard bounds stay
# wide (world & other rho: full original span; radius: capped at 1.2x retrieved)
# so the true solution is never excluded; POP_EPSILON just concentrates the
# *starting* guesses near the CV estimate. Smaller -> tighter cluster around x0.
POP_EPSILON     = 0.15
GUESS_RATIO     = 0.25     # fraction of the population seeded around x0

LAYER_VARLIMS  = ((-10, -1), (1, 1e5))                      # (y bounds, rho bounds)
CIRCLE_VARLIMS = ((-25, 25), (-10, 0), (0.5, 5), (1, 1e5))  # x, y, r, rho bounds

# Circle-radius search window. The smooth ERT inversion tends to OVER-estimate
# anomaly radius (smearing inflates the blob), so the true radius is at or
# below the retrieved one. We therefore cap the upper radius bound at
# RADIUS_UPPER_FACTOR x the retrieved (CV/x0) radius, and keep the lower bound
# at RADIUS_LOWER (the physical minimum). rho bounds are deliberately NOT
# narrowed this way -- the smooth inversion already constrains resistivity, so
# the rho hard-bounds stay at their full original span and x0 carries the
# smooth-inversion estimate.
RADIUS_UPPER_FACTOR = 1.2
RADIUS_LOWER        = 0.5    # linear metres (log10 = -0.301)

# Positional parameters (layer depth y, circle x, circle y) are searched in a
# window CENTERED ON the CV estimate ("relative to the CV input"), not across
# the whole domain. Half-width = max(POS_REL_MARGIN * |CV|, POS_ABS_MARGIN),
# then clipped to the physical domain.
#
# IMPORTANT: the CV estimate is only approximate -- for c22 it is off from the
# true position by up to ~1.75 m (layer2_y), and ~1.3-1.5 m for C1_x / C2_y. The
# window MUST stay wide enough to contain the truth, so these margins are kept
# generous on purpose. Tightening them risks putting the true position outside
# the bounds, where the optimizer can never reach it. Radius is exempt: its
# window is asymmetric [0.5, CV*1.2], and the truth sits below the (inflated)
# CV radius, so it stays contained.
POS_REL_MARGIN = 0.5        # half-width as fraction of |CV estimate|
POS_ABS_MARGIN = 2.0        # ...but at least this many metres (floor)

# A circle/interface side constraint is only imposed when the CV circle depth
# is at least this far from the CV interface depth; within it, the side is
# unreliable and the constraint is skipped (see the circle-y loop below).
CIRCLE_LAYER_MARGIN = POS_ABS_MARGIN


def _pos_window(cv, clip):
    """Search window centred on the CV estimate `cv`, clipped to the physical
    domain `clip`=(lo, hi)."""
    d = max(POS_REL_MARGIN * abs(cv), POS_ABS_MARGIN)
    return (max(cv - d, clip[0]), min(cv + d, clip[1]))

# Top layer is a KNOWN quantity in every case except c00: its resistivity is
# fixed at 50 ohm-m (ground truth), not optimized. Only its depth (y) is free.
# varflag=[True, False] -> y variable, rho fixed; when a layer has a single
# free parameter the API expects a *flat* (lb, ub) varlims for that parameter,
# hence TOP_LAYER_VARLIMS is just the y bounds (see AnomalyWorld.get_bounds).
TOP_LAYER_RHO_FIXED = 50.0
TOP_LAYER_VARFLAG   = [True, False]
TOP_LAYER_VARLIMS   = (-10, -1)     # y bounds only (flat, single free param)

# World-rho search span. Kept at the full original span for EVERY config (rho
# is not narrowed -- the smooth inversion narrows the estimate, delivered via
# x0, not the bounds). The parameters.xlsx min/max block is therefore no longer
# used to bound world-rho.
DEFAULT_WORLD_VARLIM = [1, 1e5]

# Resistivity half-range cut, keyed to the KNOWN top-layer resistivity.
# We don't know an element's exact rho, but the smooth inversion (via the CV
# estimate) tells us on which side of the 50 ohm-m top layer it sits, so we can
# halve its search span accordingly:
#     CV rho <  top  -> element is LESS resistive -> [1,   top ]
#     CV rho >= top  -> element is MORE resistive -> [top, 1e5 ]
# Applied to the bottom layer and the circles, but only for configs that HAVE a
# top layer (c01/c02/c11/c12/c21/c22). Configs with circles but no layer
# (c10, c20) have no known pivot, so their circle rho keeps the full span.
# NOTE: this trusts the CV *sign* of contrast. An element whose true rho is very
# close to 50 could be mis-sided by the CV estimate and land in the wrong half;
# NOTE: this assumption failed. CV over-predicts radius by ~2x, the compensating
# resistivity crosses the 50 ohm-m pivot, and for C2 in c21/c22 (true 25, CV 51
# and 96) the truth fell outside the feasible region. Replaced by a symmetric
# window about the CV estimate, which is both truth-containing and narrower.
RHO_REL_DECADES = 1.0   # rho bound: CV estimate x 10^(+/-1)
RHO_MIN_LINEAR = 1.0
RHO_MAX_LINEAR = 1e5


def rho_range_vs_top(cv_rho, top_rho):
    """Half-range for a resistivity element, pivoted on the known top-layer
    rho and the CV estimate's sign of contrast. Linear (lo, hi);
    AnomalyWorld.get_bounds() log10s it. top_rho=None -> full span (no pivot)."""
    if cv_rho is None or not np.isfinite(cv_rho) or cv_rho <= 0:
        return (RHO_MIN_LINEAR, RHO_MAX_LINEAR)
    lo = max(RHO_MIN_LINEAR, cv_rho * 10.0 ** (-RHO_REL_DECADES))
    hi = min(RHO_MAX_LINEAR, cv_rho * 10.0 ** (+RHO_REL_DECADES))
    return (lo, hi)


def circle_varlims(x_retrieved, y_retrieved, r_retrieved, rho_band=None):
    """Per-circle varlims. x and y are windowed around the CV estimate
    (_pos_window); radius is capped at RADIUS_UPPER_FACTOR x the retrieved
    radius with lower bound RADIUS_LOWER; rho uses rho_band (a half-range keyed
    to the top layer, or the full span if rho_band is None).
    r is passed linear; AnomalyWorld.get_bounds() log10s r and rho."""
    r_lo = min(RADIUS_LOWER, r_retrieved)          # guard: keep x0 within bounds
    r_hi = max(r_retrieved * RADIUS_UPPER_FACTOR, r_lo * 1.01)
    if rho_band is None:
        rho_band = CIRCLE_VARLIMS[3]
    return (_pos_window(x_retrieved, CIRCLE_VARLIMS[0]),   # x  window (clip -25..25)
            _pos_window(y_retrieved, CIRCLE_VARLIMS[1]),   # y  window (clip -10..0)
            (r_lo, r_hi),                                  # r  capped
            rho_band)                                      # rho half-range or full


def _clean(v):
    """Turn the xlsx's '-' placeholders into None; pass numbers through."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip() in ('-', ''):
        return None
    return float(v)


def load_cij_table(path):
    """
    Parse parameters.xlsx into:
      world_range : {'c00': (min, max), ...}          rho_world_varlim, linear space
      cv_est      : {'c00': {'rho_bg':..., 'layer1_y':..., 'layer1_rho':...,
                               'layer2_y':..., 'layer2_rho':...,
                               'c1_x':...,'c1_y':...,'c1_r':...,'c1_rho':...,
                               'c2_x':...,'c2_y':...,'c2_r':...,'c2_rho':...}, ...}
    Missing/'-' entries become None (that component is absent for the config).
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    def is_cij_key(v):
        s = str(v).strip().lower()
        return len(s) == 3 and s[0] == 'c' and s[1:].isdigit()

    # Locate the CV-estimate block's header first, so the min/max scan
    # below doesn't run into it: those rows also start with a 3-char
    # 'C00'-style code, so without this boundary the scan would eventually
    # overwrite real (min, max) pairs with (rho_bg, layer1_y) for any config
    # that has a layer1 (whose estimate row[2] is a real number too).
    header_idx = None
    for i, row in enumerate(rows):
        if row and len(row) > 1 and row[1] == 'rho bg':
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            "Could not find the 'rho bg' header row in "
            f"{path}. The sheet layout may have changed.")

    # --- world-rho min/max block (top of the sheet, before the header) ---
    world_range = {}
    for row in rows[:header_idx]:
        if row and row[0] is not None and is_cij_key(row[0]):
            mn, mx = _clean(row[1]), _clean(row[2])
            if mn is not None and mx is not None:
                world_range[str(row[0]).strip().lower()] = (mn, mx)

    # --- CV-estimate block (14 columns, one row per config) ---
    # NOTE: these are the segmentation stage's ESTIMATES, not ground truth;
    # they seed the optimizer and define its bounds.
    cv_est = {}
    for row in rows[header_idx + 1:]:
        if not row or row[0] is None or not is_cij_key(row[0]):
            continue
        key = str(row[0]).strip().lower()
        vals = [_clean(v) for v in row[1:14]]
        (rho_bg, l1_y, l1_rho, l2_y, l2_rho,
         c1_x, c1_y, c1_r, c1_rho, c2_x, c2_y, c2_r, c2_rho) = vals
        cv_est[key] = dict(
            rho_bg=rho_bg, layer1_y=l1_y, layer1_rho=l1_rho,
            layer2_y=l2_y, layer2_rho=l2_rho,
            c1_x=c1_x, c1_y=c1_y, c1_r=c1_r, c1_rho=c1_rho,
            c2_x=c2_x, c2_y=c2_y, c2_r=c2_r, c2_rho=c2_rho,
        )

    return world_range, cv_est


def build_scenarios(path=PARAMETERS_XLSX):
    world_range, cv_est = load_cij_table(path)

    scenarios = {}
    for key, t in sorted(cv_est.items()):    # 'c00', 'c01', ..., 'c22'
        n_circ, n_lay = int(key[1]), int(key[2])
        rho_bg = t['rho_bg']

        # --- layers ---
        # Top layer (layer1): resistivity is KNOWN (50 ohm-m ground truth) and
        # held fixed; only its depth is optimized. Its table 'layer1_rho' entry
        # is a CV estimate and is deliberately ignored in favour of the true 50.
        # Bottom layer (layer2), when present, stays fully variable.
        layers = []
        if n_lay >= 1:
            layers.append(dict(y=t['layer1_y'], rho=TOP_LAYER_RHO_FIXED, cnum=1,
                                name='layer1', varflag=list(TOP_LAYER_VARFLAG),
                                # top layer: only y is free -> flat (lo, hi)
                                # windowed around the CV depth estimate.
                                varlims=_pos_window(t['layer1_y'], LAYER_VARLIMS[0])))
        # top-layer resistivity is the pivot for the rho half-range cut; None
        # for configs without a top layer (c10, c20 -> circle rho stays full).
        top_rho = TOP_LAYER_RHO_FIXED if n_lay >= 1 else None

        if n_lay >= 2:
            layers.append(dict(y=t['layer2_y'], rho=t['layer2_rho'], cnum=2,
                                name='layer2', varflag=[True, True],
                                # bottom layer: y windowed around CV depth; rho
                                # cut to the half-range on the CV-indicated side
                                # of the 50 ohm-m top layer.
                                varlims=(_pos_window(t['layer2_y'], LAYER_VARLIMS[0]),
                                         rho_range_vs_top(t['layer2_rho'], top_rho))))

        # --- circles ---
        circles = []
        if n_circ >= 1:
            circles.append(dict(name='C1', x=t['c1_x'], y=t['c1_y'], r=t['c1_r'],
                                 rho=t['c1_rho'], cnum=1, varflag=[True] * 4,
                                 varlims=circle_varlims(t['c1_x'], t['c1_y'], t['c1_r'],
                                                        rho_range_vs_top(t['c1_rho'], top_rho))))
        if n_circ >= 2:
            circles.append(dict(name='C2', x=t['c2_x'], y=t['c2_y'], r=t['c2_r'],
                                 rho=t['c2_rho'], cnum=2, varflag=[True] * 4,
                                 varlims=circle_varlims(t['c2_x'], t['c2_y'], t['c2_r'],
                                                        rho_range_vs_top(t['c2_rho'], top_rho))))

        # --- relative constraints, derived from the consistent orderings
        #     observed across every row of parameters.xlsx ---
        # NOTE on units: the linear-constraint machinery mixes a variable's
        # stored value with the variable itself. rho variables are optimized in
        # LOG10 space, so a rho-vs-rho ordering is only consistent when BOTH
        # sides are free rho variables (both log10). Since the top layer's rho
        # is now a FIXED parameter (stored linearly as 50), a 'layer1.rho <=
        # world.rho' constraint would compare linear-50 against log10(world) --
        # a unit mismatch. We therefore omit it. The physical prior it encoded
        # (background more resistive than the 50 top layer) can instead be
        # imposed, if desired, by raising the world-rho lower bound to 50.
        extra_constraints = []
        if n_lay >= 2:
            # layer2 (free, log10) more resistive than the background (free,
            # log10): both sides are free rho variables, so this ordering is
            # unit-consistent. Pin it so DE can't flip the layer2/background
            # contrast.
            extra_constraints.append(dict(
                entity1='layers', idx1=1, var1='rho', op='>=',
                entity2='background', idx2=None, var2='rho'))
        if n_circ >= 2:
            # Break label-switching symmetry between the two circles: ground
            # truth always has circle1 to the right of circle2 (x1 >= x2)
            # and circle1 more resistive than circle2 (rho1 >= rho2).
            extra_constraints.append(dict(
                entity1='Circles', idx1=0, var1='x', op='>=',
                entity2='Circles', idx2=1, var2='x'))
            extra_constraints.append(dict(
                entity1='Circles', idx1=0, var1='rho', op='>=',
                entity2='Circles', idx2=1, var2='rho'))

        # Circle-centre depth bounded by the layers -- but ONLY when the CV
        # stage places the circle a safe margin clear of the interface. When a
        # circle's CV depth is within CIRCLE_LAYER_MARGIN of the CV interface,
        # the *side* is unreliable (CV interface depth can be off by ~2 m), and
        # pinning the wrong side makes the true position infeasible -- this is
        # exactly what crippled C11/C21, where CV put the interface too deep so
        # C1 looked 'above' it when the truth is below. In those ambiguous
        # cases we leave the circle free in depth (still positionally windowed).
        # y is negative-down: circle clearly 'above' -> C.y >= layer.y ; clearly
        # 'below' -> C.y <= layer.y.
        for ci in range(n_circ):
            c_cv_y = t[f'c{ci + 1}_y']
            for lj in range(n_lay):
                l_cv_y = t[f'layer{lj + 1}_y']
                if c_cv_y is None or l_cv_y is None:
                    continue
                if abs(c_cv_y - l_cv_y) < CIRCLE_LAYER_MARGIN:
                    continue                 # side ambiguous -> leave free
                op = '>=' if c_cv_y > l_cv_y else '<='
                extra_constraints.append(dict(
                    entity1='Circles', idx1=ci, var1='y', op=op,
                    entity2='layers', idx2=lj, var2='y'))

        # --- world rho bounds ---
        # world rho is cut relative to the known top layer, exactly like the
        # bottom layer and circles: the CV background estimate (rho_bg) tells us
        # which side of the 50 ohm-m top layer the background sits on. Configs
        # without a top layer (c00/c10/c20 -> top_rho is None) keep the full
        # span, since there is no known pivot.
        rho_world_varlim = list(rho_range_vs_top(rho_bg, top_rho))

        # --- x0, in the same order/space as AnomalyWorld.get_x0(): world
        # (log10), then each layer's [y, log10(rho)], then each circle's
        # [x, y, log10(r), log10(rho)] (r and rho are log10-space internally
        # by default -- see CircleAnom/Layer log_r/log_rho) ---
        x0 = [np.log10(rho_bg)]
        for layer in layers:
            # match get_x0(): append y iff its varflag is set, rho (log10) iff
            # its varflag is set. The fixed-rho top layer contributes only y.
            if layer['varflag'][0]:
                x0.append(layer['y'])
            if layer['varflag'][1]:
                x0.append(np.log10(layer['rho']))
        for circle in circles:
            x0 += [circle['x'], circle['y'],
                   np.log10(circle['r']), np.log10(circle['rho'])]

        scenarios[f'{key}_clean'] = dict(
            # stage 1 writes runs/01_forward/C{i}{j}_clean.dat; the old
            # measurements_data/a{i}{j}_c.dat layout is gone.
            dat_file=f'runs/01_forward/{key.upper()}_clean.dat',
            rho_world=rho_bg,
            rho_world_varlim=rho_world_varlim,
            rho_world_var=True,
            layers=layers,
            circles=circles,
            extra_constraints=extra_constraints,
            max_iter=GLOBAL_MAX_ITER, n=GLOBAL_N, atol=GLOBAL_ATOL, popsize=GLOBAL_POPSIZE,
            run_prefix=f'{key}_it{GLOBAL_MAX_ITER}',

            # Initial guess matching get_x0() order (see comment above).
            x0=x0,
            epsilon=POP_EPSILON,     # spread around x, y, r (relative)
            guess_ratio=GUESS_RATIO,
            target_value=TARGET_VALUE,   # early-stop threshold on MSLE
        )

    return scenarios


SCENARIOS = build_scenarios()

# ── Resource limits ───────────────────────────────────────────────────────────

RAM_PER_PROCESS_GB = 1   


def max_parallel_processes(ram_per_proc_gb: float) -> int:
    """Return the safe number of parallel worker processes."""
    try:
        import psutil
        available_ram_gb = psutil.virtual_memory().available / 1e9
        by_ram  = max(1, int(available_ram_gb / ram_per_proc_gb))
    except ImportError:
        by_ram  = 4   

    by_cpu = max(1, mp.cpu_count() - 2)   
    workers = min(by_cpu, by_ram)
    print(f"[resource check]  logical CPUs={mp.cpu_count()}  "
          f"available RAM≈{available_ram_gb:.1f} GB  "
          f"→ max parallel workers = {workers}")
    return workers


# ── World builder ─────────────────────────────────────────────────────────────

def build_aworld(cfg, meas_data):
    """Instantiate an AnomalyWorld from a scenario config dict."""
    import numpy as np
    import pygimli.meshtools as mt
    from pygimli.physics import ert
    from Anandlyn_log import AnomalyWorld, Layer, CircleAnom

    scheme = ert.createData(
        elecs=np.linspace(start=-25, stop=25, num=21),
        schemeName='dd'
    )

    layers = [
        Layer(
            y=ld['y'], rho=ld['rho'], _cnum=ld['cnum'], name=ld['name'],
            _varflag=ld['varflag'], _varlims=ld['varlims']
        )
        for ld in cfg['layers']
    ]

    circles = [
        CircleAnom(
            name=cd['name'], x=cd['x'], y=cd['y'], r=cd['r'],
            rho=cd['rho'], _c_num=cd['cnum'],
            _varflag=cd['varflag'], _varlims=cd['varlims']
        )
        for cd in cfg['circles']
    ]

    aworld = AnomalyWorld(
        _start=[-25, 0], _end=[25, -20], _scheme=scheme,
        _layers=layers, _rho_world=cfg['rho_world'],
        _circles=circles,
        _rho_world_varflag=cfg['rho_world_var'],
        _rho_world_varlim=cfg['rho_world_varlim'],
    )

    # Apply optional extra constraints dynamically
    extra_constraints = cfg.get('extra_constraints', [])
    for c in extra_constraints:
        ent1_group = getattr(aworld, c['entity1'])
        e1 = ent1_group[c['idx1']] if c['idx1'] is not None else ent1_group

        ent2_group = getattr(aworld, c['entity2'])
        e2 = ent2_group[c['idx2']] if c['idx2'] is not None else ent2_group

        aworld.constraint_group.add_constraint(
            e1, c['var1'],
            constraint_type=c['op'],
            entity2=e2, var2=c['var2']
        )

    if extra_constraints:
        aworld.refresh_constraints()

    aworld.meas = meas_data
    return aworld


def get_param_labels(aworld):
    """Human-readable labels matching get_x0() order."""
    labels = []
    if aworld.rho_world_varflag:
        labels.append('world_log10rho')
    for layer in aworld.layers:
        if layer.varflag[0]:
            labels.append(f'{layer.name}_y')
        if layer.varflag[1]:
            labels.append(f'{layer.name}_log10rho' if layer.log_rho else f'{layer.name}_rho')
    for circle in aworld.Circles:
        if circle.varflag[0]:
            labels.append(f'{circle.name}_x')
        if circle.varflag[1]:
            labels.append(f'{circle.name}_y')
        if circle.varflag[2]:
            labels.append(f'{circle.name}_log10r' if circle.log_r else f'{circle.name}_r')
        if circle.varflag[3]:
            labels.append(f'{circle.name}_log10rho' if circle.log_rho else f'{circle.name}_rho')
    return labels


def read_fevals_from_history(run_name):
    """Read total stage-1+2 fevals from the history CSV written by ls_opti_cs."""
    hist_path = run_name + "optimization_history.csv"
    if not os.path.exists(hist_path):
        return None
    try:
        data = np.genfromtxt(hist_path, delimiter=',', names=True)
        fevals = data['fevals']
        return int(fevals[-1]) if fevals.ndim > 0 else int(fevals)
    except Exception:
        return None


# ── Worker function (runs in a subprocess) ────────────────────────────────────

def single_run(args):
    """
    Execute one ls_opti_cs run for a given scenario and run index.
    Returns a dict with all collected metrics.
    """
    scenario_name, cfg, run_idx = args

    try:
        from pygimli.physics import ert
        from Anandlyn_log import generate_hybrid_population_mixed

        run_name = f"{cfg['run_prefix']}_run{run_idx + 1}_"
        print(f"[{scenario_name}] run {run_idx + 1} starting  (PID {os.getpid()})", flush=True)

        meas_manager = ert.ERTManager(cfg['dat_file'])
        # np.array(..., copy) — NOT a bare reference. data['rhoa'] is a view
        # bound to the DataContainer's lifetime; taking an owned copy here
        # decouples the measurements from meas_manager so they can't silently
        # empty out if the manager is GC'd or replaced.
        meas_data    = np.array(meas_manager.data['rhoa'], dtype=float)

        aworld = build_aworld(cfg, meas_data)
        labels = get_param_labels(aworld)

        # ── Handle Custom Hybrid Population Creation ──
        pop0 = None
        if 'x0' in cfg and cfg['x0'] is not None:
            # Gather optimization bounds from the instantiated world object
            bounds = aworld.get_bounds()
            eps = cfg.get('epsilon', 0.1)
            g_ratio = cfg.get('guess_ratio', 0.25)
            # popsize_multiplier matches the total population expected by your differential evolution 
            pmul = cfg['popsize'] 

            # x/y/r parameters get clustered near x0 (spread = eps); rho
            # parameters are sampled across their full bounds regardless of
            # x0, since resistivity contrast is what's under test. Label
            # names come from get_param_labels(): '..._log10rho'/'..._rho'
            # for resistivities, everything else is position/size.
            param_types = ['rho' if lbl.endswith('rho') else 'pos' for lbl in labels]

            print(f"[{scenario_name}] generating hybrid population around customized x0...", flush=True)
            pop0 = generate_hybrid_population_mixed(
                bounds=bounds,
                x0=cfg['x0'],
                param_types=param_types,
                epsilon=eps,
                total_pop=STAGE1_POP,      # absolute, not scaled by ndim
                popsize_multiplier=pmul,   # retained; no longer sizes the population
                guess_ratio=g_ratio,
                relative=cfg.get('relative', True)
            )

        t0_cpu  = time.process_time()
        t0_wall = time.perf_counter()

        # Pass custom pop0 population matrix if generated
        res = aworld.ls_opti_cs(
            max_iter     = cfg['max_iter'],
            _n           = cfg['n'],
            atol         = cfg['atol'],
            _popsize     = cfg['popsize'],
            _run_name    = run_name,
            _init_pop    = pop0, # ls_opti_cs's actual kwarg name (was '_pop0', a typo)
            target_value = cfg.get('target_value'),  # early-stop once MSLE <= this
        )

        t1_cpu  = time.process_time()
        t1_wall = time.perf_counter()

        fev_s12 = read_fevals_from_history(run_name)
        fev_s3  = res.nfev

        result = dict(
            scenario   = scenario_name,
            run_idx    = run_idx,
            run_name   = run_name,
            x_opti     = res.x.tolist(),
            fun        = float(res.fun),
            cpu_time   = t1_cpu  - t0_cpu,
            wall_time  = t1_wall - t0_wall,
            fev_s12    = fev_s12,
            fev_s3     = fev_s3,
            fev_total  = (fev_s12 + fev_s3) if fev_s12 is not None else None,
            labels     = labels,
            success    = True,
            error      = None,
        )
        print(f"[{scenario_name}] run {run_idx + 1} done  "
              f"obj={res.fun:.4e}  cpu={result['cpu_time']:.1f}s", flush=True)
        return result

    except Exception as e:
        print(f"[{scenario_name}] run {run_idx + 1} FAILED: {e}", flush=True)
        traceback.print_exc()
        return dict(
            scenario=scenario_name, run_idx=run_idx,
            success=False, error=str(e),
            x_opti=None, fun=None, cpu_time=None,
            wall_time=None, fev_s12=None, fev_s3=None,
            fev_total=None, labels=None, run_name=None,
        )


# ── Statistics helpers ────────────────────────────────────────────────────────

def scalar_stats(values, label, ddof=1):
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        print(f"  {label:<35}  no data")
        return
    std = np.std(arr, ddof=ddof) if arr.size > 1 else float('nan')
    print(f"  {label:<35}  mean={np.mean(arr):>12.4f}  "
          f"std={std:>12.4f}  min={np.min(arr):>12.4f}  max={np.max(arr):>12.4f}")


def summarise_scenario(scenario_name, results, out_dir):
    good = [r for r in results if r['success']]
    if not good:
        print(f"  [{scenario_name}] all runs failed — nothing to summarise.")
        return

    labels    = good[0]['labels']
    all_x     = np.array([r['x_opti']   for r in good])
    all_fun   = np.array([r['fun']       for r in good])
    all_cpu   = np.array([r['cpu_time']  for r in good])
    all_wall  = np.array([r['wall_time'] for r in good])
    all_fs3   = np.array([r['fev_s3']    for r in good])
    all_fs12  = np.array([r['fev_s12']   for r in good if r['fev_s12'] is not None])
    all_ftot  = np.array([r['fev_total'] for r in good if r['fev_total'] is not None])

    n = len(good)
    ddof = 1 if n > 1 else 0

    x_mean = np.mean(all_x, axis=0)
    x_std  = np.std(all_x,  axis=0, ddof=ddof)
    x_min  = np.min(all_x,  axis=0)
    x_max  = np.max(all_x,  axis=0)

    print(f"\n{'='*70}")
    print(f"  Scenario: {scenario_name}   ({n}/{len(results)} runs succeeded)")
    print(f"{'='*70}")
    print(f"\n  {'Parameter':<25} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("  " + "-"*67)
    for i, lbl in enumerate(labels):
        print(f"  {lbl:<25} {x_mean[i]:>12.4f} {x_std[i]:>12.4f} "
              f"{x_min[i]:>12.4f} {x_max[i]:>12.4f}")

    print(f"\n  --- Scalar run statistics ---")
    scalar_stats(all_fun,  "Objective (MSLE)",         ddof=ddof)
    scalar_stats(all_cpu,  "CPU time [s]",             ddof=ddof)
    scalar_stats(all_wall, "Wall-clock time [s]",      ddof=ddof)
    scalar_stats(all_fs3,  "Fevals Stage 3 (polish)",  ddof=ddof)
    if all_fs12.size > 0:
        scalar_stats(all_fs12, "Fevals Stages 1+2 (DE)",  ddof=ddof)
    if all_ftot.size > 0:
        scalar_stats(all_ftot, "Fevals total",             ddof=ddof)

    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, scenario_name)

    np.savetxt(f'{prefix}_all_x_opti.csv', all_x, delimiter=',', header=','.join(labels), comments='', fmt='%.6e')
    np.savetxt(f'{prefix}_x_opti_stats.csv', np.column_stack((x_mean, x_std, x_min, x_max)), delimiter=',', header='mean,std,min,max', comments='', fmt='%.6e')

    run_ids = np.arange(1, n + 1)
    if all_fs12.size == n and all_ftot.size == n:
        scalar_data   = np.column_stack((run_ids, all_fun, all_cpu, all_wall, all_fs12, all_fs3, all_ftot))
        scalar_header = 'run,objective,cpu_time_s,wall_time_s,fevals_stage12,fevals_stage3,fevals_total'
        scalar_fmt    = ['%d','%.6e','%.4f','%.4f','%d','%d','%d']
    else:
        scalar_data   = np.column_stack((run_ids, all_fun, all_cpu, all_wall, all_fs3))
        scalar_header = 'run,objective,cpu_time_s,wall_time_s,fevals_stage3'
        scalar_fmt    = ['%d','%.6e','%.4f','%.4f','%d']

    np.savetxt(f'{prefix}_run_metrics.csv', scalar_data, delimiter=',', header=scalar_header, comments='', fmt=scalar_fmt)


# ── Main ──────────────────────────────────────────────────────────────────────

def _fmt_bound(label, v):
    """Format one bound/x0 value, appending the linear equivalent for
    log10-space parameters (rho, r) so the search window is readable."""
    if 'log10rho' in label:
        return f"{v:9.3f} (rho={10**v:.4g})"
    if 'log10r' in label:            # radius (log10r, not log10rho)
        return f"{v:9.3f} (r={10**v:.3g})"
    return f"{v:9.3f}"


def _rule_for(label, lb, ub):
    """Short description of the RELATIVE DEFINITION behind a parameter's bounds,
    derived (to the CV stage / top layer) from the label and the bounds."""
    log_top = math.log10(TOP_LAYER_RHO_FIXED)
    if label.endswith('log10r'):                 # radius (not log10rho)
        return f"radius: [{RADIUS_LOWER:g}, CVx{RADIUS_UPPER_FACTOR:g}]"
    if label.endswith('log10rho'):               # world / layer2 / circle rho
        prefix = "world " if label == 'world_log10rho' else ""
        if abs(lb - log_top) < 1e-6:
            return f"{prefix}rho >= top({TOP_LAYER_RHO_FIXED:g}): [{TOP_LAYER_RHO_FIXED:g},1e5]"
        if abs(ub - log_top) < 1e-6:
            return f"{prefix}rho <  top({TOP_LAYER_RHO_FIXED:g}): [1,{TOP_LAYER_RHO_FIXED:g}]"
        return f"{prefix}rho: full span (no top pivot)"
    if label.endswith('_x') or label.endswith('_y'):
        return f"pos: CV +/- max({POS_REL_MARGIN:g}|CV|,{POS_ABS_MARGIN:g}m)"
    return ""


def display_scenario_bounds(scenarios):
    """Print, for every case, the optimization search bounds (with x0), the
    RELATIVE DEFINITION behind each bound, the fixed parameters, and the
    relative constraints in effect -- a sanity check to run BEFORE the sweep."""
    import numpy as np
    log_top = math.log10(TOP_LAYER_RHO_FIXED)
    print("\n" + "=" * 78)
    print("OPTIMIZATION BOUNDS PER CASE  (shown before run)")
    print(f"populations: exploration {STAGE1_POP}, exploitation {GLOBAL_POPSIZE} "
          f"POP_EPSILON={POP_EPSILON}  GUESS_RATIO={GUESS_RATIO}")
    print("rho / r parameters are optimized in log10 space; linear value in ()")
    print("relative definitions applied to derive the bounds:")
    print(f"  position (layer y, circle x/y): CV +/- max({POS_REL_MARGIN:g}*|CV|,"
          f" {POS_ABS_MARGIN:g} m), clipped to domain")
    print(f"  radius (circles): [{RADIUS_LOWER:g}, CV x {RADIUS_UPPER_FACTOR:g}]  "
          f"(smooth inversion inflates radius)")
    print(f"  rho (world, layer2 & circles, vs known top layer {TOP_LAYER_RHO_FIXED:g}): "
          f"CV<top -> [1,{TOP_LAYER_RHO_FIXED:g}] ; CV>=top -> [{TOP_LAYER_RHO_FIXED:g},1e5]")
    print("  rho: full span [1,1e5] when there is no top layer (c00/c10/c20)  |  "
          "top-layer rho fixed 50 (not optimized)")
    print("  circle-centre y bounded by the layer interfaces (see per-case constraints)")
    print("=" * 78)

    for name, cfg in sorted(scenarios.items()):
        try:
            aworld = build_aworld(cfg, np.zeros(1))
            labels = get_param_labels(aworld)
            bounds = aworld.get_bounds()
            x0     = aworld.get_x0()
        except Exception as e:
            print(f"\n--- {name} ---  [could not build for display: "
                  f"{type(e).__name__}: {e}]")
            continue

        ndim = len(labels)
        print(f"\n--- {name} ---  ndim={ndim}   "
              f"(populations {STAGE1_POP}/{GLOBAL_POPSIZE}, ndim={ndim})")
        print(f"    {'parameter':<15}{'x0':>21}{'lower':>21}{'upper':>21}   "
              f"relative definition")
        for lbl, xi, (lb, ub) in zip(labels, x0, bounds):
            print(f"    {lbl:<15}{_fmt_bound(lbl, xi):>21}"
                  f"{_fmt_bound(lbl, lb):>21}{_fmt_bound(lbl, ub):>21}   "
                  f"{_rule_for(lbl, lb, ub)}")

        # fixed (non-optimized) parameters
        fixed = []
        for ld in cfg['layers']:
            if not ld['varflag'][0]:
                fixed.append(f"{ld['name']}_y = {ld['y']}")
            if not ld['varflag'][1]:
                fixed.append(f"{ld['name']}_rho = {ld['rho']:g} ohm-m")
        for cd in cfg['circles']:
            for k, key in zip(range(4), ['x', 'y', 'r', 'rho']):
                if not cd['varflag'][k]:
                    fixed.append(f"{cd['name']}_{key} = {cd[key]:g}")
        if not aworld.rho_world_varflag:
            fixed.append(f"world_rho = {cfg['rho_world']:g} ohm-m")
        print(f"    fixed: {'; '.join(fixed) if fixed else '(none)'}")

        # relative constraints in effect
        alias = {'layers': 'layer', 'Circles': 'C', 'background': 'world'}
        def _ent(g, idx):
            if g == 'background':
                return 'world'
            return f"{alias[g]}{(idx or 0) + 1}"
        cons = [f"{_ent(c['entity1'], c['idx1'])}.{c['var1']} {c['op']} "
                f"{_ent(c['entity2'], c['idx2'])}.{c['var2']}"
                for c in cfg.get('extra_constraints', [])]
        print(f"    constraints: {'; '.join(cons) if cons else '(none)'}")

    print("\n" + "=" * 78 + "\n")


def main():
    N_RUNS     = 10           
    OUT_DIR    = 'results'   

    # Sanity check: show the search bounds for every case before running.
    display_scenario_bounds(SCENARIOS)

    jobs = [
        (name, cfg, run_idx)
        for name, cfg in SCENARIOS.items()
        for run_idx in range(N_RUNS)
    ]
    total_jobs = len(jobs)
    print(f"\nTotal jobs: {total_jobs}  ({len(SCENARIOS)} scenarios × {N_RUNS} runs each)\n")

    n_workers = max_parallel_processes(RAM_PER_PROCESS_GB)

    t_global_start = time.perf_counter()

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        all_results = pool.map(single_run, jobs)

    t_global_end = time.perf_counter()
    print(f"\nAll jobs finished in {t_global_end - t_global_start:.1f} s wall-clock time.")

    scenario_results = {name: [] for name in SCENARIOS}
    for r in all_results:
        scenario_results[r['scenario']].append(r)

    for name in SCENARIOS:
        summarise_scenario(name, scenario_results[name], OUT_DIR)

    print(f"\nDone. All outputs written to '{OUT_DIR}/'.")


if __name__ == '__main__':
    mp.freeze_support()
    main()