"""
config.py -- single source of truth for the C_ij regeneration pipeline.

Every stage imports from here. If a number appears in two places, it is a bug.

GEOMETRY CONVENTION (confirmed against main.tex L189 and the generator):
    Layers stack DOWNWARD FROM THE SURFACE. Layer(y=Y, rho=R) fills from the
    base of the previous layer down to depth Y with R. The World resistivity
    fills everything BELOW the last layer.

        j=0 : 0 .. -20  100                                   (homogeneous)
        j=1 : 0 .. -3    50  |  -3 .. -20  100
        j=2 : 0 .. -3    50  |  -3 .. -7   250  |  -7 .. -20  100

    The SHALLOW zone is the CONDUCTIVE one. With cMap='binary' (high value ->
    black), bright = LOW resistivity -- which is why the row profile reads
    brighter at the top.
"""
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Domain and acquisition
# ----------------------------------------------------------------------
X_MIN, X_MAX = -25.0, 25.0
Y_TOP, Y_BOT = 0.0, -20.0
WIDTH_M = X_MAX - X_MIN                 # 50
DEPTH_M = abs(Y_BOT - Y_TOP)            # 20

N_ELEC = 21
ELEC_SPACING = WIDTH_M / (N_ELEC - 1)   # 2.5 m
SCHEME_NAME = "dd"
ARRAY_LENGTH_M = WIDTH_M

TAU_M = ELEC_SPACING / 2.0              # 1.25 m nominal vertical resolution

RHO_WORLD = 100.0
RHO_WORLD_VARLIM = [50.0, 1000.0]

# ----------------------------------------------------------------------
# Forward noise
#
# pyGIMLi reads noiseLevel >= 0.5 as a PERCENT and divides by 100, giving a
# 0.5% relative floor. The absolute term at 1e-6 is then negligible:
# err_i ~ 0.005 + |1e-6/u_i|. This is the published setting; the CV input
# is produced from it.
# ----------------------------------------------------------------------
NOISE_LEVEL = 0.5
NOISE_ABS = 1e-6
FORWARD_SEED = 2351

NOISE_RMS_EXPECT = 0.005
NOISE_RMS_TOL = (0.002, 0.012) if NOISE_ABS <= 1e-5 else (0.02, 0.20)  # stage 1 asserts the perturbation IS ~0.5%
NOISE_RMS_RANGE = (0.002, 0.012) if NOISE_ABS <= 1e-5 else (0.02, 0.20)

CV_INPUT_DATASET = "noisy"
PWHG_DATASETS = ["clean", "noisy"]

# ----------------------------------------------------------------------
# Smooth inversion. cType=1 is first-order smoothness under an L2 norm --
# the "ell_2 regularized" result described in main.tex L189.
# ----------------------------------------------------------------------
INV_NX = 60
INV_NY = 60
INV_Y_FIRST = 0.5
INV_LAM = 30.0
INV_CTYPE = 1
INV_BLOCKY = False
INV_XBOUND = 15.0
INV_YBOUND = 15.0
INV_MAXITER = 20

# Area thresholds in m^2, converted to pixels at runtime. The raw pixel values
# in seg_params were calibrated at ~9.9 px/m; at PX_PER_M=20 they are 4.08x
# more permissive in physical terms.
#   one resolution cell = a * tau = 2.5 * 1.25 = 3.125 m^2
#   C1 true area = pi*1.2^2 = 4.52 m^2 ; C2 = pi*1.5^2 = 7.07 m^2
BLOB_MIN_AREA_M2 = 3.125      # one resolution cell
LAYER_MIN_AREA_M2 = 3.56      # preserves the previously calibrated 350 px
# ----------------------------------------------------------------------
# Raster export -- the normalisation fix.
#
# Fixed colour limits + cMap='binary' + logScale, written as a pure raster
# covering exactly the model domain. No axes, colourbar or whitespace, so grey
# level is an exact affine function of log10(rho) and every CV threshold is
# denominated in DECADES.
#
#   binary colormap: HIGH value -> BLACK, so bright = LOW resistivity:
#       g = 1 - (log10(rho) - log10(RHO_PLOT_MIN)) / SPAN_DECADES
# ----------------------------------------------------------------------
RHO_PLOT_MIN = 10.0
RHO_PLOT_MAX = 1000.0
SPAN_DECADES = float(np.log10(RHO_PLOT_MAX / RHO_PLOT_MIN))     # exactly 2.0

PX_PER_M = 20.0
IMG_W = int(round(WIDTH_M * PX_PER_M))      # 1000
IMG_H = int(round(DEPTH_M * PX_PER_M))      # 400
EXPORT_DPI = 100


def log10rho_to_gray(log10rho):
    """Forward export map. Grey in [0,1]; bright = low resistivity."""
    l = np.asarray(log10rho, float)
    return np.clip(1.0 - (l - np.log10(RHO_PLOT_MIN)) / SPAN_DECADES, 0.0, 1.0)


def gray_to_log10rho(g):
    """Exact inverse, used by the CV stage."""
    g = np.asarray(g, float)
    return np.log10(RHO_PLOT_MIN) + (1.0 - g) * SPAN_DECADES


def gray_to_decades(g):
    """Grey -> decades above RHO_PLOT_MIN. This is the field the CV stage
    thresholds, so bright_threshold / dark_threshold are in decades."""
    return (1.0 - np.asarray(g, float)) * SPAN_DECADES


# ----------------------------------------------------------------------
# Ground truth. L2 at -7 m in ALL j=2 scenes.
# ----------------------------------------------------------------------
L1 = dict(name="layer1", y=-3.0, rho=50.0)
L2 = dict(name="layer2", y=-7.0, rho=250.0)

C1 = dict(name="C1", x=5.0,  y=-4.0, r=1.2, rho=500.0)
C2 = dict(name="C2", x=-5.0, y=-2.5, r=1.5, rho=25.0)


def enclosing_rho(y, j):
    """Resistivity of the medium at depth y (negative down) for j interfaces."""
    if j >= 1 and y > L1["y"]:
        return L1["rho"]
    if j >= 2 and y > L2["y"]:
        return L2["rho"]
    return RHO_WORLD


def contrast_decades(rho_obj, rho_bg):
    return abs(float(np.log10(float(rho_obj) / float(rho_bg))))


def scene_contrasts(i, j):
    """Every detectable contrast in a scene, in decades. Drives the delta window."""
    out = {}
    if j >= 1:
        upper = L1["rho"]
        lower = L2["rho"] if j >= 2 else RHO_WORLD
        out["iface_-3"] = contrast_decades(lower, upper)
    if j >= 2:
        out["iface_-7"] = contrast_decades(RHO_WORLD, L2["rho"])
    for c in [C1, C2][:i]:
        out[c["name"]] = contrast_decades(c["rho"], enclosing_rho(c["y"], j))
    return out


# ----------------------------------------------------------------------
# CV thresholds, in DECADES.
#
# Minimum contrast in the suite is 0.301 decade (the -3 interface when j=1,
# C1 inside the 250 ohm-m zone when j=2, C2's upper half inside the 50 ohm-m
# layer). Offsets must sit well below half of that.
# ----------------------------------------------------------------------
# Window measured from the exported rasters (stage2b), not from true
# contrasts: the smooth inversion overshoots broad interfaces and
# attenuates compact anomalies, so true contrast is the wrong quantity.
#   layer floor   0.0078 dec = one grey level (quantisation, not physics)
#   layer ceiling 0.0863 dec = C11 gap_min, the binding scene
#   blob  floor   0.0235 dec = C00 resid_max, the artifact ceiling in a
#                 homogeneous world. The BLOB branch runs regardless of the
#                 layer gate, so this floor is real and binding.
# Measured 9/9 correct element counts at 0.05 and at 0.07. 0.07 is used:
# layer depth degrades monotonically with delta while circle radius improves,
# and radius is the larger error.
DELTA_CEILING = 0.0863
CV_SWEEP_DELTAS = [
    (0.015, 0.015),
    (0.025, 0.025),
    (0.030, 0.030),
    (0.050, 0.050),
    (0.070, 0.070),
]
CV_REFERENCE_DELTA = (0.050, 0.050)

# Data-domain structural screen (stage1b). Layer separation is 14x;
# anomaly separation is only 1.47x, so only the layer gate is used.
LAYER_SCREEN_SNR = 3.0
USE_LAYER_GATE = True
USE_ANOMALY_GATE = False

RIM_MARGIN_M = TAU_M
FLOOR_CLOSE_FRAC = 0.17
DOI_EXTRA_DEPTH = 0.1
TOP_EXCLUDE_M = 0.2


def _scene(i, j):
    """i = circular anomalies, j = layer interfaces."""
    return dict(
        name=f"C{i}{j}", key=f"c{i}{j}", i=i, j=j,
        layers=[L1, L2][:j], circles=[C1, C2][:i],
        n_params=1 + 2 * j + 4 * i,
        contrasts=scene_contrasts(i, j),
    )


SCENES = [_scene(i, j) for i in (0, 1, 2) for j in (0, 1, 2)]
SCENE_NAMES = [s["name"] for s in SCENES]
SCENE_BY_NAME = {s["name"]: s for s in SCENES}

# ----------------------------------------------------------------------
# Optimisation (mirrors run_cij_opti.py; stage 4 delegates to it)
# ----------------------------------------------------------------------
N_REPEATS = 5
OPTI_SEEDS = [1001, 1002, 1003, 1004, 1005]
GLOBAL_MAX_ITER = 20
GLOBAL_POPSIZE = 25            # scipy uses popsize * ndim
GLOBAL_N = 3
GLOBAL_ATOL = 1e-5
TARGET_VALUE = 1e-6
POP_EPSILON = 0.15
GUESS_RATIO = 0.25

TOP_LAYER_RHO_FIXED = 50.0
TOP_LAYER_VARFLAG = [True, False]
POS_REL_MARGIN = 0.5
POS_ABS_MARGIN = 2.0
CIRCLE_LAYER_MARGIN = POS_ABS_MARGIN
RADIUS_UPPER_FACTOR = 1.2
RADIUS_LOWER = 0.5

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs"
DIR_FWD = OUT / "01_forward"
DIR_INV = OUT / "02_invert"
DIR_CV = OUT / "03_cv"
DIR_OPT = OUT / "04_opti"
DIR_CMP = OUT / "05_compare"
DIR_LOG = OUT / "logs"
ALL_DIRS = [OUT, DIR_FWD, DIR_INV, DIR_CV, DIR_OPT, DIR_CMP, DIR_LOG]

ERT_INTERPRETER = ROOT / "ert_interpreter.py"
ANANDLYN = ROOT / "Anandlyn_log.py"
RUN_CIJ_OPTI = ROOT / "run_cij_opti.py"


def summary():
    lines = ["scene  i j  npar  contrasts [decade]"]
    for s in SCENES:
        c = ", ".join(f"{k}={v:.3f}" for k, v in s["contrasts"].items()) or "none"
        lines.append(f"{s['name']}   {s['i']} {s['j']}  {s['n_params']:>3}   {c}")
    vals = [v for s in SCENES for v in s["contrasts"].values()]
    lines.append(f"\nminimum contrast in suite: {min(vals):.3f} decade")
    lines.append(f"delta window: {CV_SWEEP_DELTAS}  ceiling {DELTA_CEILING}")
    lines.append(f"raster: {IMG_W}x{IMG_H} px @ {PX_PER_M} px/m, span "
                 f"{SPAN_DECADES:.3f} decade over "
                 f"[{RHO_PLOT_MIN},{RHO_PLOT_MAX}] ohm-m")
    lines.append(f"tau = {TAU_M} m = {TAU_M*PX_PER_M:.0f} px")
    lines.append(f"noise: level={NOISE_LEVEL} abs={NOISE_ABS} "
                 f"-> expect ~{NOISE_RMS_EXPECT*100:.1f}% relative")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
