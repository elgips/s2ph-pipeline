#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity- and contrast-weighted scene metric for the S2PH / PWHG ERT study.

Computes, per run and per configuration, a single scalar reconstruction-quality
score Q_scene that aggregates all ground-truth objects (layer interface + circular
anomalies) using:

    Q_scene = sum_k ( w_k * q_k ) / sum_k ( w_k )

  q_k  (quality of object k):
        - circle : IoU(recovered disk, GT disk)   in [0,1]
        - layer  : 1 - min(1, |y_rec - y_gt| / tol)  in [0,1]
        - spurious recovered object (no GT match)  : q = 0
        - missed GT object (no recovered match)    : q = 0

  w_k  (importance weight of object k) = mean over the object's pixel footprint of
        S(x) * tanh( |Δlog10 rho(x)| / rho0 )
     -  S(x)      : spatial survey sensitivity (fixed, from a homogeneous
                    reference forward model; "where can the array see").
     -  Δlog10rho : local resistivity contrast against the *enclosing medium*
                    (world above the interface, layer below), evaluated per pixel
                    so a straddling anomaly is handled automatically.
     -  tanh(./rho0) : saturating contrast response (ERT data sensitivity to
                    contrast saturates; rho0 is the saturation scale in decades).
   Normalisation is a *mean* over the footprint (so object size does not inflate
   the weight); w_max cancels in the Q ratio and is omitted.

Over/under-modeling is handled by object matching: each GT circle is matched to its
best-IoU recovered circle; unmatched recovered circles are spurious (precision hit),
unmatched GT circles are missed (recall hit). Both enter Q with q=0 and their own
weight, so over-modeling is penalised in proportion to the spurious object's
sensitivity x contrast, and under-modeling in proportion to the missed object's.

Outputs:
  - <out>/scene_metric_results.xlsx : per-configuration mean/std/CV of Q_scene and
                                      of each per-object quality.
  - <out>/sensitivity_map.png       : the cached S(x) map (sanity check).
  - <out>/example_overlay.png       : one run's recovered objects over S(x).
  - <out>/robustness_rho0_tol.csv   : Q ordering under a sweep of (rho0, tol).

Requires: numpy, scipy, openpyxl, matplotlib, pygimli (only for building S(x); the
sensitivity map is cached to sensitivity_cache.npz and reused, so pygimli is needed
only on the first run or when geometry changes).

Author: built for Elad Gips' S2PH study.
"""

import os
import numpy as np
from scipy.interpolate import griddata
import openpyxl

# =============================================================================
# CONFIG  --  edit here
# =============================================================================

# ---- input / output paths ----
XLSX_PER_RUN = "article_runs_summary.xlsx"   # per-run workbook
OUT_DIR      = "scene_metric_out"
SENS_CACHE   = os.path.join(OUT_DIR, "sensitivity_cache.npz")

# ---- survey geometry (must match the runs that produced the data) ----
GEO = dict(
    elecs_n   = 21,
    elec_a    = -25.0,
    elec_b    =  25.0,
    scheme    = "dd",
    start     = (-25.0,  0.0),
    end       = ( 25.0, -20.0),
    rho_ref   = 100.0,     # homogeneous reference resistivity for S(x)
)

# ---- pixel grid (full domain, 0.1 m) ----
GRID = dict(
    xmin=-25.0, xmax=25.0, ymin=-20.0, ymax=0.0, dx=0.1, dy=0.1,
)

# ---- ground truth for C_{1,1} (confirmed from notebook + per-run data) ----
GT = dict(
    world_log_rho = np.log10(100.0),   # medium above the interface
    layer_y       = -3.0,              # interface depth
    layer_log_rho = np.log10(50.0),    # medium below the interface
    circles = [                        # list of GT anomalies
        dict(x=5.0, y=-4.0, r=1.2, log_rho=np.log10(500.0)),
    ],
)

# ---- free metric parameters (documented defaults) ----
RHO0     = 0.378     # contrast saturation scale [decades]; GT contrast ~1.0 dec sits at the knee
TOL      = 1.25     # layer depth tolerance [m] for q_layer
LAYERBAND= 1.25     # half-thickness [m] of the interface footprint band for weighting

# ---- robustness sweep ----
RHO0_SWEEP = [0.19, 0.28, 0.378, 0.57, 0.76]
TOL_SWEEP = [0.625, 1.25, 2.5]

# =============================================================================
# 1. SENSITIVITY MAP  S(x)   (pygimli; cached)
# =============================================================================

def build_sensitivity_grid():
    """Build S(x) on the pixel grid, cached to SENS_CACHE."""
    xs = np.arange(GRID["xmin"], GRID["xmax"] + 1e-9, GRID["dx"])
    ys = np.arange(GRID["ymin"], GRID["ymax"] + 1e-9, GRID["dy"])
    XX, YY = np.meshgrid(xs, ys)

    if os.path.exists(SENS_CACHE):
        d = np.load(SENS_CACHE)
        if d["xs"].shape == xs.shape and d["ys"].shape == ys.shape:
            return xs, ys, XX, YY, d["S"]

    # ---- compute cell-wise sensitivity from a homogeneous reference ----
    import pygimli as pg
    import pygimli.meshtools as mt
    from pygimli.physics import ert

    scheme = ert.createData(
        elecs=np.linspace(GEO["elec_a"], GEO["elec_b"], GEO["elecs_n"]),
        schemeName=GEO["scheme"])
    world = mt.createWorld(start=list(GEO["start"]), end=list(GEO["end"]),
                           worldMarker=True)
    geom = world
    for s in np.array(scheme.sensorPositions())[:, :2]:
        geom.createNode(s); geom.createNode(s - [0, 0.1])
    mesh = mt.createMesh(geom, quality=34)
    rho = np.ones(mesh.cellCount()) * GEO["rho_ref"]

    fop = ert.ERTModelling(); fop.setData(scheme); fop.setMesh(mesh)
    _ = fop.response(rho)
    fop.createJacobian(rho)
    J = np.array(fop.jacobian())                       # (n_data, n_cells)
    pd = fop.regionManager().paraDomain()
    cc = np.array([[c.center().x(), c.center().y()] for c in pd.cells()])
    areas = np.array([c.size() for c in pd.cells()])
    S_cell = np.abs(J).sum(axis=0) / areas             # sensitivity density

    # interpolate onto pixel grid (linear, nearest fallback for the hull edge)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    S = griddata(cc, S_cell, pts, method="linear")
    nan = np.isnan(S)
    if nan.any():
        S[nan] = griddata(cc, S_cell, pts[nan], method="nearest")
    S = S.reshape(XX.shape)
    S = np.clip(S, 0, None)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(SENS_CACHE, xs=xs, ys=ys, S=S)
    return xs, ys, XX, YY, S


# =============================================================================
# 2. RASTERISATION HELPERS
# =============================================================================

def disk_mask(XX, YY, x, y, r):
    return (XX - x) ** 2 + (YY - y) ** 2 <= r ** 2

def background_log_rho_field(XX, YY, world_log_rho, layer_y, layer_log_rho):
    """Per-pixel log10 resistivity of the *background* (no anomalies):
    world above the interface, layer below."""
    bg = np.where(YY >= layer_y, world_log_rho, layer_log_rho)
    return bg

def iou(mask_a, mask_b):
    inter = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return inter / union if union else 0.0


# =============================================================================
# 3. WEIGHTS
# =============================================================================

def circle_weight(XX, YY, S, bg_log_rho, x, y, r, obj_log_rho, rho0):
    """Mean over the disk footprint of S * tanh(|Δlog rho| / rho0),
    with Δlog rho against the local background (per pixel)."""
    m = disk_mask(XX, YY, x, y, r)
    if not m.any():
        return 0.0
    contrast = np.abs(obj_log_rho - bg_log_rho[m])
    sat = np.tanh(contrast / rho0)
    return float(np.mean(S[m] * sat))

def layer_weight(XX, YY, S, y_layer, world_log_rho, layer_log_rho, band, rho0):
    """Mean over a band around the interface of S * tanh(jump / rho0),
    where jump = |world_log_rho - layer_log_rho| (cross-interface contrast)."""
    m = np.abs(YY - y_layer) <= band
    if not m.any():
        return 0.0
    jump = abs(world_log_rho - layer_log_rho)
    sat = np.tanh(jump / rho0)
    return float(np.mean(S[m]) * sat)


# =============================================================================
# 4. PER-RUN SCENE SCORE
# =============================================================================

def gt_circle_masks(XX, YY):
    return [disk_mask(XX, YY, c["x"], c["y"], c["r"]) for c in GT["circles"]]

def scene_score_for_run(run, XX, YY, S, bg_gt, gt_masks, rho0, tol, band):
    """
    run: dict with keys
        world_log_rho, layer_y, layer_log_rho,
        circles: list of dict(x,y,r,log_rho)   [r linear]
    Returns (Q, dict_of_object_qualities).
    """
    # ---- layer object ----
    q_layer = 1.0 - min(1.0, abs(run["layer_y"] - GT["layer_y"]) / tol)
    w_layer = layer_weight(XX, YY, S, GT["layer_y"],
                           GT["world_log_rho"], GT["layer_log_rho"], band, rho0)
    objs = [("layer", w_layer, q_layer)]

    # ---- recovered circle masks & weights (contrast vs GT background field) ----
    rec = run["circles"]
    rec_masks = [disk_mask(XX, YY, c["x"], c["y"], c["r"]) for c in rec]
    rec_w = [circle_weight(XX, YY, S, bg_gt, c["x"], c["y"], c["r"], c["log_rho"], rho0)
             for c in rec]

    # ---- greedy best-IoU matching of each GT circle to a recovered circle ----
    n_gt, n_rec = len(gt_masks), len(rec_masks)
    used = set()
    matched_quality = {}
    for gi in range(n_gt):
        best_iou, best_ri = 0.0, -1
        for ri in range(n_rec):
            if ri in used:
                continue
            v = iou(gt_masks[gi], rec_masks[ri])
            if v > best_iou:
                best_iou, best_ri = v, ri
        if best_ri >= 0:
            used.add(best_ri)
            matched_quality[f"circle{gi+1}"] = best_iou
            # weight from the *recovered* matched object
            objs.append((f"circle{gi+1}", rec_w[best_ri], best_iou))
        else:
            # missed GT circle -> weight from GT object, q=0
            gc = GT["circles"][gi]
            wg = circle_weight(XX, YY, S, bg_gt, gc["x"], gc["y"], gc["r"], gc["log_rho"], rho0)
            matched_quality[f"circle{gi+1}"] = 0.0
            objs.append((f"circle{gi+1}", wg, 0.0))

    # ---- unmatched recovered circles -> spurious, q=0 ----
    n_spurious = 0
    for ri in range(n_rec):
        if ri not in used:
            n_spurious += 1
            objs.append((f"spurious{n_spurious}", rec_w[ri], 0.0))

    # ---- aggregate ----
    wsum = sum(w for _, w, _ in objs)
    Q = sum(w * q for _, w, q in objs) / wsum if wsum > 0 else 0.0

    detail = {"Q": Q, "q_layer": q_layer}
    detail.update(matched_quality)
    detail["n_spurious"] = n_spurious
    return Q, detail


# =============================================================================
# 5. READ PER-RUN WORKBOOK
# =============================================================================

def _clean_row(r):
    return [v for v in r[1:] if isinstance(v, (int, float))]

def read_runs(sheet, ws):
    """Return a list of run dicts. Handles labeled (a11, 7 rows) and positional
    (a12, 11 rows) layouts. r is stored log10 in the sheets -> converted to linear."""
    rows = list(ws.iter_rows(values_only=True))
    labeled = any(isinstance(r[0], str) and r[0] in ("rho top", "y_layer") for r in rows)

    def rowvals(label_or_idx):
        if labeled:
            for r in rows:
                if isinstance(r[0], str) and r[0].strip().lower() == label_or_idx:
                    return _clean_row(r)
            return None
        else:
            return _clean_row(rows[label_or_idx])

    if labeled:
        world = rowvals("rho top")          # absent on "known top" sheets
        ly    = rowvals("y_layer")
        lrho  = rowvals("rho bottom")
        cx    = rowvals("x"); cy = rowvals("y")
        clr   = rowvals("r") or rowvals("log r")
        crho  = rowvals("rho_circle") or rowvals("log rho")
        blocks = [(cx, cy, clr, crho)]
        if world is None:                   # world fixed to its known GT value
            world = [GT["world_log_rho"]] * len(ly)
    else:
        # positional rows 1..: world, y_layer, layer_rho, then circle blocks of 4
        world = rowvals(1); ly = rowvals(2); lrho = rowvals(3)
        blocks = []
        idx = 4
        while idx + 3 < len(rows):
            bx = _clean_row(rows[idx]); by = _clean_row(rows[idx+1])
            br = _clean_row(rows[idx+2]); brho = _clean_row(rows[idx+3])
            if not bx:
                break
            blocks.append((bx, by, br, brho))
            idx += 4

    n = len(world)
    runs = []
    for i in range(n):
        circles = []
        for (bx, by, br, brho) in blocks:
            if i < len(bx) and i < len(by) and i < len(br) and i < len(brho):
                circles.append(dict(
                    x=bx[i], y=by[i],
                    r=10 ** br[i],          # sheets store log10(r)
                    log_rho=brho[i],        # already log10(rho)
                ))
        runs.append(dict(
            world_log_rho=world[i],
            layer_y=ly[i],
            layer_log_rho=lrho[i],
            circles=circles,
        ))
    return runs


# =============================================================================
# 6. MAIN
# =============================================================================

def aggregate(vals):
    a = np.asarray(vals, float)
    m, s = a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0
    cv = s / abs(m) if m != 0 else np.nan
    return m, s, cv

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    xs, ys, XX, YY, S = build_sensitivity_grid()
    bg_gt = background_log_rho_field(XX, YY, GT["world_log_rho"],
                                     GT["layer_y"], GT["layer_log_rho"])
    gt_masks = gt_circle_masks(XX, YY)

    wb = openpyxl.load_workbook(XLSX_PER_RUN, read_only=True, data_only=True)
    data_sheets = [s for s in wb.sheetnames if s not in ("summery table", "ploting")]

    out = openpyxl.Workbook()
    wsout = out.active; wsout.title = "scene_metric"
    header = ["config", "n_runs",
              "Q_mean", "Q_std", "Q_CV",
              "qLayer_mean", "qLayer_CV",
              "qCircle_mean", "qCircle_CV",
              "mean_n_spurious"]
    wsout.append(header)

    print(f"{'config':38s} {'Qmean':>8s} {'QCV%':>7s} {'qCircle':>8s} {'qLayer':>7s} {'nSpur':>6s}")
    for sh in data_sheets:
        runs = read_runs(sh, wb[sh])
        if not runs:
            continue
        Qs, qL, qC, nsp = [], [], [], []
        for run in runs:
            Q, det = scene_score_for_run(run, XX, YY, S, bg_gt, gt_masks,
                                         RHO0, TOL, LAYERBAND)
            Qs.append(Q); qL.append(det["q_layer"])
            # primary anomaly quality = circle1
            qC.append(det.get("circle1", 0.0))
            nsp.append(det.get("n_spurious", 0))
        Qm, Qs_, Qcv = aggregate(Qs)
        qLm, _, qLcv = aggregate(qL)
        qCm, _, qCcv = aggregate(qC)
        wsout.append([sh, len(runs),
                      Qm, Qs_, Qcv, qLm, qLcv, qCm, qCcv, float(np.mean(nsp))])
        print(f"{sh:38s} {Qm:8.3f} {Qcv*100:7.2f} {qCm:8.3f} {qLm:7.3f} {np.mean(nsp):6.2f}")

    out_path = os.path.join(OUT_DIR, "scene_metric_results.xlsx")
    out.save(out_path)
    print("\nsaved:", out_path)

    _save_figs(xs, ys, S, XX, YY, wb, data_sheets, bg_gt, gt_masks)
    _robustness(wb, data_sheets, XX, YY, S, bg_gt, gt_masks)


def _save_figs(xs, ys, S, XX, YY, wb, data_sheets, bg_gt, gt_masks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [GRID["xmin"], GRID["xmax"], GRID["ymin"], GRID["ymax"]]
    # sensitivity map (log scale for visibility)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(np.log10(S + 1e-6), origin="lower", extent=ext, aspect="equal",
                   cmap="magma")
    ax.set_title("log10 spatial sensitivity S(x)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    plt.colorbar(im, ax=ax, shrink=0.8)
    # overlay GT objects
    ax.axhline(GT["layer_y"], color="cyan", lw=1, ls="--")
    for c in GT["circles"]:
        th = np.linspace(0, 2*np.pi, 100)
        ax.plot(c["x"]+c["r"]*np.cos(th), c["y"]+c["r"]*np.sin(th), "c-", lw=1)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "sensitivity_map.png"), dpi=130)
    plt.close(fig)

    # example overlay: first run of a12_20_30 if present else first sheet
    sh = "a12_20_30" if "a12_20_30" in data_sheets else data_sheets[0]
    runs = read_runs(sh, wb[sh])
    run = runs[0]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(np.log10(S + 1e-6), origin="lower", extent=ext, aspect="equal",
                   cmap="magma", alpha=0.9)
    ax.axhline(GT["layer_y"], color="cyan", lw=1.2, ls="--", label="GT layer")
    for c in GT["circles"]:
        th = np.linspace(0, 2*np.pi, 100)
        ax.plot(c["x"]+c["r"]*np.cos(th), c["y"]+c["r"]*np.sin(th), "c-", lw=1.5,
                label="GT circle")
    ax.axhline(run["layer_y"], color="lime", lw=1.2, ls=":", label="rec layer")
    for j, c in enumerate(run["circles"]):
        th = np.linspace(0, 2*np.pi, 100)
        ax.plot(c["x"]+c["r"]*np.cos(th), c["y"]+c["r"]*np.sin(th), "w-", lw=1.2,
                label=("rec circle" if j == 0 else None))
    ax.set_title(f"{sh}  run 1  (recovered vs GT)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "example_overlay.png"), dpi=130)
    plt.close(fig)
    print("saved: sensitivity_map.png, example_overlay.png")


def _robustness(wb, data_sheets, XX, YY, S, bg_gt, gt_masks):
    """Check the config ordering by Q_mean is stable across (rho0, tol)."""
    import csv
    rows = []
    for rho0 in RHO0_SWEEP:
        for tol in TOL_SWEEP:
            order = []
            for sh in data_sheets:
                runs = read_runs(sh, wb[sh])
                if not runs:
                    continue
                Qs = [scene_score_for_run(r, XX, YY, S, bg_gt, gt_masks,
                                          rho0, tol, LAYERBAND)[0] for r in runs]
                order.append((sh, float(np.mean(Qs))))
            order.sort(key=lambda t: -t[1])
            rows.append(dict(rho0=rho0, tol=tol,
                             ranking=" > ".join(s for s, _ in order)))
    path = os.path.join(OUT_DIR, "robustness_rho0_tol.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rho0", "tol", "ranking"])
        w.writeheader(); w.writerows(rows)
    print("saved: robustness_rho0_tol.csv")


if __name__ == "__main__":
    main()
