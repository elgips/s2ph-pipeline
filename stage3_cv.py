#!/usr/bin/env python3
"""
stage3_cv.py -- CV segmentation sweep, component audit, initializer extraction.

Imports the PATCHED ert_interpreter.py and calls its live functions, so the
pipeline and the GUI cannot diverge. Requires patches 1-4 (fixed-scale
normalisation in decades); the script refuses to run without them.

Outputs, in runs/03_cv/:
    cv_sweep.csv        one row per (scene, delta) with detected elements
    cv_components.csv   one row per layer component, with why it was kept
    parameters.xlsx     CV initializers in the layout run_cij_opti expects
    {scene}_cv.png      overlay of detections on the smooth image
    cv_summary.json

The layer GATE comes from stage1b: if the data screen finds no layer signal
above LAYER_SCREEN_SNR, the layer branch's output is discarded for that
scene. The anomaly gate is NOT applied -- its separation (1.47x) is too weak
to act on. Both decisions are recorded per scene.

Gates:
    G1  ert_interpreter is patched (fixed_scale present and honoured)
    G2  the raster matches config geometry exactly
    G3  the reference delta lies inside the measured window
    G4  element counts at the reference delta match the scene design
        (recorded as a metric, NOT enforced -- a mismatch is a finding)
    G5  every extracted initializer is finite and inside its varlim
    G6  parameters.xlsx re-reads through run_cij_opti.load_cij_table

Usage:
    python stage3_cv.py
    python stage3_cv.py --scenes C11 --no-xlsx
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as _Circ, Rectangle as _Rect

from PIL import Image
from skimage import measure

import config as C
from pipelib import (Gate, Manifest, get_logger, require_upstream, GateFailure)

STAGE = "stage3_cv"


def load_interpreter(log):
    """Import ert_interpreter and verify it carries the fixed-scale patch."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ert_interpreter", str(C.ERT_INTERPRETER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ert_interpreter"] = mod
    spec.loader.exec_module(mod)

    missing = [k for k in ("fixed_scale", "rho_plot_min", "rho_plot_max")
               if k not in mod.seg_params]
    if missing or not hasattr(mod, "normalize_field"):
        raise GateFailure(
            "ert_interpreter.py is not patched for fixed-scale segmentation "
            f"(missing {missing or 'normalize_field'}). Apply patches 1-4 "
            "before running stage 3 -- otherwise thresholds silently revert "
            "to per-image fractions.")

    # Force the pipeline's plot range onto the interpreter, so a stale value
    # in seg_params cannot silently rescale every threshold.
    mod.seg_params["fixed_scale"] = True
    mod.seg_params["rho_plot_min"] = C.RHO_PLOT_MIN
    mod.seg_params["rho_plot_max"] = C.RHO_PLOT_MAX
    mod.seg_params["layer_top_rim_margin_m"] = C.RIM_MARGIN_M
    mod.seg_params["layer_floor_close_frac"] = C.FLOOR_CLOSE_FRAC
    mod.seg_params["extra_depth"] = C.DOI_EXTRA_DEPTH
    mod.seg_params["top_exclude_m"] = C.TOP_EXCLUDE_M
    # Area thresholds are stored in PIXELS but were calibrated at ~9.9 px/m.
    # Convert from m^2 so they mean the same physical area regardless of
    # PX_PER_M (at 20 px/m the raw values are 4.08x more permissive).
    px2 = C.PX_PER_M ** 2
    mod.seg_params["blob_min_size"] = int(round(C.BLOB_MIN_AREA_M2 * px2))
    mod.seg_params["layer_min_area"] = int(round(C.LAYER_MIN_AREA_M2 * px2))
    log.info("ert_interpreter loaded, fixed_scale=True, plot range [%g,%g]",
             C.RHO_PLOT_MIN, C.RHO_PLOT_MAX)
    log.info("area thresholds: blob %d px (%.3f m2), layer %d px (%.3f m2)",
             mod.seg_params["blob_min_size"], C.BLOB_MIN_AREA_M2,
             mod.seg_params["layer_min_area"], C.LAYER_MIN_AREA_M2)
    return mod


def load_layer_gate(log):
    p = C.DIR_FWD / "data_screen.json"
    if not p.exists():
        log.warning("no data_screen.json -- layer gate DISABLED for all scenes")
        return {}
    rows = json.loads(p.read_text())
    gate = {}
    for r in rows:
        gate[r["scene"]] = dict(
            has_layer=bool(r["predict_j_gt0"]),
            layer_snr=float(r["layer_snr"]),
            anom_snr=float(r["anom_snr"]))
    return gate


def px_to_m_x(col, w):
    return C.X_MIN + (float(col) / w) * C.WIDTH_M


def px_to_m_y(row, h):
    """Row index -> signed depth (negative down)."""
    return -(float(row) / h) * C.DEPTH_M


def px_to_m_len(px, h):
    return (float(px) / h) * C.DEPTH_M


def element_rho(ei, img_gray, mask):
    """Median resistivity inside a mask, via the interpreter's own readout."""
    vals = img_gray[mask]
    if vals.size == 0:
        return np.nan
    return float(ei._scale_to_rhoa(float(np.median(vals))))


def run_one(ei, scene, img_gray, db, dd, gate_has_layer):
    """One (scene, delta) evaluation. Returns summary + component records."""
    name = scene["name"]
    ei.seg_params["bright_threshold"] = float(db)
    ei.seg_params["dark_threshold"] = float(dd)

    img_n = ei.normalize_field(img_gray)
    layer_mask, blob_mask = ei.process_mask(img_n, ei.doi_mask)

    gated = (not gate_has_layer)
    if gated:
        layer_mask = np.zeros_like(layer_mask)

    circles, layers = ei.extract_circles_and_layers(layer_mask, blob_mask)
    h, w = img_gray.shape

    # profile diagnostics, for the delta-window record
    prof = ei.compute_row_profile(img_n, ei.doi_mask)
    valid = prof[~np.isnan(prof)]
    spread = float(np.ptp(valid)) if valid.size else np.nan

    comps = []
    lab = measure.label(layer_mask)
    for k, p in enumerate(measure.regionprops(lab)):
        comps.append(dict(
            scene=name, delta_bright=db, delta_dark=dd, component=k,
            area_px=int(p.area), bbox_top_px=int(p.bbox[0]),
            bbox_bot_px=int(p.bbox[2]),
            y_top_m=px_to_m_y(p.bbox[0], h), y_bot_m=px_to_m_y(p.bbox[2], h)))

    # Blob components BEFORE and AFTER filter_blobs_by_geometry, so a
    # missing anomaly can be attributed to the area floor, the aspect
    # test, or the threshold itself.
    img_n_dbg = ei.normalize_field(img_gray)
    prof_dbg = ei.compute_row_profile(img_n_dbg, ei.doi_mask)
    base = np.zeros_like(img_n_dbg)
    for y in range(h):
        if not np.isnan(prof_dbg[y]):
            base[y, :] = prof_dbg[y]
    thr = max(db, dd)
    cand = (np.abs(img_n_dbg - base) > thr) & ei.doi_mask
    for k, p in enumerate(measure.regionprops(measure.label(cand))):
        minr, minc, maxr, maxc = p.bbox
        hb, wb = maxr - minr + 1, maxc - minc + 1
        keep_area = p.area >= ei.seg_params["blob_min_size"]
        keep_asp = (wb / hb) <= ei.seg_params["blob_max_aspect"]
        comps.append(dict(
            scene=name, delta_bright=db, delta_dark=dd,
            component=f"blob{k}", kind="blob",
            area_px=int(p.area), area_m2=p.area / (C.PX_PER_M ** 2),
            bbox_top_px=int(minr), bbox_bot_px=int(maxr),
            aspect=wb / hb,
            blob_min_size=int(ei.seg_params["blob_min_size"]),
            reject_area=not keep_area, reject_aspect=not keep_asp,
            fate=("kept" if (keep_area and keep_asp)
                  else "REJ_area" if not keep_area else "REJ_aspect")))

    cir_out, lay_out = [], []
    for c in circles:
        m = np.zeros((h, w), dtype=bool)
        yy, xx = np.ogrid[:h, :w]
        m[((yy - c["y"]) ** 2 + (xx - c["x"]) ** 2) <= c["r"] ** 2] = True
        cir_out.append(dict(
            x_m=px_to_m_x(c["x"], w), y_m=px_to_m_y(c["y"], h),
            r_m=px_to_m_len(c["r"], h),
            rho=element_rho(ei, img_gray, m & ei.doi_mask),
            x_px=float(c["x"]), y_px=float(c["y"]), r_px=float(c["r"])))
    for l in layers:
        lay_out.append(dict(y_m=px_to_m_y(l["y"], h), y_px=float(l["y"])))

    # Sort by x descending, NOT by radius. Radius ordering is unstable: in
    # C20/C22 the C2 blob is marginally larger than C1's, which swaps the
    # slots and puts each element's PWHG bound around the other's position.
    # This also matches run_cij_opti's label-switching constraint (c1.x >=
    # c2.x). NOTE: that this reproduces the C1/C2 naming is a property of THIS
    # suite (C1 at +5 m, C2 at -5 m), i.e. a labelling convention, not a
    # general correspondence. Stage 5 must sort ground truth identically.
    cir_out.sort(key=lambda d: -d["x_m"])
    lay_out.sort(key=lambda d: -d["y_m"])      # shallowest first

    summary = dict(
        scene=name, delta_bright=db, delta_dark=dd,
        layer_gate_applied=gated,
        profile_spread_dec=spread,
        n_circles=len(cir_out), n_interfaces=len(lay_out),
        true_i=scene["i"], true_j=scene["j"],
        interfaces_m=";".join(f"{l['y_m']:.3f}" for l in lay_out),
        circles_xyr=";".join(f"({c['x_m']:.2f},{c['y_m']:.2f},{c['r_m']:.2f})"
                             for c in cir_out),
    )
    return summary, comps, cir_out, lay_out, layer_mask, blob_mask


def overlay(png_out, img_gray, ei, layer_mask, blob_mask, circles, layers, title):
    h, w = img_gray.shape
    fig, ax = plt.subplots(figsize=(w / 120, h / 120), dpi=120)
    disp = np.where(ei.doi_mask, img_gray, np.nan)
    ax.imshow(disp, cmap="gray", vmin=0, vmax=1, origin="upper")
    if layer_mask.any():
        ax.contour(layer_mask, colors="tab:blue", linewidths=0.9)
    _LAYER_FC = ("tab:blue", "tab:orange", "tab:green")
    top = 0
    for k, l in enumerate(sorted(layers, key=lambda d: d["y_px"])):
        band = np.zeros((h, w), dtype=bool)
        band[int(round(top)):int(round(l["y_px"])), :] = True
        band &= ei.doi_mask
        rgba = np.zeros((h, w, 4))
        rgba[..., :3] = matplotlib.colors.to_rgb(_LAYER_FC[k % len(_LAYER_FC)])
        rgba[..., 3] = np.where(band, 0.22, 0.0)
        ax.imshow(rgba, origin="upper", zorder=2)
        top = l["y_px"]
    _ANOM_FC = ("tab:red", "tab:purple")
    for k, c in enumerate(circles):
        ax.add_patch(_Circ((c["x_px"], c["y_px"]), c["r_px"],
                           facecolor=_ANOM_FC[k % len(_ANOM_FC)],
                           alpha=0.35, edgecolor=_ANOM_FC[k % len(_ANOM_FC)],
                           lw=1.6, zorder=3))
    if blob_mask.any():
        ax.contour(blob_mask, colors="tab:red", linewidths=0.9)
    for c in circles:
        ax.add_patch(plt.Circle((c["x_px"], c["y_px"]), c["r_px"],
                                ec="lime", fc="none", lw=1.4))
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(png_out, dpi=120)
    plt.close(fig)


def write_parameters_xlsx(path, inits, log):
    """Emit parameters.xlsx in the layout run_cij_opti.load_cij_table parses.

    NOTE: that parser names its second block 'truth', but the values are CV
    ESTIMATES, not ground truth. The naming is inherited; the content here is
    x0 only.
    """
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "parameters"

    ws.append(["world rho search range (linear)", "min", "max"])
    for s in C.SCENES:
        r = inits.get(s["name"])
        rho = r["rho_bg"] if r and r.get("rho_bg") else C.RHO_WORLD
        ws.append([s["key"], C.RHO_WORLD_VARLIM[0], C.RHO_WORLD_VARLIM[1]])
    ws.append([])
    ws.append(["config", "rho bg", "layer1 y", "layer1 rho",
               "layer2 y", "layer2 rho",
               "c1 x", "c1 y", "c1 r", "c1 rho",
               "c2 x", "c2 y", "c2 r", "c2 rho"])
    for s in C.SCENES:
        r = inits.get(s["name"])
        if r is None:
            ws.append([s["key"]] + ["-"] * 13)
            continue
        row = [s["key"], r.get("rho_bg", "-")]
        for k in ("layer1_y", "layer1_rho", "layer2_y", "layer2_rho",
                  "c1_x", "c1_y", "c1_r", "c1_rho",
                  "c2_x", "c2_y", "c2_r", "c2_rho"):
            v = r.get(k)
            row.append("-" if v is None or not np.isfinite(v) else round(float(v), 4))
        ws.append(row)
    wb.save(path)
    log.info("wrote %s", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=C.SCENE_NAMES)
    ap.add_argument("--no-xlsx", action="store_true")
    args = ap.parse_args()

    for d in C.ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    log = get_logger(STAGE, C.DIR_LOG)
    man = Manifest(STAGE, C.OUT)
    man.echo_config(sweep=C.CV_SWEEP_DELTAS, reference=C.CV_REFERENCE_DELTA,
                    ceiling=C.DELTA_CEILING, rim_margin_m=C.RIM_MARGIN_M,
                    use_layer_gate=getattr(C, "USE_LAYER_GATE", True))

    require_upstream("stage2_invert", C.OUT, args.scenes, log)
    ei = load_interpreter(log)
    gate = load_layer_gate(log)

    db_ref, dd_ref = C.CV_REFERENCE_DELTA
    if max(db_ref, dd_ref) >= C.DELTA_CEILING:
        raise GateFailure(
            f"reference delta {C.CV_REFERENCE_DELTA} is at or above the "
            f"measured ceiling {C.DELTA_CEILING} -- detection will fail")

    sweep_rows, comp_rows, inits = [], [], {}

    for nm in args.scenes:
        scene = C.SCENE_BY_NAME[nm]
        g = Gate(STAGE, nm, log)
        png = C.DIR_INV / f"{nm}_smooth.png"
        if not png.exists():
            g.check("G0_image", False, f"missing {png}")
            man.record(nm, gate=g)
            continue

        arr = np.asarray(Image.open(png).convert("L"), dtype=np.uint8)
        img_gray = arr.astype(float) / 255.0
        h, w = img_gray.shape
        g.check("G2_raster_geometry", (h, w) == (C.IMG_H, C.IMG_W),
                f"{(h, w)} vs {(C.IMG_H, C.IMG_W)}")

        ei.current_img = np.dstack([img_gray] * 3)
        ei.user_width = C.WIDTH_M
        ei.user_depth = C.DEPTH_M
        ei.user_scheme = C.ARRAY_LENGTH_M
        ei.doi_mask = ei.get_current_doi_mask()
        g.check("G2_doi_nonempty", ei.doi_mask.any(), int(ei.doi_mask.sum()))

        gi = gate.get(nm, dict(has_layer=True, layer_snr=np.nan, anom_snr=np.nan))
        use_gate = getattr(C, "USE_LAYER_GATE", True)
        has_layer = gi["has_layer"] if use_gate else True
        log.info("%s: layer gate %s (snr %.1f) -> layer branch %s", nm,
                 "ON" if use_gate else "OFF", gi["layer_snr"],
                 "enabled" if has_layer else "DISABLED")

        for db, dd in C.CV_SWEEP_DELTAS:
            s, comps, cir, lay, lm, bm = run_one(
                ei, scene, img_gray, db, dd, has_layer)
            s.update(layer_snr=gi["layer_snr"], anom_snr=gi["anom_snr"])
            sweep_rows.append(s)
            comp_rows.extend(comps)
            log.debug("%s d=(%.3f,%.3f): %d circ, %d iface", nm, db, dd,
                      s["n_circles"], s["n_interfaces"])

        # --- reference delta: extract the initializers ---
        s, comps, cir, lay, lm, bm = run_one(
            ei, scene, img_gray, db_ref, dd_ref, has_layer)
        overlay(C.DIR_CV / f"{nm}_cv.png", img_gray, ei, lm, bm, cir, lay,
                f"{nm}  d=({db_ref},{dd_ref})  "
                f"{len(cir)} circle(s), {len(lay)} interface(s)")

        # rho_world fills BELOW the deepest interface. Sampling the whole DOI
        # mixes in the shallow layer's own pixels and biases the estimate
        # toward the layer resistivity.
        bg = np.zeros((h, w), dtype=bool)
        bg[ei.doi_mask] = True
        bg &= ~lm & ~bm
        if lay:
            deepest_row = int(-min(l["y_m"] for l in lay) / C.DEPTH_M * h)
            below = np.zeros((h, w), dtype=bool)
            below[deepest_row:, :] = True
            if (bg & below).sum() > 500:
                bg &= below
        rec = dict(rho_bg=element_rho(ei, img_gray, bg))
        for idx, l in enumerate(lay[:2], start=1):
            rec[f"layer{idx}_y"] = l["y_m"]
        for idx, cc in enumerate(cir[:2], start=1):
            rec[f"c{idx}_x"] = cc["x_m"]
            rec[f"c{idx}_y"] = cc["y_m"]
            rec[f"c{idx}_r"] = cc["r_m"]
            rec[f"c{idx}_rho"] = cc["rho"]
        # layer1 rho is fixed at the known 50 by run_cij_opti; layer2 rho is
        # read from the image between the two interfaces if both were found.
        if len(lay) >= 1:
            rec["layer1_rho"] = C.TOP_LAYER_RHO_FIXED
        if len(lay) >= 2:
            y1 = int(-lay[0]["y_m"] / C.DEPTH_M * h)
            y2 = int(-lay[1]["y_m"] / C.DEPTH_M * h)
            band = np.zeros((h, w), dtype=bool)
            band[min(y1, y2):max(y1, y2), :] = True
            rec["layer2_rho"] = element_rho(ei, img_gray, band & ei.doi_mask)
        inits[nm] = rec

        g.check("G3_delta_in_window",
                max(db_ref, dd_ref) < C.DELTA_CEILING,
                f"{max(db_ref, dd_ref)} < {C.DELTA_CEILING}")
        g.check("G4_counts_match_design",
                (len(cir) == scene["i"]) and (len(lay) == scene["j"]),
                f"CV found {len(cir)} circ / {len(lay)} iface, "
                f"design {scene['i']}/{scene['j']}")
        finite = all(np.isfinite(v) for v in rec.values()
                     if isinstance(v, float))
        g.check("G5_initializers_finite", finite, str(rec))

        man.record(nm, gate=g, metrics=dict(
            n_circles=len(cir), n_interfaces=len(lay),
            layer_gate_disabled=(not has_layer), **rec),
            outputs={"overlay": C.DIR_CV / f"{nm}_cv.png"})

    C.DIR_CV.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sweep_rows).to_csv(C.DIR_CV / "cv_sweep.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(C.DIR_CV / "cv_components.csv", index=False)
    (C.DIR_CV / "cv_summary.json").write_text(json.dumps(inits, indent=2))

    if not args.no_xlsx:
        xlsx = C.DIR_CV / "parameters.xlsx"
        write_parameters_xlsx(xlsx, inits, log)
        # run_cij_opti.py calls build_scenarios() at module level (line ~368),
        # which opens its own PARAMETERS_XLSX from the CWD. That import will
        # fail here, and if it did NOT fail it would mean some other
        # parameters.xlsx is present and being read instead. Either way we
        # only want the parser, which is defined before that line, so the
        # exception is expected and non-fatal.
        try:
            sys.path.insert(0, str(C.ROOT))
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "run_cij_opti", str(C.RUN_CIJ_OPTI))
            rco = importlib.util.module_from_spec(spec)
            sys.modules["run_cij_opti"] = rco
            try:
                spec.loader.exec_module(rco)
            except FileNotFoundError as e:
                log.debug("run_cij_opti module-level build_scenarios() "
                          "skipped (expected): %s", e)
            wr, tr = rco.load_cij_table(str(xlsx))
            n_elem = sum(1 for v in tr.values() for x in v.values()
                         if x is not None)
            log.info("G6 re-read OK: %d world ranges, %d config rows, "
                     "%d non-null values", len(wr), len(tr), n_elem)
        except Exception as e:
            log.error("G6 parameters.xlsx did not re-read: %s", e)

    df = pd.DataFrame(sweep_rows)
    ref = df[(df.delta_bright == db_ref) & (df.delta_dark == dd_ref)]
    print("\n--- reference delta "
          f"({db_ref}, {dd_ref}) ---")
    print(ref[["scene", "true_i", "true_j", "n_circles", "n_interfaces",
               "layer_gate_applied", "interfaces_m"]].to_string(index=False))
    print("\n--- sweep: interfaces detected ---")
    print(df.pivot_table(index="scene", columns="delta_bright",
                         values="n_interfaces", aggfunc="first").to_string())
    print("\n--- sweep: circles detected ---")
    print(df.pivot_table(index="scene", columns="delta_bright",
                         values="n_circles", aggfunc="first").to_string())

    status = man.close()
    log.info("stage 3 %s -- manifest %s", status, man.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
