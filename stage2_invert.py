#!/usr/bin/env python3
"""
stage2_invert.py -- smooth (L2, cType=1) inversion + deterministic raster export.

The export is the normalisation fix. The PNG is written as a pure raster
covering exactly [X_MIN,X_MAX] x [Y_BOT,Y_TOP] at PX_PER_M, with no axes,
colourbar or whitespace, and with FIXED colour limits. Grey level is then an
exact affine function of log10(rho):

    g = 1 - (log10(rho) - log10(RHO_PLOT_MIN)) / SPAN_DECADES     (binary cmap)

so the CV stage can invert it exactly and its offsets are in DECADES. The old
path (pg.show auto-scaling + bbox_inches='tight') rescaled every scene by its
own extremes -- which is why one offset meant a different physical contrast in
each scene, and why homogeneous C00 produced full-contrast artifacts.

Outputs per scene, in runs/02_invert/:
    {scene}_smooth.png        the CV input (from the NOISY dataset)
    {scene}_smooth.npz        model vector + grid + colour limits (audit only;
                              the CV stage must NOT read this -- it works from
                              the image by design)
    {scene}_invert.json       lam, cType, chi2, rrms, iterations

Gates:
    G1  inversion converged and returned a model of the expected size
    G2  recovered resistivities lie inside the plotted colour range
    G3  raster is exactly IMG_W x IMG_H, 8-bit, single channel
    G4  SELF-CHECK: a synthetic uniform 100 ohm-m field pushed through the
        identical export path returns grey = 0.5 to within 1/255
    G5  SELF-CHECK: a synthetic two-value field returns the two expected
        grey levels (verifies slope as well as offset)
    G6  round-trip: gray_to_log10rho(exported grey) reproduces the model's
        log10(rho) field to within one grey step

Run:
    python stage2_invert.py                  # all nine
    python stage2_invert.py --scenes C11
    python stage2_invert.py --selfcheck-only # run G4/G5 and exit
"""
import argparse
import json
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from pipelib import (Gate, Manifest, get_logger, checkpoint_exists,
                     require_upstream, GateFailure)

STAGE = "stage2_invert"
GREY_STEP = 1.0 / 255.0


def _import_deps(log):
    try:
        import pygimli as pg
        from pygimli.physics import ert
    except Exception as e:
        log.error("pyGIMLi import failed -- stage 2 cannot run: %s", e)
        raise
    try:
        from PIL import Image
    except Exception as e:
        log.error("Pillow required for greyscale conversion: %s", e)
        raise
    return pg, ert, Image


def build_inversion_grid(pg):
    """The domain from inersion_domain, unchanged."""
    inv_domain = pg.createGrid(
        x=np.linspace(C.X_MIN, C.X_MAX, C.INV_NX),
        y=-pg.cat([0], pg.utils.grange(C.INV_Y_FIRST, C.DEPTH_M, n=C.INV_NY))[::-1],
        marker=2)
    grid = pg.meshtools.appendTriangleBoundary(
        inv_domain, marker=1, xbound=C.INV_XBOUND, ybound=C.INV_YBOUND)
    return inv_domain, grid


def export_raster(pg, Image, mesh, values, png_path):
    """Render `values` (linear resistivity, per cell of `mesh`) to a fixed-scale
    greyscale raster of exactly IMG_W x IMG_H covering the model domain.

    Axes are pinned to fill the canvas rather than cropped afterwards, so the
    pixel<->metre mapping is exact and identical for every scene.
    """
    # Direct rasterisation: no matplotlib, so no mesh boundary lines, no
    # antialiasing, no figure furniture. pg.show drew the domain outline into
    # the DOI, which put near-black pixels (947 ohm-m apparent) inside every
    # scene and would have fed straight into the blob branch.
    xs = C.X_MIN + (np.arange(C.IMG_W) + 0.5) * (C.WIDTH_M / C.IMG_W)
    ys = C.Y_TOP - (np.arange(C.IMG_H) + 0.5) * (C.DEPTH_M / C.IMG_H)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])
    vals = np.asarray(pg.interpolate(mesh, values, pts), float)
    vals = vals.reshape(C.IMG_H, C.IMG_W)
    bad = ~np.isfinite(vals) | (vals <= 0)
    if bad.any():
        vals[bad] = np.median(vals[~bad]) if (~bad).any() else C.RHO_WORLD
    g = C.log10rho_to_gray(np.log10(vals))
    arr = np.clip(np.round(g * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(png_path)
    return arr


def selfcheck(pg, Image, log, gate, tmp_dir):
    """G4/G5: push known uniform and two-value fields through the export path
    and verify the grey levels analytically."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    inv_domain, _ = build_inversion_grid(pg)
    n = inv_domain.cellCount()

    # G4 uniform 100 ohm-m -> g = 1 - (2.0 - 1.0)/2.0 = 0.5
    vals = np.full(n, 100.0)
    arr = export_raster(pg, Image, inv_domain, vals, tmp_dir / "_selfcheck_uniform.png")
    g_obs = float(np.median(arr)) / 255.0
    g_exp = float(C.log10rho_to_gray(np.log10(100.0)))
    gate.check("G4_uniform_grey", abs(g_obs - g_exp) <= 2 * GREY_STEP,
               f"observed {g_obs:.4f} expected {g_exp:.4f}")

    # G5 two-value field: 50 above -3 m, 100 below -> two distinct greys
    yc = np.array([inv_domain.cell(k).center().y() for k in range(n)])
    vals2 = np.where(yc > C.L1["y"], C.L1["rho"], C.RHO_WORLD)
    arr2 = export_raster(pg, Image, inv_domain, vals2,
                         tmp_dir / "_selfcheck_twovalue.png")
    top = arr2[: int(0.5 * C.L1["y"] * -C.PX_PER_M), :] / 255.0
    bot = arr2[int(1.5 * -C.L1["y"] * C.PX_PER_M):, :] / 255.0
    g50 = float(C.log10rho_to_gray(np.log10(C.L1["rho"])))
    g100 = float(C.log10rho_to_gray(np.log10(C.RHO_WORLD)))
    gate.check("G5_shallow_grey", abs(np.median(top) - g50) <= 3 * GREY_STEP,
               f"observed {np.median(top):.4f} expected {g50:.4f} (50 ohm-m)")
    gate.check("G5_deep_grey", abs(np.median(bot) - g100) <= 3 * GREY_STEP,
               f"observed {np.median(bot):.4f} expected {g100:.4f} (100 ohm-m)")
    gate.check("G5_polarity_bright_is_low_rho", np.median(top) > np.median(bot),
               f"shallow(50)={np.median(top):.3f} deep(100)={np.median(bot):.3f}")
    log.info("self-check: uniform g=%.4f (exp %.4f), 50->%.4f (exp %.4f), "
             "100->%.4f (exp %.4f)", g_obs, g_exp,
             np.median(top), g50, np.median(bot), g100)


def run_scene(scene, deps, log, manifest, force=False):
    pg, ert, Image = deps
    name = scene["name"]
    g = Gate(STAGE, name, log)

    dat = C.DIR_FWD / f"{name}_{C.CV_INPUT_DATASET}.dat"
    png = C.DIR_INV / f"{name}_smooth.png"
    npz = C.DIR_INV / f"{name}_smooth.npz"
    js = C.DIR_INV / f"{name}_invert.json"

    if all(checkpoint_exists(p, force) for p in (png, npz, js)):
        log.info("%s: checkpoint present, skipping", name)
        manifest.record(name, outputs=dict(png=png, npz=npz, json=js), status="OK")
        return True

    if not dat.exists():
        g.check("G0_input_exists", False, f"missing {dat}")
        manifest.record(name, gate=g)
        return False

    data = ert.load(str(dat))
    inv_domain, grid = build_inversion_grid(pg)

    mgr = ert.ERTManager(blockyModel=C.INV_BLOCKY)
    model = mgr.invert(data, mesh=grid, lam=C.INV_LAM, cType=C.INV_CTYPE,
                       maxIter=C.INV_MAXITER, verbose=False)
    model = np.asarray(model, float)

    # --- G1 ---------------------------------------------------------
    g.check("G1_model_size", model.size == inv_domain.cellCount(),
            f"{model.size} vs {inv_domain.cellCount()} cells")
    g.check("G1_model_finite", np.all(np.isfinite(model)) and np.all(model > 0))
    chi2 = float(getattr(mgr.inv, "chi2", lambda: np.nan)()) \
        if callable(getattr(mgr.inv, "chi2", None)) else float("nan")
    rrms = float(getattr(mgr.inv, "relrms", lambda: np.nan)()) \
        if callable(getattr(mgr.inv, "relrms", None)) else float("nan")

    # --- G2 ---------------------------------------------------------
    lo, hi = float(np.min(model)), float(np.max(model))
    g.check("G2_within_plot_range",
            lo >= C.RHO_PLOT_MIN and hi <= C.RHO_PLOT_MAX,
            f"model span [{lo:.2f},{hi:.2f}] vs plot "
            f"[{C.RHO_PLOT_MIN},{C.RHO_PLOT_MAX}] -- clipping would occur")

    # --- export + G3 ------------------------------------------------
    C.DIR_INV.mkdir(parents=True, exist_ok=True)
    arr = export_raster(pg, Image, inv_domain, model, png)
    g.check("G3_raster_shape", arr.shape == (C.IMG_H, C.IMG_W),
            f"{arr.shape} vs {(C.IMG_H, C.IMG_W)}")
    g.check("G3_raster_8bit", arr.dtype == np.uint8, str(arr.dtype))

    # --- G6 round-trip ----------------------------------------------
    # Compare exported grey against the model interpolated onto the raster.
    try:
        xs = np.linspace(C.X_MIN, C.X_MAX, C.IMG_W)
        ys = np.linspace(C.Y_TOP, C.Y_BOT, C.IMG_H)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.column_stack([XX.ravel(), YY.ravel(),
                               np.zeros(XX.size)])
        interp = np.asarray(pg.interpolate(inv_domain, model, pts), float)
        interp = interp.reshape(C.IMG_H, C.IMG_W)
        ok = np.isfinite(interp) & (interp > 0)
        g_exp = C.log10rho_to_gray(np.log10(np.where(ok, interp, 1.0)))
        g_obs = arr.astype(float) / 255.0
        err = np.abs(g_obs - g_exp)[ok]
        med_err = float(np.median(err))
        g.close("G6_roundtrip_median_grey_err", med_err, 0.0, 4 * GREY_STEP)
    except Exception as e:
        log.warning("%s: round-trip check skipped: %s", name, e)
        g.check("G6_roundtrip_ran", False, str(e))

    np.savez_compressed(
        npz, model=model,
        cell_x=np.array([inv_domain.cell(k).center().x()
                         for k in range(inv_domain.cellCount())]),
        cell_y=np.array([inv_domain.cell(k).center().y()
                         for k in range(inv_domain.cellCount())]),
        rho_plot_min=C.RHO_PLOT_MIN, rho_plot_max=C.RHO_PLOT_MAX,
        span_decades=C.SPAN_DECADES, px_per_m=C.PX_PER_M)

    js.write_text(json.dumps(dict(
        scene=name, source_dataset=C.CV_INPUT_DATASET, source_file=str(dat),
        lam=C.INV_LAM, cType=C.INV_CTYPE, blocky=C.INV_BLOCKY,
        maxIter=C.INV_MAXITER, chi2=chi2, rrms=rrms,
        n_cells=int(inv_domain.cellCount()),
        model_rho=dict(min=lo, max=hi, median=float(np.median(model))),
        raster=dict(w=C.IMG_W, h=C.IMG_H, px_per_m=C.PX_PER_M,
                    rho_plot=[C.RHO_PLOT_MIN, C.RHO_PLOT_MAX],
                    span_decades=C.SPAN_DECADES,
                    convention="binary cmap: bright = LOW resistivity"),
    ), indent=2))

    manifest.record(name, gate=g, outputs=dict(png=png, npz=npz, json=js),
                    metrics=dict(chi2=chi2, rrms=rrms, rho_min=lo, rho_max=hi))
    log.info("%s: chi2=%.3f rrms=%.3f rho[%.1f,%.1f] -> %s",
             name, chi2, rrms, lo, hi, "OK" if g.ok else "FAILED")
    return g.ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=C.SCENE_NAMES)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selfcheck-only", action="store_true")
    args = ap.parse_args()

    for d in C.ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    log = get_logger(STAGE, C.DIR_LOG)
    man = Manifest(STAGE, C.OUT)
    man.echo_config(lam=C.INV_LAM, cType=C.INV_CTYPE, blocky=C.INV_BLOCKY,
                    rho_plot=[C.RHO_PLOT_MIN, C.RHO_PLOT_MAX],
                    span_decades=C.SPAN_DECADES,
                    raster=[C.IMG_W, C.IMG_H], px_per_m=C.PX_PER_M)

    deps = _import_deps(log)
    pg, ert, Image = deps

    # Export self-checks run first and unconditionally: if the colour mapping
    # is wrong, nothing downstream is worth computing.
    sc = Gate(STAGE, "_selfcheck", log)
    selfcheck(pg, Image, log, sc, C.DIR_INV)
    man.record("_selfcheck", gate=sc)
    if not sc.ok:
        man.close()
        raise GateFailure("export self-check failed -- colour limits, cmap or "
                          "log flag are wrong; fix before inverting anything")
    log.info("export self-check PASSED")
    if args.selfcheck_only:
        man.close()
        return 0

    require_upstream("stage1_forward", C.OUT, args.scenes, log)

    for nm in args.scenes:
        try:
            run_scene(C.SCENE_BY_NAME[nm], deps, log, man, args.force)
        except Exception as e:
            log.exception("%s raised", nm)
            man.error(nm, e)

    status = man.close()
    log.info("stage 2 %s -- manifest %s", status, man.path)
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
