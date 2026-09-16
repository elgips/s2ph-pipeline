#!/usr/bin/env python3
"""
stage1_forward.py -- build the nine C_ij worlds and compute forward responses.

Outputs per scene, in runs/01_forward/:
    {scene}_clean.dat     noise-free apparent resistivities
    {scene}_noisy.dat     noiseLevel=0.5 (-> 0.5% floor), noiseAbs=1e-6
    {scene}_geom.json     the geometry actually built, for downstream GT
    {scene}_mesh.png      mesh/geometry render for eyeball checking

Gates (all must pass or the scene is marked FAILED):
    G1  the world builds and the parameter count matches config
    G2  clean forward returns finite, positive rhoa for every quadrupole
    G3  the median clean rhoa is within a physically sane band
    G4  noisy differs from clean by ~NOISE_RMS_EXPECT (catches a silent
        clean-for-noisy substitution in EITHER direction)
    G5  the two .dat files are written, non-empty, and re-readable

Run:
    python stage1_forward.py                 # all nine
    python stage1_forward.py --scenes C11 C22
    python stage1_forward.py --force         # ignore existing checkpoints
"""
import argparse
import json
import sys

import numpy as np

import config as C
from pipelib import Gate, Manifest, get_logger, checkpoint_exists

STAGE = "stage1_forward"


def _import_deps(log):
    """Imported lazily so --check-config works without pyGIMLi installed."""
    try:
        import pygimli as pg
        from pygimli.physics import ert
    except Exception as e:
        log.error("pyGIMLi import failed -- stage 1 cannot run: %s", e)
        raise
    sys.path.insert(0, str(C.ROOT))
    try:
        from Anandlyn_log import AnomalyWorld, Layer, CircleAnom
    except Exception as e:
        log.error("could not import Anandlyn_log from %s: %s", C.ROOT, e)
        raise
    return pg, ert, AnomalyWorld, Layer, CircleAnom


def build_world(scene, ert, AnomalyWorld, Layer, CircleAnom):
    """Construct the AnomalyWorld for one scene.

    varflags are NOT set here: they are a PWHG concern only, and the forward
    solve ignores them. Stage 4 sets them via run_cij_opti.py.
    """
    scheme = ert.createData(
        elecs=np.linspace(C.X_MIN, C.X_MAX, C.N_ELEC),
        schemeName=C.SCHEME_NAME)

    layers = [Layer(name=l["name"], y=l["y"], rho=l["rho"], _cnum=k + 1)
              for k, l in enumerate(scene["layers"])]
    circles = [CircleAnom(name=c["name"], x=c["x"], y=c["y"], r=c["r"],
                          rho=c["rho"], _c_num=k + 1)
               for k, c in enumerate(scene["circles"])]

    world = AnomalyWorld(
        _start=[C.X_MIN, C.Y_TOP], _end=[C.X_MAX, C.Y_BOT],
        _scheme=scheme, _layers=layers, _circles=circles,
        _rho_world=C.RHO_WORLD,
        _rho_world_varflag=True, _rho_world_varlim=list(C.RHO_WORLD_VARLIM))
    return world, scheme


def rel_rms(a, b):
    """RMS relative difference between two rhoa vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b) & (a > 0)
    if not m.any():
        return np.nan
    return float(np.sqrt(np.mean(((b[m] - a[m]) / a[m]) ** 2)))


def run_scene(scene, deps, log, manifest, force=False):
    pg, ert, AnomalyWorld, Layer, CircleAnom = deps
    name = scene["name"]
    g = Gate(STAGE, name, log)

    f_clean = C.DIR_FWD / f"{name}_clean.dat"
    f_noisy = C.DIR_FWD / f"{name}_noisy.dat"
    f_geom = C.DIR_FWD / f"{name}_geom.json"
    f_mesh = C.DIR_FWD / f"{name}_mesh.png"

    if all(checkpoint_exists(p, force) for p in (f_clean, f_noisy, f_geom)):
        log.info("%s: checkpoint present, skipping (use --force to redo)", name)
        manifest.record(name, outputs=dict(clean=f_clean, noisy=f_noisy,
                                           geom=f_geom, mesh=f_mesh),
                        status="OK")
        return True

    log.info("%s: i=%d j=%d, %d free params, contrasts %s",
             name, scene["i"], scene["j"], scene["n_params"],
             {k: round(v, 3) for k, v in scene["contrasts"].items()})

    # --- G1 build ---------------------------------------------------
    world, scheme = build_world(scene, ert, AnomalyWorld, Layer, CircleAnom)
    g.check("G1_world_built", world is not None)
    g.check("G1_n_layers", len(world.layers) == scene["j"],
            f"{len(world.layers)} vs {scene['j']}")
    g.check("G1_n_circles", len(world.Circles) == scene["i"],
            f"{len(world.Circles)} vs {scene['i']}")
    g.check("G1_n_quadrupoles", scheme.size() > 0, scheme.size())

    # geometry render, purely for eyeballing
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = world.show_mesh()
        (fig if hasattr(fig, "savefig") else plt.gcf()).savefig(
            f_mesh, dpi=110, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        log.warning("%s: mesh render failed (non-fatal): %s", name, e)

    # --- G2/G3 clean forward ---------------------------------------
    d_clean = world.get_forward_solution(_noise=False)
    rhoa_c = np.asarray(d_clean["rhoa"], float)
    g.check("G2_finite", np.all(np.isfinite(rhoa_c)),
            f"{np.sum(~np.isfinite(rhoa_c))} non-finite")
    g.check("G2_positive", np.all(rhoa_c > 0),
            f"min={np.nanmin(rhoa_c):.4g}")
    med = float(np.median(rhoa_c))
    g.close("G3_median_rhoa", med, 5.0, 2000.0, " ohm-m")

    # --- G4 noisy forward ------------------------------------------
    d_noisy = world.get_forward_solution(
        _noise=True, _noise_level=C.NOISE_LEVEL,
        _noiseAbs=C.NOISE_ABS, _seed=C.FORWARD_SEED)
    rhoa_n = np.asarray(d_noisy["rhoa"], float)
    g.check("G4_noisy_finite", np.all(np.isfinite(rhoa_n)))
    rms = rel_rms(rhoa_c, rhoa_n)
    lo, hi = C.NOISE_RMS_TOL
    g.close("G4_noise_rms", rms, lo, hi)
    if np.isfinite(rms) and rms < 1e-6:
        g.check("G4_noise_actually_applied", False,
                "noisy is identical to clean -- noiseLevel not taking effect")

    # --- G5 write and re-read --------------------------------------
    C.DIR_FWD.mkdir(parents=True, exist_ok=True)
    d_clean.save(str(f_clean))
    d_noisy.save(str(f_noisy))
    for tag, path, ref in (("clean", f_clean, rhoa_c), ("noisy", f_noisy, rhoa_n)):
        try:
            back = ert.load(str(path))
            ok = back.size() == len(ref)
            g.check(f"G5_reread_{tag}", ok, f"{back.size()} vs {len(ref)}")
        except Exception as e:
            g.check(f"G5_reread_{tag}", False, str(e))

    f_geom.write_text(json.dumps(dict(
        scene=name, i=scene["i"], j=scene["j"],
        n_params=scene["n_params"], contrasts=scene["contrasts"],
        layers=scene["layers"], circles=scene["circles"],
        rho_world=C.RHO_WORLD,
        zones=describe_zones(scene["j"]),
        n_data=int(len(rhoa_c)),
        noise=dict(level=C.NOISE_LEVEL, abs=C.NOISE_ABS,
                   seed=C.FORWARD_SEED, realised_rel_rms=rms),
        rhoa=dict(median=med, min=float(np.min(rhoa_c)),
                  max=float(np.max(rhoa_c))),
    ), indent=2))

    manifest.record(name, gate=g,
                    outputs=dict(clean=f_clean, noisy=f_noisy,
                                 geom=f_geom, mesh=f_mesh),
                    metrics=dict(n_data=len(rhoa_c), rhoa_median=med,
                                 noise_rel_rms=rms))
    log.info("%s: %d data, median rhoa %.1f, noise rms %.4f -> %s",
             name, len(rhoa_c), med, rms, "OK" if g.ok else "FAILED")
    return g.ok


def describe_zones(j):
    if j == 0:
        return [dict(top=0.0, bot=C.Y_BOT, rho=C.RHO_WORLD)]
    if j == 1:
        return [dict(top=0.0, bot=C.L1["y"], rho=C.L1["rho"]),
                dict(top=C.L1["y"], bot=C.Y_BOT, rho=C.RHO_WORLD)]
    return [dict(top=0.0, bot=C.L1["y"], rho=C.L1["rho"]),
            dict(top=C.L1["y"], bot=C.L2["y"], rho=C.L2["rho"]),
            dict(top=C.L2["y"], bot=C.Y_BOT, rho=C.RHO_WORLD)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=C.SCENE_NAMES)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for d in C.ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    log = get_logger(STAGE, C.DIR_LOG)
    log.info("config:\n%s", C.summary())

    man = Manifest(STAGE, C.OUT)
    man.echo_config(noise_level=C.NOISE_LEVEL, noise_abs=C.NOISE_ABS,
                    seed=C.FORWARD_SEED, n_elec=C.N_ELEC,
                    scheme=C.SCHEME_NAME, L2_y=C.L2["y"])

    deps = _import_deps(log)
    for nm in args.scenes:
        if nm not in C.SCENE_BY_NAME:
            log.error("unknown scene %s", nm)
            continue
        try:
            run_scene(C.SCENE_BY_NAME[nm], deps, log, man, args.force)
        except Exception as e:
            log.exception("%s raised", nm)
            man.error(nm, e)

    status = man.close()
    log.info("stage 1 %s -- manifest %s", status, man.path)
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
