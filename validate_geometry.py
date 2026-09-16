# -*- coding: utf-8 -*-
"""
validate_geometry.py
====================
Follow-up to validate_sensitivity.py, targeting the two things that run
exposed:

  (C) a robust, non-fatal version of the analytic rho-column check. The
      original crashed because pygimli's Jacobian is over the PARAMETER domain
      (fop.paraDomain()), whose cell count differs from world.mesh.cellCount()
      -- the refined H2 forward mesh made it worse. Fixed by masking para-domain
      cells, and wrapped so a pygimli quirk can't kill the harness.

  (G) check_forward_smoothness: the decisive test for the remesh problem. For
      each parameter it sweeps a small window, records the response AND
      world.mesh.cellCount() at every step, and flags where the response jumps
      against where the mesh retriangulates. Output: the largest jump-free
      window per parameter = the actual achievable central-FD step. Resistivity
      params should show ZERO remesh events (they don't change geometry) -- a
      built-in sanity check on the whole diagnosis.

Run in ERT_GUI. Needs pwhg_wrapper.py on the path.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point

import pwhg_wrapper as W


def _jac_to_numpy(jac):
    try:
        A = np.array(jac)
        if A.ndim == 2 and A.shape[0] > 0 and A.shape[1] > 0:
            return A
    except Exception:
        pass
    nr, nc = jac.rows(), jac.cols()
    A = np.zeros((nr, nc))
    for i in range(nr):
        A[i, :] = jac[i]
    return A


# ---------------------------------------------------------------------------
# C -- analytic rho check, paraDomain-aware and non-fatal
# ---------------------------------------------------------------------------
def check_rho_columns(world):
    """
    Compare FD d ln(rhoa)/d log10(rho_circle) against the analytic chain-rule
    column  ln(10) * rho_c * (1/rhoa) * sum_{c in region} Jcell[:,c], with region
    membership taken on the PARAMETER domain (aligned with Jacobian columns).

    Interpretation of the reported ratio = FD / analytic:
      mean ~ 1, CV ~ 0  -> column correct.
      mean != 1, CV ~ 0 -> constant geometric-factor / model-transform convention
                           (e.g. response is resistance not rhoa, or a log-model
                           Jacobian), NOT a sensitivity error.
      CV large           -> genuine per-datum disagreement, investigate.
    """
    try:
        import pygimli.physics.ert as ert
    except Exception as e:
        print("[C] skipped (import):", e)
        return None

    theta0 = np.asarray(world.get_x0(), float)
    labels = W.param_labels(world)
    steps = W.default_steps(world)
    Jfd, ln0, rhoa0, _ = W.jacobian(world, theta0, steps=steps)
    world.update(list(theta0))

    try:
        fop = ert.ERTModelling()
        fop.setData(world.scheme)
        fop.setMesh(world.mesh)
        res = np.asarray(world.resistivity_map, float)
        _ = np.asarray(fop.response(res), float)
        fop.createJacobian(res)
        Jcell = _jac_to_numpy(fop.jacobian())          # (n_data, n_para)
        para = fop.paraDomain()
        npara = para.cellCount()
        if Jcell.shape[1] != npara:
            print(f"[C] Jacobian cols ({Jcell.shape[1]}) != paraDomain cells "
                  f"({npara}); analytic check skipped.")
            return None
        centers = np.array([[para.cell(i).center().x(),
                             para.cell(i).center().y()] for i in range(npara)])
    except Exception as e:
        print("[C] pixel Jacobian failed:", e)
        return None

    out = []
    for j, (nm, kind) in enumerate(labels):
        if kind != "rho" or not nm.startswith("C"):
            continue
        ci = int(nm[1:])
        poly = world.Circles[ci].polygon
        mask = np.array([poly.contains(Point(x, y)) for (x, y) in centers])
        if mask.sum() == 0:
            print(f"[C] {nm}: no paraDomain cells inside polygon; skipped")
            continue
        rho_c = float(world.Circles[ci].rho)
        col_rhoa = np.log(10.0) * rho_c * Jcell[:, mask].sum(axis=1)
        col_ln = col_rhoa / rhoa0
        ratio = Jfd[:, j] / (col_ln + 1e-300)
        cv = np.std(ratio) / (abs(np.mean(ratio)) + 1e-300)
        out.append((nm, float(np.mean(ratio)), float(cv)))
        print(f"[C] rho {nm}: mean(FD/analytic) = {np.mean(ratio):.4f}, "
              f"CV = {cv:.2e}  ({int(mask.sum())} cells)")
    if not out:
        print("[C] no circle-rho columns to check.")
    return out


# ---------------------------------------------------------------------------
# G -- forward smoothness / remesh diagnostic  (the decisive one)
# ---------------------------------------------------------------------------
def check_forward_smoothness(world, out="forward_smoothness.png",
                             windows=None, n=41):
    """
    Sweep each parameter through +/- window around theta0. At each step record
    the full response and world.mesh.cellCount(). Remesh events = steps where the
    cell count changes; these are where the (piecewise-smooth) forward map can
    jump. The largest jump-free sub-window is the biggest central-FD step that
    stays on one smooth piece.

    windows: half-width per kind. Position/depth in metres, r/rho in log10.
    """
    theta0 = np.asarray(world.get_x0(), float)
    labels = W.param_labels(world)
    if windows is None:
        windows = {"x": 0.5, "y": 0.5, "r": 0.15, "rho": 0.05}

    fig, axes = plt.subplots(len(labels), 2,
                             figsize=(10, 2.3 * len(labels)), squeeze=False)
    report = {}

    for row, (nm, kind) in enumerate(labels):
        hw = windows[kind]
        svals = np.linspace(-hw, hw, n)
        resp = []
        ncell = []
        for s in svals:
            th = theta0.copy(); th[row] += s
            _, rhoa = W.forward(world, th)
            resp.append(rhoa)
            ncell.append(world.mesh.cellCount())
        world.update(list(theta0))
        resp = np.array(resp)
        ncell = np.array(ncell)

        pk = resp.max(0) - resp.min(0)
        d = int(np.argmax(pk))                         # most-perturbed datum
        events = np.where(np.diff(ncell) != 0)[0]

        epos = svals[events] if events.size else np.array([])
        bounds = np.concatenate(([svals[0]], epos, [svals[-1]]))
        max_gap = float(np.max(np.diff(bounds))) if bounds.size > 1 else 2 * hw
        report[f"{nm}/{kind}"] = dict(n_events=int(events.size),
                                      max_smooth_window=max_gap)

        axL, axR = axes[row][0], axes[row][1]
        axL.plot(svals, resp[:, d], ".-", ms=3)
        for e in events:
            axL.axvline(svals[e], color="C3", alpha=0.3, lw=0.7)
        axL.set_title(f"{nm} {kind}: datum {d}, {events.size} remesh events",
                      fontsize=9)
        axL.set_xlabel(f"delta {kind}"); axL.set_ylabel("rhoa")
        axR.plot(svals, ncell, ".-", ms=3)
        axR.set_title("mesh cell count", fontsize=9)
        axR.set_xlabel(f"delta {kind}"); axR.set_ylabel("cells")

    fig.suptitle("Forward smoothness under single-parameter perturbation",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)

    print("\n[G] forward smoothness:")
    for k, v in report.items():
        flag = "smooth-ish" if v["n_events"] <= 2 else "REMESH-LIMITED"
        print(f"    {k:<10} remesh events = {v['n_events']:<3} "
              f"max jump-free window = {v['max_smooth_window']:.3e}   {flag}")
    print(f"    figure: {out}")
    print("    Resistivity rows should show 0 events. If geometry rows show many "
          "events with a small jump-free window, no clean FD step exists and the "
          "forward needs a fixed topology (see notes).")
    return report


def main():
    from validate_sensitivity import build_c11_world
    world = build_c11_world()
    print("param order:", W.param_labels(world))
    check_rho_columns(world)
    check_forward_smoothness(world)


if __name__ == "__main__":
    main()
