#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seq_report.py -- persistence, single-run audit, and cross-run summary for the
sequential (seq_*) Stage-3/3b/4 runs: greedy acquisition, the prior-drawn
sweep, and (once added) the dynamic/tunnel-advancement runs.

WHY THIS EXISTS
----------------
seq_greedy.run_greedy() and seq_sweep.sweep()/run_one() return a rich result
dict (theta, Sigma, hist, gamma trajectory, per-draw errors...) but NEVER
WRITE IT TO DISK -- only printed to stdout. A finished run is therefore lost
the moment the terminal scrolls or the process exits. This module is the
missing save step, plus the audit/summary reporting that Article 1 had
(stage2b_contrast_audit.py: per-item PASS/WARN table against known tolerances;
q_scene_report.py / q_scene_cij.py: median[IQR] aggregation across many runs,
grouped "by type") but that never existed for the seq_* (Stage 3/3b/4) side of
the pipeline.

USAGE (three lines added to an existing seq_*.py script; see the bottom of
this file for exact patch locations in seq_greedy.py and seq_sweep.py):

    from seq_report import save_run, audit_run, summarize_runs

    res = run_greedy(theta_hat0, Sigma0, theta_true, pool, prior_stds, ...)
    path = save_run(res, kind="greedy_single", scene="c11",
                     data_forward="soft", theta_true=theta_true,
                     theta_hat0=theta_hat0, prior_stds=prior_stds)
    audit_run(path)                      # prints + writes the single-run audit

Later, to see everything that has accumulated under runs/06_sequential/:

    python seq_report.py summarize                 # all kinds
    python seq_report.py summarize --kind sweep_draw
    python seq_report.py audit runs/06_sequential/greedy_single/<run_id>.json

STORAGE LAYOUT
--------------
    runs/06_sequential/<kind>/<run_id>.json     one file per run, full record
    runs/06_sequential/index.csv                one row per run, append-only,
                                                 cheap to scan for summarize()
    runs/06_sequential/<kind>/<run_id>_audit.json   single-run audit (on demand)

<kind> is the caller-chosen run type, e.g. "greedy_single", "sweep_draw",
"dynamic_step" (one time-step of a tunnel-advancement run), "radius_sweep".
This is the "type" that summarize_runs() groups by, matching the "by
config"/"by cell" grouping in q_scene_report.py / seq_sweep.report().

Everything here is plain numpy + json + csv -- no pygimli import, so
audit_run()/summarize_runs() can be run standalone, later, on a different
machine than the one that produced the runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Paths -- mirrors config.py's runs/ layout (DIR_FWD=01_forward, ...
# DIR_CMP=05_compare) without importing config, so this module has zero
# hard dependency on the rest of the pipeline being importable.
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs" / "06_sequential"
INDEX_CSV = RUNS_DIR / "index.csv"

THETA_NAMES = ['world_log10rho', 'L0_y', 'L0_log10rho',
               'C0_x', 'C0_y', 'C0_log10r', 'C0_log10rho']
I_X, I_Y, I_R, I_RHO = 3, 4, 5, 6
GEOM = [I_X, I_Y, I_R, I_RHO]

# Tolerances already used inline in seq_greedy.run_greedy() /
# seq_sweep.run_one() -- centralised here so the audit and the original run
# can never silently drift apart.
TOL = dict(
    whit_lo=0.3, whit_hi=1.6,
    dy_hi=0.15,          # |depth error| (m)
    dx_hi=0.15,          # |lateral position error| (m), same bar as dy
    dlog10rho_hi=0.15,   # tight bar (matches seq_greedy's lg.check)
    dlog10rho_loose_hi=0.30,   # loose bar (matches seq_sweep's rho_ok)
    pos_rel_hi=0.50,     # |d(x,y)| / r_true  (matches seq_sweep's geom_ok)
    r_rel_hi=0.25,       # |dr| / r_true
)


# ----------------------------------------------------------------------
# JSON-safety
# ----------------------------------------------------------------------
def _jsonable(obj):
    """Recursively convert numpy scalars/arrays (and dict/list containers)
    into plain Python types that json.dump can handle."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ----------------------------------------------------------------------
# SAVE -- the missing step
# ----------------------------------------------------------------------
def save_run(result, kind, scene=None, data_forward=None, theta_true=None,
             theta_hat0=None, prior_stds=None, extra_meta=None,
             run_id=None, runs_dir=RUNS_DIR, keep_full_hist=True):
    """Persist one run of run_greedy() / run_one() / a dynamic-step driver.

    Parameters
    ----------
    result : dict
        Whatever run_greedy() (or an equivalent driver) returned: expected
        to carry at least `theta` (final estimate) and, when available,
        `Sigma`, `acquired`, `hist`, `gamma`/`gamma0`. Anything present is
        stored; nothing here is required beyond `theta`.
    kind : str
        Run type, e.g. "greedy_single", "sweep_draw", "dynamic_step",
        "radius_sweep". This is the grouping key for summarize_runs().
    scene : str, optional
        Scene name (e.g. "c11") if applicable.
    data_forward : str, optional
        "soft" or "remesh", per run_greedy's data_forward argument.
    theta_true, theta_hat0, prior_stds : array-like, optional
        Recorded alongside the result so the audit can recompute every
        error from scratch rather than trusting only what run_greedy()
        happened to print.
    extra_meta : dict, optional
        Anything run-specific: e.g. for a dynamic/tunnel run, the time
        step index, elapsed tunnel length, forgetting factor tau, drift
        rate, direction (perpendicular/parallel/oblique), etc. Stored
        verbatim under record["meta"].
    run_id : str, optional
        Defaults to "<kind>_<scene>_<UTC-compact-timestamp>_<8-hex>", unique
        and sortable by time.
    keep_full_hist : bool
        If False, only the LAST hist entry's scalar fields are kept (drops
        the per-step theta/Sigma trace) to keep file size down on very long
        sweeps. Default True -- these runs are cheap to store and the
        trajectory is exactly what the audit plots/inspects.

    Returns
    -------
    Path to the written JSON file.
    """
    runs_dir = Path(runs_dir)
    out_dir = runs_dir / kind
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if run_id is None:
        run_id = f"{kind}_{scene or 'na'}_{ts}_{uuid.uuid4().hex[:8]}"

    theta_final = np.asarray(result.get("theta"), float) if result.get("theta") is not None else None
    theta_true_a = np.asarray(theta_true, float) if theta_true is not None else None
    err = (theta_final - theta_true_a) if (theta_final is not None and theta_true_a is not None) else None

    hist = result.get("hist", [])
    if not keep_full_hist and hist:
        hist = [hist[-1]]

    record = dict(
        run_id=run_id, kind=kind, scene=scene, data_forward=data_forward,
        saved_at=ts,
        theta_true=theta_true_a, theta_hat0=theta_hat0, theta_final=theta_final,
        error=err, prior_stds=prior_stds,
        Sigma_final=result.get("Sigma"),
        acquired=result.get("acquired"),
        n_acquisitions=len(result.get("acquired", []) or []),
        gamma0=result.get("gamma0"), gamma_final=result.get("gamma"),
        # run_greedy() results carry whit only inside hist[-1]; run_drift()/
        # run_drift_multi() results (no hist list) carry it at the top level.
        whit_final=(hist[-1].get("whit") if hist else result.get("whit")),
        hist=hist,
        meta=extra_meta or {},
    )
    record = _jsonable(record)

    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2))

    _append_index_row(record, path, runs_dir)
    print(f"[seq_report] saved {path}")
    return path


def _flat_row_for_index(record, path):
    err = record.get("error")
    row = dict(
        run_id=record["run_id"], kind=record["kind"], scene=record.get("scene"),
        data_forward=record.get("data_forward"), saved_at=record.get("saved_at"),
        path=str(path),
        n_acquisitions=record.get("n_acquisitions"),
        gamma0=record.get("gamma0"), gamma_final=record.get("gamma_final"),
        whit_final=record.get("whit_final"),
    )
    if err is not None:
        row.update(dx=err[I_X], dy=err[I_Y],
                   dlog10r=err[I_R], dlog10rho=err[I_RHO])
        row["pos_err_m"] = float(np.hypot(err[I_X], err[I_Y]))
    # flatten scalar meta fields (e.g. dynamic-run step/tau/direction) so the
    # csv index doubles as the "by type" grouping key without re-reading json
    for k, v in (record.get("meta") or {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            row[f"meta_{k}"] = v
    return row


def _append_index_row(record, path, runs_dir):
    row = _flat_row_for_index(record, path)
    index_csv = Path(runs_dir) / "index.csv"
    is_new = not index_csv.exists()
    # union of columns seen so far, so a new meta_* field doesn't break the file
    existing_fields = []
    if not is_new:
        with open(index_csv, newline="") as f:
            existing_fields = next(csv.reader(f), [])
    fields = list(dict.fromkeys(existing_fields + list(row.keys())))
    if not is_new and fields != existing_fields:
        _rewrite_index_with_new_columns(index_csv, fields)
    with open(index_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def _rewrite_index_with_new_columns(index_csv, fields):
    with open(index_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(index_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


# ----------------------------------------------------------------------
# AUDIT -- single run, PASS/WARN table (article-1 style: stage2b_contrast_audit.py)
# ----------------------------------------------------------------------
def _check(name, ok, detail):
    return dict(name=name, ok=bool(ok), detail=detail)


def audit_run(path_or_record, write=True):
    """Single-run audit: PASS/WARN table against the tolerances already used
    inline in seq_greedy/seq_sweep (TOL above), plus the convergence
    trajectory and per-parameter error table.

    Accepts either a path to a saved run JSON or an already-loaded record
    dict (as returned by save_run's argument, i.e. NOT yet json-round-tripped
    -- both are handled by re-serialising through _jsonable first).
    """
    if isinstance(path_or_record, (str, Path)):
        path = Path(path_or_record)
        record = json.loads(path.read_text())
    else:
        path = None
        record = _jsonable(path_or_record)

    checks = []
    err = record.get("error")
    whit = record.get("whit_final")
    hist = record.get("hist") or []

    if whit is not None:
        checks.append(_check("whit ~ 1 (final)", TOL["whit_lo"] <= whit <= TOL["whit_hi"],
                              f"whit={whit:.3f} expect [{TOL['whit_lo']},{TOL['whit_hi']}]"))
    if err is not None:
        checks.append(_check("|dy| depth error", abs(err[I_Y]) <= TOL["dy_hi"],
                              f"dy={err[I_Y]:+.3f} m  |dy|<= {TOL['dy_hi']}"))
        checks.append(_check("|dx| lateral error", abs(err[I_X]) <= TOL["dx_hi"],
                              f"dx={err[I_X]:+.3f} m  |dx|<= {TOL['dx_hi']}"))
        checks.append(_check("|d log10 rho| contrast error",
                              abs(err[I_RHO]) <= TOL["dlog10rho_hi"],
                              f"dlog10rho={err[I_RHO]:+.3f}  <= {TOL['dlog10rho_hi']} "
                              f"(loose bar {TOL['dlog10rho_loose_hi']})"))
        r_true = 10 ** record["theta_true"][I_R] if record.get("theta_true") else None
        if r_true:
            r_final = 10 ** record["theta_final"][I_R]
            r_rel = abs(r_final - r_true) / r_true
            pos_rel = float(np.hypot(err[I_X], err[I_Y])) / r_true
            checks.append(_check("|dr|/r", r_rel <= TOL["r_rel_hi"],
                                  f"dr/r={r_rel:.3f} <= {TOL['r_rel_hi']}"))
            checks.append(_check("|d(x,y)|/r", pos_rel <= TOL["pos_rel_hi"],
                                  f"pos/r={pos_rel:.3f} <= {TOL['pos_rel_hi']}"))

    n_pass = sum(c["ok"] for c in checks)
    print(f"\n=== single-run audit: {record['run_id']} "
          f"({record['kind']}, scene={record.get('scene')}, "
          f"data_forward={record.get('data_forward')}) ===")
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'WARN'}] {c['name']:<28s} {c['detail']}")
    print(f"  {n_pass}/{len(checks)} checks passed  "
          f"n_acquisitions={record.get('n_acquisitions')}  "
          f"Gamma0={_fmt_sci(record.get('gamma0'))} -> "
          f"Gamma_final={_fmt_sci(record.get('gamma_final'))}")

    if err is not None:
        print("  per-parameter error (final - true):")
        for k, name in enumerate(THETA_NAMES):
            print(f"    {name:<16s} {err[k]:+.4f}")

    if len(hist) >= 2:
        gammas = [h.get("gamma") for h in hist if h.get("gamma") is not None]
        if gammas:
            print(f"  Gamma trajectory: {gammas[0]:.2e} -> {gammas[-1]:.2e} "
                  f"over {len(hist)} steps")

    audit = dict(run_id=record["run_id"], checks=checks,
                 n_pass=n_pass, n_checks=len(checks),
                 status=("OK" if n_pass == len(checks) else "WARN"))
    if write and path is not None:
        audit_path = path.parent / f"{path.stem}_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2))
        print(f"  wrote {audit_path}")
    return audit


def _fmt_sci(v):
    return "n/a" if v is None else f"{v:.2e}"


# ----------------------------------------------------------------------
# SUMMARY -- across runs, grouped by kind (article-1 style:
# q_scene_report.py / q_scene_cij.py's median[IQR]-by-config table)
# ----------------------------------------------------------------------
def _median_iqr(vals):
    a = np.asarray([v for v in vals if v not in (None, "")], float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    q1, q3 = np.percentile(a, [25, 75])
    return float(np.median(a)), float(q3 - q1)


def summarize_runs(runs_dir=RUNS_DIR, kind=None, group_by=("kind", "scene", "data_forward")):
    """Aggregate every run recorded in index.csv, grouped by `group_by`
    (default: kind x scene x data_forward -- "by type" as the user asked).
    Prints a median[IQR] table (robust, per Article 1's stated preference
    for these heavy-tailed error distributions) and writes summary.csv.
    """
    index_csv = Path(runs_dir) / "index.csv"
    if not index_csv.exists():
        print(f"[seq_report] no runs recorded yet at {index_csv}")
        return []

    with open(index_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if kind:
        rows = [r for r in rows if r.get("kind") == kind]
    if not rows:
        print(f"[seq_report] no runs matching kind={kind!r}")
        return []

    groups = {}
    for r in rows:
        key = tuple(r.get(g, "") for g in group_by)
        groups.setdefault(key, []).append(r)

    out_rows = []
    print(f"\n{'  '.join(group_by):<40s}{'n':>4s}{'nq_med':>8s}"
          f"{'|dxy|/r n/a':>12s}{'|dy| med[IQR]':>18s}{'|dlog10rho| med[IQR]':>22s}"
          f"{'whit med':>10s}")
    for key, grp in sorted(groups.items()):
        label = "/".join(str(k) for k in key)
        nq_med, _ = _median_iqr([g.get("n_acquisitions") for g in grp])
        dy_med, dy_iqr = _median_iqr([abs(float(g["dy"])) if g.get("dy") not in (None, "") else None
                                       for g in grp])
        dr_med, dr_iqr = _median_iqr([abs(float(g["dlog10rho"])) if g.get("dlog10rho") not in (None, "") else None
                                       for g in grp])
        whit_med, _ = _median_iqr([g.get("whit_final") for g in grp])
        print(f"{label:<40s}{len(grp):>4d}{nq_med:>8.0f}"
              f"{'':>12s}{dy_med:>7.3f}[{dy_iqr:.3f}]{'':>4s}"
              f"{dr_med:>10.3f}[{dr_iqr:.3f}]{'':>7s}{whit_med:>10.2f}")
        out_rows.append(dict(zip(group_by, key), n=len(grp),
                             n_acquisitions_median=nq_med,
                             abs_dy_median=dy_med, abs_dy_iqr=dy_iqr,
                             abs_dlog10rho_median=dr_med, abs_dlog10rho_iqr=dr_iqr,
                             whit_median=whit_med))

    out_path = Path(runs_dir) / "summary.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out_path}")
    return out_rows


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("audit", help="single-run audit from a saved run JSON")
    sp.add_argument("path")

    sp = sub.add_parser("summarize", help="cross-run summary, grouped by type")
    sp.add_argument("--kind", default=None)
    sp.add_argument("--group-by", nargs="+", default=["kind", "scene", "data_forward"])

    args = ap.parse_args()
    if args.cmd == "audit":
        audit_run(args.path)
    elif args.cmd == "summarize":
        summarize_runs(kind=args.kind, group_by=tuple(args.group_by))


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# PATCH LOCATIONS (apply by hand -- three lines each, not auto-applied here
# since this environment does not run the ERT_GUI/pygimli stack)
# ----------------------------------------------------------------------
#
# seq_greedy.py, end of __main__ (currently just calls run_greedy and returns):
#
#     from seq_report import save_run, audit_run
#     res = run_greedy(theta_hat0, Sigma0, theta_true, pool, prior_stds,
#                       data_forward="soft", n_max_steps=30, gamma_target=0.10,
#                       min_steps=6)
#     path = save_run(res, kind="greedy_single", scene=scene, data_forward="soft",
#                      theta_true=theta_true, theta_hat0=theta_hat0,
#                      prior_stds=prior_stds)
#     audit_run(path)
#
# seq_sweep.py, inside run_one() right after `res = run_greedy(...)`:
#
#     from seq_report import save_run
#     save_run(res, kind="sweep_draw", scene=None, data_forward=data_forward,
#              theta_true=theta_true, prior_stds=prior_stds,
#              extra_meta=dict(seed_kind="cv_like"))
#
#   (run_one()'s own return dict is unaffected; sweep()/report() keep working
#   exactly as before -- this only adds the missing disk write alongside it.
#   Afterwards: `python seq_report.py summarize --kind sweep_draw`.)
#
# A future dynamic/tunnel-advancement driver (see the Article-3 3D scenario
# work) should call save_run(..., kind="dynamic_step", extra_meta=dict(
#     direction=..., step=t, tunnel_length_m=..., tau_forget=...)) once per
# time step, so summarize_runs(group_by=("kind","meta_direction")) reports
# convergence quality by advancement direction.
