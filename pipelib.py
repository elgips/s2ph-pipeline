"""
pipelib.py -- checkpointing, manifests, logging and validation gates.

Design rule: a stage may only be marked OK if its validate() passed. A stage
that writes files but fails validation is marked FAILED and the orchestrator
stops. Nothing downstream is allowed to consume an unvalidated artifact.
"""
import hashlib
import json
import logging
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def get_logger(stage, log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(stage)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
                            "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(Path(log_dir) / f"{stage}.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(sh)
    log.addHandler(fh)
    return log


def sha256(path, nbytes=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        data = f.read() if nbytes is None else f.read(nbytes)
        h.update(data)
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------
# Validation gates
# ----------------------------------------------------------------------
class GateFailure(Exception):
    pass


class Gate:
    """Collects named checks. Never raises mid-collection so that a run
    reports ALL failures rather than only the first."""

    def __init__(self, stage, scene=None, log=None):
        self.stage, self.scene, self.log = stage, scene, log
        self.checks = []

    def check(self, name, condition, detail=""):
        ok = bool(condition)
        self.checks.append(dict(name=name, ok=ok, detail=str(detail)))
        tag = f"{self.stage}" + (f"/{self.scene}" if self.scene else "")
        if self.log:
            (self.log.debug if ok else self.log.error)(
                f"[{tag}] {'PASS' if ok else 'FAIL'} {name} {detail}")
        return ok

    def close(self, name, value, lo, hi, unit=""):
        """Range check with the observed value always recorded."""
        ok = (value is not None) and np.isfinite(value) and (lo <= value <= hi)
        return self.check(name, ok, f"{value}{unit} expected [{lo},{hi}]{unit}")

    @property
    def failed(self):
        return [c for c in self.checks if not c["ok"]]

    @property
    def ok(self):
        return not self.failed

    def raise_if_failed(self):
        if self.failed:
            msg = "; ".join(f"{c['name']} ({c['detail']})" for c in self.failed)
            raise GateFailure(f"{self.stage}"
                              + (f"/{self.scene}" if self.scene else "")
                              + f": {len(self.failed)} gate(s) failed -> {msg}")


# ----------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------
class Manifest:
    """One JSON per stage. Records inputs, outputs, gates, timings, env."""

    def __init__(self, stage, out_dir):
        self.stage = stage
        self.path = Path(out_dir) / f"_manifest_{stage}.json"
        self.data = dict(
            stage=stage, status="RUNNING",
            started=datetime.now().isoformat(timespec="seconds"),
            finished=None, duration_s=None,
            env=dict(python=platform.python_version(),
                     numpy=np.__version__, platform=platform.platform()),
            config_echo={}, scenes={}, errors=[])
        self._t0 = time.time()

    def echo_config(self, **kw):
        self.data["config_echo"].update(kw)

    def scene(self, name):
        return self.data["scenes"].setdefault(
            name, dict(status="PENDING", outputs={}, gates=[], metrics={}))

    def record(self, name, gate=None, outputs=None, metrics=None, status=None):
        s = self.scene(name)
        if gate is not None:
            s["gates"] = gate.checks
            s["status"] = "OK" if gate.ok else "FAILED"
        if outputs:
            for k, p in outputs.items():
                p = Path(p)
                s["outputs"][k] = dict(
                    path=str(p),
                    exists=p.exists(),
                    bytes=p.stat().st_size if p.exists() else 0,
                    sha256_16=sha256(p) if p.exists() else None)
        if metrics:
            s["metrics"].update(_jsonable(metrics))
        if status:
            s["status"] = status

    def error(self, scene, exc):
        self.data["errors"].append(dict(
            scene=scene, type=type(exc).__name__, msg=str(exc),
            traceback=traceback.format_exc()))
        if scene:
            self.scene(scene)["status"] = "FAILED"

    def close(self):
        self.data["finished"] = datetime.now().isoformat(timespec="seconds")
        self.data["duration_s"] = round(time.time() - self._t0, 2)
        stats = [s["status"] for s in self.data["scenes"].values()]
        self.data["status"] = ("OK" if stats and all(s == "OK" for s in stats)
                               else "FAILED" if any(s == "FAILED" for s in stats)
                               else "EMPTY")
        self.data["summary"] = dict(
            n_scenes=len(stats), n_ok=stats.count("OK"),
            n_failed=stats.count("FAILED"), n_errors=len(self.data["errors"]))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))
        return self.data["status"]


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def load_manifest(stage, out_dir):
    p = Path(out_dir) / f"_manifest_{stage}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def require_upstream(stage, out_dir, scenes, log):
    """Refuse to run if the upstream stage did not finish OK for these scenes."""
    m = load_manifest(stage, out_dir)
    if m is None:
        raise GateFailure(f"upstream stage '{stage}' has no manifest in {out_dir}"
                          f" -- run it first")
    bad = [s for s in scenes
           if m["scenes"].get(s, {}).get("status") != "OK"]
    if bad:
        raise GateFailure(f"upstream stage '{stage}' not OK for: {bad}. "
                          f"Fix or re-run before continuing.")
    log.info(f"upstream '{stage}' OK for {len(scenes)} scene(s)")
    return m


def checkpoint_exists(path, force=False):
    """True if the artifact is present and we are not forcing recomputation."""
    return (not force) and Path(path).exists() and Path(path).stat().st_size > 0
