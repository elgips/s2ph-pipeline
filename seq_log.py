"""
seq_log.py -- shared, consistent state narration for the sequential-loop files.

Goal: every seq_* script should say, as it runs, WHAT stage it is in, WHAT it
just computed, and WHETHER that value is in the expected range -- so the run is
followable without reading the code. Import and use:

    from seq_log import Log
    log = Log("handoff")           # names the component
    log.stage("read CV seed")      # a numbered pipeline stage banner
    log.info("seed theta", theta)  # a labelled value (arrays summarized)
    log.check("whit ~ 1", whit, lo=0.5, hi=1.5)   # value + PASS/WARN verdict
    log.done("Sigma0 ready")       # closes a stage

Design: no dependencies beyond numpy; single-line, greppable, aligned output;
a global VERBOSE switch so the same scripts can run quiet inside the sweep.
"""
import time
import numpy as np

VERBOSE = True          # set False (or Log(quiet=True)) to silence in sweeps


def _fmt(v):
    """Compact one-line summary of a value (scalars, arrays, dicts)."""
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        av = abs(float(v))
        return f"{v:.4g}" if (av == 0 or 1e-3 <= av < 1e4) else f"{v:.3e}"
    a = np.asarray(v)
    if a.ndim == 0:
        return _fmt(a.item())
    if a.size <= 8:
        return "[" + " ".join(_fmt(x) for x in a.ravel()) + "]"
    return (f"shape{tuple(a.shape)} "
            f"min={_fmt(a.min())} max={_fmt(a.max())} "
            f"mean={_fmt(a.mean())}")


class Log:
    def __init__(self, component, quiet=False):
        self.c = component
        self.quiet = quiet
        self._stage = 0
        self._t0 = time.time()
        self._tstage = self._t0

    def _emit(self, kind, msg):
        if self.quiet or not VERBOSE:
            return
        dt = time.time() - self._t0
        print(f"[{self.c:<10}|{dt:6.1f}s|{kind:<5}] {msg}", flush=True)

    def stage(self, name):
        self._stage += 1
        self._tstage = time.time()
        self._emit("STAGE", f"{self._stage}. {name}")

    def info(self, label, value=None):
        self._emit("info", f"{label}: {_fmt(value)}" if value is not None else label)

    def check(self, label, value, lo=None, hi=None):
        """Print value with a PASS/WARN verdict against an expected range."""
        ok = True
        if lo is not None and value < lo:
            ok = False
        if hi is not None and value > hi:
            ok = False
        rng = ""
        if lo is not None or hi is not None:
            rng = f"  (expect [{'' if lo is None else _fmt(lo)},"\
                  f"{'' if hi is None else _fmt(hi)}])"
        verdict = "PASS" if ok else "WARN"
        self._emit(verdict, f"{label} = {_fmt(value)}{rng}")
        return ok

    def done(self, msg=""):
        dt = time.time() - self._tstage
        self._emit("done", f"{msg}  ({dt:.1f}s)")

    def warn(self, msg):
        self._emit("WARN", msg)
