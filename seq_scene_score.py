"""
seq_scene_score.py -- score the sequential loop with ARTICLE 1's metric.

Instead of the ad-hoc criteria used earlier (pos/r, dr/r), use the published
Q_scene from scene_metric.py, generalized to arbitrary ground truth so it works
on prior-drawn theta:

    Q = sum_k w_k q_k / sum_k w_k
        q_circle = IoU(recovered disk, GT disk)
        q_layer  = 1 - min(1, |dy| / TOL)
        w_k      = mean over footprint of  S(x) * tanh(|dlog10 rho| / RHO0)

TWO USES, and the second is the interesting one:

  (a) SCORING   -- q_circle (IoU) replaces pos/r and dr/r with one scale-aware
      number, and keeps article 1 and article 3 metrically consistent.

  (b) PREDICTION -- w evaluated at the GROUND TRUTH object is an a-priori
      detectability score: it needs no inversion, only S(x) and the contrast.
      Plotting achieved Q against predicted w tests whether detectability is
      forecastable before acquiring anything.

MEASURED FACT worth knowing before interpreting w:
  RHO0 = 0.378 dec, so tanh saturates 2.3x faster than the 2-D cylinder
  polarization factor K = (u-1)/(u+1) (which corresponds to rho0 = 0.869).
  Across the sweep's contrast range (0.5-1.4 dec) the tanh factor only spans
  0.87-1.00, so w is DOMINATED BY S(x): the metric predicts detectability
  essentially from POSITION/DEPTH, not contrast. That is a testable claim.

Run standalone for a demo:  python seq_scene_score.py
"""
import os
import numpy as np

# scene_metric provides S(x), the weights, IoU and the published constants
import scene_metric as sm

I_X, I_Y, I_R, I_RHO = 3, 4, 5, 6      # indices in the 7-param theta

# ---------------------------------------------------------------------------
# CONTRAST SATURATION SCALE
# ---------------------------------------------------------------------------
# Article 1 used RHO0 = 0.378 dec, calibrated empirically so the C_1,1 contrast
# (1.0 dec) sat at the knee. Article 3 uses the PHYSICAL value instead:
#
#   for an infinite circular cylinder in a uniform transverse field the
#   polarization factor is
#       K = (rho1 - rho0)/(rho1 + rho0) = (10^d - 1)/(10^d + 1)
#         = tanh( d * ln10 / 2 )                      d = log10(rho1/rho0)
#   so  tanh(d / RHO0_PHYS)  with  RHO0_PHYS = 2/ln10 = 0.8686  IS the
#   polarization factor exactly.
#
# This makes the contrast term a physical amplitude factor rather than a
# heuristic squashing function. CAVEAT (state it when reporting): the identity
# holds in the isolated-body, uniform-field, far-field limit. check_saturation
# measured this configuration's rho-sensitivity decaying with slope ~-0.28
# rather than the -1 that limit implies, because the anomaly sits near the
# surface and near the layer -- so 0.869 is principled, not exact, for this
# geometry. Both values are exposed so results can be reported under each and
# the ordering checked for stability (scene_metric._robustness does this).
RHO0_PHYS = 2.0 / np.log(10.0)          # 0.8686 dec -- cylinder K factor
RHO0_ART1 = sm.RHO0                     # 0.378 dec  -- article 1, empirical
RHO0_DEFAULT = RHO0_PHYS                # article 3 default

_GRID = {"cache": None}


def grid():
    """Build (and cache) the pixel grid and sensitivity map S(x)."""
    if _GRID["cache"] is None:
        _GRID["cache"] = sm.build_sensitivity_grid()   # xs, ys, XX, YY, S
    return _GRID["cache"]


def theta_to_scene(theta):
    """7-param theta -> the dicts scene_metric works with (r linear)."""
    t = np.asarray(theta, float)
    return dict(world_log_rho=float(t[0]),
                layer_y=float(t[1]),
                layer_log_rho=float(t[2]),
                circles=[dict(x=float(t[I_X]), y=float(t[I_Y]),
                              r=float(10.0 ** t[I_R]),
                              log_rho=float(t[I_RHO]))])


def background_field(XX, YY, scene):
    return sm.background_log_rho_field(XX, YY, scene["world_log_rho"],
                                       scene["layer_y"], scene["layer_log_rho"])


def score_pair(theta_true, theta_est, rho0=None, tol=None, band=None):
    """Q and per-object detail for one (truth, estimate) pair.

    Generalized version of scene_metric.scene_score_for_run: the GT is taken
    from theta_true instead of the module-level C_1,1 constant, so this works
    for prior-drawn scenes.
    """
    rho0 = RHO0_DEFAULT if rho0 is None else rho0
    tol = sm.TOL if tol is None else tol
    band = sm.LAYERBAND if band is None else band
    xs, ys, XX, YY, S = grid()

    gt = theta_to_scene(theta_true)
    est = theta_to_scene(theta_est)
    bg_gt = background_field(XX, YY, gt)

    # --- layer object ---
    q_layer = 1.0 - min(1.0, abs(est["layer_y"] - gt["layer_y"]) / tol)
    w_layer = sm.layer_weight(XX, YY, S, gt["layer_y"], gt["world_log_rho"],
                              gt["layer_log_rho"], band, rho0)
    objs = [("layer", w_layer, q_layer)]

    # --- circle: IoU between recovered and GT disks ---
    gc, ec = gt["circles"][0], est["circles"][0]
    m_gt = sm.disk_mask(XX, YY, gc["x"], gc["y"], gc["r"])
    m_es = sm.disk_mask(XX, YY, ec["x"], ec["y"], ec["r"])
    q_circle = sm.iou(m_gt, m_es)
    w_circle = sm.circle_weight(XX, YY, S, bg_gt, ec["x"], ec["y"], ec["r"],
                                ec["log_rho"], rho0)
    if q_circle <= 0.0:          # no overlap -> treat as missed, weight from GT
        w_circle = sm.circle_weight(XX, YY, S, bg_gt, gc["x"], gc["y"], gc["r"],
                                    gc["log_rho"], rho0)
    objs.append(("circle1", w_circle, q_circle))

    wsum = sum(w for _, w, _ in objs)
    Q = sum(w * q for _, w, q in objs) / wsum if wsum > 0 else 0.0
    return float(Q), dict(Q=float(Q), q_layer=float(q_layer),
                          q_circle=float(q_circle),
                          w_layer=float(w_layer), w_circle=float(w_circle))


def predict_w(theta_true, rho0=None):
    """A-PRIORI detectability of the GT anomaly: w at the true object.

    No inversion involved -- only S(x) at the object's footprint and its
    contrast against the local background. Also returns the two factors
    separately so you can see which one is doing the work.
    """
    rho0 = RHO0_DEFAULT if rho0 is None else rho0
    xs, ys, XX, YY, S = grid()
    gt = theta_to_scene(theta_true)
    bg = background_field(XX, YY, gt)
    c = gt["circles"][0]
    m = sm.disk_mask(XX, YY, c["x"], c["y"], c["r"])
    if not m.any():
        return dict(w=0.0, S_mean=0.0, tanh_mean=0.0, contrast=0.0)
    contrast = np.abs(c["log_rho"] - bg[m])
    sat = np.tanh(contrast / rho0)
    return dict(w=float(np.mean(S[m] * sat)),
                S_mean=float(np.mean(S[m])),
                tanh_mean=float(np.mean(sat)),
                contrast=float(np.mean(contrast)))


def compare_rho0(theta_true, theta_est):
    """Report Q under BOTH saturation scales, so the choice is auditable and
    the conclusion's dependence on it is visible."""
    out = {}
    for name, r0 in (("article1_0.378", RHO0_ART1), ("physical_0.869", RHO0_PHYS)):
        Q, d = score_pair(theta_true, theta_est, rho0=r0)
        out[name] = dict(Q=Q, w_circle=d["w_circle"], q_circle=d["q_circle"])
    return out


def spread(values):
    """Article-1 'spread': coefficient of variation across runs (Q_CV)."""
    a = np.asarray(values, float)
    if a.size < 2:
        return dict(mean=float(a.mean()) if a.size else np.nan, std=0.0, cv=np.nan)
    m, s = float(a.mean()), float(a.std(ddof=1))
    return dict(mean=m, std=s, cv=(s / abs(m) if m != 0 else np.nan))


def _demo():
    print("constants from scene_metric: "
          f"RHO0={sm.RHO0} dec  TOL={sm.TOL} m  LAYERBAND={sm.LAYERBAND} m")
    print(f"rho0 matching the cylinder K factor would be {2/np.log(10):.4f} dec "
          f"-> article 1 saturates {(2/np.log(10))/sm.RHO0:.2f}x faster\n")

    th_true = np.array([2.0, -3.0, np.log10(50), 5.0, -4.0,
                        np.log10(1.2), np.log10(500)])

    # a-priori detectability vs depth: does w track S(x) alone?
    print("a-priori detectability w at the GT object, vs depth (r=1.2, rho=500):")
    print(f"{'depth':>7}{'S_mean':>12}{'tanh':>8}{'w':>12}{'w/w(-4)':>9}")
    ref = None
    for y in [-2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]:
        t = th_true.copy(); t[I_Y] = y
        p = predict_w(t)
        if ref is None or y == -4.0:
            ref = p["w"] if y == -4.0 else ref
        print(f"{y:>7.1f}{p['S_mean']:>12.4g}{p['tanh_mean']:>8.3f}"
              f"{p['w']:>12.4g}{(p['w']/ref if ref else np.nan):>9.2f}")

    # and vs radius at fixed depth
    print("\nsame, vs radius (depth -4, rho=500):")
    print(f"{'r':>7}{'S_mean':>12}{'tanh':>8}{'w':>12}")
    for r in [0.6, 0.9, 1.2, 1.8, 2.5]:
        t = th_true.copy(); t[I_R] = np.log10(r)
        p = predict_w(t)
        print(f"{r:>7.2f}{p['S_mean']:>12.4g}{p['tanh_mean']:>8.3f}{p['w']:>12.4g}")

    # scoring demo: a good and a bad estimate
    print("\nscoring (Q, IoU) for two estimates of the same truth:")
    good = th_true.copy(); good[I_X] += 0.05; good[I_R] += 0.01
    bad = th_true.copy(); bad[I_X] += 1.2; bad[I_R] += 0.30
    for name, est in (("near-exact", good), ("displaced+inflated", bad)):
        Q, d = score_pair(th_true, est)
        print(f"  {name:<20} Q={Q:.3f}  IoU={d['q_circle']:.3f}  "
              f"q_layer={d['q_layer']:.3f}")


if __name__ == "__main__":
    _demo()
