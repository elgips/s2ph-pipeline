#!/usr/bin/env python
"""
wk_report.py — sensitivity-contrast weight w_k of recovered objects in the
over-specified C_2,1 runs, separating the true anomaly from the spurious one.

    python wk_report.py

Verifies the claim in Section~\\ref{sec:fieldtransfer} that a fabricated object
carries markedly lower w_k than a genuine one. Each recovered circle is matched
to the ground-truth anomaly by IoU; the better-matching circle is labelled TRUE
and the other SPURIOUS. Runs in which neither circle overlaps the truth are
reported separately, since there the labelling is not meaningful.

Writes wk_pairs.csv (one row per run) and prints the summary quoted in the text.
"""
import csv
import glob
import json
import numpy as np
import scene_metric as sm

GT = dict(x=5.0, y=-4.0, r=1.2)


def iou(x, y, r, x0=GT['x'], y0=GT['y'], r0=GT['r']):
    d = float(np.hypot(x - x0, y - y0))
    if d >= r + r0:
        return 0.0
    if d <= abs(r - r0):
        return (min(r, r0) ** 2) / (max(r, r0) ** 2)
    a1 = r * r * np.arccos((d * d + r * r - r0 * r0) / (2 * d * r))
    a2 = r0 * r0 * np.arccos((d * d + r0 * r0 - r * r) / (2 * d * r0))
    a3 = 0.5 * np.sqrt((-d + r + r0) * (d + r - r0) * (d - r + r0) * (d + r + r0))
    inter = a1 + a2 - a3
    return float(inter / (np.pi * r * r + np.pi * r0 * r0 - inter))


def main():
    xs, ys, XX, YY, S = sm.build_sensitivity_grid()

    rows = []
    for f in sorted(glob.glob('article_runs/two_anom_*/run*/result.json')):
        r = json.load(open(f))
        if r.get('status') != 'ok':
            continue
        g = dict(zip(r['param_labels'], r['x']))
        world = 10 ** g['world_log10rho']
        bg = sm.background_log_rho_field(XX, YY, g['world_log10rho'],
                                         g['layer1_y'], g['layer1_log10rho'])
        objs = []
        for p in ('C1', 'C2'):
            cx, cy = g[f'{p}_x'], g[f'{p}_y']
            cr = 10 ** g[f'{p}_log10r']
            lrho = g[f'{p}_log10rho']
            w = sm.circle_weight(XX, YY, S, bg, cx, cy, cr, lrho, sm.RHO0)
            objs.append(dict(name=p, w=float(w), iou=iou(cx, cy, cr),
                             x=cx, y=cy, r=cr, rho=10 ** lrho))
        objs.sort(key=lambda o: -o['iou'])
        best, other = objs
        rows.append(dict(experiment=r['experiment'], run=r['run'],
                         w_true=best['w'], iou_true=best['iou'],
                         w_spur=other['w'], iou_spur=other['iou'],
                         both_missed=int(best['iou'] <= 1e-9)))

    with open('wk_pairs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote wk_pairs.csv ({len(rows)} runs)')

    ok = [r for r in rows if not r['both_missed']]
    miss = [r for r in rows if r['both_missed']]
    wt = np.array([r['w_true'] for r in ok])
    ws = np.array([r['w_spur'] for r in ok])
    lower = int((ws < wt).sum())

    print(f'\nruns where the true anomaly was recovered by at least one circle: '
          f'{len(ok)} of {len(rows)}')
    print(f'  w_k, matched (true)   median {np.median(wt):.3f}  '
          f'IQR {np.percentile(wt,75)-np.percentile(wt,25):.3f}')
    print(f'  w_k, unmatched (spur) median {np.median(ws):.3f}  '
          f'IQR {np.percentile(ws,75)-np.percentile(ws,25):.3f}')
    print(f'  spurious weight lower in {lower}/{len(ok)} runs '
          f'({100*lower/len(ok):.0f}%)')
    if miss:
        print(f'\nruns where neither circle overlapped the truth: {len(miss)} '
              f'(labels not meaningful; excluded above)')

    print('\nby cell:')
    for e in sorted({r['experiment'] for r in ok}):
        v = [r for r in ok if r['experiment'] == e]
        a = np.array([[r['w_true'], r['w_spur']] for r in v])
        n_lower = int((a[:, 1] < a[:, 0]).sum())
        print(f'  {e:22s} n={len(v):2d}  true {np.median(a[:,0]):.3f}  '
              f'spur {np.median(a[:,1]):.3f}  lower in {n_lower}/{len(v)}')


if __name__ == '__main__':
    main()
