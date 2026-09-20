#!/usr/bin/env python
"""Merged RadioML capacity-law analysis: cap (alpha 1..1000) + cap2 (alpha 0.003..1) + cap3 (dense 0.1..0.8).

Per width m: pool (alpha, rel) points from the three sweeps, denser sweeps
taking precedence at shared alphas, then alpha_c via the paper's half-height
criterion (log-interp crossing to 0.5) with a healthy-anchor guard: a curve
whose shallowest point already sits below 0.6 is discarded rather than read off
the left edge of the grid.  The 0.6 margin is deliberately stricter than the
0.5 target -- a curve starting between 0.5 and 0.6 would put the crossing
within one grid step of that edge -- and on the shipped sweeps it never binds
(rel0 = 0.979, 1.023, 0.991, 0.957 for m = 32, 64, 128, 256).
gamma_RML = slope of log(alpha_c) vs log(m)
(alpha_c ~ m^gamma, wider m needs stronger aux -> gamma > 0), OLS + R^2.

The threshold is the absolute 0.5, which is the paper's rule for every real
domain: rel_scatter is normalised by the same seed's single-task scatter, so a
healthy cell sits at 1.0 and "half height" is 0.5.  Reading the threshold
relative to the curve's own first point instead (0.5 * rel0, which is 0.48-0.51
here) moves the crossings to {0.196, 0.378, 0.248, 0.251} and the fit to
gamma = 0.047 (R^2 = 0.024); that variant is not the one the paper reports.
The same rule on the same data is in repro/scripts/capacity_exponent.py.
"""

import os
R = os.environ.get("REPFIX_RESULTS",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "results"))

import json, glob, os
import numpy as np

BASE = os.path.join(R, 'E2c')

def load_records(path):
    d = json.load(open(path))
    if isinstance(d, dict):
        for k in ('records', 'results', 'data'):
            if k in d and isinstance(d[k], list):
                return d[k], d
        return [], d
    return d, {}

def collect_width_points():
    """returns {m: {alpha: (rel, [files])}} pooling cap3/cap2/cap sweeps"""
    widths = {}
    for tag in ('cap3', 'cap2', 'cap'):  # denser/more-seeded sweeps take precedence
        for f in sorted(glob.glob(os.path.join(BASE, f'E2c_radioml_{tag}_m*.json'))):
            recs, _ = load_records(f)
            m = int(os.path.basename(f).split('_m')[-1].split('.')[0])
            for r in recs:
                al = r.get('alpha'); rl = r.get('rel_scatter_mean', r.get('rel_scatter'))
                if al is None or rl is None:
                    continue
                widths.setdefault(m, {})
                if tag in ('cap3', 'cap2'):
                    widths[m][float(al)] = (float(rl), os.path.basename(f))
                else:
                    widths[m].setdefault(float(al), (float(rl), os.path.basename(f)))
    return widths

def half_height_alpha(alphas, rels):
    a = np.asarray(alphas, float); r = np.asarray(rels, float)
    order = np.argsort(a)
    a = a[order]; r = r[order]
    rel0 = float(r[0])
    if rel0 < 0.6:        # strict margin on the 0.5 target; see the module docstring
        return None, rel0, 'UNHEALTHY_ANCHOR'
    target = 0.5          # absolute: rel_scatter is normalised, healthy == 1.0
    for i in range(len(a) - 1):
        r1, r2 = r[i], r[i + 1]
        if r1 >= target >= r2:
            t = np.log10(a[i]) + (np.log10(a[i + 1]) - np.log10(a[i])) * (r1 - target) / (r1 - r2)
            return float(10 ** t), rel0, 'ok'
    return None, rel0, 'NO_CROSSING'

widths = collect_width_points()
summary = {}
for m in sorted(widths):
    pts = sorted(widths[m].items())
    alphas = [p[0] for p in pts]; rels = [p[1][0] for p in pts]
    src = sorted(set(p[1][1] for p in pts))
    ac, rel0, status = half_height_alpha(alphas, rels)
    print(f'--- width m={m}: n={len(alphas)} alpha_range=[{min(alphas):g},{max(alphas):g}] src={src}')
    print('    rel curve: ' + ' '.join(f'{a:g}:{r:.3f}' for a, r in zip(alphas, rels)))
    print(f'    rel0={rel0:.3f} status={status} alpha_c={ac}')
    summary[m] = (ac, rel0, status)

usable = sorted([(m, v[0]) for m, v in summary.items() if v[0] is not None and v[2] == 'ok'])
print('\n=== gamma_RML ===')
if len(usable) >= 2:
    lgm = np.log10([p[0] for p in usable])
    lga = np.log10([p[1] for p in usable])
    slope, intercept = np.polyfit(lgm, lga, 1)
    pred = slope * lgm + intercept
    ss_res = float(np.sum((lga - pred) ** 2))
    ss_tot = float(np.sum((lga - np.mean(lga)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print(f'points={usable}')
    print(f'gamma_RML = {slope:.3f}  (R^2={r2:.3f})')
    if len(usable) < 4:
        print(f'(provisional: {len(usable)}/4 widths)')
else:
    print(f'only {len(usable)} usable width(s); cap2 sweeps pending')
