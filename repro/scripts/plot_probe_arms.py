"""Plot matched retrospective certificate ablations and exact-flux margins."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from certificate_matched_audit import auc

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'repro/results/PBA/PBA_har.json'
MATCHED = ROOT / 'repro/results/CPUCONT/certificate_matched.json'
OUT = ROOT / 'paper/figures/fig_probe_arms'

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
                     'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

summary = json.loads(MATCHED.read_text())['arms']
raw = json.loads(RAW.read_text())
points = [dict(p, B=r['audit_batch']) for r in raw['runs'] for p in r['points'] if not p.get('degenerate')]
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.65))
x = np.arange(3)
fixed = [summary[k]['fired'] for k in ['frozen_canonical_triangle', 'corrected_canonical_triangle', 'corrected_canonical_combined']]
optimized = [summary[k]['fired'] for k in ['frozen_optimized_triangle', 'corrected_optimized_triangle', 'corrected_optimized_combined']]
for offset, values, color, label in [(-.18, fixed, '#8C9BAB', 'canonical scale'),
                                      (.18, optimized, '#C44E52', 'optimized scale')]:
    bars = axes[0].bar(x + offset, values, width=.34, color=color, label=label)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 3, str(value), ha='center', fontsize=8)
axes[0].set_xticks(x, ['frozen\ntriangle', 'corrected\ntriangle', 'corrected\ncombined'])
axes[0].set_ylabel('certified checkpoints (of 405)')
axes[0].set_ylim(0, 180)
axes[0].set_title('(a) predictor, envelope, and scale separated')
axes[0].legend(loc='upper left', frameon=False)


def scores(kind):
    values, labels = [], []
    for p in points:
        alpha = p['alpha']
        actual = p['r'] - alpha * p['c']
        if kind == 'frozen':
            pred = 1 - alpha * p['frozen']['q']
            delta = p['frozen']['Delta']
        else:
            pred = p['d'] - alpha * p['a']
            if kind == 'corrected_optimized':
                delta = abs(p['d']) * p['corrected']['Delta_star_min']
            else:
                base = p['nu'] * p['n_p'] / p['B']
                delta = abs(actual / (p['k_tr'] * base) - pred)
        values.append(abs(pred) / max(delta, 1e-300))
        labels.append(pred * actual > 0)
    return np.array(values), np.array(labels)


for kind, color, label in [('frozen', '#4C72B0', 'frozen, canonical'),
                            ('corrected', '#55A868', 'corrected, canonical'),
                            ('corrected_optimized', '#C44E52', 'corrected, optimized')]:
    score, correct = scores(kind)
    order = np.argsort(-score, kind='stable')
    end = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(score) - 1]
    tpr = np.r_[0, np.cumsum(correct[order])[end] / correct.sum()]
    fpr = np.r_[0, np.cumsum(~correct[order])[end] / (~correct).sum()]
    axes[1].plot(fpr, tpr, color=color, lw=1.4, label=f'{label} ({auc(score, correct):.2f})')
axes[1].plot([0, 1], [0, 1], '--', color='#999999', lw=.8)
axes[1].set(xlabel='false-positive rate', ylabel='true-positive rate',
            title='(b) retrospective margin (uses exact flux)')
axes[1].legend(loc='lower right', frameon=False, fontsize=7)
fig.tight_layout(pad=.5, w_pad=1.1)
fig.savefig(OUT.with_suffix('.pdf'), bbox_inches='tight')
fig.savefig(OUT.with_suffix('.png'), dpi=220, bbox_inches='tight')
print('counts', fixed, optimized)
