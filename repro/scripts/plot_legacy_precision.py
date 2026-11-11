"""Held-out legacy decisions and matched physical CPU timing."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style import COLORS as C, apply_style, panel, save

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'repro/results/CPUCONT'


def main():
    audit = json.loads((RESULTS / 'legacy_precision_audit_20261111.json').read_text())
    timing = json.loads((RESULTS / 'legacy_precision_timing_20261111.json').read_text())
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.7))
    fig.subplots_adjust(left=.14, right=.975, bottom=.2, top=.885, wspace=.48)

    ax = axes[0]
    ax.axvline(5, color=C['reference'], lw=.85, ls='--')
    for i, row in enumerate(audit['rows']):
        old = 100 * row['maximum_disagreement']['previous']
        new = 100 * row['maximum_disagreement']['prefix']
        ax.plot([old, new], [i, i], color=C['grid'], lw=.95)
        ax.scatter(old, i, s=18, color=C['reference'], label='Prior referral' if i == 0 else None)
        ax.scatter(new, i, s=18, color=C['repair'], marker='s', label='INT8 old block' if i == 0 else None)
    ax.set(yticks=range(6), yticklabels=[r['reference_seed'] for r in audit['rows']],
           xlim=(-.15, 5.35), ylim=(5.65, -.65), xlabel='Old decisions changed (%)')
    ax.grid(axis='x')
    ax.legend(loc='lower right', fontsize=7.1, handletextpad=.27)
    panel(ax, 'a', 'Held-out compatibility')

    ax = axes[1]
    keys = ('prefix_over_current', 'previous_over_current', 'exact_over_current')
    ax.axvline(1., color=C['reference'], ls='--', lw=.8)
    for y, (key, color) in enumerate(zip(keys, (C['repair'], C['reference'], C['current']))):
        row = timing['ratios'][key]
        low, high = row['ci95']
        mean = row['mean']
        ax.scatter(row['reference_values'], y + np.linspace(.12, .26, 6), s=15,
                   color=color, alpha=.55, linewidths=0, zorder=3)
        ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], color=color,
                    fmt='D', ms=4.2, capsize=2.2, lw=1.15, zorder=4)
    ax.set(yticks=range(3), yticklabels=['INT8', 'Prior', 'FP32'],
           ylim=(2.56, -.38), xlabel='Latency / current only')
    ax.set_xticks([1., 1.05, 1.10, 1.15, 1.20])
    ax.set_xlim(.988, 1.218)
    ax.grid(axis='x')
    panel(ax, 'b', 'Batch-128 CPU latency')
    save(fig, ROOT / 'paper/figures/fig_legacy_precision')
    plt.close(fig)


if __name__ == '__main__':
    main()
