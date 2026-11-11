"""Checkpoint-level replication, compatibility, and physical serving costs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from figure_style import COLORS as C, apply_style, panel, save

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'repro/results/CPUCONT'


def horizontal_interval(ax, y, statistic, color):
    values = statistic['values']
    ax.scatter(values, y + np.linspace(.10, .21, len(values)), s=14,
               color=color, alpha=.52, zorder=3, linewidths=0)
    mean = statistic['mean']
    ax.errorbar(mean, y, xerr=[[mean - statistic['lower']], [statistic['upper'] - mean]],
                color=color, fmt='D', ms=4.2, lw=1.25, capsize=2.4, zorder=4)


def plot(replication, timing, native, output):
    apply_style()
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 3.95))
    fig.subplots_adjust(left=.157, right=.976, top=.922, bottom=.18,
                        wspace=.39, hspace=.66)
    for ax, letter, title, key, baseline, unit in [
        (axs[0, 0], 'a', 'Primary benefit', 'gain_pp', 0, 'Accuracy gain (points)'),
        (axs[0, 1], 'b', 'Auxiliary progress', 'loss_ratio', 1, 'Contrastive loss / supervised'),
    ]:
        ax.axvline(baseline, color=C['reference'], ls='--', lw=.75)
        for i, (prefix, color) in enumerate([('original_matched_', C['current']), ('', C['purple'])]):
            horizontal_interval(ax, i, replication[prefix + key], color)
        ax.set(yticks=[0, 1], yticklabels=['TorchVision', 'AugReg'], ylim=(1.49, -.36), xlabel=unit)
        if letter == 'b':
            ax.tick_params(labelleft=False, left=False)
        ax.grid(axis='x')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        panel(ax, letter, title)

    ax = axs[1, 0]
    for j, key in enumerate(('classification', 'coverage80', 'coverage50')):
        for i, (role, color, marker) in enumerate([('direct', C['frozen'], 'o'),
                                                    ('protected', C['repair'], 's')]):
            values = [100 * row['disagreements'][role][key] for row in replication['rows']]
            ypos = j + (-.15 if i == 0 else .15)
            ax.scatter(values, ypos + np.linspace(-.06, .06, len(values)),
                       color=color, marker=marker, s=14, alpha=.55, linewidths=0, zorder=3,
                       label=('Unchanged head' if i == 0 else 'Retained FP32') if j == 0 else None)
    ax.set(yticks=range(3), yticklabels=['Class', 'Gate (80%)', 'Gate (50%)'],
           ylim=(2.51, -.47), xlabel='Old decisions changed (%)')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis='x')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.37), ncol=2,
              fontsize=7.2, handletextpad=.3, columnspacing=.75, borderaxespad=0)
    panel(ax, 'c', 'AugReg compatibility')

    ax = axs[1, 1]
    previous = native['cost']['1']['chosen']['ratio']
    batch_one = dict(mean=previous['mean'], lower=previous['ci95'][0], upper=previous['ci95'][1],
                     values=previous['reference_values'])
    for i, (stats, budget, color) in enumerate([
        (batch_one, 1.20, C['reference']),
        (timing['ratios']['autocast'], 1.10, C['repair']),
        (timing['ratios']['fp32'], 1.10, C['current']),
    ]):
        ax.plot([budget, budget], [i - .29, i + .32], color=C['reference'], ls='--', lw=.75)
        horizontal_interval(ax, i, stats, color)
    ax.axvline(1, color=C['grid'], lw=.8)
    ax.set(yticks=range(3), yticklabels=['1 · BF16', '128 · BF16', '128 · FP32'],
           ylim=(2.49, -.37), xlabel='Latency / current only')
    ax.set_xlim(left=min(ax.get_xlim()[0], .99))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis='x')
    panel(ax, 'd', 'TorchVision serving cost')
    save(fig, output)
    plt.close(fig)


def main():
    replication, timing, native = [json.loads((RESULTS / name).read_text()) for name in (
        'augreg_replication_20261111.json', 'matched_batch_latency_20261111.json',
        'contrastive_native_summary_20261111.json')]
    plot(replication, timing, native, ROOT / 'paper/figures/fig_contrastive_benefit')


if __name__ == '__main__':
    main()
