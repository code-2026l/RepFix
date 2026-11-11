"""Publication-width views of early warnings, query use, and paired outcomes."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

from figure_style import COLORS as C, apply_style, panel, save

ROOT = Path(__file__).resolve().parents[2]
SCHEDULES = ('stable', 'mild', 'nominal', 'strong')
STYLE = ((C['repair'], 'o'), (C['current'], 's'), (C['purple'], '^'), (C['frozen'], 'D'))


def main():
    source = ROOT / 'repro/results/CPUCONT/early_referral_confirmation_20261111.json'
    rows = json.loads(source.read_text())['episodes']
    apply_style()
    fig = plt.figure(figsize=(5.5, 4.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.2, 1))
    fig.subplots_adjust(left=.145, right=.973, bottom=.12, top=.94, wspace=.47, hspace=.66)

    ax = fig.add_subplot(grid[0, 0])
    limit = max(35, 100 * max(max(r['early_direct'], r['actual_direct']) for r in rows) + 2)
    ax.plot([0, limit], [0, limit], color=C['grid'], linewidth=.85)
    ax.axhline(5, color=C['reference'], ls='--', lw=.8)
    ax.axvline(5, color=C['reference'], ls='--', lw=.8)
    for name, (color, marker) in zip(SCHEDULES, STYLE):
        group = [r for r in rows if r['rate'] == name]
        ax.scatter([100 * r['early_direct'] for r in group], [100 * r['actual_direct'] for r in group],
                   s=19, color=color, marker=marker, alpha=.82, linewidths=0,
                   label=name.capitalize(), zorder=3)
    ax.set(xlim=(-.8, limit), ylim=(-.8, limit), xlabel='Epoch-2 disagreement (%)',
           ylabel='Epoch-8 disagreement (%)')
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.legend(loc='upper left', ncol=2, columnspacing=.6, handletextpad=.2, fontsize=7.2)
    panel(ax, 'a', 'Early warning')

    ax = fig.add_subplot(grid[0, 1])
    jitter = np.linspace(-.09, .09, 6)
    for i, name in enumerate(SCHEDULES):
        group = [r for r in rows if r['rate'] == name]
        for policy, delta, color, marker in [
            ('margin', -.17, C['repair'], 'o'), ('uniform', .17, C['reference'], 's')
        ]:
            values = [r['policies'][policy]['fraction'] for r in group]
            ax.scatter(values, i + delta + jitter, s=15, color=color, marker=marker, alpha=.62,
                       linewidths=0, label=policy.capitalize() if i == 0 else None, zorder=3)
    ax.set(yticks=range(4), yticklabels=[name.capitalize() for name in SCHEDULES],
           ylim=(3.52, -.42), xlim=(-.065, 1.07), xlabel='Examples sent to reference')
    ax.set_xticks([0, .25, .5, .75, 1.])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis='x')
    ax.legend(loc='upper right', ncol=2, fontsize=7.2, handletextpad=.25, columnspacing=.5)
    panel(ax, 'b', 'Reference calls')

    ax = fig.add_subplot(grid[1, :])
    ax.axvline(5, color=C['reference'], ls='--', lw=.85)
    for i, name in enumerate(SCHEDULES):
        group = [r for r in rows if r['rate'] == name]
        raw = [100 * r['actual_direct'] for r in group]
        fixed = [100 * r['policies']['margin']['maximum_disagreement'] for r in group]
        ypos = i + np.linspace(-.17, .17, len(group))
        for a, b, y in zip(raw, fixed, ypos):
            ax.plot([a, b], [y, y], color=C['reference'], lw=.63, alpha=.57, zorder=2)
        ax.scatter(raw, ypos, s=17, color=C['frozen'], alpha=.8, linewidths=0,
                   label='Unchanged head' if i == 0 else None, zorder=3)
        ax.scatter(fixed, ypos, s=16, color=C['repair'], marker='s', alpha=.8, linewidths=0,
                   label='Selected repair' if i == 0 else None, zorder=4)
    ax.set(yticks=range(4), yticklabels=[name.capitalize() for name in SCHEDULES],
           ylim=(3.55, -.45), xlim=(-.7, limit), xlabel='Epoch-8 disagreement (%)')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.grid(axis='x')
    ax.legend(loc='upper right', ncol=2, fontsize=7.2, handletextpad=.25, columnspacing=.7)
    panel(ax, 'c', 'Paired old-decision outcomes')
    save(fig, ROOT / 'paper/figures/fig_early_update')
    plt.close(fig)


if __name__ == '__main__':
    main()
