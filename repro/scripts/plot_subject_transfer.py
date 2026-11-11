"""Paired forecast errors and cross-subject auxiliary gains at print size."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style import COLORS as C, apply_style, panel, save

ROOT = Path(__file__).resolve().parents[2]


def read(name):
    path = ROOT / f'repro/results/CPUCONT/{name}_20261111.json'
    return json.loads(path.read_text())


def main():
    risk = read('har_transfer_confirmation')
    graph = read('har_graph_confirmation')
    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.33))
    fig.subplots_adjust(left=.17, right=.975, bottom=.19, top=.86, wspace=.38)
    for i, (key, color) in enumerate((('forecast', C['repair']), ('schedule_only', C['current']),
                                      ('persistence', C['reference']))):
        result = risk['mae_pp'][key]
        mean, (low, high) = result['mean'], result['ci95']
        values = result['reference_values']
        y = i + np.linspace(-.145, .145, len(values))
        a.scatter(values, y, s=10, color=color, alpha=.44, edgecolors='none')
        a.errorbar(mean, i, xerr=[[mean - low], [high - mean]], fmt='D', markersize=4,
                   color=color, markeredgecolor='white', markeredgewidth=.55,
                   linewidth=1.15, capsize=2.4, zorder=4)
    a.set(yticks=range(3), yticklabels=('Trajectory', 'Schedule', 'Persistence'), ylim=(2.47, -.47),
          xlim=(0, 19), xticks=(0, 5, 10, 15), xlabel='Risk forecast MAE (pp)')
    a.grid(axis='x')
    panel(a, 'a', 'Early forecast error')

    subjects = sorted(graph['subject_gains_pp'], key=int)
    y = np.arange(len(subjects))
    control = [graph['subject_gains_pp'][s]['control/0.0'] for s in subjects]
    warm = [graph['subject_gains_pp'][s]['warm/0.1'] for s in subjects]
    b.axvline(0, lw=.8, color=C['reference'], zorder=1)
    for j, (x0, x1) in enumerate(zip(control, warm)):
        b.plot([x0, x1], [j, j], lw=.65, color=C['reference'], alpha=.6, zorder=2)
    b.scatter(control, y, s=16, color=C['repair'], linewidths=0, label='Supervised', zorder=3)
    b.scatter(warm, y, s=16, color=C['purple'], marker='s', linewidths=0, label='Warm teacher', zorder=3)
    b.set(yticks=y, yticklabels=subjects, ylim=(9.6, -.65), xlim=(-4.2, 4.2), xticks=(-4, -2, 0, 2, 4),
          xlabel='Graph accuracy gain (pp)', ylabel='Held-out subject')
    b.grid(axis='x')
    b.legend(loc='lower center', ncol=2, handletextpad=.15, columnspacing=.65, fontsize=7.1)
    panel(b, 'b', 'Auxiliary gain by subject')
    save(fig, ROOT / 'paper/figures/fig_subject_transfer')
    plt.close(fig)


if __name__ == '__main__':
    main()
