"""Render the utility and legacy-risk trade-off in the matched SNR studies."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from figure_style import COLORS as C, apply_style, panel, save

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'repro/results/CPUCONT'


def read(study, stage):
    return json.loads((RESULTS / f'rml_{study}_{stage}_20261111.json').read_text())


def main():
    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.85))
    fig.subplots_adjust(left=.112, right=.978, bottom=.17, top=.77, wspace=.44)
    styles = {
        'side': ('o', C['reference'], 'Fixed features'),
        'dual': ('s', C['current'], 'Dual student'),
        'guard': ('^', C['purple'], 'Margin loss'),
        'exact': ('D', C['repair'], 'Exact reuse'),
    }
    seen = set()
    a.axvspan(0, 4, color=C['repair'], alpha=.06)
    a.axvline(4, lw=.65, color=C['reference'], linestyle='--')
    a.axhline(-.15, lw=.65, color=C['reference'], linestyle='--')
    for study in ('one_pass', 'anchored'):
        result = read(study, 'choice')
        for record in result['candidates']:
            family = record['name'].split('/')[0]
            marker, color, label = styles[family]
            x, y = 100 * record['max_disagreement'], record['gain_vs_joint_pp']
            selected = record['name'] == result['selected']
            if selected:
                a.scatter(x, y, s=50, marker='o', facecolor='none', edgecolor=color, lw=.85, zorder=3)
            a.scatter(x, y, s=21, marker=marker, color=color, edgecolors='white', linewidths=.35,
                      label=label if family not in seen else None, zorder=4)
            seen.add(family)
    a.set(xlim=(-1, 19), xticks=(0, 4, 8, 12, 16), xlabel='Worst old-rule disagreement (%)',
          ylabel='Accuracy vs. full joint (pp)')
    a.grid(axis='y')
    a.legend(loc='lower left', ncol=2, bbox_to_anchor=(-.02, 1.15), borderaxespad=0,
             columnspacing=.8, handletextpad=.3, fontsize=7.1)
    panel(a, 'a', 'Development candidates')

    b.axvspan(0, 5, color=C['repair'], alpha=.06)
    b.axvline(5, lw=.65, color=C['reference'], linestyle='--')
    b.axhline(0, lw=.6, color=C['reference'])
    for study in ('one_pass', 'anchored'):
        result = read(study, 'confirmation')
        name = result['selected']
        marker, color, label = styles[name.split('/')[0]]
        xs = [100 * max(row['disagreement'][name]) for row in result['references']]
        ys = [row['accuracy_pp'][name] - row['accuracy_pp']['detach'] for row in result['references']]
        b.scatter(xs, ys, s=20, marker=marker, color=color, alpha=.66, edgecolors='white', linewidths=.3,
                  label=label, zorder=3)
    b.set(xlim=(-.55, 8), xticks=(0, 2, 4, 6, 8), xlabel='Old-rule disagreement (%)',
          ylabel='Accuracy vs. detachment (pp)')
    b.grid(axis='y')
    b.legend(loc='lower left', ncol=2, bbox_to_anchor=(-.02, 1.15), borderaxespad=0,
             columnspacing=.8, handletextpad=.3, fontsize=7.1)
    panel(b, 'b', 'Independent confirmation')
    save(fig, ROOT / 'paper/figures/fig_rml_one_pass')
    plt.close(fig)


if __name__ == '__main__':
    main()
