"""All declared benefit/compatibility accuracy contrasts, both fixed tasks."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style import COLORS as C, WIDTH, apply_style, panel, save

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'repro/results/CPUCONT'


def main():
    apply_style()
    summary_path = BASE / 'compiled_interface_summary.json'
    summary = json.loads(summary_path.read_text())
    paths = sorted((BASE / 'compiled_confirmation').glob('t???.json'))
    assert len(paths) == 20
    records = [json.loads(path.read_text()) for path in paths]
    contrasts = [('ordinary_auxiliary_gain', 'J', 'D', C['current']),
                 ('compiled_auxiliary_gain', 'CJ', 'CD', C['purple']),
                 ('compiled_total_gain', 'CJ', 'D', C['repair']),
                 ('compiled_vs_joint', 'CJ', 'J', C['amber'])]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.65))
    fig.subplots_adjust(left=.105, right=.975, top=.84, bottom=.22, wspace=.38)
    for ax, domain, letter, label in zip(axes, ['radioml', 'har'], ['a', 'b'],
                                        ['RadioML SNR', 'HAR reconstruction']):
        panel(ax, letter, label)
        rows = [record for record in records if record['domain'] == domain]
        def accuracy(record, mode):
            return next(row['test']['own']['acc'] for row in record['branches'] if row['mode'] == mode)
        limits = [-.5, 0.]
        for index, (name, left, right, color) in enumerate(contrasts):
            values = np.array([100*(accuracy(record, left)-accuracy(record, right)) for record in rows])
            estimate = summary['domains'][domain]['contrasts'][name]
            mean = 100*estimate['mean']
            lo, hi = np.array(estimate['mean_ci95'])*100
            y = 3-index
            ax.scatter(values, y+np.linspace(-.11, .11, len(values)), s=8, color=color, alpha=.35, linewidths=0)
            ax.errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], fmt='D', markersize=3.8,
                        color=color, capsize=2, elinewidth=1.15, zorder=6)
            limits.extend([lo, hi, *values])
        ax.axvline(0, color=C['reference'], lw=.75, ls='--')
        ax.plot([-.5, -.5], [-.22, .22], color=C['amber'], lw=.9, ls=(0, (1, 2)))
        span = max(limits)-min(limits)
        ax.set(xlim=(min(limits)-.12*span, max(limits)+.12*span), ylim=(-.45, 3.45),
               yticks=[3, 2, 1, 0], yticklabels=['J - D', 'CJ - CD', 'CJ - D', 'CJ - J'],
               xlabel='Primary-accuracy difference (pp)')
        ax.grid(axis='x', alpha=.65)
        ax.tick_params(axis='y', length=0)
    fig.text(.5, .035, 'Small points: all 10 seeds per task; diamonds / bars: paired means / 95% CI',
             ha='center', fontsize=7.1)
    result = save(fig, ROOT / 'paper/figures/fig_compiled_frontier')
    manifest = dict(figure=result, input_sha256={str(path.relative_to(ROOT)):
        hashlib.sha256(path.read_bytes()).hexdigest() for path in [summary_path, *paths]},
        complete_confirmation_seeds_per_task=10, selected_seeds_or_endpoints=False)
    (ROOT / 'paper/figures/compiled_figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n',
                                                                    encoding='utf-8', newline='\n')
    plt.close(fig)
    return result


if __name__ == '__main__':
    main()
