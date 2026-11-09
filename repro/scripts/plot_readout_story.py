"""Render the opening evidence and companion plots from complete archived records.

All accuracies in the opening panels use the full original test set. The
reconstruction alignment maps must match the independently archived consumer
bank exactly; no monitor/remainder result is substituted for a full-test score.
No training, test-label model selection or fitted performance curves.
"""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style import COLORS as C, WIDTH, apply_style, panel, save
from summarize_cpu_evidence import describe
from plot_paper_diagrams import consumer_interface
from spatial_diagrams import validate_layout

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'repro/results/CPUCONT'
OUT = ROOT / 'paper/figures'
SOURCES, AUDIT = {}, {}


def read(path):
    path = Path(path)
    blob = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(blob).hexdigest()
    return json.loads(blob)


def branch(row, mode, dose):
    return next(b for b in row['branches'] if b['mode'] == mode and b['alpha'] == dose)


def export(fig, name):
    AUDIT[name] = save(fig, OUT / name)
    plt.close(fig)


def overview(old, repairs, recon, mechanism):
    fig = plt.figure(figsize=(WIDTH, 4.0))
    a = fig.add_axes([.008,.54,.984,.415])
    consumer_interface(a)
    fig.text(.018,.98,'(a) Shared updates and retained interfaces',fontsize=8.8,
             fontfamily='Times New Roman',fontweight='bold',va='top')
    b = fig.add_axes([.105,.16,.365,.29])
    c = fig.add_axes([.605,.16,.365,.29])

    iso = [branch(r, 'joint', 3)['trajectory'][-1] for r in old]
    iso_rep = [branch(r, 'joint', 3) for r in repairs]
    rec = [branch(r, 'joint', 100) for r in recon]
    iso_values = np.array([[b['frozen']['acc'], b['own']['acc'],
                            p['repairs']['scalar_only']['acc']] for b, p in zip(iso, iso_rep)])
    rec_values = np.array([[b['repairs']['frozen']['acc'], b['archived_own']['acc'],
                            b['repairs']['paired_ridge']['acc']] for b in rec])
    healthy = [np.mean([r['healthy']['frozen']['acc'] for r in old]),
               np.mean([r['healthy']['acc'] for r in recon])]
    for ax, values, healthy_acc, letter, heading, labels in [
        (b, iso_values, healthy[0], 'b', 'Variance auxiliary', ['Frozen', 'Current', 'Scale']),
        (c, rec_values, healthy[1], 'c', 'Reconstruction', ['Frozen', 'Current', 'Linear*'])]:
        panel(ax, letter, heading)
        ax.axhline(healthy_acc, color=C['reference'], ls='--', lw=.8, zorder=0)
        for row in values:
            ax.plot([0, 1, 2], row, color='#D8DDE2', lw=.55, zorder=1)
        for j, color in enumerate([C['frozen'], C['current'], C['repair']]):
            jitter = np.linspace(-.08, .08, len(values))
            ax.scatter(j + jitter, values[:, j], s=10, color=color, alpha=.45, edgecolors='none', zorder=2)
            mean, sd = values[:, j].mean(), values[:, j].std()
            ax.errorbar(j, mean, yerr=sd, fmt='D', color=color, ms=4.2, capsize=2.4,
                        elinewidth=1.2, markeredgecolor='white', markeredgewidth=.5, zorder=3)
            ax.text(j, 1.055, f'{mean:.3f}', ha='center', va='top', fontsize=7.8, color=color)
        ax.set(xlim=(-.42, 2.42), ylim=(0, 1.075), xticks=[0, 1, 2], xticklabels=labels,
               yticks=[0, .5, 1])
        ax.grid(axis='y')
        ax.tick_params(axis='x', labelsize=7.5)
        for t, color in zip(ax.get_xticklabels(), [C['frozen'], C['current'], C['repair']]):
            t.set_color(color)
        ax.set_ylabel('Test accuracy' if ax is b else '')
        if ax is c:
            ax.tick_params(axis='y', labelleft=False)
    b.set_xticklabels(['Frozen\nconsumer','Current\nhead','Scale\nrepair'])
    c.set_xticklabels(['Frozen\nconsumer','Current\nhead','Linear\nrepair*'])
    b.text(.45, .10, '$\\alpha=3$ / 10 seeds\nNo new labels', transform=b.transAxes, fontsize=7.2)
    real = branch(mechanism['reconstruction']['har'], 'joint', 100)
    mse_ratio = real['test_reconstruction_ratio_to_detached']['mean']
    c.text(.40, .10, '$\\alpha=100$ / 5 seeds\nRecon. MSE $\\times$%.3f' % mse_ratio,
           transform=c.transAxes, fontsize=7.2)
    fig.text(.50, .035, 'Points: seeds   Diamonds / bars: mean ± SD   Dashed: healthy reference',
             ha='center', fontsize=7.1)
    validate_layout(fig)
    export(fig, 'fig_readout_overview')
    AUDIT['overview_values'] = dict(isotropic=iso_values.tolist(), reconstruction=rec_values.tolist(),
                                  reconstruction_mse_ratio=mse_ratio,
                                  uncertainty='Population SD across 10 / 5 seeds; full test corpus')
    summary = dict(protocol='full-test real HAR reconstruction repair; maps match fixed consumer bank',
                   n=5, paired_ridge=describe(rec_values[:, 2]), frozen=describe(rec_values[:, 0]),
                   own=describe(rec_values[:, 1]),
                   input_record_sha256={k: v for k, v in SOURCES.items() if '/recon_repairs/' in k})
    (BASE / 'reconstruction_repair_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (ROOT / 'paper/story_numbers.tex').write_text(
        '%% Generated by plot_readout_story.py from complete full-test records.\n'
        r'\newcommand{\ReconHarPairedLinearHigh}{%.3f}' % rec_values[:, 2].mean() + '\n')


def paired(old, mechanism):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), gridspec_kw={'width_ratios': [1, 1.02]})
    fig.subplots_adjust(left=.10, right=.985, top=.86, bottom=.20, wspace=.70)
    epochs = [e['epoch'] for e in branch(old[0], 'joint', 3)['trajectory']]
    for mode, field, color, label, ls in [
        ('joint', 'frozen', C['frozen'], 'Frozen consumer', '-'),
        ('joint', 'own', C['current'], 'Current task head', '-'),
        ('detach', 'frozen', C['reference'], 'Detached, frozen', '--')]:
        values = np.array([[e[field]['acc'] for e in branch(r, mode, 3)['trajectory']] for r in old])
        mean, sd = values.mean(0), values.std(0)
        axes[0].plot(epochs, mean, color=color, lw=1.6, label=label, ls=ls)
        axes[0].fill_between(epochs, mean - sd, mean + sd, color=color, alpha=.12, lw=0)
    axes[0].set(xlabel='Continuation epoch', ylabel='Test accuracy', ylim=(.2, 1.01),
                yticks=[.2, .4, .6, .8, 1], xticks=[0, 10, 20, 30])
    axes[0].legend(loc='lower left', fontsize=7.2, handlelength=1.8, borderpad=.1)
    panel(axes[0], 'a', 'Matched continuation')
    selected = ['frozen', 'mean_only', 'scalar_only', 'mean_scalar', 'diagonal_stats', 'paired_ridge']
    y = np.arange(len(selected))
    for case, delta, color, label, marker in [
        ('har_variance', -.13, C['repair'], 'Isotropic (10 seeds)', 'o'),
        ('har_projected', .13, C['purple'], 'Projected (5 seeds)', 's')]:
        b = branch(mechanism['repairs'][case], 'joint', 3)
        means = [b['repairs'][arm]['acc']['mean'] for arm in selected]
        sd = [b['repairs'][arm]['acc']['sd'] for arm in selected]
        axes[1].errorbar(means, y + delta, xerr=sd, fmt=marker, color=color, ms=3.8,
                         lw=1, capsize=2, label=label)
    axes[1].set_yticks(y, ['Frozen', 'Mean only', 'Scale only', 'Mean + scale', 'Per-feature', 'Paired linear*'])
    axes[1].set(xlabel='Test accuracy after repair', xlim=(.25, 1.01), xticks=[.3, .5, .7, .9],
                ylim=(len(selected) + 1.2, -.6))
    axes[1].legend(loc='lower left', fontsize=7.2, labelspacing=.5, borderpad=.1)
    panel(axes[1], 'b', 'Fixed repair hierarchy')
    axes[0].grid(axis='y')
    axes[1].grid(axis='x')
    export(fig, 'fig_paired_and_repair')


def scale():
    data = read(ROOT / 'results/READOUT/readout_scale_law.json')
    cs = np.array(data['c_grid'])
    rows = data['rows']
    def values(key, sub):
        return np.array([[e[key][sub] for e in r['curve']] for r in rows])
    cstar = np.median([r['critical']['c_star_median'] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.65))
    fig.subplots_adjust(left=.10, right=.965, top=.85, bottom=.30, wspace=.38)
    for ax, metric in zip(axes, ['acc', 'ece']):
        arms = [('frozen', metric, C['frozen'], 'Frozen', '-'),
                ('recalibrated', metric, C['repair'], 'Input recalibration', '-')]
        if metric == 'ece':
            arms.insert(1, ('frozen', 'ece_temp', C['purple'], 'Frozen + healthy $T$', '--'))
        for key, sub, color, label, ls in arms:
            v = values(key, sub)
            ax.plot(cs, v.mean(0), color=color, ls=ls, label=label)
            ax.fill_between(cs, v.mean(0) - v.std(0), v.mean(0) + v.std(0), color=color, alpha=.12, lw=0)
        ax.axvline(cstar, color=C['reference'], ls=':', lw=1)
        ax.set(xscale='log', xlim=(min(cs), max(cs)), xlabel='Representation scale $c$',
               ylabel='Accuracy' if metric == 'acc' else 'ECE',
               ylim=(.1, 1.02) if metric == 'acc' else (0, .55))
        ax.grid(axis='y')
    axes[0].text(.06, .08, '$c^{\\mathrm{bias}}=%.3f$' % cstar, transform=axes[0].transAxes, fontsize=7.8)
    panel(axes[0], 'a', 'A sufficient scale mechanism')
    panel(axes[1], 'b', 'Calibration is not input alignment')
    fig.legend(*axes[1].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.54, .015),
               ncol=3, fontsize=7.2, columnspacing=1, handlelength=1.6)
    export(fig, 'fig_readout_scale_law')


def consumer():
    summary = read(BASE / 'consumer_frontier_summary.json')
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.85))
    fig.subplots_adjust(left=.10, right=.985, top=.83, bottom=.23, wspace=.36)
    cases = list(summary['cases'].values())
    labels = ['HAR\niso.', 'HAR\nproj.', 'HAR\nrecon.', 'RML\nSNR', 'RML\nMLP', 'RML\nCNN']
    x = np.arange(len(cases))
    for ax, field, letter, title, color in [
        (axes[0], 'scatter_false_safe', 'a', 'Scatter misses consumer harm', C['frozen']),
        (axes[1], 'selected_certified', 'b', 'Certify a choice or abstain', C['repair'])]:
        counts = np.array([c['budgets']['0.1'][field] for c in cases])
        n = np.array([c['endpoints'] for c in cases])
        ax.bar(x, counts / n, width=.64, color=color, alpha=.9)
        if field == 'selected_certified':
            ax.bar(x, 1 - counts / n, bottom=counts / n, width=.64, color='#E7EBEF',
                   hatch='///', edgecolor='#C0C8CF', linewidth=.4)
            ax.text(.98, .96, 'Hatched: abstain', transform=ax.transAxes, ha='right', fontsize=7.2,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=.95))
        for i, (count, denominator) in enumerate(zip(counts, n)):
            ax.text(i, count / denominator + .035, f'{count}/{denominator}', ha='center', fontsize=7)
        ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.16), yticks=[0, .5, 1],
               ylabel='Fraction of joint endpoints')
        ax.tick_params(axis='x', labelsize=7.1)
        ax.grid(axis='y')
        panel(ax, letter, title)
    fig.text(.54, .02, 'Fixed 10-percentage-point relative-loss budget', ha='center', fontsize=7.6)
    export(fig, 'fig_consumer_audit')


def main():
    apply_style()
    old = [read(p) for folder, prefix in [('har', 'h'), ('replication', 'v')]
           for p in sorted((BASE / folder).glob(prefix + '???.json'))]
    repair_rows = [read(p) for p in sorted((BASE / 'mechanism').glob('d???.json'))]
    repairs = sorted([r for r in repair_rows if r['domain'] == 'har' and r['auxiliary'] == 'variance'], key=lambda r: r['seed'])
    recon = [read(p) for p in sorted((BASE / 'recon_repairs').glob('k???.json'))]
    assert len(old) == len(repairs) == 10 and len(recon) == 5
    for r in recon:
        bank = read(BASE / 'frontier' / ('b%03d.json' % (20 + r['seed'])))
        assert r['complete'] and r['repairs_fitted_on'] == 'fitting features only; no labels or evaluation statistics'
        for b in r['branches']:
            counterpart = branch(bank, b['mode'], b['alpha'])
            assert b['checkpoint_sha256'] == counterpart['checkpoint_sha256']
            for arm, rec in b['repairs'].items():
                assert rec['map_sha256'] == counterpart['candidates'][arm]['map_sha256']
        assert max(r['healthy_reconstruction_error'].values()) <= 1e-7
    mechanism = read(BASE / 'mechanism_summary.json')
    overview(old, repairs, recon, mechanism)
    paired(old, mechanism)
    scale()
    consumer()
    AUDIT['source_sha256'] = SOURCES
    AUDIT['reconstruction_maps_match_consumer_bank'] = True
    (OUT / 'story_figure_manifest.json').write_text(json.dumps(AUDIT, indent=2) + '\n')
    print(json.dumps({k: v for k, v in AUDIT.items() if k.startswith('fig_')}, indent=2))


if __name__ == '__main__':
    main()
