"""Build the manuscript figures from recorded results, without synthetic anchors.

Run with Python, NumPy and Matplotlib. Outputs go directly to paper/figures.
The adjacent manifest records all input hashes and the displayed width fits.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/figures'
SOURCES = {}
AUDIT = {}
COL = {'joint': '#596574', 'stf0': '#2778B5', 'stfhard': '#C78D25',
       'stfsoft': '#14856C', 'stfcal': '#BC5836'}
MODES = ['joint', 'stf0', 'stfhard', 'stfsoft']
LABELS = ['Joint', 'STF-0', 'STF-hard', 'STF-soft']
MARKERS = ['o', 's', '^', 'D']
DOMAINS = ['har', 'radioml', 'battery']
DNAMES = ['HAR', 'RadioML', 'Battery']
DCOL = ['#2778B5', '#C78D25', '#14856C']
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.2, 'axes.labelsize': 7.2,
    'axes.titlesize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'legend.fontsize': 6.2, 'axes.linewidth': .55, 'lines.linewidth': 1.2,
    'text.color': '#202D3D', 'axes.labelcolor': '#344255',
    'xtick.color': '#526071', 'ytick.color': '#526071',
    'axes.edgecolor': '#9AA5B1', 'axes.axisbelow': True,
    'grid.color': '#E2E7EC', 'grid.linewidth': .45,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.pad': 2, 'ytick.major.pad': 2,
})


def read(relative):
    path = ROOT / relative
    blob = path.read_bytes()
    SOURCES[relative] = hashlib.sha256(blob).hexdigest()
    return json.loads(blob)


def title(ax, letter, text):
    ax.set_title(r'$\bf{' + letter + '}$  ' + text, loc='left', pad=7)
    ax.grid(axis='y')
    ax.tick_params(direction='out', length=2)


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Check text stays within the exported canvas; final overlaps are checked visually.
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text().strip():
            box = text.get_window_extent(renderer)
            if box.width and box.height:
                assert box.x0 >= -1 and box.y0 >= -1, text.get_text()
                assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, text.get_text()
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=320, facecolor='white')
    assert b'/Type3' not in (OUT / f'{name}.pdf').read_bytes()
    plt.close(fig)


def crossing(rows, width):
    rows = sorted((r for r in rows if r['m'] == width), key=lambda r: r['alpha'])
    if rows[0]['rel_scatter_mean'] < .5:
        return None
    for left, right in zip(rows, rows[1:]):
        y0, y1 = left['rel_scatter_mean'], right['rel_scatter_mean']
        if y1 < .5 <= y0:
            t = (.5-y0)/(y1-y0)
            return float(np.exp(np.log(left['alpha']) + t*np.log(right['alpha']/left['alpha'])))
    return None


def sweeps():
    out = {}
    for domain in ('har', 'battery'):
        files = sorted((ROOT/'results/BSCALE').glob(f'xcap_{domain}_m*.json'))
        assert len(files) == 9
        out[domain] = [r for p in files for r in read(p.relative_to(ROOT).as_posix())['results']]
    out['radioml'] = read('repro/results/xcap2_radioml.json')['results']
    return out


def unified():
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.48))
    fig.subplots_adjust(left=.092, right=.985, bottom=.12, top=.87, wspace=.64, hspace=.85)
    a, b, c, d, e, f = axes.flat
    data = sweeps()
    for domain, label, color, marker in zip(DOMAINS, DNAMES, DCOL, MARKERS):
        rows = sorted((r for r in data[domain] if r['m'] == 128), key=lambda r: r['alpha'])
        assert rows
        a.semilogx([r['alpha'] for r in rows], [r['rel_scatter_mean'] for r in rows],
                   marker=marker, ms=2.5, mew=.4, color=color, label=label)
        widths = sorted({r['m'] for r in data[domain]})
        pts = [(m, crossing(data[domain], m)) for m in widths]
        pts = [(m, y) for m, y in pts if y is not None]
        x, y = np.array(pts).T
        fit = np.polyfit(np.log(x), np.log(y), 1)
        AUDIT[domain] = {'crossings': pts, 'width_slope': float(fit[0])}
        b.loglog(x, y, marker, color=color, ms=3)
        b.plot(x, np.exp(np.polyval(fit, np.log(x))), '--', color=color, lw=.9)
    a.axhspan(0, .3, color='#F5EDE8', zorder=-2)
    a.axhline(.5, color='#929AA4', ls=':', lw=.8)
    a.set(xlabel=r'Auxiliary weight $\alpha$', ylabel='Relative scatter', ylim=(0, 1.08), yticks=[0, .5, 1])
    a.set_xticks([1e-4, 1e-1, 1e2])
    a.minorticks_off()
    title(a, 'a', 'Dose response')
    # Shared domain key, outside the data regions of panels a and b.
    handles = [Line2D([], [], color=col, marker=mk, ms=3, lw=1, label=lab)
               for col, mk, lab in zip(DCOL, MARKERS, DNAMES)]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.078, .992),
               ncol=3, columnspacing=1.4, handlelength=1.6)
    b.set(xlabel=r'Width $m$', ylabel=r'Half-height $\alpha_c$')
    b.set_xticks([16, 64, 256], ['16', '64', '256'])
    b.set_yticks([1e-3, 1e-1, 1e1])
    b.minorticks_off()
    title(b, 'b', 'Width dependence')

    raw = read('results/R7/R7_dose_nyu_a0.5.json')['raw']
    stf = {r['seed']: r for r in raw if r['mode'] == 'stfcal'}
    joint = {r['seed']: r for r in raw if r['mode'] == 'joint'}
    assert len(stf) == len(joint) == 10 and set(stf) == set(joint)
    for seed, r in sorted(stf.items()):
        x = r['cal_rho_u']
        assert bool(r['cal_gate']) == (x > 2)
        # Both outcomes use the exact shared warm-up probe coordinate.
        c.plot([x, x], [joint[seed]['rel_scatter'], r['rel_scatter']], color='.82', lw=.55, zorder=1)
        c.plot(x, joint[seed]['rel_scatter'], 'o', mfc='white', mec=COL['joint'], ms=3.5, mew=.8, zorder=2)
        c.plot(x, r['rel_scatter'], 's', color=COL['stfcal'], ms=2.3, zorder=3)
    c.axhspan(0, .3, color='#F5EDE8', zorder=-2)
    c.axvline(2, ls=':', color='#929AA4', lw=.8)
    c.set(xlabel=r'Probe $\hat\rho$', ylabel='Relative scatter', ylim=(-.05, 1.35),
          xlim=(1.88, 2.22), xticks=[1.9, 2, 2.2], yticks=[0, .5, 1])
    c.legend(handles=[Line2D([], [], marker='o', mfc='white', color=COL['joint'], ls='', ms=3, label='Joint'),
                      Line2D([], [], marker='s', color=COL['stfcal'], ls='', ms=3, label='STFcal')],
             loc='upper left', bbox_to_anchor=(-.05, 1.015), ncol=2,
             columnspacing=.5, handletextpad=.2, borderaxespad=.1)
    title(c, 'c', 'NYUv2 gate')

    repair = {dm: {r['mode']: r for r in read(file)['results']} for dm, file in zip(DOMAINS,
              ['repro/results/xrep_har.json', 'repro/results/xrep_radioml.json', 'repro/results/xrep2_battery.json'])}
    for ax, key, letter, heading in [(d, 'collapse_rate', 'd', 'Collapse'), (e, 'recovery_rate', 'e', 'Task recovery')]:
        for i, (mode, label, marker) in enumerate(zip(MODES, LABELS, MARKERS)):
            # Offsets separate categorical methods only; zero outcomes stay visible.
            ax.plot(np.arange(3)+(i-1.5)*.13, [repair[dm][mode][key] for dm in DOMAINS],
                    ls='', marker=marker, color=COL[mode], ms=4.0, label=label,
                    markeredgecolor='white', markeredgewidth=.35)
        ax.set(xticks=np.arange(3), xticklabels=DNAMES, ylim=(-.09, 1.09), yticks=[0, .5, 1], ylabel='Fraction of seeds')
        ax.tick_params(axis='x', labelsize=6)
        title(ax, letter, heading)
    d.legend(loc='center', ncol=1, fontsize=6.0, labelspacing=.28,
             handletextpad=.5, handlelength=.8)

    folds = []
    for mode in MODES:
        matrix = []
        for seed in range(10):
            rows = sorted(read(f'results/R5fin/R5fin_{mode}_s{seed}.json'), key=lambda r: r['fold'])
            assert [r['fold'] for r in rows] == list(range(10))
            matrix.append([r['oos_ic'] for r in rows])
        folds.append(np.mean(matrix, axis=0))
    folds = np.array(folds)
    for fold in folds.T:
        f.plot(np.arange(4), fold, '-', color='.82', lw=.5, zorder=1)
    for i, mode in enumerate(MODES):
        f.plot(np.full(10, i), folds[i], '.', color=COL[mode], ms=2.5, alpha=.65)
        f.errorbar(i, folds[i].mean(), yerr=folds[i].std(), fmt=MARKERS[i], color=COL[mode],
                   ms=3.7, capsize=2, elinewidth=.9, zorder=3)
    f.axhline(0, color='#9AA5B1', lw=.6, zorder=0)
    f.set(xticks=np.arange(4), xticklabels=['Joint', 'STF-0', 'STF\nhard', 'STF\nsoft'], ylabel='OOS rank-IC')
    f.tick_params(axis='x', labelsize=6)
    title(f, 'f', 'Financial ranking')
    save(fig, 'unified_all')


def battery():
    data = read('repro/results/xrep2_battery.json')
    summary = {r['mode']: r for r in data['results']}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(5.5, 2.05))
    fig.subplots_adjust(left=.085, right=.985, top=.81, bottom=.23, wspace=.62)
    for i, mode in enumerate(MODES):
        rows = [r for r in data['raw'] if r['mode'] == mode]
        assert len(rows) == 20
        jitter = np.random.default_rng(i).uniform(-.12, .12, len(rows))
        for ax, key in ((a, 'metric'), (c, 'rel_scatter')):
            v = np.array([r[key] for r in rows])
            expected = summary[mode]['metric_mean' if key == 'metric' else 'rel_scatter_mean']
            assert np.isclose(v.mean(), expected)
            ax.plot(i+jitter, v, '.', color=COL[mode], ms=2.4, alpha=.45)
            ax.errorbar(i, v.mean(), yerr=v.std(), color=COL[mode], fmt=MARKERS[i],
                        ms=4, capsize=2, elinewidth=1, zorder=4)
    for ax in (a, b, c):
        ax.set_xticks(range(4), ['Joint', 'STF-0', 'STF\nhard', 'STF\nsoft'])
        ax.set_xlim(-.45, 3.45)
    a.set(ylabel='Log-SOH RMSE', ylim=(.23, .66), yticks=[.3, .45, .6])
    title(a, 'a', 'Prediction error')
    for off, key, marker, color, label in [(-.10, 'collapse_rate', 'x', COL['joint'], 'Collapse'),
                                          (.10, 'recovery_rate', 'o', COL['stfsoft'], 'Recovery')]:
        b.plot(np.arange(4)+off, [summary[m][key] for m in MODES], ls='', marker=marker, color=color, ms=4, label=label)
    b.set(ylabel='Fraction of seeds', ylim=(-.09, 1.1), yticks=[0, .5, 1])
    b.legend(loc='center', fontsize=6, handlelength=1, handletextpad=.4)
    title(b, 'b', 'Outcome rates')
    c.axhspan(0, .3, color='#F5EDE8', zorder=-2)
    c.axhline(1, ls=':', color='#929AA4', lw=.8)
    c.set(ylabel='Relative scatter', ylim=(-.05, 1.15), yticks=[0, .5, 1])
    title(c, 'c', 'Relative scatter')
    save(fig, 'battery_repair')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    unified()
    battery()
    (OUT/'figure_data_manifest.json').write_text(json.dumps({'sources': SOURCES, 'width_fits': AUDIT}, indent=2)+'\n')
    print(json.dumps(AUDIT, indent=2))
