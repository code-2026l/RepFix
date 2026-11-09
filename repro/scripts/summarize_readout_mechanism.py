"""Local analysis/figure generation for the fixed mechanism and real-head audit."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from readout_alignment import ARMS
from summarize_cpu_evidence import describe, experiment_summary, load_records

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'repro/results/CPUCONT'


def get_branch(row, mode, alpha):
    return next(b for b in row['branches'] if b['mode'] == mode and b['alpha'] == alpha)


def read_complete(paths, expected):
    if len(paths) != expected:
        raise AssertionError('Expected %d records, found %d' % (expected, len(paths)))
    rows = [json.loads(p.read_text()) for p in paths]
    for row in rows:
        if not row['complete'] or row['device'] != 'cpu' or row['torch_threads'] != 1 or row['cuda_visibility'] != '':
            raise AssertionError('A completed single-thread CPU record is required')
    return rows


def repair_summary(rows):
    if len({r['seed'] for r in rows}) != len(rows):
        raise AssertionError('Duplicate seeds')
    summary = dict(n=len(rows), seeds=sorted(r['seed'] for r in rows), branches=[])
    for template in rows[0]['branches']:
        group = [get_branch(r, template['mode'], template['alpha']) for r in rows]
        result = dict(mode=template['mode'], alpha=template['alpha'], repairs={},
                      retrained=describe([b['archived_retrained']['acc'] for b in group]))
        for arm in ARMS:
            entries = [b['repairs'][arm] for b in group]
            values = {k: describe([e[k] for e in entries])
                      for k in ('acc', 'nll', 'ece', 'reference_prediction_agreement')}
            values['alignment_residual'] = {split: describe([e['alignment_residual'][split] for e in entries])
                                             for split in ('fitting', 'calibration', 'test')}
            values['paired_accuracy_difference_from_frozen'] = describe([
                b['repairs'][arm]['acc'] - b['repairs']['frozen']['acc'] for b in group])
            result['repairs'][arm] = values
        summary['branches'].append(result)
    return summary


def reconstruction_summary(rows, task_rows):
    result = experiment_summary(rows)
    tasks = {r['seed']: r for r in task_rows}
    if set(tasks) != {r['seed'] for r in rows}:
        raise AssertionError('Reconstruction task records do not match the encoder seeds')
    result['healthy_reconstruction_mse'] = describe([tasks[r['seed']]['healthy']['test_mse'] for r in rows])
    for branch in result['branches']:
        primary = [get_branch(r, 'primary', 0)['trajectory'][-1] for r in rows]
        endpoints = [get_branch(r, branch['mode'], branch['alpha'])['trajectory'][-1] for r in rows]
        task = [get_branch(tasks[r['seed']], branch['mode'], branch['alpha']) for r in rows]
        branch['test_reconstruction_mse'] = describe([t['test_mse'] for t in task])
        branch['paired_own_accuracy_difference_from_primary'] = describe([
            e['own']['acc'] - p['own']['acc'] for e, p in zip(endpoints, primary)])
        scatter_ratios = [e['geometry']['relative_scatter'] / p['geometry']['relative_scatter']
                          for e, p in zip(endpoints, primary)]
        branch['scatter_ratio_to_primary'] = describe(scatter_ratios)
        branch['scatter_collapse_count_to_primary'] = sum(value < .3 for value in scatter_ratios)
        branch['paired_retrained_accuracy_difference_from_primary'] = describe([
            e['retrained']['acc'] - p['retrained']['acc'] for e, p in zip(endpoints, primary)])
        if branch['mode'] != 'primary':
            detached = [get_branch(tasks[r['seed']], 'detach', branch['alpha']) for r in rows]
            branch['test_reconstruction_ratio_to_detached'] = describe([
                t['test_mse'] / d['test_mse'] for t, d in zip(task, detached)])
    return result


def macro_text(summary):
    names = {}
    for case, prefix, doses in [('har_variance', 'Har', [(3, 'Mid'), (100, 'High')]),
                                 ('har_projected', 'Proj', [(3, 'Mid')]),
                                 ('radioml_snr', 'Snr', [(30, 'Thirty')])]:
        for dose, suffix in doses:
            b = get_branch(summary['repairs'][case], 'joint', dose)
            for arm, name in [('mean_only', 'Mean'), ('scalar_only', 'Scalar'),
                               ('mean_scalar', 'MeanScalar'), ('diagonal_stats', 'Diagonal'),
                               ('paired_ridge', 'Ridge'), ('covariance', 'Covariance')]:
                names['Mech' + prefix + name + suffix] = '%.3f' % b['repairs'][arm]['acc']['mean']
    for case, prefix, doses in [('har', 'Har', [(1, 'Low'), (10, 'Mid'), (100, 'High')]),
                                ('radioml', 'Rml', [(30, 'Low'), (300, 'Mid'), (3000, 'High')])]:
        item = summary['reconstruction'][case]
        names['Recon' + prefix + 'Primary'] = '%.3f' % get_branch(item, 'primary', 0)['frozen']['mean']
        names['Recon' + prefix + 'PrimaryOwn'] = '%.3f' % get_branch(item, 'primary', 0)['own']['mean']
        for dose, suffix in doses:
            b = get_branch(item, 'joint', dose)
            for key, name in [('frozen', 'Frozen'), ('recalibrated', 'Recal'), ('retrained', 'Retrained'),
                               ('own', 'Own'), ('relative_trace', 'Trace'), ('relative_scatter', 'Scatter'),
                               ('scatter_ratio_to_primary', 'ScatterToPrimary'),
                               ('test_reconstruction_ratio_to_detached', 'MseRatio')]:
                names['Recon' + prefix + name + suffix] = '%.3f' % b[key]['mean']
            if case == 'har' and dose == 100:
                ci = b['paired_frozen_difference_from_primary']['mean_ci95']
                names['ReconHarFrozenGapHigh'] = '%.3f' % b['paired_frozen_difference_from_primary']['mean']
                names['ReconHarFrozenGapLowerHigh'] = '%.3f' % ci[0]
                names['ReconHarFrozenGapUpperHigh'] = '%.3f' % ci[1]
    return ('% Generated locally from the full fixed-grid CPU results.\n' +
            '\n'.join('\\newcommand{\\' + k + '}{' + v + '}' for k, v in names.items()) + '\n')


def tables(summary):
    lines = [r'\begin{table}[t]', r'\centering',
        r'\caption{Fixed repair hierarchy on archived paired checkpoints. Means across 10 HAR isotropic, 5 HAR projected, and 5 RadioML SNR seeds. All maps use fitting features only, without labels or test-statistic fitting. PD, Proc., and ridge require paired old/new anchors; covariance alignment uses only marginal second moments. Ridge is fixed at $10^{-3}$ in standardized coordinates; covariance eigenvalue flooring is $10^{-6}$ of the largest eigenvalue. Full seed dispersions, NLL, ECE, prediction agreement, and fitting/held-out residuals accompany the records.}',
        r'\label{tab:repair_hierarchy}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{lr' + 'r' * len(ARMS) + '}', r'\toprule',
        r'Case & $\alpha$ & frozen & mean & scale & \shortstack{mean\\+scale} & diag. & PD & Proc. & cov. & ridge \\', r'\midrule']
    for case, label in [('har_variance', 'HAR isotropic'), ('har_projected', 'HAR projected'),
                        ('radioml_snr', 'RadioML SNR')]:
        for b in summary['repairs'][case]['branches']:
            if b['mode'] not in ('joint', 'primary'):
                continue
            lines.append(label + ' & %g & ' % b['alpha'] + ' & '.join(
                '$%.3f$' % b['repairs'][arm]['acc']['mean'] for arm in ARMS) + r' \\')
        lines.append(r'\midrule')
    lines[-1:] = [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (ROOT / 'paper/repair_hierarchy_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def reconstruction_table(full):
        lines = [r'\begin{table}[t]', r'\centering',
            (r'\caption{Paired continuation with real reconstruction auxiliaries (five seeds per domain). '
             r'All branches share the healthy encoder, primary head, Adam state, and future batches. '
             r'MSE ratio compares held-out reconstruction error with the dose-matched detached decoder; '
             r'values below one indicate improved reconstruction. Trace is relative to the healthy encoder. '
             r'All entries are seed means.}'),
            r'\label{tab:reconstruction_' + ('full' if full else 'paired') + '}',
            r'\footnotesize' if full else r'\small',
            r'\begin{tabular}{lrlrrrrrr}' if full else r'\begin{tabular}{lrrrrrrr}', r'\toprule',
            (r'Case & $\alpha$ & arm & frozen & recal. & refit & own & rel. trace & MSE ratio \\' if full else
             r'Case & $\alpha$ & frozen & recal. & refit & own & rel. trace & MSE ratio \\'), r'\midrule']
        for case, label in [('har', 'HAR recon.'), ('radioml', 'RadioML I/Q')]:
            for b in summary['reconstruction'][case]['branches']:
                if b['mode'] != 'joint' and not full:
                    continue
                prefix = label + ' & %g & ' % b['alpha']
                if full:
                    prefix += {'primary': 'primary', 'joint': 'joint', 'detach': 'detach', 'stfcal': 'STF-cal'}[b['mode']] + ' & '
                values = ['$%.3f$' % b[k]['mean'] for k in ('frozen', 'recalibrated', 'retrained', 'own', 'relative_trace')]
                values.append('$%.3f$' % b['test_reconstruction_ratio_to_detached']['mean']
                              if b['mode'] != 'primary' else '--')
                lines.append(prefix + ' & '.join(values) + r' \\')
            lines.append(r'\midrule')
        lines[-1:] = [r'\bottomrule', r'\end{tabular}', r'\end{table}']
        return '\n'.join(lines) + '\n'
    for full in (False, True):
        (ROOT / ('paper/reconstruction_' + ('full' if full else 'paired') + '_table.tex')).write_text(
            reconstruction_table(full), encoding='utf-8')


def figure(summary):
    from figure_style import apply_style
    from plot_readout_story import paired
    old = [json.loads(p.read_text()) for folder, prefix in [('har', 'h'), ('replication', 'v')]
           for p in sorted((BASE / folder).glob(prefix + '???.json'))]
    apply_style()
    paired(old, summary)


def main():
    rp = sorted((BASE / 'mechanism').glob('d???.json'))
    repairs = read_complete(rp, 20)
    recon = {case: load_records(['mechanism'], prefix)[0] for case, prefix in [('har', 'u'), ('radioml', 'w')]}
    if any(len(v) != 5 for v in recon.values()):
        raise AssertionError('Both reconstruction domains require all five seeds')
    tp = sorted((BASE / 'task_validation').glob('e???.json'))
    task = read_complete(tp, 10)
    groups = [('har_variance', 'har', 'variance', 10), ('har_projected', 'har', 'projected', 5),
              ('radioml_snr', 'radioml', 'snr', 5)]
    summary = dict(protocol='fixed label-free repair hierarchy and predeclared real-reconstruction grids',
        uncertainty='Population seed SD and descriptive seed-level t intervals conditional on each fixed dataset/recipe.',
        repairs={}, reconstruction={})
    for name, domain, auxiliary, count in groups:
        rows = sorted([r for r in repairs if r['domain'] == domain and r['auxiliary'] == auxiliary], key=lambda r: r['seed'])
        if len(rows) != count:
            raise AssertionError('Missing repair seeds')
        summary['repairs'][name] = repair_summary(rows)
    for case, rows in recon.items():
        summary['reconstruction'][case] = reconstruction_summary(rows, [r for r in task if r['domain'] == case])
    paths = rp + tp + sorted((BASE / 'mechanism').glob('u???.json')) + sorted((BASE / 'mechanism').glob('w???.json'))
    summary['input_record_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (BASE / 'mechanism_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (ROOT / 'paper/mechanism_numbers.tex').write_text(macro_text(summary), encoding='utf-8')
    tables(summary)
    figure(summary)
    print(json.dumps(dict(repair_runs=len(repairs), reconstruction_runs=sum(map(len, recon.values())),
                          task_validation_runs=len(task), output='mechanism_summary.json')))


if __name__ == '__main__':
    main()
