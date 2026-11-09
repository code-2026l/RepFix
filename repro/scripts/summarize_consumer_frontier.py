"""Local-only analysis of CNN replication and fixed unlabeled consumer audits."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from readout_alignment import ARMS
from summarize_cpu_evidence import describe, experiment_summary

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'repro/results/CPUCONT'


def branch(row, mode, dose):
    return next(b for b in row['branches'] if b['mode'] == mode and b['alpha'] == dose)


def audit_case(rows, mode='joint'):
    entries = [b for r in rows for b in r['branches'] if b['mode'] == mode]
    result = dict(n_seeds=len(rows), endpoints=len(entries), budgets={})
    for budget in (.05, .10):
        decisions = [b['decisions'][str(budget)] for b in entries]
        frozen = [b['candidates']['frozen'] for b in entries]
        selected = [d for d in decisions if d['certified']]
        result['budgets'][str(budget)] = dict(
            harmful_frozen=sum(c['loss_increase'] > budget + 1e-12 for c in frozen),
            frozen_certified=sum(c['upper_remaining_disagreement'] <= budget for c in frozen),
            frozen_certificate_false_safe=sum(c['upper_remaining_disagreement'] <= budget and
                                               c['loss_increase'] > budget + 1e-12 for c in frozen),
            scatter_safe=sum(b['scatter_safe_heuristic'] for b in entries),
            scatter_false_safe=sum(b['scatter_safe_heuristic'] and b['candidates']['frozen']['loss_increase'] > budget + 1e-12
                                   for b in entries),
            probe_safe=sum(b['probe_safe_heuristic'] for b in entries),
            probe_false_safe=sum(b['probe_safe_heuristic'] and b['candidates']['frozen']['loss_increase'] > budget + 1e-12
                                 for b in entries),
            selected_certified=len(selected), abstained=len(entries) - len(selected),
            selected_false_safe=sum(d['loss_increase'] > budget + 1e-12 for d in selected),
            selected_arm_counts=dict(Counter(d['selected_arm'] or 'abstain' for d in decisions)),
            selected_remainder_accuracy=dict(mean=float(np.mean([d['remainder_accuracy'] for d in selected])),
                sd=float(np.std([d['remainder_accuracy'] for d in selected])),
                n=len(selected), interpretation='descriptive selected-endpoint summary; no independent-trial interval')
                if selected else None)
    return result


def cnn_summary(rows, audit):
    result = experiment_summary(rows)
    by_seed = {r['seed']: r for r in audit}
    for b in result['branches']:
        task = [branch(by_seed[r['seed']], b['mode'], b['alpha']) for r in rows]
        b['test_reconstruction_mse'] = describe([t['raw_head_test_reconstruction_mse'] for t in task])
        if b['mode'] != 'primary':
            detached = [branch(by_seed[r['seed']], 'detach', b['alpha']) for r in rows]
            b['test_reconstruction_ratio_to_detached'] = describe([
                t['raw_head_test_reconstruction_mse'] / d['raw_head_test_reconstruction_mse']
                for t, d in zip(task, detached)])
    return result


def main():
    paths = sorted((BASE / 'frontier').glob('b???.json')) + sorted((BASE / 'cnn_consumer').glob('n???.json'))
    training_paths = sorted((BASE / 'frontier').glob('c???.json'))
    if len(paths) != 35 or len(training_paths) != 5:
        raise AssertionError('All 30 archived audits, five CNN audits, and five CNN training records are required')
    audits = [json.loads(p.read_text()) for p in paths]
    training = [json.loads(p.read_text()) for p in training_paths]
    for r in audits + training:
        if not r['complete'] or r['device'] != 'cpu' or r['torch_threads'] != 1 or r['cuda_visibility'] != '':
            raise AssertionError('Incomplete or non-CPU numerical evidence')
    candidates = sum(len(r['branches']) * len(ARMS) for r in audits)
    if candidates != 3015 or any(r['family_size'] != candidates for r in audits):
        raise AssertionError('The predeclared simultaneous family does not match the complete bank')
    cases = [('har_iso', 'HAR isotropic', 'har', 'variance', 'mlp', 10),
             ('har_proj', 'HAR projected', 'har', 'projected', 'mlp', 5),
             ('har_recon', 'HAR recon.', 'har', 'recon', 'mlp', 5),
             ('rml_snr', 'RadioML SNR', 'radioml', 'snr', 'mlp', 5),
             ('rml_recon', 'RadioML I/Q MLP', 'radioml', 'recon', 'mlp', 5),
             ('rml_cnn', 'RadioML I/Q CNN', 'radioml', 'recon', 'waveform_cnn', 5)]
    summary = dict(protocol='fixed 3015-candidate bank; 1024 unlabeled uniform monitor inputs per corpus',
        bank_delta=.05, candidate_functions=candidates,
        guarantee_scope='random finite-corpus partition, not a new population or a future update',
        reference_access='healthy predictions on monitor inputs; paired fitting anchors for paired repairs',
        uncertainty='seed SD and descriptive t intervals conditional on each fixed recipe; branches within a seed are dependent',
        cases={}, primary_controls={}, total_disagreement_bound_violations=0)
    for key, label, domain, auxiliary, arch, count in cases:
        group = [r for r in audits if (r['domain'], r['auxiliary'], r['architecture']) == (domain, auxiliary, arch)]
        if len(group) != count or len({r['seed'] for r in group}) != count:
            raise AssertionError('Missing/duplicated audit seeds for ' + key)
        summary['cases'][key] = dict(label=label, **audit_case(group))
        summary['primary_controls'][key] = audit_case(group, 'primary')
    cnn_audits = [r for r in audits if r['architecture'] == 'waveform_cnn']
    summary['cnn'] = cnn_summary(training, cnn_audits)
    summary['total_disagreement_bound_violations'] = sum(
        c['disagreement_bound_violated'] for r in audits for b in r['branches'] for c in b['candidates'].values())
    for budget in (.05, .10):
        items = [c['budgets'][str(budget)] for c in summary['cases'].values()]
        summary['joint_totals_' + str(budget)] = {key: sum(item[key] for item in items)
            for key in ['harmful_frozen', 'frozen_certified', 'frozen_certificate_false_safe',
                        'scatter_safe', 'scatter_false_safe', 'probe_safe', 'probe_false_safe',
                        'selected_certified', 'abstained', 'selected_false_safe']}
    summary['input_record_sha256'] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in paths + training_paths}
    (BASE / 'consumer_frontier_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    numbers = dict(ConsumerFamily=str(candidates), ConsumerJointN=str(sum(c['endpoints'] for c in summary['cases'].values())))
    for budget, suffix in [(.05, 'Five'), (.10, 'Ten')]:
        totals = summary['joint_totals_' + str(budget)]
        for field, name in [('selected_certified', 'Selected'), ('abstained', 'Abstain'),
                             ('selected_false_safe', 'FalseSafe'), ('scatter_false_safe', 'ScatterFalseSafe'),
                             ('scatter_safe', 'ScatterSafe')]:
            numbers['Consumer' + name + suffix] = str(totals[field])
        numbers['ConsumerModified' + suffix] = str(totals['selected_certified'] - totals['frozen_certified'])
        numbers['ConsumerUnchanged' + suffix] = str(totals['frozen_certified'])
    numbers['ConsumerHarSelectedTen'] = str(sum(summary['cases'][key]['budgets']['0.1']['selected_certified']
                                                for key in ['har_iso', 'har_proj', 'har_recon']))
    numbers['ConsumerRmlSelectedTen'] = str(sum(summary['cases'][key]['budgets']['0.1']['selected_certified']
                                                for key in ['rml_snr', 'rml_recon', 'rml_cnn']))
    numbers['ConsumerRmlJointN'] = str(sum(summary['cases'][key]['endpoints']
                                         for key in ['rml_snr', 'rml_recon', 'rml_cnn']))
    numbers['CnnHealthy'] = '%.3f' % summary['cnn']['healthy']['mean']
    numbers['CnnPrimary'] = '%.3f' % branch(summary['cnn'], 'primary', 0)['frozen']['mean']
    for dose, suffix in [(30, 'Low'), (300, 'Mid'), (3000, 'High')]:
        b = branch(summary['cnn'], 'joint', dose)
        for field, label in [('frozen', 'Frozen'), ('recalibrated', 'Recal'), ('retrained', 'Refit'),
                              ('own', 'Own'), ('relative_trace', 'Trace'),
                              ('test_reconstruction_ratio_to_detached', 'MseRatio')]:
            numbers['Cnn' + label + suffix] = '%.3f' % b[field]['mean']
    (ROOT / 'paper/consumer_numbers.tex').write_text('% Generated from the complete fixed bank locally.\n' +
        '\n'.join('\\newcommand{\\' + k + '}{' + v + '}' for k, v in numbers.items()) + '\n', encoding='utf-8')

    table = [r'\begin{table}[t]', r'\centering',
        r'\caption{Fixed-family consumer audit. Each corpus supplies 1,024 uniformly sampled unlabeled monitor inputs; remaining labels score the decisions afterward. The 3,015 fixed endpoint/repair candidates share total tail probability $0.05$. \emph{Scatter false safe} uses a scatter ratio $\ge0.3$ relative to healthy features on the same monitor, despite frozen-consumer loss increase above budget. Selection uses the first certified candidate, including identity/frozen; uncertified cases abstain. Counts describe dependent checkpoints, not independent trials.}',
        r'\label{tab:consumer_audit}', r'\footnotesize', r'\begin{tabular}{lrrrrrr}', r'\toprule',
        r'Case & budget & joint & \shortstack{scatter\\false safe} & \shortstack{frozen\\certified} & \shortstack{candidate\\selected} & \shortstack{selected\\false safe} \\', r'\midrule']
    for case in summary['cases'].values():
        for budget in (.05, .10):
            b = case['budgets'][str(budget)]
            table.append(r'%s & %.2f & %d & %d & %d & %d & %d \\' % (
                case['label'], budget, case['endpoints'], b['scatter_false_safe'], b['frozen_certified'],
                b['selected_certified'], b['selected_false_safe']))
    table.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (ROOT / 'paper/consumer_audit_table.tex').write_text('\n'.join(table) + '\n', encoding='utf-8')

    table = [r'\begin{table}[t]', r'\centering',
        r'\caption{Waveform CNN reconstruction continuation on the same prepared RadioML inputs and split as the MLP study. Five seeds; 25 healthy plus 15 continuation epochs; the original heads, optimizer, cloned-state and batch controls are retained. Entries are means, with population seed SD for frozen and own accuracy; all dispersions are archived. MSE ratio compares reconstruction error with the matched detached decoder. All metrics here use the full original test corpus, separately from the monitor/remainder audit.}',
        r'\label{tab:cnn_continuation}', r'\footnotesize', r'\begin{tabular}{rlrrrrrr}', r'\toprule',
        r'$\alpha$ & arm & frozen & recal. & refit & own & rel. trace & MSE ratio \\', r'\midrule']
    for b in summary['cnn']['branches']:
        cells = [(r'$%.3f\pm%.3f$' % (b[k]['mean'], b[k]['sd']) if k in ('frozen', 'own')
                  else '$%.3f$' % b[k]['mean']) for k in ['frozen', 'recalibrated', 'retrained', 'own']]
        cells += ['$%.3f$' % b['relative_trace']['mean'],
                  '$%.3f$' % b['test_reconstruction_ratio_to_detached']['mean'] if b['mode'] != 'primary' else '--']
        table.append('%g & %s & ' % (b['alpha'], b['mode']) + ' & '.join(cells) + r' \\')
    table.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (ROOT / 'paper/cnn_continuation_table.tex').write_text('\n'.join(table) + '\n', encoding='utf-8')

    from figure_style import apply_style
    from plot_readout_story import consumer
    apply_style()
    consumer()
    print(json.dumps(dict(cnn_runs=5, audit_runs=35, candidate_functions=candidates,
        budget_10pp=summary['joint_totals_0.1'], bound_violations=summary['total_disagreement_bound_violations']), indent=2))


if __name__ == '__main__':
    main()
