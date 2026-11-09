"""Local factorial verification and analysis; no new model/repair selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from auxiliary_state_factorial import CELLS
from paired_readout_continuation import state_hash
from summarize_cpu_evidence import describe

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'repro/results/CPUCONT'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    paths = sorted((BASE / 'head_state_first').glob('j???.json')) + sorted((BASE / 'head_state').glob('j???.json'))
    if len(paths) != 5:
        raise AssertionError('All five predeclared factorial seeds are required')
    rows = [json.loads(p.read_text()) for p in paths]
    states_count = 0
    for p, r in zip(paths, rows):
        assert r['complete'] and r['device'] == 'cpu' and r['torch_threads'] == 1 and r['cuda_visibility'] == ''
        assert r['dose'] == 30 and r['continuation_epochs'] == 15 and r['outside_consumer_certificate_bank']
        assert r['source_sha256'] == sha(ROOT / 'repro/scripts/auxiliary_state_factorial.py')
        assert r['paired_source_sha256'] == sha(ROOT / 'repro/scripts/paired_readout_continuation.py')
        original = BASE / 'real' / r['source_record']
        assert r['source_record_sha256'] == sha(original)
        reference_path = original.parent / (original.stem + '_states') / 'reference.pt'
        assert r['reference_checkpoint_sha256'] == sha(reference_path)
        reference = torch.load(reference_path, map_location='cpu', weights_only=True)
        shared = {k: v for k, v in reference['model'].items() if k.startswith(('encoder.', 'primary.'))}
        assert state_hash(shared) == r['initial_shared_sha256']
        names = list(reference['model'])
        identifiers = reference['optimizer']['param_groups'][0]['params']
        assert len(names) == len(identifiers)
        shared_opt = {name: reference['optimizer']['state'].get(identifier, {})
                      for name, identifier in zip(names, identifiers) if name.startswith(('encoder.', 'primary.'))}
        assert state_hash(shared_opt) == r['initial_shared_optimizer_sha256']
        warm_aux = {k[4:]: v for k, v in reference['model'].items() if k.startswith('aux.')}
        assert max(r['healthy_reconstruction_error'].values()) <= 1e-7
        assert [(b['weight_state'], b['moment_state']) for b in r['branches']] == CELLS
        for b in r['branches']:
            assert b['initial_shared_sha256'] == r['initial_shared_sha256']
            assert b['initial_shared_optimizer_sha256'] == r['initial_shared_optimizer_sha256']
            assert b['initial_auxiliary_sha256'] == (state_hash(warm_aux) if b['weight_state'] == 'warm'
                                                    else r['initial_auxiliary_sha256'])
            state_path = p.parent / (p.stem + '_states') / (b['weight_state'] + '_' + b['moment_state'] + '.pt')
            saved = torch.load(state_path, map_location='cpu', weights_only=True)
            assert state_hash(saved['model']) == b['final_model_sha256']
            assert state_hash(saved['optimizer']) == b['final_optimizer_sha256']
            assert all(t.device.type == 'cpu' for t in saved['model'].values())
            states_count += 1
        warm = r['branches'][0]
        assert warm['archived_endpoint_exact_replay']
        archived_final = torch.load(original.parent / (original.stem + '_states') / 'joint_30.pt',
                                    map_location='cpu', weights_only=True)
        assert warm['final_model_sha256'] == state_hash(archived_final['model'])
        assert warm['final_optimizer_sha256'] == state_hash(archived_final['optimizer'])
        assert len({b['minibatch_order_sha256'] for b in r['branches']}) == 1
        first = [b['update_audits'][0] for b in r['branches']]
        for i, j in [(0, 1), (2, 3)]:
            for key in ('V_before', 'c', 'r', 'actual_delta_V', 'primary_counterfactual_V_after'):
                assert abs(first[i][key] - first[j][key]) < 1e-10
        assert max(a['primary_counterfactual_V_after'] for a in first) - min(a['primary_counterfactual_V_after'] for a in first) < 1e-10
    summary = dict(protocol='five healthy SNR references; auxiliary-weight by auxiliary-Adam-state factorial',
        dose=30, seeds=[r['seed'] for r in rows], cells=[], contrasts={},
        uncertainty='Descriptive seed-paired t intervals at fixed data/recipe; sampled updates are dependent.',
        separate_from_consumer_bank=True)
    for i, (weights, moments) in enumerate(CELLS):
        branches = [r['branches'][i] for r in rows]
        endpoints = [b['trajectory'][-1] for b in branches]
        primaries = [next(b for b in json.loads((BASE / 'real' / r['source_record']).read_text())['branches']
                          if b['mode'] == 'primary')['trajectory'][-1] for r in rows]
        scatter_ratios = [e['geometry']['relative_scatter'] / p['geometry']['relative_scatter']
                          for e, p in zip(endpoints, primaries)]
        early = [a for b in branches for a in b['update_audits'] if a['epoch'] <= 2]
        entry = dict(weights=weights, moments=moments,
            **{k: describe([e[k]['acc'] for e in endpoints]) for k in ['frozen', 'recalibrated', 'retrained', 'own']},
            relative_trace=describe([e['geometry']['relative_trace'] for e in endpoints]),
            relative_scatter=describe([e['geometry']['relative_scatter'] for e in endpoints]),
            scatter_ratio_to_primary=describe(scatter_ratios),
            scatter_collapse_count_to_primary=sum(x < .3 for x in scatter_ratios),
            auxiliary_test_mse=describe([e['auxiliary_test_mse'] for e in endpoints]),
            first_step_c=describe([b['update_audits'][0]['c'] for b in branches]),
            first_step_auxiliary_counterfactual=describe([b['update_audits'][0]['auxiliary_counterfactual_delta_V'] for b in branches]),
            early_sampled_updates=len(early), early_positive_c=sum(a['c'] > 0 for a in early),
            early_auxiliary_counterfactual_contraction=sum(a['auxiliary_counterfactual_delta_V'] < 0 for a in early),
            early_flow_step_disagreement=sum(a['gradient_flow_dV'] * a['actual_delta_V'] < 0 for a in early))
        summary['cells'].append(entry)
    for name, i, j in [('weights_at_retained_moments', 2, 0), ('weights_at_reset_moments', 3, 1),
                        ('moments_at_warm_weights', 1, 0), ('moments_at_initial_weights', 3, 2),
                        ('initial_reset_vs_warm_retained', 3, 0)]:
        summary['contrasts'][name] = describe([r['branches'][i]['trajectory'][-1]['frozen']['acc'] -
                                              r['branches'][j]['trajectory'][-1]['frozen']['acc'] for r in rows])
    summary['input_record_sha256'] = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    (BASE / 'head_state_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    queues = [json.loads((BASE / folder / 'queue.json').read_text()) for folder in ['head_state_first', 'head_state']]
    for q, n in zip(queues, [1, 4]):
        assert q['finished'] and not q['failed'] and not q['running'] and not q['pending'] and len(q['completed']) == n
    report = dict(records=5, endpoint_states=states_count, exact_archived_replays=5, new_conditional_trajectories=15,
        all_shared_weights_and_moments_matched=True, all_future_orders_match=True,
        all_first_step_factorial_invariants_pass=True, all_saved_parameters_cpu=True, all_queues_finished=True,
        monitored_peak_aggregate_rss_bytes=max(q['peak_aggregate_rss_bytes'] for q in queues),
        sum_job_wall_seconds=sum(j['wall_s'] for q in queues for j in q['completed']),
        sampled_updates=sum(len(b['update_audits']) for r in rows for b in r['branches']),
        no_endpoint_scatter_collapses=all(c['scatter_collapse_count_to_primary'] == 0 for c in summary['cells']),
        input_record_sha256=summary['input_record_sha256'])
    (BASE / 'head_state_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = [r'\begin{table}[t]', r'\centering',
        r'\caption{Conditional SNR head-state factorial at dose $30$ (five seeds). Shared encoder/primary parameters and their Adam states, the frozen consumer, and future minibatches are identical at branching. Auxiliary weights and auxiliary Adam state form the four cells. Warm/retained replays the original model/Adam endpoint bitwise. Fresh probes share a fixed initialization seed across cells. Means with population seed SD for frozen accuracy; full paired intervals and early actual-step audits are archived.}',
        r'\label{tab:head_state}', r'\footnotesize', r'\begin{tabular}{llrrrrrr}', r'\toprule',
        r'Weights & aux. Adam & frozen & recal. & refit & own & rel. trace & SNR MSE \\', r'\midrule']
    macros = {}
    for i, c in enumerate(summary['cells']):
        vals = [r'$%.3f\pm%.3f$' % (c['frozen']['mean'], c['frozen']['sd'])]
        vals += ['$%.3f$' % c[k]['mean'] for k in ['recalibrated', 'retrained', 'own', 'relative_trace', 'auxiliary_test_mse']]
        lines.append(c['weights'] + ' & ' + c['moments'] + ' & ' + ' & '.join(vals) + r' \\')
        suffix = ['WarmKeep', 'WarmReset', 'InitialKeep', 'InitialReset'][i]
        macros['HeadFrozen' + suffix] = '%.3f' % c['frozen']['mean']
        macros['HeadTrace' + suffix] = '%.3f' % c['relative_trace']['mean']
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (ROOT / 'paper/head_state_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (ROOT / 'paper/head_state_numbers.tex').write_text('% Generated locally from all factorial cells.\n' +
        '\n'.join('\\newcommand{\\' + k + '}{' + v + '}' for k, v in macros.items()) + '\n', encoding='utf-8')
    print(json.dumps(dict(verification={k: v for k, v in report.items() if k != 'input_record_sha256'},
        cells=[{k: c[k] for k in ['weights', 'moments', 'frozen', 'relative_trace', 'early_positive_c', 'early_sampled_updates']}
               for c in summary['cells']], contrasts=summary['contrasts']), indent=2))


if __name__ == '__main__':
    main()
