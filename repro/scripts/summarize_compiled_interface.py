"""Local integrity audit and all predeclared compiled-interface comparisons."""
import hashlib
import json
from pathlib import Path

import torch

from consumer_risk import upper_total_disagreements
from paired_readout_continuation import state_hash
from summarize_cpu_evidence import describe

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'repro/results/CPUCONT'
MODES = ['D', 'J', 'LD', 'LJ', 'CD', 'CJ']
CONTRASTS = [('ordinary_auxiliary_gain', 'J', 'D'),
             ('compiled_auxiliary_gain', 'CJ', 'CD'),
             ('compiled_total_gain', 'CJ', 'D'),
             ('compiled_vs_joint', 'CJ', 'J'),
             ('frozen_detach_cost', 'LD', 'D'),
             ('frozen_joint_cost', 'LJ', 'J'),
             ('constraint_joint_cost', 'CJ', 'LJ'),
             ('constraint_detach_cost', 'CD', 'LD'),
             ('frozen_trunk_auxiliary_gain', 'LJ', 'LD')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def branch(record, mode):
    return next(row for row in record['branches'] if row['mode'] == mode)


def main():
    torch.set_num_threads(1)
    plan = ROOT / 'docs/plans/2026-11-09-compiled-interface.md'
    paths = sorted((BASE / 'compiled_confirmation').glob('t???.json'))
    replays = sorted((BASE / 'compiled_replay').glob('u???.json'))
    assert len(paths) == len(replays) == 20
    records = [json.loads(path.read_text()) for path in paths]
    state_count = 0
    for index, (path, record) in enumerate(zip(paths, records)):
        assert record['complete'] and record['device'] == 'cpu' and record['cuda_visibility'] == ''
        assert record['torch_threads'] == 1 and record['plan_sha256'] == sha(plan)
        expected_domain = 'har' if index < 10 else 'radioml'
        assert record['domain'] == expected_domain and record['seed'] == 400 + index % 10
        assert record['configuration']['dose'] == (10. if expected_domain == 'har' else 1.)
        assert record['configuration']['final_affine_dtype'] == 'float64'
        assert record['splits']['primary_label_fraction'] == (.1 if expected_domain == 'har' else .2)
        assert not record['splits']['validation_used_for_encoder']
        assert not record['splits']['validation_used_for_healthy_head']
        for name, digest in record['source_sha256'].items():
            assert digest == sha(ROOT / 'repro/scripts' / name), name
        assert [row['mode'] for row in record['branches']] == MODES
        assert len({row['minibatch_order_sha256'] for row in record['branches']}) == 1
        states = path.parent / (path.stem + '_states')
        reference = torch.load(states / 'reference.pt', map_location='cpu', weights_only=True)
        assert state_hash(reference['model']) == record['reference_model_sha256']
        assert state_hash(reference['optimizer']) == record['reference_optimizer_sha256']
        a = reference['head']['weight'] / reference['sd'].reshape(-1)
        p = reference['projector']
        assert float((a @ p).norm() / a.norm()) < 1e-10
        assert float((p @ p-p).norm()) < 1e-10
        trunk = {k: v for k, v in reference['model'].items() if k.startswith('trunk.')}
        trunk_count = len(trunk)
        state_count += 1
        for row in record['branches']:
            mode = row['mode']
            assert row['initial_model_sha256'] == record['reference_model_sha256']
            assert row['initial_optimizer_sha256'] == record['reference_optimizer_sha256']
            saved = torch.load(states / (mode + '.pt'), map_location='cpu', weights_only=True)
            assert state_hash(saved['model']) == row['final_model_sha256']
            assert state_hash(saved['optimizer']) == row['final_optimizer_sha256']
            assert all(value.device.type == 'cpu' for value in saved['model'].values())
            assert row['folding_max_absolute_feature_error'] < 1e-10
            if mode[0] in 'LC':
                frozen = {k: v for k, v in saved['model'].items() if k.startswith('trunk.')}
                assert state_hash(frozen) == state_hash(trunk)
                for key in range(trunk_count):
                    assert state_hash(saved['optimizer']['state'][key]) == state_hash(reference['optimizer']['state'][key])
            if mode.startswith('C'):
                assert torch.equal(saved['model']['anchor_weight'], reference['model']['final.weight'])
                assert torch.equal(saved['model']['anchor_bias'], reference['model']['final.bias'])
                deployed = torch.load(states / (mode + '_compiled.pt'), map_location='cpu', weights_only=True)['model']
                assert state_hash(deployed) == row['compiled_model_sha256']
                assert not any('anchor' in key or 'projector' in key for key in deployed)
                assert {k: list(v.shape) for k, v in deployed.items() if k.startswith(('trunk.', 'final.'))} == record['deployment']['deployed_encoder_shapes']
                for split in ['validation', 'test']:
                    metrics = row[split]
                    assert metrics['max_absolute_healthy_logit_error'] < 1e-8
                    assert metrics['prediction_changes'] == 0
                    for name in ['acc', 'nll', 'ece']:
                        assert abs(metrics['frozen'][name] - metrics['reference'][name]) < 1e-9
                assert all(trace['prediction_changes'] == 0 for trace in row['diagnostic_trace'])
                state_count += 1
            audit = row['test']['consumer_audit']
            upper = upper_total_disagreements(audit['population_examples'], audit['monitor_examples'],
                                              audit['observed_disagreements'], .05 / 120)
            assert upper == audit['upper_total_disagreements']
            assert abs(audit['upper_remaining_disagreement'] - (upper-audit['observed_disagreements']) /
                       (audit['population_examples']-audit['monitor_examples'])) < 1e-12
            state_count += 1
        assert record['deployment']['deployed_encoder_affine_calls'] == 3
        assert not record['deployment']['inference_reference_required']
        assert not record['deployment']['inference_projection_required']
        replay = json.loads(replays[index].read_text())
        assert replay['complete'] and replay['record_sha256'] == sha(path)
        assert replay['verifier_source_sha256'] == sha(ROOT / 'repro/scripts/verify_compiled_consumers_cpu.py')
        assert len(replay['branches']) == 6
        assert all(row['all_scores_replayed'] and row['bounds_replayed'] for row in replay['branches'])
    assert state_count == 180
    domains = {}
    for domain in ['radioml', 'har']:
        rows = [record for record in records if record['domain'] == domain]
        contrasts = {name: describe([branch(r, left)['test']['own']['acc'] -
                                    branch(r, right)['test']['own']['acc'] for r in rows])
                     for name, left, right in CONTRASTS}
        criteria = {name: contrasts[name]['mean_ci95'][0] > 0 for name in
                    ['ordinary_auxiliary_gain', 'compiled_auxiliary_gain', 'compiled_total_gain']}
        criteria['noninferior_within_half_percentage_point'] = contrasts['compiled_vs_joint']['mean_ci95'][0] > -.005
        protected = [branch(record, mode)['test'] for record in rows for mode in ['CD', 'CJ']]
        criteria['compiled_consumer_preserved'] = all(p['prediction_changes'] == 0 and
            p['max_absolute_healthy_logit_error'] < 1e-8 for p in protected)
        arms = {}
        for mode in MODES:
            endpoints = [branch(record, mode)['test'] for record in rows]
            arms[mode] = {key: describe([endpoint[key]['acc'] for endpoint in endpoints])
                          for key in ['own', 'frozen', 'reference']}
            arms[mode].update({key: describe([e[key] for e in endpoints]) for key in
                              ['disagreement', 'auxiliary_mse', 'relative_trace']})
            arms[mode].update(frozen_nll=describe([e['frozen']['nll'] for e in endpoints]),
                frozen_ece=describe([e['frozen']['ece'] for e in endpoints]),
                certified_count=sum(e['consumer_audit']['certified'] for e in endpoints),
                max_absolute_logit_error=max(e['max_absolute_healthy_logit_error'] for e in endpoints),
                upper_disagreement=describe([e['consumer_audit']['upper_remaining_disagreement'] for e in endpoints]))
        latency = {}
        for batch in ['1', '1024']:
            latency[batch] = {}
            for mode in ['compiled', 'two_encoders']:
                ratios = [r['deployment']['timings'][batch][mode]['median_seconds'] /
                          r['deployment']['timings'][batch]['ordinary']['median_seconds'] for r in rows]
                latency[batch][mode] = describe(ratios)
        domains[domain] = dict(seeds=[r['seed'] for r in rows], contrasts=contrasts,
            criteria=criteria, all_benefit_compatibility_criteria_pass=all(criteria.values()),
            arms=arms, latency_ratios=latency,
            mean_batch1024_latency_below_1p2=latency['1024']['compiled']['mean_ci95'][1] < 1.2,
            protected_rank=describe([r['projector']['protected_rank'] for r in rows]),
            free_dimensions=describe([r['projector']['free_dimensions'] for r in rows]),
            compiled_auxiliary_mse_ratio=describe([branch(r, 'CJ')['test']['auxiliary_mse'] /
                                                  branch(r, 'CD')['test']['auxiliary_mse'] for r in rows]))
    summary = dict(plan_sha256=sha(plan), domains=domains, model_optimizer_and_compiled_states_verified=state_count,
        confirmation_records=20, independently_replayed_consumers=120,
        maximum_replay_metric_error=max(b['maximum_metric_error'] for p in replays
            for b in json.loads(p.read_text())['branches']),
        maximum_folding_feature_error=max(b['folding_max_absolute_feature_error'] for r in records for b in r['branches']),
        family=dict(candidates=120, delta=.05, monitor=1024, budget=.05),
        uncertainty='Paired descriptive 95% t intervals over ten new seeds per task, fixed data/split/recipe.',
        input_record_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    (BASE / 'compiled_interface_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8', newline='\n')
    queues = [json.loads((BASE / folder / 'queue.json').read_text()) for folder in ['compiled_confirmation', 'compiled_replay']]
    assert all(q['finished'] and not q['failed'] and not q['running'] for q in queues)
    verification = dict(confirmation_records=20, model_optimizer_and_compiled_states_verified=180,
        sources_plan_pairing_frozen_trunk_and_Adam_verified=True, saved_consumers_replayed=120,
        all_compiled_outputs_preserved=True, compiled_encoder_has_no_reference_or_projection=True,
        deployment_affine_calls=3, maximum_replay_metric_error=summary['maximum_replay_metric_error'],
        peak_aggregate_rss_bytes=max(q['peak_aggregate_rss_bytes'] for q in queues),
        sum_job_wall_seconds=sum(job['wall_s'] for q in queues for job in q['completed']),
        criteria={domain: value['criteria'] for domain, value in domains.items()})
    (BASE / 'compiled_interface_verification.json').write_text(json.dumps(verification, indent=2)+'\n', encoding='utf-8', newline='\n')
    macros = {}
    for domain, label in [('radioml', 'Rml'), ('har', 'Har')]:
        item = domains[domain]
        for name, suffix in [('ordinary_auxiliary_gain', 'JointGain'), ('compiled_auxiliary_gain', 'AuxGain'),
                             ('compiled_total_gain', 'TotalGain'), ('compiled_vs_joint', 'JointCost')]:
            value = item['contrasts'][name]
            macros['Compile'+label+suffix] = '%.2f' % (100*value['mean'])
            macros['Compile'+label+suffix+'Low'] = '%.2f' % (100*value['mean_ci95'][0])
            macros['Compile'+label+suffix+'High'] = '%.2f' % (100*value['mean_ci95'][1])
        for mode in MODES:
            macros['Compile'+label+'Own'+mode] = '%.3f' % item['arms'][mode]['own']['mean']
        macros['Compile'+label+'Latency'] = '%.2f' % item['latency_ratios']['1024']['compiled']['mean']
        macros['Compile'+label+'SingleLatency'] = '%.2f' % item['latency_ratios']['1']['compiled']['mean']
        macros['Compile'+label+'MseRatio'] = '%.3f' % item['compiled_auxiliary_mse_ratio']['mean']
    error = max(item['arms'][mode]['max_absolute_logit_error'] for item in domains.values() for mode in ['CD', 'CJ'])
    coefficient, exponent = ('%.1e' % error).split('e')
    macros['CompileLogitError'] = coefficient+r'\times10^{'+str(int(exponent))+'}'
    (ROOT / 'paper/compiled_numbers.tex').write_text('% Generated from all twenty fixed confirmation runs.\n' +
        '\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key, value in macros.items())+'\n', encoding='utf-8', newline='\n')
    table = [r'\begin{table}[t]', r'\centering',
        r'\caption{Folded single-encoder continuation, ten new seeds per task. D/J update the full encoder; LD/LJ freeze the nonlinear trunk; CD/CJ additionally constrain the last affine map to preserve the normalized healthy consumer. All use the same native float64 final affine and float32 trunk. Means with population seed SD; the 120-consumer bank uses a five-point relative-error budget.}',
        r'\label{tab:compiled_full}', r'\small', r'\begin{tabular}{llrrrrr}', r'\toprule',
        r'Task & arm & own accuracy & frozen accuracy & frozen NLL & ECE & certified \\', r'\midrule']
    for domain, label in [('radioml', 'SNR'), ('har', 'HAR recon.')]:
        for index, mode in enumerate(MODES):
            arm = domains[domain]['arms'][mode]
            cells = [r'$%.3f\pm%.3f$' % (arm[key]['mean'], arm[key]['sd']) for key in ['own', 'frozen']]
            cells += ['$%.3f$' % arm['frozen_nll']['mean'], '$%.3f$' % arm['frozen_ece']['mean'], str(arm['certified_count'])+'/10']
            table.append((label if index == 0 else '')+' & '+mode+' & '+' & '.join(cells)+r' \\')
        table.append(r'\midrule' if domain == 'radioml' else r'\bottomrule')
    table += [r'\end{tabular}', r'\end{table}']
    (ROOT / 'paper/compiled_interface_table.tex').write_text('\n'.join(table)+'\n', encoding='utf-8', newline='\n')
    contrasts = [r'\begin{table}[t]', r'\centering',
        r'\caption{All declared folded-control accuracy contrasts in percentage points, mean with paired descriptive 95\% interval. Conditional on each fixed dataset/split/recipe. The first four rows have required lower limits $0,0,0,-0.5$ respectively; all other rows diagnose freezing and constraint costs regardless of sign.}',
        r'\label{tab:compiled_contrasts}', r'\small', r'\begin{tabular}{lrr}', r'\toprule',
        r'Contrast & RadioML SNR & HAR reconstruction \\', r'\midrule']
    for name, left, right in CONTRASTS:
        cells = []
        for domain in ['radioml', 'har']:
            value = domains[domain]['contrasts'][name]
            pattern = (r'$%+.3f\ [%.3f,\,%.3f]$' if name in
                       ['constraint_joint_cost', 'constraint_detach_cost', 'frozen_trunk_auxiliary_gain']
                       else r'$%+.2f\ [%.2f,\,%.2f]$')
            cells.append(pattern % (100*value['mean'], 100*value['mean_ci95'][0], 100*value['mean_ci95'][1]))
        contrasts.append(left+'--'+right+' & '+' & '.join(cells)+r' \\')
    contrasts += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (ROOT / 'paper/compiled_contrasts_table.tex').write_text('\n'.join(contrasts)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(verification=verification, domains={name: dict(criteria=item['criteria'],
        contrasts=item['contrasts'], latency=item['latency_ratios']) for name, item in domains.items()}), indent=2))


if __name__ == '__main__':
    main()
