"""Fit and replay the fixed RadioML trajectory and release-rule study."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_har_transfer import ROOT, RESULTS, digest, interval, save, state_hash
from consumer_routing import decisions
from har_early_forecast import BASELINES, SCALES, examples, fit, predict, scores

PLAN = ROOT / 'docs/plans/2026-11-11-rml-trajectory.md'
FORECASTER = RESULTS / 'rml_trajectory_forecaster_20261111.json'
CONFIRMATION = RESULTS / 'rml_trajectory_confirmation_20261111.json'
REPAIR_PLAN = ROOT / 'docs/plans/2026-11-11-rml-early-repair.md'
REPAIR_FORECASTER = RESULTS / 'rml_repair_forecaster_20261111.json'
REPAIR_CONFIRMATION = RESULTS / 'rml_repair_confirmation_20261111.json'


def load(stage, records, arrays, repair=False):
    confirm = stage == 'confirmation'
    prefix, start, count = ('c', 8600, 16) if confirm else ('d', 8500, 12)
    plan = REPAIR_PLAN if repair else PLAN
    forecaster = REPAIR_FORECASTER if repair else FORECASTER
    if repair:
        start += 200
    for offset in range(count):
        key = f'{prefix}{offset:03d}'
        row = json.loads((records / f'{key}.json').read_text())
        receipt_path = records / f'{key}.forecast.json'
        receipt = json.loads(receipt_path.read_text())
        assert row['complete'] and row['device'] == 'cpu' and row['stage'] == stage
        assert row['seed'] == receipt['seed'] == start + offset
        assert row['plan_sha256'] == receipt['plan_sha256'] == digest(plan)
        assert row['test_loaded'] == confirm and not receipt['test_loaded'] and receipt['early_epoch'] == 6
        assert digest(receipt_path) == row['forecast_sha256']
        assert receipt['forecaster_sha256'] == (digest(forecaster) if confirm else None)
        path = arrays / f'{key}_states/replay.pt'
        assert digest(path) == row['replay_sha256']
        replay = torch.load(path, map_location='cpu', weights_only=True)
        outputs = replay['outputs']
        assert receipt['early_sha256'] == state_hash(tuple(outputs[k] for k in ('old_early', 'early', 'thresholds')))
        assert torch.equal(outputs['old']['modulation'], outputs['retained']['modulation'])
        assert len(replay['labels']) == (4400 if confirm else 1600)
        assert row.get('repair', False) == repair
        yield row, outputs, receipt, replay['labels']


def calibrate(rows, plan=PLAN):
    examples_by_seed = [row[:3] for row in rows]
    fitted = fit(examples_by_seed[:8])
    errors = {scale: [] for scale in SCALES}
    for _, outputs, _, _ in rows[8:]:
        for program, head, x, y in examples(outputs):
            prior, delta = scores(fitted, program, head, x)
            target = y.double().mean(0).max()
            for scale in SCALES:
                errors[scale].append(float(abs((prior + scale * delta).clamp(0, 1).max() - target)))
    means = {str(scale): float(np.mean(errors[scale])) for scale in SCALES}
    return dict(plan_sha256=digest(plan), scale=min(SCALES, key=lambda s: means[str(s)]),
                tuning_mae=means, model=fit(examples_by_seed))


def confirm(rows):
    config = json.loads(FORECASTER.read_text())
    methods = ('forecast', 'unshrunk', *BASELINES)
    references = []
    for row, outputs, receipt, labels in rows:
        thresholds = outputs['thresholds']['modulation']
        old = decisions(outputs['old']['modulation'], thresholds)
        retained = decisions(outputs['retained']['modulation'], thresholds)
        episodes = []
        for program, (first, sixth) in outputs['early'].items():
            estimated = predict(config, program, outputs['old_early'], first, sixth, outputs['thresholds'])
            final = decisions(outputs['terminal'][program]['modulation'], thresholds)
            actual = (final != old).double().mean(0).tolist()
            forecasts = {name: values['modulation'] for name, values in estimated.items()}
            outcomes = {}
            for name, values in forecasts.items():
                np.testing.assert_allclose(values, receipt['forecasts'][program][name]['modulation'],
                                           rtol=0, atol=1e-9)
                uses_reference = max(values) > .05
                decisions_served = retained if uses_reference else final
                violation = (decisions_served != old).double().mean(0).max().item() > .05
                outcomes[name] = dict(uses_reference=uses_reference, violation=violation)
            episodes.append(dict(program=program, actual=actual, forecasts=forecasts, release=outcomes,
                                 primary_pp=(outputs['primary'][program].argmax(1) == labels).double().mean().item() * 100))
        references.append(dict(seed=row['seed'], labeled=row['labeled_examples'], episodes=episodes))
    all_episodes = [e for r in references for e in r['episodes']]
    mae = {method: np.array([np.mean([abs(max(e['forecasts'][method]) - max(e['actual']))
                          for e in r['episodes']]) * 100 for r in references]) for method in methods}
    differences = {method: interval(mae['forecast'] - mae[method]) for method in BASELINES}

    def counts(group, method):
        return dict(episodes=len(group), exceedances=sum(max(e['actual']) > .05 for e in group),
                    reference=sum(e['release'][method]['uses_reference'] for e in group),
                    violations=sum(e['release'][method]['violation'] for e in group),
                    false_alarms=sum(e['release'][method]['uses_reference'] and max(e['actual']) <= .05 for e in group))

    release = {method: counts(all_episodes, method) for method in methods}
    by_rate = {rate: {method: counts([e for e in all_episodes if e['program'].endswith('/' + rate)], method)
                     for method in methods} for rate in ('1e-05', '0.0001', '0.001')}
    primary = {}
    for rate in by_rate:
        deltas = []
        for ref in references:
            arms = {e['program']: e['primary_pp'] for e in ref['episodes']}
            deltas.append(arms['joint/' + rate] - arms['detach/' + rate])
        primary[rate] = interval(deltas)
    return dict(plan_sha256=digest(PLAN), forecaster_sha256=digest(FORECASTER), scale=config['scale'],
                forecast_passed=all(v['ci95'][1] < 0 for v in differences.values()),
                mae_pp={method: interval(values) for method, values in mae.items()},
                paired_mae_difference_pp=differences, release=release, release_by_rate=by_rate,
                joint_gain_pp=primary, references=references)



def confirm_repair(rows):
    from rml_trajectory_cpu import POLICIES, REPAIR_RATES, release_choices

    config = json.loads(REPAIR_FORECASTER.read_text())
    references = []
    for row, outputs, receipt, labels in rows:
        thresholds = outputs['thresholds']['modulation']
        old = decisions(outputs['old']['modulation'], thresholds)
        forecasts, actual = {}, {}
        for key, (first, sixth) in outputs['early'].items():
            estimate = predict(config, key, outputs['old_early'], first, sixth, outputs['thresholds'])
            for method, values in estimate.items():
                np.testing.assert_allclose(values['modulation'], receipt['forecasts'][key][method]['modulation'],
                                           atol=1e-9, rtol=0)
            forecasts[key] = estimate
            output = decisions(outputs['terminal'][key]['modulation'], thresholds)
            actual[key] = float((output != old).double().mean(0).max())
        policy = release_choices(forecasts, outputs['primary'])
        assert policy == receipt['policy']
        actual['reference'] = 0.
        scores_pp = {key: 100 * (z.argmax(1) == labels).double().mean().item()
                     for key, z in outputs['primary'].items()}
        references.append(dict(seed=row['seed'], actual=actual, policy=policy, scores_pp=scores_pp))
    methods = {}
    for method in POLICIES:
        risks = [ref['actual'][choice] for ref in references for choice in ref['policy'][method].values()]
        choices = [choice for ref in references for choice in ref['policy'][method].values()]
        methods[method] = dict(episodes=len(choices), reference=choices.count('reference'),
                               aligned=sum(key.endswith('/aligned') for key in choices),
                               direct=sum(key != 'reference' and not key.endswith('/aligned') for key in choices),
                               violations=sum(risk > .05 for risk in risks), maximum_disagreement=max(risks))
    differences = {}
    for rate in REPAIR_RATES:
        differences[str(rate)] = interval([ref['scores_pp'][f'joint/{rate}'] -
                                          ref['scores_pp'][f'detach/{rate}'] for ref in references])
    return dict(plan_sha256=digest(REPAIR_PLAN), forecaster_sha256=digest(REPAIR_FORECASTER),
                scale=config['scale'], policy=methods, joint_gain_pp=differences,
                practical_passed=methods['forecast']['violations'] == 0 and
                methods['forecast']['reference'] < methods['schedule_only']['reference'],
                references=references)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('calibration', 'confirmation'))
    parser.add_argument('--records', type=Path)
    parser.add_argument('--arrays', type=Path)
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    kind = 'contrastiverml' + ('repair' if args.repair else '') + args.stage
    arrays = args.arrays or ROOT / 'tmp' / f'conditional_{kind}'
    torch.set_num_threads(4)
    rows = list(load(args.stage, args.records or arrays, arrays, args.repair))
    if args.repair:
        result, output = ((calibrate(rows, REPAIR_PLAN), REPAIR_FORECASTER) if args.stage == 'calibration' else
                          (confirm_repair(rows), REPAIR_CONFIRMATION))
    else:
        result, output = ((calibrate(rows), FORECASTER) if args.stage == 'calibration' else
                          (confirm(rows), CONFIRMATION))
    save(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('model', 'references')}, indent=2))


if __name__ == '__main__':
    main()
