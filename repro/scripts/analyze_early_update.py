"""Replay sealed early choices and score the fixed prospective comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
import torch

from paired_readout_continuation import state_hash
from prospective_vision_cpu import EVALUATION_BUDGET, FAMILIES, MULTIPLIERS, RATES, choose, disagreement, split


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values):
    data = np.asarray(values, dtype=float)
    mean = float(data.mean())
    radius = float(t.ppf(.975, len(data) - 1) * data.std(ddof=1) / np.sqrt(len(data)))
    return dict(mean=mean, ci95=[mean - radius, mean + radius], reference_values=data.tolist())


def load(args):
    rows = []
    plan_sha = digest(args.plan)
    for path in sorted(args.records.glob('[cd]???.json')):
        outcome = json.loads(path.read_text())
        receipt_path = path.with_suffix('.forecast.json')
        receipt = json.loads(receipt_path.read_text())
        assert outcome['complete'] and outcome['device'] == 'cpu'
        assert not outcome['official_test_loaded'] and outcome['epochs'] == list(range(1, 9))
        assert outcome['plan_sha256'] == receipt['plan_sha256'] == plan_sha
        assert outcome['stage'] == receipt['stage'] == args.stage
        assert all(outcome[k] == receipt[k] for k in ('reference_seed', 'rate', 'alpha'))
        assert receipt['early_epoch'] == 2 and receipt['evaluated_examples'] == 0
        assert digest(receipt_path) == outcome['forecast_sha256']
        array_path = args.arrays / (path.stem + '_states') / 'replay.pt'
        assert digest(array_path) == outcome['replay_sha256']
        data = torch.load(array_path, map_location='cpu', weights_only=True)
        assert state_hash((data['monitor_old'], data['forecasts'], data['thresholds'])) == receipt['monitor_logits_sha256']
        assert len(data['monitor_old']) == outcome['monitor_examples'] == 512
        assert len(data['labels']) == outcome['validation_examples'] == 2000
        assert torch.equal(data['validation_indices'], split(outcome['reference_seed'])[-1])
        risks = {k: {f: disagreement(z, data['monitor_old'], data['thresholds'])
                     for f, z in candidates.items()} for k, candidates in data['forecasts'].items()}
        assert list(risks) == list(map(str, MULTIPLIERS)) and risks == receipt['risks']
        assert {k: choose(v) for k, v in risks.items()} == receipt['actions']
        actual = {f: disagreement(z, data['old'], data['thresholds']) for f, z in data['outcomes'].items()}
        assert list(actual) == list(FAMILIES) and actual == outcome['outcomes']
        assert not any(actual['reference'])
        for name in ('current', 'stop'):
            accuracy = float((data[name].argmax(1) == data['labels']).double().mean())
            assert accuracy == outcome[name + '_accuracy']
        rows.append(dict(file=path.name, receipt=receipt, record=outcome, record_sha256=digest(path),
                         measured={f: max(r) for f, r in actual.items()}))
    refs = range(7410, 7412) if args.stage == 'development' else range(7420, 7426)
    keys = {(r['record']['reference_seed'], r['record']['rate'], r['record']['alpha']) for r in rows}
    expected = {(ref, rate, .1) for ref in refs for rate in RATES}
    if args.stage == 'confirmation':
        expected |= {(ref, 'nominal', 0.) for ref in refs}
    assert keys == expected and len(rows) == len(keys)
    return rows


def develop(rows, plan):
    errors = {str(k): float(np.mean([
        abs(max(row['receipt']['risks'][str(k)][family]) - row['measured'][family])
        for row in rows for family in FAMILIES[:-1]])) for k in MULTIPLIERS}
    selected = min(MULTIPLIERS, key=lambda k: (errors[str(k)], k))
    schedule_only = {}
    for rate in RATES:
        group = [row for row in rows if row['record']['rate'] == rate]
        schedule_only[rate] = {f: [float(np.mean([r['measured'][f] for r in group]))]
                               for f in FAMILIES[:-1]}
    return dict(complete=True, plan_sha256=digest(plan), multiplier=selected,
                development_mae=errors, schedule_only_risks=schedule_only,
                source_records={r['file']: r['record_sha256'] for r in rows})


def scores(rows, policy):
    p = np.array([r['prediction'][policy] > EVALUATION_BUDGET for r in rows])
    y = np.array([r['unsafe'] for r in rows])
    true_positive, true_negative = int((p & y).sum()), int((~p & ~y).sum())
    return dict(true_positive=true_positive, true_negative=true_negative,
                false_positive=int((p & ~y).sum()), false_negative=int((~p & y).sum()),
                sensitivity=true_positive / int(y.sum()) if y.any() else None,
                specificity=true_negative / int((~y).sum()) if (~y).any() else None,
                unsafe_repaired_without_reference=sum(r['unsafe'] and
                    r['actions'][policy] not in ('reference', 'direct') and
                    r['chosen_risk'][policy] <= EVALUATION_BUDGET for r in rows),
                chosen_reference=sum(r['actions'][policy] == 'reference' for r in rows),
                false_safe_actions=sum(r['chosen_risk'][policy] > EVALUATION_BUDGET for r in rows))


def confirm(rows, choice_path):
    choice = json.loads(choice_path.read_text())
    choice_sha = digest(choice_path)
    k = str(choice['multiplier'])
    episodes, utility = [], []
    for row in rows:
        record, receipt, actual = row['record'], row['receipt'], row['measured']
        assert receipt['multiplier'] == choice['multiplier'] and receipt['choice_sha256'] == choice_sha
        schedule = choice['schedule_only_risks'][record['rate']]
        assert schedule == receipt['schedule_only_risks'] and choose(schedule) == receipt['schedule_only_action']
        if not record['alpha']:
            continue
        actions = dict(forecast=receipt['actions'][k], persistence=receipt['actions']['1.0'],
                       schedule_only=receipt['schedule_only_action'])
        episodes.append(dict(reference_seed=record['reference_seed'], rate=record['rate'],
            prediction=dict(forecast=max(receipt['risks'][k]['direct']),
                            persistence=max(receipt['risks']['1.0']['direct']),
                            schedule_only=max(schedule['direct'])),
            unsafe=actual['direct'] > EVALUATION_BUDGET, actions=actions,
            chosen_risk={name: actual[arm] for name, arm in actions.items()},
            actual_risk=actual, actual_contract_rates=record['outcomes'],
            oracle=next(arm for arm in FAMILIES if actual[arm] <= EVALUATION_BUDGET),
            current_accuracy=record['current_accuracy'], stop_accuracy=record['stop_accuracy']))
    policies = ('forecast', 'persistence', 'schedule_only')
    refs = range(7420, 7426)
    results = {name: scores(episodes, name) for name in policies}
    mae = {}
    for name in policies:
        values = [float(np.mean([abs(row['prediction'][name] - row['actual_risk']['direct'])
                                 for row in episodes if row['reference_seed'] == ref])) for ref in refs]
        mae[name] = interval(values)
    differences = {name: interval(np.asarray(mae['forecast']['reference_values']) -
                                  np.asarray(mae[name]['reference_values'])) for name in policies[1:]}
    unsafe = sum(e['unsafe'] for e in episodes)
    safe = len(episodes) - unsafe
    policy = results['forecast']
    forecast_gate = (safe >= 3 and unsafe >= 3 and policy['sensitivity'] >= .8 and
                     policy['specificity'] >= .8 and differences['persistence']['ci95'][1] < 0)
    intervention_gate = (unsafe > 0 and policy['false_safe_actions'] == 0 and
                         policy['unsafe_repaired_without_reference'] >= unsafe / 2)
    for ref in refs:
        joint, control = sorted([r['record'] for r in rows if r['record']['reference_seed'] == ref
                                 and r['record']['rate'] == 'nominal'], key=lambda r: -r['alpha'])
        utility.append(dict(reference_seed=ref,
            gain_pp=100 * (joint['current_accuracy'] - control['current_accuracy']),
            loss_ratio=joint['auxiliary_loss'] / control['auxiliary_loss']))
    stop_gain = [float(np.mean([100 * (r['current_accuracy'] - r['stop_accuracy']) for r in episodes
                               if r['reference_seed'] == ref])) for ref in refs]
    return dict(complete=True, choice_sha256=choice_sha, multiplier=choice['multiplier'],
                true_safe=safe, true_unsafe=unsafe, forecasts_replayed=len(rows),
                forecast_gate_passed=forecast_gate, intervention_gate_passed=intervention_gate,
                current_predictions_shared=True, policies=results, direct_mae=mae,
                forecast_minus_comparator_mae=differences, episodes=episodes,
                continued_minus_early_stop_pp=interval(stop_gain),
                nominal_auxiliary_utility=dict(gain_pp=interval([v['gain_pp'] for v in utility]),
                    loss_ratio=interval([v['loss_ratio'] for v in utility]), rows=utility))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('development', 'confirmation'), required=True)
    for name in ('records', 'arrays', 'plan', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--choice', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = load(args)
    result = develop(rows, args.plan) if args.stage == 'development' else confirm(rows, args.choice)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    fields = ('multiplier', 'forecast_gate_passed', 'intervention_gate_passed', 'true_unsafe')
    print(json.dumps({key: result[key] for key in fields if key in result}))
