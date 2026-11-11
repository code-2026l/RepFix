"""Replay width selection, the held-out access gate, and matched CPU timing."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from analyze_early_update import digest, interval
from prospective_vision_cpu import disagreement
from stable_prefix_cpu import WIDTHS, select_fraction


def records(args, prefix, count):
    paths = sorted(args.records.glob(prefix + '???.json'))
    assert len(paths) == count
    for path in paths:
        row = json.loads(path.read_text())
        assert row['complete'] and row['device'] == 'cpu'
        assert row['plan_sha256'] == digest(args.plan)
        yield path, row


def replay(args, path, row):
    raw = args.arrays / f'{path.stem}_states/replay.pt'
    assert digest(raw) == row['replay_sha256']
    return torch.load(raw, map_location='cpu', weights_only=True)


def development(args):
    candidates = []
    for path, row in records(args, 'k', len(WIDTHS)):
        assert row['stage'] == 'fit' and not row['test_loaded']
        assert row['teacher_images'] == 12000 and len(row['losses']) == 12
        data = replay(args, path, row)
        values = []
        for r in row['rows']:
            saved = data[str(r['reference_seed'])]
            fraction, rates = select_fraction(saved['cheap'], saved['exact'], saved['thresholds'])
            assert fraction == r['fraction'] and rates == r['rates']
            values.append(fraction)
        assert [r['reference_seed'] for r in row['rows']] == [7410, 7411]
        cost = float(np.mean([1. if q == 1. else row['width'] / 512 + q for q in values]))
        candidates.append(dict(width=row['width'], job=path.stem, state_sha256=row['state_sha256'],
            fractions=values, mean_cost_proxy=cost, passed=max(values) <= .5 and cost < 1.))
    assert [r['width'] for r in candidates] == list(WIDTHS)
    eligible = [r for r in candidates if r['passed']]
    selected = min(eligible or candidates, key=lambda r: (r['mean_cost_proxy'], r['width']))
    return dict(complete=True, plan_sha256=digest(args.plan), candidates=candidates, **selected)


def audit(args, force_zero=False):
    choice = json.loads(args.choice.read_text())
    assert choice['passed'] and choice['plan_sha256'] == digest(args.plan)
    rows = []
    permutation = torch.randperm(10000, generator=torch.Generator().manual_seed(7590))
    for path, row in records(args, 'c', 6):
        assert row['stage'] == 'audit' and row['test_loaded'] and row['width'] == choice['width']
        assert row['current_predictions_shared'] and row['test_examples'] == 1500
        receipt_path = path.with_suffix('.forecast.json')
        assert digest(receipt_path) == row['forecast_sha256']
        receipt = json.loads(receipt_path.read_text())
        assert receipt['early_epoch'] == 2 and not receipt['test_loaded']
        assert receipt['choice_sha256'] == digest(args.choice) and receipt['plan_sha256'] == digest(args.plan)
        assert receipt['reference_seed'] == row['reference_seed']
        data = replay(args, path, row)
        start = (row['reference_seed'] - 7420) * 1500
        assert torch.equal(data['test_indices'], permutation[start:start + 1500])
        fraction, rates = select_fraction(data['monitor_cheap'], data['monitor_exact'], data['thresholds'])
        assert (0. if force_zero else fraction) == receipt['prefix_fraction']
        assert rates == receipt['monitor_rates']
        for name in ('prefix', 'previous'):
            q = receipt[name + '_fraction']
            expected = sum(math.ceil(q * len(b)) for b in data['labels'].split(128))
            assert expected == row['calls'][name]
        assert receipt['previous_fraction'] == .9
        logits = data['logits']
        actual = {name: disagreement(logits[name], logits['exact'], data['thresholds'])
                  for name in ('prefix', 'previous', 'prefix_cheap')}
        assert actual == row['outcomes']
        rows.append(dict(reference_seed=row['reference_seed'], contract_rates=actual,
            maximum_disagreement={name: max(values) for name, values in actual.items()},
            planned_fraction={name: receipt[name + '_fraction'] for name in ('prefix', 'previous')},
            physical_fraction={name: count / 1500 for name, count in row['calls'].items()},
            current_accuracy=float((logits['current'].argmax(1) == data['labels']).double().mean())))
    assert [r['reference_seed'] for r in rows] == list(range(7420, 7426))
    passed = all(r['maximum_disagreement']['prefix'] <= .05 and
                 r['physical_fraction']['prefix'] <= (0. if force_zero else .50) for r in rows)
    query = {name: interval([r['physical_fraction'][name] for r in rows]) for name in ('prefix', 'previous')}
    return dict(complete=True, width=choice['width'], access_gate_passed=passed, official_test_examples=9000,
        current_predictions_shared=True, rows=rows, exact_referral=query,
        prefix_cost_proxy=interval([1. if r['planned_fraction']['prefix'] == 1.
                                   else choice['width'] / 512 + r['physical_fraction']['prefix']
                                   for r in rows]))


def timing(args):
    rows = []
    for path, row in records(args, 'n', 6):
        assert row['stage'] == 'timing' and row['batch'] == 128 and row['threads'] == 12
        audit_path = args.audit_records / ('c' + path.stem[1:] + '.json')
        assert digest(audit_path) == row['audit_sha256']
        assert json.loads(audit_path.read_text())['reference_seed'] == row['reference_seed']
        assert all(len(s) == 31 and min(s) > 0 for s in row['samples_seconds'].values())
        medians = {name: float(np.median(s)) for name, s in row['samples_seconds'].items()}
        assert medians == row['medians_seconds']
        rows.append(dict(reference_seed=row['reference_seed'], seconds=medians))
    assert [r['reference_seed'] for r in rows] == list(range(7420, 7426))
    ratios = {f'{num}_over_{den}': interval([r['seconds'][num] / r['seconds'][den] for r in rows])
              for num, den in [('prefix', 'previous'), ('prefix', 'current'), ('prefix', 'exact'),
                               ('previous', 'current'), ('exact', 'current')]}
    return dict(complete=True, rows=rows, ratios=ratios,
                latency_gate_passed=ratios['prefix_over_previous']['ci95'][1] < 1.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('development', 'audit', 'timing'), required=True)
    for name in ('records', 'arrays', 'plan', 'out', 'choice', 'audit-records'):
        parser.add_argument('--' + name, type=Path, required=name in ('records', 'plan', 'out'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {'development': development, 'audit': audit, 'timing': timing}[args.stage](args)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({key: value for key, value in result.items() if key in
        ('width', 'candidates', 'passed', 'access_gate_passed', 'latency_gate_passed', 'ratios')}))
