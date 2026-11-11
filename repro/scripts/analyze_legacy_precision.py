"""Select and replay a zero-exact-call legacy precision policy."""

import argparse
import json
from pathlib import Path

import torch

import analyze_stable_prefix as common
from analyze_early_update import digest
from legacy_precision_cpu import METHODS
from prospective_vision_cpu import disagreement


def development(args):
    [(path, row)] = list(common.records(args, 'k', 1))
    assert row['stage'] == 'fit' and row['calibration_images'] == 1024 and not row['test_loaded']
    data = common.replay(args, path, row)
    assert tuple(r['method'] for r in row['methods']) == METHODS
    candidates = []
    for method in row['methods']:
        name = method['method']
        values = []
        assert [r['reference_seed'] for r in method['outcomes']] == [7410, 7411]
        for ref in method['outcomes']:
            saved = data[name][str(ref['reference_seed'])]
            rates = disagreement(saved['cheap'], saved['exact'], saved['thresholds'])
            assert rates == ref['rates']
            values.append(max(rates))
        candidates.append(dict(method=name, state_sha256=method['state_sha256'],
                               maximum_disagreements=values, passed=max(values) <= .03))
    winner = next((r for r in candidates if r['passed']), None)
    return dict(complete=True, plan_sha256=digest(args.plan), width=512,
                job=path.stem, candidates=candidates, **(winner or dict(passed=False)))


def audit(args):
    result = common.audit(args, force_zero=True)
    result.pop('prefix_cost_proxy')
    choice = json.loads(args.choice.read_text())
    return dict(**result, method=choice['method'], approximate_block_fraction=1.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('development', 'audit', 'timing'), required=True)
    for name in ('records', 'arrays', 'plan', 'out', 'choice', 'audit-records'):
        parser.add_argument('--' + name, type=Path, required=name in ('records', 'plan', 'out'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {'development': development, 'audit': audit, 'timing': common.timing}[args.stage](args)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    fields = ('method', 'candidates', 'passed', 'access_gate_passed', 'latency_gate_passed', 'ratios')
    print(json.dumps({key: result[key] for key in fields if key in result}))
