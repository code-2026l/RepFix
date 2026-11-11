"""Replay fixed early warning and limited-reference policies on sealed outputs."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from analyze_early_update import digest, interval, load
from consumer_routing import decision_margin, route_logits
from prospective_vision_cpu import EVALUATION_BUDGET, FAMILIES, disagreement

FRACTIONS = (.25, .50, .75, .90)
MONITOR_BUDGET = .025


def ranking(logits, thresholds, policy, seed):
    if policy == 'margin':
        return torch.argsort(decision_margin(logits, thresholds), stable=True)
    return torch.randperm(len(logits), generator=torch.Generator().manual_seed(seed))


def route(cheap, legacy, order, fraction):
    referred = torch.zeros(len(cheap), dtype=torch.bool)
    referred[order[:math.ceil(fraction * len(cheap))]] = True
    return route_logits(cheap, legacy, referred), int(referred.sum())


def select(early, legacy, thresholds, policy, seed):
    for family in FAMILIES[:-1]:
        if max(disagreement(early[family], legacy, thresholds)) <= MONITOR_BUDGET:
            return family, 0.
    order = ranking(early['ridge'], thresholds, policy, seed)
    for fraction in FRACTIONS:
        candidate, _ = route(early['ridge'], legacy, order, fraction)
        if max(disagreement(candidate, legacy, thresholds)) <= MONITOR_BUDGET:
            return 'ridge', fraction
    return 'reference', 1.


def apply(data, family, fraction, policy, seed):
    legacy = data['outcomes']['reference']
    if family == 'reference':
        result, calls = legacy, len(legacy)
    elif not fraction:
        result, calls = data['outcomes'][family], 0
    else:
        cheap = data['outcomes'][family]
        order = ranking(cheap, data['thresholds'], policy, seed)
        result, calls = route(cheap, legacy, order, fraction)
    rates = disagreement(result, data['old'], data['thresholds'])
    return dict(family=family, fraction=fraction, reference_calls=calls,
                actual_rates=rates, maximum_disagreement=max(rates))


def warning(rows):
    unsafe = [r for r in rows if r['actual_direct'] > EVALUATION_BUDGET]
    safe = [r for r in rows if r['actual_direct'] <= EVALUATION_BUDGET]
    detected = sum(r['early_direct'] > EVALUATION_BUDGET for r in unsafe)
    correctly_clear = sum(r['early_direct'] <= EVALUATION_BUDGET for r in safe)
    sensitivity = detected / len(unsafe) if unsafe else None
    specificity = correctly_clear / len(safe) if safe else None
    return dict(unsafe=len(unsafe), safe=len(safe), detected=detected,
                correctly_clear=correctly_clear, sensitivity=sensitivity,
                specificity=specificity,
                passed=len(unsafe) >= 3 and len(safe) >= 3 and sensitivity >= .8 and specificity >= .8)


def analyze(args):
    groups = [
        (args.original_records, args.original_arrays, args.original_plan, 'confirmation'),
        (args.stable_records, args.stable_arrays, args.plan, 'stability'),
    ]
    rows = []
    for records, arrays, plan, stage in groups:
        for row in load(SimpleNamespace(records=records, arrays=arrays, plan=plan, stage=stage)):
            record, receipt = row['record'], row['receipt']
            if not record['alpha']:
                continue
            ref = record['reference_seed']
            if stage == 'confirmation':
                assert receipt['multiplier'] == 1.
            stem = Path(row['file']).stem
            data = torch.load(arrays / f'{stem}_states/replay.pt', map_location='cpu', weights_only=True)
            early, old, thresholds = data['forecasts']['1.0'], data['monitor_old'], data['thresholds']
            result = {}
            for policy in ('margin', 'uniform'):
                family, fraction = select(early, old, thresholds, policy, ref + 90000)
                result[policy] = apply(data, family, fraction, policy, ref + 90001)
            selected = result['margin']
            matched = apply(data, selected['family'], selected['fraction'], 'uniform', ref + 90001)
            original = receipt['actions']['1.0']
            rows.append(dict(reference_seed=ref, rate=record['rate'],
                early_direct=max(receipt['risks']['1.0']['direct']), actual_direct=row['measured']['direct'],
                original_policy=apply(data, original, float(original == 'reference'), 'uniform', ref + 90001),
                policies=result, uniform_at_margin_budget=matched))
    assert len(rows) == 24
    unsafe = [r for r in rows if r['actual_direct'] > EVALUATION_BUDGET]
    base = [r for r in rows if r['rate'] != 'stable']
    summary = {}
    for policy in ('margin', 'uniform'):
        violations = sum(r['policies'][policy]['maximum_disagreement'] > EVALUATION_BUDGET for r in rows)
        median = float(np.median([r['policies'][policy]['fraction'] for r in unsafe])) if unsafe else None
        summary[policy] = dict(false_safe_actions=violations, median_unsafe_reference_fraction=median,
            maximum_selected_disagreement=max(r['policies'][policy]['maximum_disagreement'] for r in rows),
            mean_reference_fraction=float(np.mean([r['policies'][policy]['fraction'] for r in rows])),
            original_grid_reference_fraction=float(np.mean([r['policies'][policy]['fraction'] for r in base])),
            passed=bool(unsafe) and violations == 0 and median <= .50)
    refs = range(7420, 7426)
    mae = [float(np.mean([abs(r['early_direct'] - r['actual_direct'])
                         for r in rows if r['reference_seed'] == ref])) for ref in refs]
    differences = [float(np.mean([
        r['policies']['margin']['maximum_disagreement'] - r['uniform_at_margin_budget']['maximum_disagreement']
        for r in rows if r['reference_seed'] == ref])) for ref in refs]
    return dict(complete=True, plan_sha256=digest(args.plan), episodes=rows,
                warning=warning(rows), direct_mae=interval(mae), policies=summary,
                by_schedule={rate: warning([r for r in rows if r['rate'] == rate])
                             for rate in ('stable', 'mild', 'nominal', 'strong')},
                original_policy_reference_fraction=float(np.mean([r['original_policy']['fraction'] for r in base])),
                original_policy_budget_violations=sum(
                    r['original_policy']['maximum_disagreement'] > EVALUATION_BUDGET for r in base),
                margin_minus_matched_uniform_disagreement=interval(differences))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('original-records', 'original-arrays', 'original-plan', 'stable-records',
                 'stable-arrays', 'plan', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = analyze(args)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({name: result[name] for name in ('warning', 'policies')}))
