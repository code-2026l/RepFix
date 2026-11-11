"""Independently replay one-pass RadioML compatibility and matched latency."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_har_transfer import ROOT, RESULTS, digest, interval, save
from consumer_routing import decisions
from rml_one_pass_cpu import CANDIDATES

PLAN = ROOT / 'docs/plans/2026-11-11-rml-one-pass.md'
CHOICE = RESULTS / 'rml_one_pass_choice_20261111.json'
STAGES = dict(development=('d', 8900, 8), confirmation=('c', 9000, 16), timing=('t', 9000, 16))



def verify_shared_legacy(program):
    chosen, reference = (program[name] for name in ('chosen', 'reference'))
    old_width = reference['2.weight'].shape[0]
    old_classes = reference['4.weight'].shape[0]
    for suffix in ('weight', 'bias'):
        assert torch.equal(chosen['0.' + suffix], reference['0.' + suffix])
        assert torch.equal(chosen['2.' + suffix][:old_width], reference['2.' + suffix])
    assert torch.equal(chosen['4.weight'][-old_classes:, :old_width], reference['4.weight'])
    assert torch.equal(chosen['4.bias'][-old_classes:], reference['4.bias'])
    assert not torch.count_nonzero(chosen['4.weight'][-old_classes:, old_width:])


def load(stage, records, arrays, confirmation, plan, choice, anchored):
    prefix, start, count = STAGES[stage]
    start += 200 * anchored
    for offset in range(count):
        name = f'{prefix}{offset:03d}'
        row = json.loads((records / f'{name}.json').read_text())
        assert row['complete'] and row['device'] == 'cpu' and row['stage'] == stage
        assert row['plan_sha256'] == digest(plan) and row['seed'] == start + offset
        if stage == 'timing':
            reference = json.loads((confirmation / f'c{offset:03d}.json').read_text())
            assert row['program_sha256'] == reference['program_sha256']
            assert row['threads'] == 1
            assert all(len(blocks) == 9 for batch in row['timings'].values() for blocks in batch.values())
            yield row
            continue
        assert row['test_loaded'] == (stage == 'confirmation')
        assert row['choice_sha256'] == (digest(choice) if stage == 'confirmation' else None)
        folder = arrays / f'{name}_states'
        path = folder / 'replay.pt'
        assert digest(path) == row['replay_sha256']
        exact_selected = False
        if stage == 'confirmation':
            program_file = folder / 'program.pt'
            assert digest(program_file) == row['program_sha256']
            exact_selected = json.loads(choice.read_text())['selected'].startswith('exact/')
            if exact_selected:
                verify_shared_legacy(torch.load(program_file, map_location='cpu', weights_only=True))
        value = torch.load(path, map_location='cpu', weights_only=True)
        assert len(value['labels']) == (4400 if stage == 'confirmation' else 1600)
        thresholds = value['thresholds']
        old = decisions(value['old'], thresholds)
        risk = {name: (decisions(z, thresholds) != old).double().mean(0).tolist()
                for name, z in value['legacy'].items()}
        assert not any(risk['exact'])
        labels = value['labels']
        accuracy = {name: (z.argmax(1) == labels).double().mean().item() * 100
                    for name, z in value['current'].items()}
        changed = {name: int((z.argmax(1) != value['current']['joint'].argmax(1)).sum())
                   for name, z in value['current'].items()}
        residual = {name: (value['current'][name] - value['current']['joint']).abs().max().item()
                    for name in value['current']}
        result = dict(seed=row['seed'], accuracy_pp=accuracy, disagreement=risk, changed=changed,
                      max_current_logit_difference=residual, labeled=row['labeled'])
        if stage == 'confirmation':
            result['legacy_subgraph_verified'] = exact_selected
            result['max_legacy_logit_difference'] = {
                name: (z - value['old']).abs().max().item() for name, z in value['legacy'].items()
            }
        yield result


def added_macs(name):
    width = int(name.split('/')[1]) if name.startswith(('side/', 'exact/')) else 0
    return 64 * 11 + width * (64 + 22)


def choose(rows, candidates, plan):
    scored = []
    for name in candidates:
        max_risk = max(max(row['disagreement'][name]) for row in rows)
        gain = float(np.mean([row['accuracy_pp'][name] - row['accuracy_pp']['joint'] for row in rows]))
        scored.append(dict(name=name, added_macs=added_macs(name), max_disagreement=max_risk,
                               gain_vs_joint_pp=gain, qualifies=max_risk <= .04 and gain >= -.15))
    eligible = [row for row in scored if row['qualifies']]
    if eligible:
        best = min(eligible, key=lambda row: (row['added_macs'], row['max_disagreement']))
    else:
        usable = [row for row in scored if row['gain_vs_joint_pp'] >= -.15]
        best = min(usable or scored, key=lambda row: (row['max_disagreement'], row['added_macs']))
    return dict(plan_sha256=digest(plan), selected=best['name'], passed=bool(eligible),
                candidates=scored, references=rows)


def confirm(rows, plan, choice):
    chosen = json.loads(choice.read_text())['selected']
    candidates = {}
    for name in rows[0]['disagreement']:
        risks = [row['disagreement'][name] for row in rows]
        endpoint = name if name in rows[0]['accuracy_pp'] else 'joint'
        max_risk = max(max(risk) for risk in risks)
        change_vs_joint = interval([row['accuracy_pp'][endpoint] - row['accuracy_pp']['joint'] for row in rows])
        gain_vs_detach = interval([row['accuracy_pp'][endpoint] - row['accuracy_pp']['detach'] for row in rows])
        candidates[name] = dict(max_rule_disagreement=np.max(risks, axis=0).tolist(),
                                references_over_budget=sum(max(risk) > .05 for risk in risks),
                                gain_vs_detach_pp=gain_vs_detach, change_vs_joint_pp=change_vs_joint,
                                passed=max_risk <= .05 and gain_vs_detach['ci95'][0] > 0 and
                                change_vs_joint['ci95'][0] > -.25)
    return dict(plan_sha256=digest(plan), choice_sha256=digest(choice), selected=chosen,
                inference_compatibility_utility_passed=candidates[chosen]['passed'],
                candidates=candidates, references=rows)


def timing(rows, plan):
    summaries = {}
    for batch, cap in (('1', 1.20), ('128', 1.10)):
        blocks = [row['timings'][batch] for row in rows]
        estimates = {}
        for baseline in ('current', 'exact'):
            ratios = [np.median(np.array(run['chosen']) / np.array(run[baseline])) for run in blocks]
            estimates['ratio_to_' + baseline] = interval(ratios)
        estimates['passed'] = estimates['ratio_to_current']['ci95'][1] <= cap
        estimates['median_us'] = {name: float(np.median([np.median(run[name]) for run in blocks]) / 1000)
                                  for name in ('current', 'chosen', 'exact')}
        summaries[batch] = estimates
    return dict(plan_sha256=digest(plan), passed=all(s['passed'] for s in summaries.values()),
                batches=summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--records', type=Path)
    parser.add_argument('--arrays', type=Path)
    parser.add_argument('--confirmation', type=Path)
    parser.add_argument('--anchored', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    group = 'anchored' if args.anchored else 'onepass'
    name = 'anchored' if args.anchored else 'one_pass'
    plan = ROOT / 'docs/plans/2026-11-11-rml-anchored-one-pass.md' if args.anchored else PLAN
    choice = RESULTS / f'rml_{name}_choice_20261111.json'
    default = ROOT / 'tmp' / f'conditional_contrastiverml{group}{args.stage}'
    confirmation = args.confirmation or ROOT / 'tmp' / f'conditional_contrastiverml{group}confirmation'
    rows = list(load(args.stage, args.records or default, args.arrays or default,
                     confirmation, plan, choice, args.anchored))
    if args.stage == 'development':
        candidates = CANDIDATES
        if args.anchored:
            from rml_anchored_one_pass_cpu import CANDIDATES as anchored_candidates
            candidates = anchored_candidates
        result, path = choose(rows, candidates, plan), choice
    else:
        result = confirm(rows, plan, choice) if args.stage == 'confirmation' else timing(rows, plan)
        path = RESULTS / f'rml_{name}_{args.stage}_20261111.json'
    save(path, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'references'}, indent=2))


if __name__ == '__main__':
    main()
