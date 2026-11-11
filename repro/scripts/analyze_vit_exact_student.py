"""Replay predictions and fixed reference-level contrasts for exact ViT students."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_har_transfer import ROOT, RESULTS, digest, interval, save
from consumer_routing import decisions

PLAN = ROOT / 'docs/plans/2026-11-11-vit-exact-student.md'
CHOICE = RESULTS / 'vit_exact_student_choice_20261111.json'


def loss(row):
    return float(np.average(row['losses'], weights=row['counts']))


def load(stage, records, arrays):
    count, prefix = (1, 'p') if stage == 'pilot' else (12, 'c')
    for index in range(count):
        name = f'{prefix}{index:03d}'
        row = json.loads((records / (name + '.json')).read_text())
        assert row['complete'] and row['stage'] == stage and row['device'] == 'cpu'
        assert row['plan_sha256'] == digest(PLAN) and not row['official_test_loaded']
        assert row['fitting_examples'] == 10000 and row['fitting_views'] == 2 and row['validation_examples'] == 5000
        if stage == 'confirmation':
            assert row['reference_seed'] == 6201 + index // 3 and row['seed'] == 6310 + index % 3
            assert row['choice_sha256'] == digest(CHOICE)
        else:
            assert row['reference_seed'] == 6200 and row['seed'] == 6300
        replay = arrays / (name + '_states/replay.pt')
        assert digest(replay) == row['replay_sha256']
        value = torch.load(replay, map_location='cpu', weights_only=True)
        assert not (decisions(value['old'], value['thresholds']) != decisions(value['protected'], value['thresholds'])).any()
        for outcome in row['outcomes']:
            expected = (value['predictions'][outcome['key']] == value['labels']).double().mean().item()
            assert expected == outcome['accuracy']
            teacher = (value['teachers'][outcome['teacher']] == value['labels']).double().mean().item()
            assert teacher == outcome['teacher_accuracy']
        yield row, value


def select(rows):
    row, _ = rows[0]
    candidates = []
    for coefficient in sorted(set(x['coefficient'] for x in row['outcomes'])):
        arms = {x['teacher']: x for x in row['outcomes'] if x['coefficient'] == coefficient}
        joint = arms['joint']
        comparisons = {name: 100 * (joint['accuracy'] - arms[name]['accuracy'])
                       for name in ('two_view', 'double_epoch')}
        retention = 100 * (joint['accuracy'] - joint['teacher_accuracy'])
        ratios = {name: loss(joint['auxiliary']) / loss(arms[name]['auxiliary']) for name in comparisons}
        passed = retention >= -.25 and min(comparisons.values()) >= .2 and max(ratios.values()) < 1
        candidates.append(dict(coefficient=coefficient, gains_pp=comparisons, vs_joint_teacher_pp=retention,
                               auxiliary_ratios=ratios, qualifies=passed))
    eligible = [r for r in candidates if r['qualifies']]
    chosen = eligible[0] if eligible else max(candidates, key=lambda r: min(r['gains_pp'].values()))
    return dict(plan_sha256=digest(PLAN), pilot_passed=bool(eligible), coefficient=chosen['coefficient'],
                candidates=candidates)


def confirm(rows):
    references = []
    for row, replay in rows:
        arms = {x['teacher']: x for x in row['outcomes']}
        joint = arms['joint']
        refs = dict(reference_seed=row['reference_seed'], seed=row['seed'],
                    gains_pp={name: 100 * (joint['accuracy'] - arms[name]['accuracy'])
                              for name in ('two_view', 'double_epoch')},
                    vs_joint_teacher_pp=100 * (joint['accuracy'] - joint['teacher_accuracy']),
                    auxiliary_ratios={name: loss(joint['auxiliary']) / loss(arms[name]['auxiliary'])
                                      for name in ('two_view', 'double_epoch')},
                    accuracy_pp=100 * joint['accuracy'],
                    rich_accuracy_pp=100 * (replay['rich'] == replay['labels']).double().mean().item(),
                    folded_prediction_changes=sum(x['folded_changes'] for x in row['outcomes']))
        references.append(refs)
    groups = []
    for seed in range(6201, 6205):
        group = [r for r in references if r['reference_seed'] == seed]
        assert len(group) == 3
        groups.append(dict(reference_seed=seed,
                           gains_pp={name: np.mean([r['gains_pp'][name] for r in group]).item()
                                     for name in ('two_view', 'double_epoch')},
                           auxiliary_ratios={name: np.mean([r['auxiliary_ratios'][name] for r in group]).item()
                                             for name in ('two_view', 'double_epoch')},
                           vs_joint_teacher_pp=np.mean([r['vs_joint_teacher_pp'] for r in group]).item(),
                           versus_rich_pp=np.mean([r['accuracy_pp'] - r['rich_accuracy_pp'] for r in group]).item()))
    gains = {name: interval([r['gains_pp'][name] for r in groups]) for name in ('two_view', 'double_epoch')}
    auxiliary = {name: interval([r['auxiliary_ratios'][name] for r in groups]) for name in gains}
    retention = interval([r['vs_joint_teacher_pp'] for r in groups])
    return dict(plan_sha256=digest(PLAN), choice_sha256=digest(CHOICE), gains_pp=gains,
                auxiliary_ratios=auxiliary, versus_joint_teacher_pp=retention,
                versus_rich_pp=interval([r['versus_rich_pp'] for r in groups]),
                passed=all(g['ci95'][0] > 0 for g in gains.values()) and retention['ci95'][0] > -.25,
                auxiliary_passed=all(g['ci95'][1] < 1 for g in auxiliary.values()),
                references=references, reference_groups=groups)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('pilot', 'confirmation'))
    parser.add_argument('--records', type=Path)
    parser.add_argument('--arrays', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    local = ROOT / 'tmp' / ('conditional_contrastivevitstudent' + args.stage)
    rows = list(load(args.stage, args.records or local, args.arrays or local))
    result = select(rows) if args.stage == 'pilot' else confirm(rows)
    out = CHOICE if args.stage == 'pilot' else RESULTS / 'vit_exact_student_confirmation_20261111.json'
    save(out, result)
    print(json.dumps({key: val for key, val in result.items() if key != 'references'}, indent=2))
