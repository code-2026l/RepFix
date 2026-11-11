"""Replay all paired controls at a matched suffix-forward budget."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analyze_small_budget_confirmation import interval


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accuracy(logits, labels):
    assert logits.ndim == 2 and len(logits) == len(labels) == 5000
    assert torch.isfinite(logits).all()
    return float((logits.argmax(1) == labels).double().mean())


def analyze(args):
    plan_sha = digest(args.plan)
    expected = {(ref, seed, role) for ref in range(6201, 6205)
                for seed in range(6310, 6313) for role in ('two_view', 'twenty_epochs')}
    found = set()
    rows = []
    for path in sorted(args.records.glob('k???.json')):
        record = json.loads(path.read_text())
        assert record['complete'] and record['device'] == 'cpu'
        assert not record['official_test_loaded'] and record['alpha'] == 0
        assert record['plan_sha256'] == plan_sha
        role = 'two_view' if record['primary_two_view'] else 'twenty_epochs'
        assert record['epochs'] == (10 if role == 'two_view' else 20)
        assert len(record['trace']) == record['epochs']
        ref, seed = record['reference_seed'], record['seed']
        key = (ref, seed, role)
        assert key in expected and key not in found
        found.add(key)
        index = (ref - 6201) * 3 + seed - 6310
        original = json.loads((args.joint_records / f'c{index:03d}.json').read_text())
        native = json.loads((args.native_records / f'n{index:03d}.json').read_text())
        assert (original['reference_seed'], original['seed'], original['alpha']) == (ref, seed, .1)
        assert (native['reference_seed'], native['seed']) == (ref, seed)
        control_path = args.arrays / (path.stem + '_states') / 'replay.pt'
        assert digest(control_path) == record['replay_sha256']
        control = torch.load(control_path, map_location='cpu', weights_only=True)
        assert (control['reference_seed'], control['seed']) == (ref, seed)
        joint_path = args.joint_arrays / f'n{index:03d}_states' / 'replay.pt'
        assert digest(joint_path) == native['state_sha256']
        joint = torch.load(joint_path, map_location='cpu', weights_only=True)
        assert torch.equal(control['labels'], joint['labels'])
        control_accuracy = accuracy(control['logits'], control['labels'])
        joint_accuracy = accuracy(joint['current_logits'], joint['labels'])
        assert control_accuracy == record['validation']['acc']
        assert joint_accuracy == original['validation']['acc'] == native['native']['current']['acc']
        rows.append(dict(reference_seed=ref, seed=seed, role=role,
                         joint_accuracy=joint_accuracy, control_accuracy=control_accuracy,
                         gain_correct=round(5000 * joint_accuracy) - round(5000 * control_accuracy),
                         gain_pp=100 * (joint_accuracy - control_accuracy),
                         auxiliary_loss_ratio=original['contrastive_loss'] / record['contrastive_loss'],
                         control_elapsed_seconds=record['elapsed_s'],
                         original_joint_elapsed_seconds=original['elapsed_s'],
                         control_output_sha256=digest(control_path),
                         joint_output_sha256=digest(joint_path)))
    assert found == expected
    contrasts = {}
    for role in ('two_view', 'twenty_epochs'):
        groups = []
        for ref in range(6201, 6205):
            selected = [row for row in rows if row['reference_seed'] == ref and row['role'] == role]
            assert len(selected) == 3
            groups.append(dict(reference_seed=ref,
                gain_pp=float(np.mean([r['gain_pp'] for r in selected])),
                auxiliary_loss_ratio=float(np.mean([r['auxiliary_loss_ratio'] for r in selected])),
                joint_accuracy=float(np.mean([r['joint_accuracy'] for r in selected])),
                control_accuracy=float(np.mean([r['control_accuracy'] for r in selected]))))
        gain = interval([g['gain_pp'] for g in groups])
        contrasts[role] = dict(gain_pp=gain,
            auxiliary_loss_ratio=interval([g['auxiliary_loss_ratio'] for g in groups]),
            positive_lower_accuracy_bound=gain['ci95'][0] > 0,
            reference_groups=groups)
    return dict(complete=True, plan_sha256=plan_sha, official_test_loaded=False,
                control_endpoints=len(rows), matched_joint_endpoints=12,
                all_accuracy_counts_recomputed=True, contrasts=contrasts,
                matched_work_gate_passed=all(c['positive_lower_accuracy_bound'] for c in contrasts.values()),
                suffix_forwards_per_arm=1580, training_image_suffix_evaluations=200000,
                optimizer_steps=dict(joint=790, two_view=790, twenty_epochs=1580),
                elapsed_time_scope='separate numerical executions with different host loads',
                rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--arrays', type=Path, required=True)
    parser.add_argument('--joint-records', type=Path, required=True)
    parser.add_argument('--native-records', type=Path, required=True)
    parser.add_argument('--joint-arrays', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    torch.set_num_threads(1)
    args = parser.parse_args()
    result = analyze(args)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(complete=True, passed=result['matched_work_gate_passed'],
                         contrasts={name: value['gain_pp'] for name, value in result['contrasts'].items()})))
