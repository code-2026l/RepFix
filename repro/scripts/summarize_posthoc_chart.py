"""Local selection and statistical summary for the frozen chart protocol."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = ['diagonal_stats', 'paired_ridge']


def contrast(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
    return dict(n=len(values), mean=mean, low=mean-half if half is not None else None,
                high=mean+half if half is not None else None, values=values.tolist())


def development(directory):
    plan_sha = hashlib.sha256((ROOT/'docs/plans/2026-11-09-deployment-chart.md').read_bytes()).hexdigest()
    records = [json.loads(path.read_text()) for path in sorted(directory.glob('a???.json'))]
    assert len(records)==6 and all(row['complete'] and row['phase']=='development' for row in records)
    assert all(row['plan_sha256']==plan_sha and not row['splits']['test_metrics_computed'] for row in records)
    result = dict(plan_sha256=plan_sha, selected_before_confirmation=True,
                  rule='first fixed candidate passing every declared development mean and every identity check', tasks={})
    for domain in ['har', 'radioml']:
        rows = sorted([row for row in records if row['domain']==domain], key=lambda row:row['seed'])
        assert [row['seed'] for row in rows]==[600, 601, 602]
        gain = [row['branches'][1]['validation']['own']['acc']-row['branches'][0]['validation']['own']['acc'] for row in rows]
        item = dict(seeds=[row['seed'] for row in rows], ordinary_gain=contrast(gain), candidates={}, selected=None, confirm=False)
        for candidate in CANDIDATES:
            joint = [row['branches'][1]['repairs'][candidate] for row in rows]
            all_arms = [branch['repairs'][candidate] for row in rows for branch in row['branches']]
            metrics = dict(
                old_accuracy_difference=contrast([r['validation']['frozen']['acc']-r['validation']['reference']['acc'] for r in joint]),
                old_nll_increase=contrast([r['validation']['frozen']['nll']-r['validation']['reference']['nll'] for r in joint]),
                old_ece_increase=contrast([r['validation']['frozen']['ece']-r['validation']['reference']['ece'] for r in joint]),
                batch1_ratio=contrast([r['deployment']['timings']['1']['ratio'] for r in joint]),
                batch1024_ratio=contrast([r['deployment']['timings']['1024']['ratio'] for r in joint]),
                max_current_logit_error=max(r['validation']['current_max_absolute_logit_error'] for r in all_arms),
                current_prediction_changes=sum(r['validation']['current_prediction_changes'] for r in all_arms),
                identical_encoder_shapes=all(r['deployment']['identical_encoder_shapes'] for r in all_arms),
                nonlinear_trunks_changed=all(branch['nonlinear_trunk_changed'] for row in rows for branch in row['branches']))
            checks = dict(ordinary_gain=item['ordinary_gain']['mean']>0,
                old_accuracy=metrics['old_accuracy_difference']['mean']>=-.005,
                old_nll=metrics['old_nll_increase']['mean']<=.05,
                old_ece=metrics['old_ece_increase']['mean']<=.02,
                current_identity=metrics['current_prediction_changes']==0 and metrics['max_current_logit_error']<1e-8,
                shapes=metrics['identical_encoder_shapes'], nonlinear_learning=metrics['nonlinear_trunks_changed'],
                batch1_cost=metrics['batch1_ratio']['mean']<=1.2,
                batch1024_cost=metrics['batch1024_ratio']['mean']<=1.1)
            item['candidates'][candidate] = dict(metrics=metrics, checks=checks, passes=all(checks.values()))
            if all(checks.values()) and item['selected'] is None:
                item.update(selected=candidate, confirm=True)
        result['tasks'][domain] = item
    return result


def confirmation(directory, choice):
    records = [json.loads(path.read_text()) for path in sorted(directory.glob('c???.json'))]
    result = dict(plan_sha256=choice['plan_sha256'], phase='confirmation', tasks={})
    for domain, selected in choice['tasks'].items():
        if not selected['confirm']:
            continue
        rows = sorted([row for row in records if row['domain']==domain], key=lambda row:row['seed'])
        assert [row['seed'] for row in rows]==list(range(700, 710))
        assert all(row['complete'] and row['phase']=='confirmation' and row['plan_sha256']==choice['plan_sha256'] for row in rows)
        candidate = selected['selected']
        joints = [row['branches'][1]['repairs'][candidate] for row in rows]
        gains = [row['branches'][1]['test']['own']['acc']-row['branches'][0]['test']['own']['acc'] for row in rows]
        repaired_gains = [joint['test']['own']['acc']-row['branches'][0]['test']['own']['acc'] for row,joint in zip(rows,joints)]
        accuracy_diff = [joint['test']['frozen']['acc']-joint['test']['reference']['acc'] for joint in joints]
        metrics = dict(ordinary_gain=contrast(gains), repaired_current_gain=contrast(repaired_gains),
            old_accuracy_difference=contrast(accuracy_diff),
            old_nll_increase=contrast([r['test']['frozen']['nll']-r['test']['reference']['nll'] for r in joints]),
            old_ece_increase=contrast([r['test']['frozen']['ece']-r['test']['reference']['ece'] for r in joints]),
            certified=sum(r['test']['consumer_audit']['certified'] for r in joints),
            disagreements=contrast([r['test']['disagreement'] for r in joints]),
            batch1_ratio=contrast([r['deployment']['timings']['1']['ratio'] for r in joints]),
            batch1024_ratio=contrast([r['deployment']['timings']['1024']['ratio'] for r in joints]),
            max_current_logit_error=max(r['test']['current_max_absolute_logit_error'] for r in joints),
            current_prediction_changes=sum(r['test']['current_prediction_changes'] for r in joints),
            identical_encoder_shapes=all(r['deployment']['identical_encoder_shapes'] for r in joints))
        checks = dict(ordinary_gain=metrics['ordinary_gain']['low']>0,
            repaired_gain=metrics['repaired_current_gain']['low']>0,
            old_accuracy=metrics['old_accuracy_difference']['low']>=-.005,
            old_nll=metrics['old_nll_increase']['mean']<=.05, old_ece=metrics['old_ece_increase']['mean']<=.02,
            current_identity=metrics['current_prediction_changes']==0 and metrics['max_current_logit_error']<1e-8,
            shapes=metrics['identical_encoder_shapes'], batch1_cost=metrics['batch1_ratio']['mean']<=1.2,
            batch1024_cost=metrics['batch1024_ratio']['mean']<=1.1, consumer_certificate=metrics['certified']==10)
        result['tasks'][domain] = dict(selected=candidate, metrics=metrics, checks=checks, passes=all(checks.values()))
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['development', 'confirmation'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--choice', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = development(args.directory) if args.phase=='development' else confirmation(args.directory, json.loads(args.choice.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
