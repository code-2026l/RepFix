"""Recompute paired gains, old decisions, and matched batch-128 costs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
import torch

from consumer_routing import decisions


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    margin = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return dict(mean=mean, lower=mean - margin, upper=mean + margin, values=values.tolist())


def replay(directory, record_path, plan):
    result = json.loads(record_path.read_text())
    assert result['complete'] and result['device'] == 'cpu'
    assert result['seed'] == 6310 and not result['official_test_loaded']
    assert result['plan_sha256'] == hashlib.sha256(plan.read_bytes()).hexdigest()
    path = directory / (record_path.stem + '_states') / 'replay.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result['replay_sha256']
    return result, torch.load(path, map_location='cpu', weights_only=True)


def compare_decisions(current, old, thresholds, recorded):
    changed = decisions(current, list(thresholds.values())) != decisions(old, list(thresholds.values()))
    names = ['classification', *thresholds]
    assert list(recorded) == names
    for i, name in enumerate(names):
        assert int(changed[:, i].sum()) == round(len(old) * recorded[name]['disagreement'])
    return {name: float(changed[:, i].double().mean()) for i, name in enumerate(names)}


def replication(records, tensors, original):
    rows = []
    plan = Path('docs/plans/2026-11-11-augreg-replication.md')
    preparation = json.loads((original.parent / 'vision_auxiliary_preparation/q002.json').read_text())
    config = None
    for i in range(4):
        result, data = replay(tensors, records / f'a{i:03d}.json', plan)
        control = json.loads((original / f'c{3 * i:03d}.json').read_text())
        assert result['reference_seed'] == control['reference_seed'] == 6201 + i
        assert control['seed'] == 6310 and control['alpha'] == .1 and control['epochs'] == 10
        candidate = {k: result[k] for k in ('checkpoint_sha256', 'data_sha256', 'augmentation')}
        assert candidate['data_sha256'] == preparation['data_sha256']
        if config is None:
            config = candidate
        assert candidate == config
        assert len(data['labels']) == result['validation_images'] == 5000
        accuracies = {name: float((data[name].argmax(1) == data['labels']).double().mean())
                      for name in ('supervised', 'contrastive', 'native')}
        for name in ('supervised', 'contrastive'):
            assert abs(accuracies[name] - result['outcome'][name]['accuracy']) < 1e-7
        native_changes = int((data['native'].argmax(1) != data['contrastive'].argmax(1)).sum())
        assert native_changes == result['current_prediction_changes']
        contracts = {}
        for name in ('protected', 'direct'):
            contracts[name] = compare_decisions(data[name], data['old'],
                                                data['thresholds'], result[name])
        error = float((data['protected'] - data['old']).abs().max())
        assert error == result['old_max_logit_error']
        auxiliary = result['outcome']
        rows.append(dict(reference_seed=6201 + i, accuracy=accuracies,
                         native_prediction_changes=native_changes, disagreements=contracts,
                         old_max_logit_error=error,
                         loss_ratio=auxiliary['contrastive']['loss'] / auxiliary['supervised']['loss'],
                         original_gain_pp=100 * (control['validation']['acc'] -
                                                 control['baseline']['accuracy']),
                         original_loss_ratio=control['contrastive_loss'] /
                                             control['baseline']['contrastive_loss']))
    gain = interval([100 * (r['accuracy']['native'] - r['accuracy']['supervised']) for r in rows])
    loss = interval([r['loss_ratio'] for r in rows])
    exact = all(not any(r['disagreements']['protected'].values()) and
                not r['native_prediction_changes'] for r in rows)
    return dict(reference_pairs=len(rows), config=config, gain_pp=gain, loss_ratio=loss,
                protected_exact_decisions=exact, passed=gain['lower'] > 0 and loss['upper'] < 1 and exact,
                original_matched_gain_pp=interval([r['original_gain_pp'] for r in rows]),
                original_matched_loss_ratio=interval([r['original_loss_ratio'] for r in rows]), rows=rows)


def latency(records, tensors, original):
    rows = []
    plan = Path('docs/plans/2026-11-11-matched-batch-latency.md')
    for i in range(4):
        result, data = replay(tensors, records / f'n{3*i:03d}.json', plan)
        control = json.loads((original / f'c{3*i:03d}.json').read_text())
        assert result['reference_seed'] == 6201 + i and result['batch'] == 128
        assert result['checkpoint_sha256'] == control['state_sha256']
        durations = result['samples_seconds']
        assert all(len(samples) == 31 and min(samples) > 0 for samples in durations.values())
        medians = {name: float(np.median(samples)) for name, samples in durations.items()}
        assert medians == result['medians_seconds']
        contracts = {}
        for name in ('fp32', 'autocast'):
            contracts[name] = compare_decisions(data['logits'][name], data['logits']['reference'],
                                                data['thresholds'], result['contracts'][name])
        rows.append(dict(reference_seed=result['reference_seed'], median_seconds=medians,
                         prefix_median_seconds=float(np.median(result['prefix_samples_seconds'])),
                         suffix_median_seconds={name: float(np.median(samples)) for name, samples
                                                in result['suffix_samples_seconds'].items()},
                         host_load_average=result['host_load_average'], disagreements=contracts,
                         current_prediction_changes=result['current_prediction_changes']))
    cost = {name: interval([r['median_seconds'][name] / r['median_seconds']['current'] for r in rows])
            for name in ('autocast', 'fp32')}
    exact = all(not any(r['disagreements']['fp32'].values()) for r in rows)
    maximum = max(max(r['disagreements']['autocast'].values()) for r in rows)
    same = all(not any(r['current_prediction_changes'].values()) for r in rows)
    return dict(ratios=cost, fp32_exact_decisions=exact, current_predictions_unchanged=same,
                maximum_autocast_disagreement=maximum,
                passed=cost['autocast']['upper'] <= 1.10 and maximum <= .002 and same, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['replication', 'latency'])
    parser.add_argument('--records', required=True, type=Path)
    parser.add_argument('--tensors', required=True, type=Path)
    parser.add_argument('--original', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.kind == 'replication':
        summary = replication(args.records, args.tensors, args.original)
    else:
        summary = latency(args.records, args.tensors, args.original)
    args.out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
