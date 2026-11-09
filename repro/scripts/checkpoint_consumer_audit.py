"""CPU numerical audit of fixed readout candidates on an unlabeled monitor.

Maps fit training features only. Candidate selection is completed using only
monitor predictions and finite-corpus bounds before evaluation labels are read
for scoring. This is current-checkpoint control, not future-step forecasting.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

import paired_readout_continuation as paired
from checkpoint_repair_audit import assert_metrics
from consumer_risk import certificate, first_certified
from readout_alignment import ARMS, apply_map, fit_repairs, map_hash


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def reconstruction_mse(model, features, inputs, batch=2048):
    total = 0.
    for start in range(0, len(features), batch):
        total += float((model.aux(features[start:start + batch].float()) -
                        inputs[start:start + batch]).square().sum())
    return total / inputs.numel()


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    old = json.loads(args.record.read_text())
    if not old['complete'] or old['device'] != 'cpu':
        raise ValueError('A completed paired CPU record is required')
    config = argparse.Namespace(**old['config'])
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if getattr(config, 'architecture', None) == 'waveform_cnn':
        from cnn_reconstruction_continuation import make_cnn_data
        model, data, _, splits = make_cnn_data(config)
    else:
        model, data, _, splits = paired.make_data(config)
    if splits != old['splits']:
        raise AssertionError('Data or fitting/calibration split changed')
    xfit, yfit, _, _, xtest, ytest = data
    states = args.record.parent / (args.record.stem + '_states')
    reference_path = states / 'reference.pt'
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    if paired.state_hash(reference['model']) != old['reference_model_sha256']:
        raise AssertionError('Reference state mismatch')
    model.load_state_dict(reference['model'])
    model.eval()
    ref_fit, ref_test = paired.embed(model, xfit), paired.embed(model, xtest)
    head, mu, sd = paired.fit_head(ref_fit, yfit)
    with torch.no_grad():
        ref_logits = head((ref_test - mu) / sd)
        ref_prediction = ref_logits.argmax(1).numpy()
    healthy_error = assert_metrics(paired.metrics(ref_logits, ytest), old['healthy']['frozen'], 'healthy')
    monitor_seed = 790001 + config.seed * 104729 + (1000000 if config.domain == 'radioml' else 0)
    monitor_count = min(1024, len(xtest) // 2)
    monitor = np.random.default_rng(monitor_seed).permutation(len(xtest))[:monitor_count]
    remainder = np.ones(len(xtest), dtype=bool)
    remainder[monitor] = False
    tail = .05 / args.family_size
    result = dict(protocol='fixed-family finite-corpus unlabeled consumer audit',
        source_record=args.record.name, source_record_sha256=sha(args.record),
        source_sha256=sha(Path(__file__)), risk_source_sha256=sha(Path(__file__).with_name('consumer_risk.py')),
        alignment_source_sha256=sha(Path(__file__).with_name('readout_alignment.py')),
        reference_checkpoint_sha256=sha(reference_path), reference_head_sha256=paired.state_hash(head.state_dict()),
        domain=config.domain, auxiliary=config.aux, architecture=getattr(config, 'architecture', 'mlp'),
        seed=config.seed, device='cpu', torch_threads=torch.get_num_threads(),
        cuda_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=torch.__version__,
        python_version=platform.python_version(), family_size=args.family_size, bank_delta=.05,
        allocated_tail_probability=tail, risk_budgets=[.05, .10], monitor_seed=monitor_seed,
        monitor_examples=monitor_count, remainder_examples=int(remainder.sum()),
        monitor_indices_sha256=hashlib.sha256(monitor.tobytes()).hexdigest(),
        maps_fit_on='fitting split only', monitor_labels_used_for_selection=False,
        monitor_used_for_encoder_training=False, forecast_future_updates=False,
        guarantee_scope='fixed finite corpus, random uniform monitor, fixed candidate family',
        healthy_reconstruction_error=healthy_error, branches=[], complete=False)
    if config.aux == 'recon':
        result['healthy_reconstruction_mse'] = reconstruction_mse(model, ref_test, xtest)
    for archived in old['branches']:
        path = states / ('%s_%g.pt' % (archived['mode'], archived['alpha']))
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        shared = {k: v for k, v in state['model'].items() if k.startswith(('encoder.', 'primary.'))}
        if paired.state_hash(shared) != archived['final_shared_sha256']:
            raise AssertionError('Final checkpoint mismatch')
        current_fit, current_test = paired.embed(model, xfit), paired.embed(model, xtest)
        maps = fit_repairs(ref_fit, current_fit, mu, sd)
        predictions, certificates = {}, {}
        frozen_logits = None
        # No evaluation label is used in this block or the subsequent choice.
        for arm in ARMS:
            with torch.no_grad():
                logits = head(apply_map(current_test, maps[arm]))
                predictions[arm] = logits.argmax(1).numpy()
            certificates[arm] = certificate(predictions[arm], ref_prediction, monitor, tail)
            certificates[arm]['map_sha256'] = map_hash(maps[arm])
            if arm == 'frozen':
                frozen_logits = logits
        choices = {str(budget): first_certified(certificates, budget, ARMS) for budget in (.05, .10)}
        # Labels enter only after all choices are frozen, for external scoring.
        labels = ytest.numpy()
        healthy_accuracy = float(np.mean(ref_prediction[remainder] == labels[remainder]))
        for arm, prediction in predictions.items():
            accuracy = float(np.mean(prediction[remainder] == labels[remainder]))
            disagreement = float(np.mean(prediction[remainder] != ref_prediction[remainder]))
            certificates[arm].update(remainder_accuracy=accuracy,
                healthy_remainder_accuracy=healthy_accuracy, loss_increase=healthy_accuracy - accuracy,
                remainder_disagreement=disagreement,
                disagreement_bound_violated=disagreement > certificates[arm]['upper_remaining_disagreement'] + 1e-12)
        c = current_test[monitor]
        r = ref_test[monitor]
        cc, rc = c - c.mean(0), r - r.mean(0)
        scatter = float(cc.std(1).mean() / rc.std(1).mean().clamp_min(1e-30))
        branch = dict(mode=archived['mode'], alpha=archived['alpha'], checkpoint_sha256=sha(path),
            monitor_relative_scatter=scatter, scatter_safe_heuristic=scatter >= .3,
            probe_safe_heuristic=not archived['gate']['gate'], candidates=certificates,
            decisions={budget: dict(selected_arm=arm, certified=arm is not None,
                remainder_accuracy=None if arm is None else certificates[arm]['remainder_accuracy'],
                loss_increase=None if arm is None else certificates[arm]['loss_increase'])
                for budget, arm in choices.items()},
            frozen_reconstruction_error=assert_metrics(paired.metrics(frozen_logits, ytest),
                archived['trajectory'][-1]['frozen'], path.name),
            archived_own=archived['trajectory'][-1]['own'], archived_retrained=archived['trajectory'][-1]['retrained'])
        if config.aux == 'recon':
            branch['raw_head_test_reconstruction_mse'] = reconstruction_mse(model, current_test, xtest)
        result['branches'].append(branch)
        paired.save_json(args.out.with_suffix('.partial.json'), result)
        print(json.dumps(dict(mode=branch['mode'], alpha=branch['alpha'], choices=choices,
            frozen_loss=round(certificates['frozen']['loss_increase'], 4),
            elapsed_s=round(time.time() - started, 2))), flush=True)
    result.update(complete=True, elapsed_s=time.time() - started)
    paired.save_json(args.out, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--family-size', type=int, default=3015)
    args = parser.parse_args()
    if args.family_size < 1:
        parser.error('A positive predeclared family size is required')
    run(args)


if __name__ == '__main__':
    main()
