"""CPU numerical evaluation of fixed repairs on archived paired checkpoints.

Never train an encoder, select a hyperparameter, or fit a map on evaluation
features. Reconstruct and verify the archived healthy readout before repairs.
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
import torch.nn.functional as F

import paired_readout_continuation as paired
from readout_alignment import (ARMS, EIGEN_FLOOR, RIDGE, apply_map, fit_repairs,
                               map_hash, relative_residual)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_metrics(actual, archived, name):
    differences = {key: abs(actual[key] - archived[key]) for key in ('acc', 'nll', 'ece')}
    if differences['acc'] > 1e-12 or max(differences.values()) > 1e-7:
        raise AssertionError('Cannot reproduce archived readout at %s: %s' % (name, differences))
    return differences


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.time()
    old = json.loads(args.record.read_text())
    if old.get('complete') is not True or old.get('device') != 'cpu':
        raise ValueError('A completed CPU paired-continuation record is required')
    config = argparse.Namespace(**old['config'])
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model, data, _, splits = paired.make_data(config)
    if splits != old['splits']:
        raise AssertionError('Dataset or fitting/calibration split changed')
    xfit, yfit, xcal, _, xtest, ytest = data
    states = args.record.parent / (args.record.stem + '_states')
    reference_path = states / 'reference.pt'
    state = torch.load(reference_path, map_location='cpu', weights_only=True)
    if paired.state_hash(state['model']) != old['reference_model_sha256']:
        raise AssertionError('Reference checkpoint mismatch')
    model.load_state_dict(state['model'])
    model.eval()
    ref_fit = paired.embed(model, xfit)
    ref_cal = paired.embed(model, xcal)
    ref_test = paired.embed(model, xtest)
    # Warm training used an independent minibatch RNG and no stochastic layer.
    # The global initialization sequence is thus the original head-fit sequence.
    head, mu0, sd0 = paired.fit_head(ref_fit, yfit)
    with torch.no_grad():
        ref_logits = head((ref_test - mu0) / sd0)
    healthy = paired.metrics(ref_logits, ytest)
    reconstruction = assert_metrics(healthy, old['healthy']['frozen'], 'healthy')
    result = dict(protocol='fixed unlabeled repairs of archived paired checkpoints',
                  source_record=args.record.name, source_record_sha256=file_hash(args.record),
                  reference_checkpoint_sha256=file_hash(reference_path),
                  source_sha256=file_hash(Path(__file__)),
                  alignment_source_sha256=file_hash(Path(__file__).with_name('readout_alignment.py')),
                  paired_source_sha256=file_hash(Path(paired.__file__)),
                  domain=config.domain, auxiliary=config.aux, seed=config.seed,
                  device='cpu', cuda_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  torch_threads=torch.get_num_threads(), torch_version=torch.__version__,
                  python_version=platform.python_version(), splits=splits,
                  repairs_fitted_on='fitting features only; no labels or evaluation statistics',
                  ridge=RIDGE, covariance_relative_eigen_floor=EIGEN_FLOOR,
                  healthy=healthy, healthy_reconstruction_error=reconstruction,
                  reference_head=dict(weight=head.weight.detach().tolist(), bias=head.bias.detach().tolist(),
                                      mean=mu0.tolist(), sd=sd0.tolist(),
                                      sha256=paired.state_hash(head.state_dict())),
                  branches=[], complete=False)
    targets = dict(fitting=(ref_fit - mu0) / sd0,
                   calibration=(ref_cal - mu0) / sd0, test=(ref_test - mu0) / sd0)
    for archived in old['branches']:
        name = '%s_%g.pt' % (archived['mode'], archived['alpha'])
        path = states / name
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        shared = {k: v for k, v in model.state_dict().items() if k.startswith(('encoder.', 'primary.'))}
        if paired.state_hash(shared) != archived['final_shared_sha256']:
            raise AssertionError('Final checkpoint hash mismatch')
        current = dict(fitting=paired.embed(model, xfit), calibration=paired.embed(model, xcal),
                       test=paired.embed(model, xtest))
        maps = fit_repairs(ref_fit, current['fitting'], mu0, sd0)
        branch = dict(mode=archived['mode'], alpha=archived['alpha'], checkpoint_sha256=file_hash(path),
                      archived_retrained=archived['trajectory'][-1]['retrained'],
                      archived_own=archived['trajectory'][-1]['own'], repairs={})
        if config.aux == 'recon':
            with torch.no_grad():
                branch['reconstruction_mse'] = dict(
                    fitting=float(F.mse_loss(model.aux(current['fitting'].float()), xfit)),
                    test=float(F.mse_loss(model.aux(current['test'].float()), xtest)))
        for arm in ARMS:
            affine = maps[arm]
            standardized_test = apply_map(current['test'], affine)
            with torch.no_grad():
                logits = head(standardized_test)
            measures = paired.metrics(logits, ytest)
            measures.update(reference_prediction_agreement=float(
                (logits.argmax(1) == ref_logits.argmax(1)).double().mean()),
                paired_anchors=affine['paired_anchors'], map_sha256=map_hash(affine),
                matrix_frobenius_norm=float(affine['matrix'].norm()),
                alignment_residual={split: relative_residual(
                    apply_map(values, affine), targets[split]) for split, values in current.items()})
            branch['repairs'][arm] = measures
        endpoint = archived['trajectory'][-1]
        branch['frozen_reconstruction_error'] = assert_metrics(
            branch['repairs']['frozen'], endpoint['frozen'], name)
        branch['diagonal_reconstruction_error'] = assert_metrics(
            branch['repairs']['diagonal_stats'], endpoint['recalibrated'], name + ' diagonal')
        result['branches'].append(branch)
        paired.save_json(args.out.with_suffix('.partial.json'), result)
        print(json.dumps(dict(mode=branch['mode'], alpha=branch['alpha'],
                              accuracies={k: round(v['acc'], 4) for k, v in branch['repairs'].items()},
                              elapsed_s=round(time.time() - start, 2))), flush=True)
    result.update(complete=True, elapsed_s=time.time() - start)
    paired.save_json(args.out, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    if args.data_root:
        os.environ['REPFIX_DATA_X'] = str(args.data_root)
    run(args)


if __name__ == '__main__':
    main()
