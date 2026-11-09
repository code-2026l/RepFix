"""Evaluate reconstruction on fixed archived models, using CPU only.

No fitting or selection occurs here. A healthy decoder was trained on detached
features, making it possible to distinguish encoder/consumer effects from
whether joint training actually reduces held-out reconstruction error.
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

import numpy as np
import torch

import paired_readout_continuation as paired


@torch.no_grad()
def reconstruction(model, features, mode='joint', gate=False, batch=256):
    total = 0.
    for begin in range(0, len(features), batch):
        x = features[begin:begin + batch]
        h = paired.filtered_input(model(x), mode, gate)
        total += float((model.aux(h) - x).square().sum())
    return total / features.numel()


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    old = json.loads(args.record.read_text())
    config = argparse.Namespace(**old['config'])
    if not old['complete'] or old['device'] != 'cpu' or config.aux != 'recon':
        raise ValueError('A completed CPU reconstruction-continuation record is required')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model, data, _, splits = paired.make_data(config)
    if splits != old['splits']:
        raise AssertionError('Data/splits changed')
    states = args.record.parent / (args.record.stem + '_states')
    reference = torch.load(states / 'reference.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(reference['model'])
    model.eval()

    def evaluate(mode='joint', gate=False):
        return dict(fitting_mse=reconstruction(model, data[0], mode, gate),
                    calibration_mse=reconstruction(model, data[2], mode, gate),
                    test_mse=reconstruction(model, data[4], mode, gate),
                    raw_head_test_mse=reconstruction(model, data[4]))

    result = dict(protocol='held-out reconstruction evaluation; fixed final checkpoints',
                  domain=config.domain, seed=config.seed, device='cpu',
                  torch_threads=torch.get_num_threads(), cuda_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  source_record=args.record.name,
                  source_record_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  evaluation_batch_size=256, reconstruction_uses_declared_forward_filter=True,
                  healthy=evaluate(), branches=[], complete=False)
    for branch in old['branches']:
        path = states / ('%s_%g.pt' % (branch['mode'], branch['alpha']))
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        shared = {k: v for k, v in state['model'].items() if k.startswith(('encoder.', 'primary.'))}
        if paired.state_hash(shared) != branch['final_shared_sha256']:
            raise AssertionError('Checkpoint changed')
        result['branches'].append(dict(mode=branch['mode'], alpha=branch['alpha'],
            gate=branch['gate']['gate'], checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            **evaluate(branch['mode'], branch['gate']['gate'])))
    result['complete'] = True
    paired.save_json(args.out, result)
    print(json.dumps(dict(record=args.record.name, healthy=result['healthy'], branches=result['branches'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
