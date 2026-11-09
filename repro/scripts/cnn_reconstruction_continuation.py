"""CPU waveform CNN using the unchanged paired-continuation engine.

Only the encoder architecture changes. Inputs, heads, optimizer, independent
calibration, cloned states, future orders, and audits are the existing protocol.
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

import torch
from torch import nn

import paired_readout_continuation as paired

BASE_MAKE_DATA = paired.make_data
SPEC = dict(input_channels=2, input_length=128, channels=[8, 16], kernels=[7, 5],
            pool=2, feature_width=64, stochastic_layers=False, normalization=False)


class WaveformModel(nn.Module):
    def __init__(self, classes=11, m=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Unflatten(1, (2, 128)),
            nn.Conv1d(2, 8, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(), nn.Linear(16 * 32, m))
        self.primary = nn.Linear(m, classes)
        self.aux = nn.Sequential(nn.Linear(m, 256), nn.ReLU(), nn.Linear(256, 256))

    def forward(self, x):
        return self.encoder(x)


def make_cnn_data(args):
    if args.domain != 'radioml' or args.aux != 'recon' or args.m != 64:
        raise ValueError('The fixed waveform CNN study uses RadioML reconstruction at width 64')
    # Reuse the audited loader/split without consuming an unrelated MLP init.
    rng = torch.get_rng_state()
    base, data, auxiliary, splits = BASE_MAKE_DATA(args)
    torch.set_rng_state(rng)
    if data[0].shape[1] != 256:
        raise ValueError('Expected unchanged flattened two-channel I/Q inputs')
    return WaveformModel(base.primary.out_features, args.m), data, auxiliary, splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='radioml', choices=['radioml'])
    parser.add_argument('--aux', default='recon', choices=['recon'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warm-epochs', type=int, default=25)
    parser.add_argument('--cont-epochs', type=int, default=15)
    parser.add_argument('--m', type=int, default=64)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--doses', default='30,300,3000')
    parser.add_argument('--modes', default='joint,detach')
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--audit-every', type=int, default=5)
    parser.add_argument('--limit-train', type=int, default=0)
    parser.add_argument('--limit-test', type=int, default=0)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.architecture = 'waveform_cnn'
    if args.data_root:
        os.environ['REPFIX_DATA_X'] = str(args.data_root)
    if args.warm_epochs < 1 or args.cont_epochs < 1 or set(args.modes.split(',')) - {'joint', 'detach'}:
        parser.error('Positive lengths and joint/detach modes required')
    paired.make_data = make_cnn_data
    paired.run(args)
    for path in (args.out, args.out.with_suffix('.partial.json')):
        result = json.loads(path.read_text())
        result.update(architecture='waveform_cnn', architecture_spec=SPEC,
                      architecture_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        paired.save_json(path, result)


if __name__ == '__main__':
    main()
