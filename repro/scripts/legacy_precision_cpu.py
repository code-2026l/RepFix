"""Quantize the old block while keeping the shared prefix and current output FP32."""

import argparse
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from torch import nn
from torch.ao import quantization as quant

import late_vision_cpu as vision
import prospective_vision_cpu as early
import stable_prefix_cpu as shared
import vision_auxiliary_cpu as base

METHODS = ('int8_residual', 'int8_full', 'bf16_residual')


class LegacyBlock(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.first = nn.Conv2d(512, 512, 3, padding=1)
        self.relu = nn.ReLU()
        self.last = nn.Conv2d(512, 512, 3, padding=1)
        if method.startswith('int8'):
            self.quant = quant.QuantStub()
            self.dequant = quant.DeQuantStub()
        if method == 'int8_full':
            self.add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.method == 'bf16_residual':
            residual = self.last(self.relu(self.first(x.bfloat16())))
            return (x + residual.float()).relu().mean((-2, -1))
        q = self.quant(x)
        residual = self.last(self.relu(self.first(q)))
        if self.method == 'int8_full':
            return self.dequant(self.add.add_relu(q, residual)).mean((-2, -1))
        return (x + self.dequant(residual)).relu().mean((-2, -1))


@torch.no_grad()
def build(method, calibration, teacher=None):
    model = LegacyBlock(method).eval().to(memory_format=torch.channels_last)
    if teacher is not None:
        for layer, source in zip((model.first, model.last), teacher.maps):
            layer.weight.copy_(source.weight)
            layer.bias.copy_(source.bias)
    if method == 'bf16_residual':
        return model.bfloat16().requires_grad_(False)
    torch.backends.quantized.engine = 'x86'
    quant.fuse_modules(model, [['first', 'relu']], inplace=True)
    model.qconfig = quant.get_default_qconfig('x86')
    quant.prepare(model, inplace=True)
    for x in calibration.split(128):
        model(x.contiguous(memory_format=torch.channels_last))
    return quant.convert(model, inplace=True).eval().requires_grad_(False)


def load_precision(args, choice):
    assert base.digest(args.student) == choice['state_sha256']
    dummy = torch.zeros(1, 512, 2, 2)
    model = build(choice['method'], dummy)
    model.load_state_dict(torch.load(args.student, map_location='cpu', weights_only=True))
    return model


@torch.no_grad()
def fit(args):
    cache = torch.load(args.population, map_location='cpu', weights_only=True, mmap=True)
    _, teacher = shared.teacher_block(cache)
    pool = torch.randperm(50000, generator=torch.Generator().manual_seed(7400))[14000:28000]
    index = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(7550))[:1024]]
    calibration = cache['x'][0, index]
    references = {}
    for seed in (7410, 7411):
        legacy, thresholds, *_, monitor = shared.reference(cache, teacher, seed)
        exact = vision.linear(base.embeds(teacher, monitor), legacy)
        references[seed] = (legacy, thresholds, monitor, exact)
    out = shared.folder(args)
    data, rows = {}, []
    for method in METHODS:
        model = build(method, calibration, teacher)
        state = out / (method + '.pt')
        torch.save(model.state_dict(), state)
        data[method] = {}
        outcomes = []
        for seed, (legacy, thresholds, monitor, exact) in references.items():
            cheap = vision.linear(base.embeds(model, monitor), legacy)
            rates = early.disagreement(cheap, exact, thresholds)
            outcomes.append(dict(reference_seed=seed, rates=rates))
            data[method][str(seed)] = dict(cheap=cheap, exact=exact, thresholds=thresholds)
        rows.append(dict(method=method, outcomes=outcomes, state_sha256=base.digest(state)))
    replay = out / 'replay.pt'
    torch.save(data, replay)
    shared.record(args, dict(methods=rows, calibration_images=1024, test_loaded=False,
        replay_sha256=base.digest(replay), precision_source_sha256=base.digest(Path(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('fit', 'audit', 'timing'), required=True)
    parser.add_argument('--reference-seed', type=int)
    parser.add_argument('--plan-sha256', required=True)
    for key in ('population', 'out', 'archive', 'choice', 'student', 'serving', 'audit-record'):
        parser.add_argument('--' + key, type=Path, required=key in ('population', 'out'))
    args = parser.parse_args()
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    if args.stage == 'fit':
        fit(args)
    elif args.stage == 'audit':
        shared.audit(args, load_precision, force_zero=True)
    else:
        shared.timing(args, load_precision)
