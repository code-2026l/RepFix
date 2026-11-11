"""Batch-128 timing with an identical FP32 current graph in every program."""

import argparse
import copy
import json
import os
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
from torch import nn

import late_vision_cpu as vision
import nonlinear_suffix_cpu as factory
import vision_auxiliary_cpu as base
from consumer_contracts import evaluate_contracts
from contrastive_serving_cpu import trace
from suffix_serving import SuffixReadout
from whole_amp import WholeAMPReadout


def measure(programs, sample, repetitions):
    names = list(programs)
    for name in names:
        for _ in range(12):
            programs[name](sample)
    durations = {name: [] for name in names}
    for index in range(repetitions):
        rotation = index % len(names)
        for name in names[rotation:] + names[:rotation]:
            start = time.perf_counter()
            programs[name](sample)
            durations[name].append(time.perf_counter() - start)
    return durations


@torch.no_grad()
def main(args):
    record = json.loads(args.record.read_text())
    endpoint = args.record.parent / f'{args.record.stem}_states' / 'endpoint.pt'
    assert base.digest(endpoint) == record['state_sha256']
    state = torch.load(endpoint, map_location='cpu', weights_only=True)
    cache = torch.load(args.population, map_location='cpu', weights_only=True, mmap=True)
    prefix, old = factory.build_for(cache)
    old.load_state_dict(cache['healthy'])
    old.eval().requires_grad_(False)
    current = copy.deepcopy(old)
    current.load_state_dict(state['model'])
    ref = state['reference']
    images, _, coarse, data_sha = vision.images(True)
    assert data_sha == cache['data_sha256']
    val = ref['validation']
    image_size = cache['metadata']['image_size']
    sample = vision.preprocess(images[val[:128]], image_size)
    legacy, reader = ref['heads']['coarse'], state['reader']
    pfx = trace(prefix, sample[:1])
    tokens = pfx(sample)
    tail = trace(SuffixReadout(old, current, reader, legacy, 'U'), tokens[:1])
    exact = trace(SuffixReadout(old, old, legacy, legacy, 'F'), tokens[:1])
    approx = trace(WholeAMPReadout(nn.Identity(), old, legacy), tokens[:1], amp=True)

    def protected(h, reference):
        return tail(h), reference(h)

    programs = dict(current=lambda x: tail(pfx(x)),
                    autocast=lambda x: protected(pfx(x), approx),
                    fp32=lambda x: protected(pfx(x), exact))
    torch.testing.assert_close(tokens, prefix(sample))
    host_load = list(os.getloadavg())
    durations = measure(programs, sample, 31)
    medians = {name: float(np.median(values)) for name, values in durations.items()}
    prefix_time = measure({'prefix': pfx}, sample, 9)
    suffix_times = measure(dict(current=tail, autocast=approx, fp32=exact), tokens, 9)
    outputs = {name: [] for name in ['reference', 'fp32', 'autocast']}
    current_errors = dict(autocast=0, fp32=0)
    for indices in val.split(128):
        h = pfx(vision.preprocess(images[indices], image_size))
        expected = tail(h)
        outputs['reference'].append(vision.linear(old(h), legacy))
        for name, reference in [('autocast', approx), ('fp32', exact)]:
            new, previous = protected(h, reference)
            outputs[name].append(previous)
            current_errors[name] += int((new.argmax(1) != expected.argmax(1)).sum())
    outputs = {name: torch.cat(values) for name, values in outputs.items()}
    contracts = {name: evaluate_contracts(outputs[name], outputs['reference'],
                                          coarse[val], ref['thresholds'])
                 for name in ('autocast', 'fp32')}
    folder = args.out.parent / f'{args.out.stem}_states'
    folder.mkdir(exist_ok=True, parents=True)
    replay = folder / 'replay.pt'
    torch.save(dict(logits=outputs, coarse=coarse[val], thresholds=ref['thresholds']), replay)
    base.save(args.out, dict(complete=True, device='cpu', threads=12, batch=128,
        official_test_loaded=False, plan_sha256=args.plan_sha256,
        reference_seed=record['reference_seed'], seed=record['seed'],
        checkpoint_sha256=record['state_sha256'], replay_sha256=base.digest(replay),
        host_load_average=host_load, samples_seconds=durations, medians_seconds=medians,
        prefix_samples_seconds=prefix_time['prefix'], suffix_samples_seconds=suffix_times,
        current_prediction_changes=current_errors, contracts=contracts, branches=[]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    main(parser.parse_args())
