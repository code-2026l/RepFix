"""Distil a narrower legacy residual block and audit actual CPU referral."""

import argparse
import copy
import hashlib
import json
import math
import os
import tarfile
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import late_vision_cpu as vision
import paired_readout_continuation as paired
import prospective_vision_cpu as early
import vision_auxiliary_cpu as base
from analyze_early_referral import select as select_previous
from conditional_reference_cpu import ridge
from consumer_contracts import confidence_thresholds
from consumer_routing import decision_margin, route_logits
from nonlinear_suffix import build_full
from shared_reference_vision import NormalizedHead

WIDTHS = (64, 128, 256)
FRACTIONS = tuple(i / 10 for i in range(11))


class CompressedBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.first = nn.Conv2d(512, width, 3, padding=1)
        self.last = nn.Conv2d(width, 512, 3, padding=1)

    def preactivation(self, x):
        return x + self.last(self.first(x).relu())

    def forward(self, x):
        return self.preactivation(x).relu().mean((-2, -1))


def initialise(teacher, width, activation_energy):
    contribution = activation_energy * teacher.maps[1].weight.square().sum((0, 2, 3))
    indices = torch.argsort(contribution, descending=True, stable=True)[:width]
    student = CompressedBlock(width).to(memory_format=torch.channels_last)
    with torch.no_grad():
        student.first.weight.copy_(teacher.maps[0].weight[indices])
        student.first.bias.copy_(teacher.maps[0].bias[indices])
        student.last.weight.copy_(teacher.maps[1].weight[:, indices])
        student.last.bias.copy_(teacher.maps[1].bias)
    return student


def teacher_block(cache):
    prefix, teacher = build_full('resnet18')
    teacher.load_state_dict(cache['healthy'])
    return prefix, teacher.eval().requires_grad_(False)


def reference(cache, teacher, seed):
    fitting, mask, anchors, monitor, _ = early.split(seed)
    xx, yy = cache['x'][:, fitting], cache['y'][fitting]
    torch.manual_seed(seed)
    initial = base.embeds(teacher, xx[:, mask].flatten(0, 1))
    clean = initial[:int(mask.sum())]
    legacy = base.fit_affine(clean, yy[mask])
    thresholds = list(confidence_thresholds(vision.linear(clean, legacy)).values())
    mean, sd = initial.mean(0), initial.std(0, unbiased=False)
    sd = sd.clamp_min(.1 * sd[sd > 0].median())
    head = NormalizedHead(legacy['weight'] * sd, legacy['bias'] + legacy['weight'] @ mean, mean, sd)
    return legacy, thresholds, head, xx, yy, mask, cache['x'][0, anchors], cache['x'][0, monitor]


def indices(logits, thresholds, fraction):
    order = torch.argsort(decision_margin(logits, thresholds), stable=True)
    return order[:math.ceil(fraction * len(logits))]


def select_fraction(cheap, exact, thresholds):
    ranking = indices(cheap, thresholds, 1.)
    results = {}
    for fraction in FRACTIONS:
        mask = torch.zeros(len(cheap), dtype=torch.bool)
        mask[ranking[:math.ceil(fraction * len(cheap))]] = True
        results[str(fraction)] = early.disagreement(route_logits(cheap, exact, mask), exact, thresholds)
    selected = next(fraction for fraction in FRACTIONS if max(results[str(fraction)]) <= .03)
    return selected, results


def folder(args):
    path = args.out.parent / f'{args.out.stem}_states'
    path.mkdir(exist_ok=True, parents=True)
    return path


def record(args, values):
    base.save(args.out, dict(complete=True, device='cpu', stage=args.stage,
        plan_sha256=args.plan_sha256, source_sha256={Path(__file__).name: base.digest(Path(__file__))},
        branches=[], **values))


def fit(args):
    cache = torch.load(args.population, map_location='cpu', weights_only=True, mmap=True)
    _, teacher = teacher_block(cache)
    fit_pool = torch.randperm(50000, generator=torch.Generator().manual_seed(7400))[14000:28000]
    fitting = fit_pool[torch.randperm(len(fit_pool), generator=torch.Generator().manual_seed(7550))[:6000]]
    xx = cache['x'][:, fitting].flatten(0, 1)
    with torch.no_grad():
        target, energy = [], torch.zeros(512)
        for x in xx.split(128):
            hidden = teacher.hidden(x)
            energy += hidden.square().sum((0, 2, 3))
            target.append((x + teacher.maps[1](hidden)).contiguous())
        target = torch.cat(target)
        pooled = target.relu().mean((-2, -1))
        sd = pooled.std(0)
        sd = sd.clamp_min(.1 * sd[sd > 0].median())
        variance = target.var(unbiased=False).clamp_min(1e-6)
    torch.manual_seed(7560)
    student = initialise(teacher, args.width, energy)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
    order = torch.Generator().manual_seed(7561)
    losses = []
    for epoch in range(12):
        total = 0.
        for idx in torch.randperm(len(xx), generator=order).split(128):
            x = xx[idx].contiguous(memory_format=torch.channels_last)
            pre = student.preactivation(x)
            feature_loss = ((pre.relu().mean((-2, -1)) - pooled[idx]) / sd).square().mean()
            loss = feature_loss + .25 * (pre - target[idx]).square().mean() / variance
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        losses.append(total / len(xx))
        print(json.dumps(dict(epoch=epoch + 1, width=args.width, loss=losses[-1])), flush=True)
    data, rows = {}, []
    with torch.no_grad():
        for seed in (7410, 7411):
            legacy, thresholds, *_, monitor = reference(cache, teacher, seed)
            cheap = vision.linear(base.embeds(student, monitor), legacy)
            exact = vision.linear(base.embeds(teacher, monitor), legacy)
            fraction, rates = select_fraction(cheap, exact, thresholds)
            data[str(seed)] = dict(cheap=cheap, exact=exact, thresholds=thresholds)
            rows.append(dict(reference_seed=seed, fraction=fraction, rates=rates))
    out = folder(args)
    state, replay = out / 'student.pt', out / 'replay.pt'
    torch.save(student.state_dict(), state)
    torch.save(data, replay)
    record(args, dict(width=args.width, teacher_images=len(xx), losses=losses, rows=rows,
                      test_loaded=False, state_sha256=base.digest(state), replay_sha256=base.digest(replay)))


def load_student(args, choice):
    assert base.digest(args.student) == choice['state_sha256']
    student = CompressedBlock(choice['width']).to(memory_format=torch.channels_last)
    student.load_state_dict(torch.load(args.student, map_location='cpu', weights_only=True))
    return student.eval().requires_grad_(False)


def test_images(path, seed):
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == early.ARCHIVE_SHA256
    with tarfile.open(path, 'r:gz') as archive:
        member = archive.getmember('cifar-10-batches-bin/test_batch.bin')
        assert member.isfile() and member.size == 10000 * 3073
        with archive.extractfile(member) as stream:
            raw = np.frombuffer(stream.read(), dtype=np.uint8).reshape(10000, 3073)
    start = (seed - 7420) * 1500
    order = torch.randperm(10000, generator=torch.Generator().manual_seed(7590))[start:start + 1500]
    selected = raw[order.numpy()]
    images = torch.from_numpy(selected[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy())
    labels = torch.from_numpy(selected[:, 0].astype(np.int64))
    return images, labels, order


def serve_old(cheap, features, teacher, legacy, thresholds, fraction):
    if fraction == 0:
        return cheap, 0
    if fraction == 1:
        return vision.linear(teacher(features), legacy), len(features)
    selected = indices(cheap, thresholds, fraction)
    result = cheap.clone()
    result[selected] = vision.linear(teacher(features[selected]), legacy)
    return result, len(selected)


def fit_current(args, cache, teacher):
    seed = args.reference_seed
    legacy, thresholds, head, xx, yy, mask, anchors, monitor = reference(cache, teacher, seed)
    current = copy.deepcopy(teacher).requires_grad_(True)
    optimizer = torch.optim.AdamW([dict(params=current.parameters(), lr=early.RATES['strong']),
                                  dict(params=head.parameters(), lr=1e-4)], weight_decay=1e-4)
    rng = torch.Generator().manual_seed(seed + 21000)
    views = torch.Generator().manual_seed(seed + 72000)
    early.train_epochs(current, head, optimizer, xx, yy, mask, .1, rng, views, 0, 2)
    old_anchors = base.embeds(teacher, anchors)
    exact_monitor = vision.linear(base.embeds(teacher, monitor), legacy)
    early_logits = early.candidate_logits(old_anchors, base.embeds(current, anchors),
                                         base.embeds(current, monitor), legacy)
    family, fraction = select_previous(early_logits, exact_monitor, thresholds, 'margin', seed + 90000)
    assert (family, fraction) == ('ridge', .9)
    return dict(legacy=legacy, thresholds=thresholds, head=head, xx=xx, yy=yy, mask=mask,
                anchors=anchors, monitor=monitor, exact_monitor=exact_monitor, old_anchors=old_anchors,
                current=current, optimizer=optimizer, rng=rng, views=views, previous_fraction=fraction)


def audit(args, load_model=load_student, force_zero=False):
    choice = json.loads(args.choice.read_text())
    assert choice['passed'] and choice['plan_sha256'] == args.plan_sha256
    cache = torch.load(args.population, map_location='cpu', weights_only=True, mmap=True)
    prefix, teacher = teacher_block(cache)
    student = load_model(args, choice)
    state = fit_current(args, cache, teacher)
    legacy, thresholds, current = state['legacy'], state['thresholds'], state['current']
    cheap_monitor = vision.linear(base.embeds(student, state['monitor']), legacy)
    fraction, rates = select_fraction(cheap_monitor, state['exact_monitor'], thresholds)
    if force_zero:
        fraction = 0.
    previous = state['previous_fraction']
    seal = args.out.with_suffix('.forecast.json')
    base.save(seal, dict(early_epoch=2, test_loaded=False, plan_sha256=args.plan_sha256,
        choice_sha256=base.digest(args.choice), reference_seed=args.reference_seed,
        previous_fraction=previous, prefix_fraction=fraction, monitor_rates=rates))
    sealed = base.digest(seal)
    print(json.dumps(dict(forecast_fixed=True)), flush=True)
    early.train_epochs(current, state['head'], state['optimizer'], state['xx'], state['yy'],
                       state['mask'], .1, state['rng'], state['views'], 2, 8)
    assert base.digest(seal) == sealed
    images, labels, order = test_images(args.archive, args.reference_seed)
    new_anchors = base.embeds(current, state['anchors'])
    residual = vision.linear(state['old_anchors'], legacy) - vision.linear(new_anchors, legacy)
    correction, _, _ = ridge(new_anchors, residual - residual.mean(1, keepdim=True), .001)
    torch.manual_seed(args.reference_seed + 40000)
    xx, yy, mask = state['xx'], state['yy'], state['mask']
    reader = base.fit_affine(base.embeds(current, xx[:, mask].flatten(0, 1)), yy[mask].repeat(2))
    calls = dict(previous=0, prefix=0)
    logits = {name: [] for name in ('exact', 'current', 'previous', 'prefix', 'prefix_cheap')}
    with torch.no_grad():
        for batch in images.split(128):
            h = prefix(vision.preprocess(batch, 64))
            features = current(h)
            direct = vision.linear(features, legacy) + vision.linear(features, correction)
            compressed = vision.linear(student(h), legacy)
            logits['exact'].append(vision.linear(teacher(h), legacy))
            logits['current'].append(vision.linear(features, reader))
            logits['prefix_cheap'].append(compressed)
            for name, z, q in (('previous', direct, previous), ('prefix', compressed, fraction)):
                served, used = serve_old(z, h, teacher, legacy, thresholds, q)
                logits[name].append(served)
                calls[name] += used
    logits = {name: torch.cat(values) for name, values in logits.items()}
    outcomes = {name: early.disagreement(logits[name], logits['exact'], thresholds)
                for name in ('previous', 'prefix', 'prefix_cheap')}
    out = folder(args)
    replay, serving = out / 'replay.pt', out / 'serving.pt'
    torch.save(dict(logits=logits, monitor_exact=state['exact_monitor'], monitor_cheap=cheap_monitor,
                    thresholds=thresholds, test_indices=order, labels=labels), replay)
    torch.save(dict(current=current.state_dict(), legacy=legacy, reader=reader, correction=correction,
                    thresholds=thresholds, prefix_fraction=fraction, previous_fraction=previous,
                    sample=vision.preprocess(images[:128], 64)), serving)
    record(args, dict(reference_seed=args.reference_seed, width=choice['width'], forecast_sha256=sealed,
                      test_loaded=True, test_examples=len(labels), calls=calls, outcomes=outcomes,
                      current_predictions_shared=True, replay_sha256=base.digest(replay),
                      serving_sha256=base.digest(serving)))


@torch.no_grad()
def timing(args, load_model=load_student):
    choice = json.loads(args.choice.read_text())
    previous = json.loads(args.audit_record.read_text())
    assert base.digest(args.serving) == previous['serving_sha256']
    state = torch.load(args.serving, map_location='cpu', weights_only=True)
    cache = torch.load(args.population, map_location='cpu', weights_only=True, mmap=True)
    prefix, teacher = teacher_block(cache)
    student = load_model(args, choice)
    current = copy.deepcopy(teacher)
    current.load_state_dict(state['current'])
    legacy, reader, correction = state['legacy'], state['reader'], state['correction']
    thresholds = state['thresholds']
    sample = state['sample'].contiguous(memory_format=torch.channels_last)

    def program(name):
        h = prefix(sample)
        features = current(h)
        current_logits = vision.linear(features, reader)
        if name == 'current':
            return current_logits
        if name == 'exact':
            return current_logits, vision.linear(teacher(h), legacy)
        if name == 'previous':
            z = vision.linear(features, legacy) + vision.linear(features, correction)
        else:
            z = vision.linear(student(h), legacy)
        served, _ = serve_old(z, h, teacher, legacy, thresholds, state[name + '_fraction'])
        return current_logits, served

    names = ['current', 'previous', 'prefix', 'exact']
    expected = program('current')
    assert all(torch.equal(program(name)[0], expected) for name in names[1:])
    for name in names:
        for _ in range(12):
            program(name)
    samples = {name: [] for name in names}
    for i in range(31):
        shift = i % len(names)
        for name in names[shift:] + names[:shift]:
            start = time.perf_counter()
            program(name)
            samples[name].append(time.perf_counter() - start)
    record(args, dict(reference_seed=previous['reference_seed'], batch=128, threads=12,
                      audit_sha256=base.digest(args.audit_record), samples_seconds=samples,
                      medians_seconds={name: float(np.median(s)) for name, s in samples.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('fit', 'audit', 'timing'), required=True)
    parser.add_argument('--width', type=int, choices=WIDTHS)
    parser.add_argument('--reference-seed', type=int)
    parser.add_argument('--plan-sha256', required=True)
    for key in ('population', 'out', 'archive', 'choice', 'student', 'serving', 'audit-record'):
        parser.add_argument('--' + key, type=Path, required=key in ('population', 'out'))
    args = parser.parse_args()
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    {'fit': fit, 'audit': audit, 'timing': timing}[args.stage](args)
