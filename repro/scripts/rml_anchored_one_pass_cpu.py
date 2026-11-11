"""Distill current updates while anchoring the healthy legacy function."""

import argparse
import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from consumer_routing import decisions
from har_early_forecast import signed_margins
from rml_one_pass_cpu import affine, benchmark, distill, fit_repair, folded, ridge, run, teacher

CANDIDATES = [f'exact/{n}' for n in (0, 4, 8, 16)] + [f'guard/{w}' for w in (4, 12, 36)]


def exact(healthy, joint, reader, x, name, seed):
    width = int(name.split('/')[1])
    with torch.no_grad():
        first = healthy.encoder[:2](x)
        second = healthy.encoder[2:4](first)
        z = joint.primary(joint(x))
    current = nn.Linear(64 + width, 11)
    start = nn.Linear(64, 11)
    ridge(second, z, start)
    with torch.no_grad():
        current.weight.copy_(F.pad(start.weight, (0, width)))
        current.bias.copy_(start.bias)
    hidden = nn.Linear(64, width) if width else None

    def forward(h, ids):
        features = second[ids]
        if hidden is not None:
            features = torch.cat((features, hidden(h).relu()), 1)
        return current(features), None

    params = list(current.parameters()) + (list(hidden.parameters()) if hidden is not None else [])
    fit_repair(params, forward, first, teacher(z), None, seed, .003, 1)
    first, _, second, _, last = healthy.encoder
    old_w = reader['weight'].float()
    old_b = reader['bias'].float()
    second_w, second_b = second.weight, second.bias
    if hidden is not None:
        second_w = torch.cat((second_w, hidden.weight))
        second_b = torch.cat((second_b, hidden.bias))
    out = affine(torch.cat((current.weight, F.pad(old_w @ last.weight, (0, width)))),
                 torch.cat((current.bias, old_b + old_w @ last.bias)))
    return nn.Sequential(copy.deepcopy(first), nn.ReLU(), affine(second_w, second_b), nn.ReLU(), out).eval()


def guarded(healthy, joint, reader, thresholds, x, z_old, name, seed):
    model = copy.deepcopy(healthy)
    old = affine(reader['weight'].float(), reader['bias'].float())
    weight = int(name.split('/')[1])
    with torch.no_grad():
        current_target = teacher(joint.primary(joint(x)))
        old_target = teacher(z_old)
        old_decision = decisions(z_old, thresholds)
        target_margin = signed_margins(z_old, old_decision, thresholds).clamp(0., .25)
    optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.primary.parameters()) +
                                 list(old.parameters()), lr=.0003)
    generator = torch.Generator().manual_seed(seed + 5000)
    for _ in range(36):
        for ids in torch.randperm(len(x), generator=generator).split(256):
            h = model(x[ids])
            current, legacy = model.primary(h), old(h)
            observed_margin = signed_margins(legacy, old_decision[ids], thresholds)
            shortfall = (target_margin[ids] - observed_margin).relu().mean()
            loss = distill(current, current_target, ids) + weight * (distill(legacy, old_target, ids) + 2 * shortfall)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return folded(model, old)


def candidate(healthy, joint, reader, thresholds, x, z_old, name, seed):
    if name.startswith('exact/'):
        return exact(healthy, joint, reader, x, name, seed)
    return guarded(healthy, joint, reader, thresholds, x, z_old, name, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('development', 'confirmation', 'timing'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--choice', type=Path)
    parser.add_argument('--program', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args = parser.parse_args()
    benchmark(args) if args.stage == 'timing' else run(args, CANDIDATES, candidate)
