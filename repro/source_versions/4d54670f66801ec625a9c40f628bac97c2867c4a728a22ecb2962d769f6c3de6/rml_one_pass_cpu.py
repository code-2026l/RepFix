"""Fit and audit a single-forward compatibility program at the useful SNR rate."""

import argparse
import copy
import json
import platform
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from consumer_contracts import confidence_thresholds
from rml_trajectory_cpu import aligned_logits, base, fit_stream, fresh_state, legacy_logits, paired

WIDTHS = (0, 8, 16, 32)
DUAL = ((.0003, 1), (.0003, 4), (.001, 1), (.001, 4))
CANDIDATES = [f'side/{width}' for width in WIDTHS] + [f'dual/{lr}/{weight}' for lr, weight in DUAL]


def teacher(z):
    centered = z.float() - z.float().mean(1, keepdim=True)
    return centered, centered.softmax(1), centered.square().mean().clamp_min(1e-6)


def distill(prediction, target, indices):
    z, probability, variance = target
    centered = prediction - prediction.mean(1, keepdim=True)
    return F.mse_loss(centered, z[indices]) / variance + F.kl_div(
        centered.log_softmax(1), probability[indices], reduction='batchmean')


def ridge(h, z, layer):
    x, y = h.double(), z.double()
    xm, ym = x.mean(0), y.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(.001)
    a = (x - xm) / scale
    q = torch.linalg.solve(a.T @ a + torch.eye(a.shape[1], dtype=a.dtype) * (.001 * len(a)),
                           a.T @ (y - ym))
    weight = (q / scale[:, None]).T
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(ym - weight @ xm)


def fit_repair(params, forward, x, old, current, seed, rate, weight):
    optimizer = torch.optim.Adam(params, lr=rate)
    generator = torch.Generator().manual_seed(seed + 5000)
    for _ in range(36):
        for ids in torch.randperm(len(x), generator=generator).split(256):
            z_old, z_current = forward(x[ids], ids)
            loss = weight * distill(z_old, old, ids)
            if current is not None:
                loss = loss + distill(z_current, current, ids)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def compose(outer, inner):
    return outer.weight @ inner.weight, outer.bias + outer.weight @ inner.bias


def folded(model, old=None, hidden=None):
    first, _, second, _, last = model.encoder
    current_weight, current_bias = compose(model.primary, last)
    middle_weight, middle_bias = second.weight, second.bias
    if old is None:
        weight, bias = current_weight, current_bias
    elif hidden is None:
        old_weight, old_bias = compose(old, last)
        weight = torch.cat((current_weight, old_weight))
        bias = torch.cat((current_bias, old_bias))
    else:
        width = hidden.out_features
        middle_weight = torch.cat((middle_weight, hidden.weight))
        middle_bias = torch.cat((middle_bias, hidden.bias))
        weight = torch.cat((F.pad(current_weight, (0, width)), old.weight))
        bias = torch.cat((current_bias, old.bias))
    return nn.Sequential(copy.deepcopy(first), nn.ReLU(),
                         affine(middle_weight, middle_bias), nn.ReLU(), affine(weight, bias)).eval()


def affine(weight, bias):
    layer = nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    return layer


def side_repair(model, x, z, name, seed):
    width = int(name.split('/')[1])
    with torch.no_grad():
        first = model.encoder[:2](x)
        second = model.encoder[2:4](first)
    old = nn.Linear(64 + width, 11)
    start = nn.Linear(64, 11)
    ridge(second, z, start)
    with torch.no_grad():
        old.weight.copy_(F.pad(start.weight, (0, width)))
        old.bias.copy_(start.bias)
    hidden = nn.Linear(64, width) if width else None

    def forward(h, ids):
        features = second[ids]
        if hidden is not None:
            features = torch.cat((features, hidden(h).relu()), 1)
        return old(features), None

    params = list(old.parameters()) + (list(hidden.parameters()) if hidden is not None else [])
    fit_repair(params, forward, first, teacher(z), None, seed, .003, 1)
    if hidden is not None:
        return folded(model, old, hidden)
    program = folded(model)
    program[-1] = affine(torch.cat((program[-1].weight, old.weight)),
                         torch.cat((program[-1].bias, old.bias)))
    return program


def dual_repair(model, x, z, name, seed):
    _, rate, weight = name.split('/')
    clone = copy.deepcopy(model)
    old = nn.Linear(64, 11)
    with torch.no_grad():
        h = model(x)
        ridge(h, z, old)
        target = teacher(model.primary(h))

    def forward(batch, _):
        h = clone(batch)
        return old(h), clone.primary(h)

    fit_repair(list(clone.encoder.parameters()) + list(clone.primary.parameters()) + list(old.parameters()),
               forward, x, teacher(z), target, seed, float(rate), int(weight))
    return folded(clone, old)


def split_and_warm(seed):
    raw = torch.load(paired.cdm._path('RML2016.10a_pre.pt'), map_location='cpu', mmap=True, weights_only=True)
    generator = torch.Generator().manual_seed(seed + 10000)
    order = torch.randperm(len(raw['y_tr']), generator=generator)
    fit = order[:12000]
    x = raw['X_tr'][fit].float()
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-8)
    x = (x - mean) / scale
    y, snr = raw['y_tr'][fit].long(), raw['aux_tr'][fit].float()
    snr = (snr - snr.mean()) / snr.std(unbiased=False)
    known = torch.zeros(len(y), dtype=torch.bool)
    for label in range(11):
        ids = (y == label).nonzero().flatten()
        known[ids[torch.randperm(len(ids), generator=generator)[:round(.03 * len(ids))]]] = True
    torch.manual_seed(seed)
    healthy = paired.ram.RealAuxModel(x.shape[1], 11, 64, 'reg', 1)
    optimizer = torch.optim.Adam(healthy.parameters(), lr=.001)
    data = x, y, snr, known
    fit_stream(dict(model=healthy, optimizer=optimizer, objective='detach', stream=100), data, seed, 0, 12)
    return healthy, optimizer.state_dict(), data, raw, order, generator, mean, scale


def run(args):
    confirm = args.stage == 'confirmation'
    choice = json.loads(args.choice.read_text()) if confirm else None
    if confirm:
        assert choice['plan_sha256'] == args.plan_sha256 and choice['selected'] in CANDIDATES
    seed = args.seed
    healthy, moments, data, raw, order, generator, mean, scale = split_and_warm(seed)
    x, y, _, known = data
    h = base.embeds(healthy, x[known])
    reader = base.fit_affine(h, y[known])
    thresholds = list(confidence_thresholds(base.late.linear(h, reader)).values())
    readers = dict(modulation=reader)
    old_teacher = legacy_logits(healthy, x, readers)['modulation']
    endpoints = {}
    for objective in ('detach', 'joint'):
        state = fresh_state(healthy, moments, objective, 1., .0001, 300)
        fit_stream(state, data, seed, 0, 24)
        endpoints[objective] = state['model']
    joint, detach = (endpoints[name] for name in ('joint', 'detach'))
    names = CANDIDATES if not confirm else list(dict.fromkeys(('side/0', choice['selected'])))
    programs = {}
    for name in names:
        torch.manual_seed(seed + 2000)
        fit = side_repair if name.startswith('side/') else dual_repair
        programs[name] = fit(joint, x, old_teacher, name, seed)
    baseline = folded(joint)
    old_program = folded(healthy)
    old_program[-1] = affine(reader['weight'].float() @ healthy.encoder[-1].weight,
                              reader['bias'].float() + reader['weight'].float() @ healthy.encoder[-1].bias)
    print(json.dumps(dict(programs_frozen=True, stage=args.stage)), flush=True)
    if confirm:
        indices = torch.randperm(len(raw['y_te']), generator=generator)[:4400]
        x_eval, labels = (raw['X_te'][indices].float() - mean) / scale, raw['y_te'][indices].long()
    else:
        indices = order[13600:15200]
        x_eval, labels = (raw['X_tr'][indices].float() - mean) / scale, raw['y_tr'][indices].long()
    anchors = (raw['X_tr'][order[15200:15712]].float() - mean) / scale
    old_anchor = legacy_logits(healthy, anchors, readers)['modulation']
    with torch.no_grad():
        current = dict(joint=joint.primary(base.embeds(joint, x_eval)),
                       detach=detach.primary(base.embeds(detach, x_eval)), folded=baseline(x_eval))
        old = legacy_logits(healthy, x_eval, readers)['modulation']
        legacy = dict(direct=legacy_logits(joint, x_eval, readers)['modulation'],
                      ridge512=aligned_logits(joint, x_eval, anchors, old_anchor, reader)['modulation'],
                      exact=legacy_logits(healthy, x_eval, readers)['modulation'],
                      folded_exact=old_program(x_eval))
        for name, program in programs.items():
            current[name], legacy[name] = program(x_eval).split(11, dim=1)
    folder = args.out.parent / f'{args.out.stem}_states'
    folder.mkdir(exist_ok=True, parents=True)
    replay = folder / 'replay.pt'
    torch.save(dict(labels=labels, old=old, current=current, legacy=legacy, thresholds=thresholds), replay)
    program_path = folder / 'program.pt'
    if confirm:
        torch.save(dict(chosen=programs[choice['selected']].state_dict(), baseline=baseline.state_dict(),
                        reference=old_program.state_dict()), program_path)
    base.save(args.out, dict(complete=True, device='cpu', stage=args.stage, seed=seed, branches=[],
        plan_sha256=args.plan_sha256, test_loaded=confirm, labeled=int(known.sum()),
        choice_sha256=base.digest(args.choice) if confirm else None, replay_sha256=base.digest(replay),
        program_sha256=base.digest(program_path) if confirm else None,
        source_sha256={Path(__file__).name: base.digest(Path(__file__))}))


def from_state(state):
    return nn.Sequential(affine(state['0.weight'], state['0.bias']), nn.ReLU(),
                         affine(state['2.weight'], state['2.bias']), nn.ReLU(),
                         affine(state['4.weight'], state['4.bias'])).eval()


class ExactPair(nn.Module):
    def __init__(self, current, reference):
        super().__init__()
        self.current = current
        self.reference = reference

    def forward(self, x):
        return torch.cat((self.current(x), self.reference(x)), 1)


@torch.inference_mode()
def benchmark(args):
    payload = torch.load(args.program, map_location='cpu', weights_only=True)
    current, chosen, reference = (from_state(payload[name]) for name in ('baseline', 'chosen', 'reference'))
    generator = torch.Generator().manual_seed(args.seed)
    raw = dict(current=current, chosen=chosen, exact=ExactPair(current, reference).eval())
    results = {}
    for batch, repeats in ((1, 3000), (128, 500)):
        x = torch.randn(batch, current[0].in_features, generator=generator)
        programs = {name: torch.jit.freeze(torch.jit.trace(model, x)) for name, model in raw.items()}
        for fn in programs.values():
            for _ in range(100):
                fn(x)
        blocks = {name: [] for name in programs}
        for block in range(9):
            order = list(programs)
            if block % 2:
                order.reverse()
            for name in order:
                fn = programs[name]
                start = time.perf_counter_ns()
                for _ in range(repeats):
                    fn(x)
                blocks[name].append((time.perf_counter_ns() - start) / repeats)
        results[str(batch)] = blocks
    base.save(args.out, dict(complete=True, device='cpu', stage='timing', seed=args.seed, branches=[],
        plan_sha256=args.plan_sha256, program_sha256=base.digest(args.program),
        torch=torch.__version__, machine=platform.machine(), threads=torch.get_num_threads(), timings=results,
        source_sha256={Path(__file__).name: base.digest(Path(__file__))}))


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
    benchmark(args) if args.stage == 'timing' else run(args)
