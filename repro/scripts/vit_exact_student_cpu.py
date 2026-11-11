"""Fit a one-pass visual student from frozen hidden features and saved teachers."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

import contrastive_benefit_cpu as contrastive
import nonlinear_suffix_cpu as factory
import vision_auxiliary_cpu as base
from consumer_routing import decisions

COEFFICIENTS = (.001, .01, .1, 1.)
TEACHERS = ('joint', 'two_view', 'double_epoch')


class ExactVisualStudent(nn.Module):
    def __init__(self, healthy, weight, bias, legacy, both=True):
        super().__init__()
        self.healthy = healthy
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)
        self.register_buffer('old_weight', legacy['weight'])
        self.register_buffer('old_bias', legacy['bias'])
        self.both = both

    def forward(self, tokens):
        h, hidden = self.healthy.features(tokens)
        z = F.linear(torch.cat((h, hidden), 1), self.weight, self.bias)
        if self.both:
            return z, F.linear(h, self.old_weight, self.old_bias)
        return z


@torch.no_grad()
def target_features(model, inputs):
    return torch.stack([base.embeds(model, view) for view in inputs])


@torch.no_grad()
def auxiliary(features, mean, sd):
    values, counts = [], []
    for first in range(0, features.shape[1], 128):
        batch = features[:, first:first + 128]
        if batch.shape[1] < 2:
            continue
        values.append(contrastive.contrastive(batch[0], batch[1], mean, sd).item())
        counts.append(batch.shape[1])
    return dict(losses=values, counts=counts)


def accuracy(z, labels):
    return float((z.argmax(1) == labels).double().mean())


@torch.no_grad()
def run(args):
    start = time.perf_counter()
    cache = torch.load(args.population, map_location='cpu', weights_only=True, mmap=True)
    paths = [args.joint, args.two_view, args.double_epoch]
    states, records = {}, {}
    for name, record_path in zip(TEACHERS, paths):
        record = json.loads(record_path.read_text())
        assert record['complete'] and record['reference_seed'] == args.reference_seed
        assert record['seed'] == args.seed
        endpoint = record_path.parent / (record_path.stem + '_states') / 'endpoint.pt'
        assert base.digest(endpoint) == record['state_sha256']
        records[name] = record
        states[name] = torch.load(endpoint, map_location='cpu', weights_only=True)
    ref = states['joint']['reference']
    assert all(torch.equal(state['reference']['fitting'], ref['fitting']) and
               torch.equal(state['reference']['validation'], ref['validation']) for state in states.values())
    _, healthy = factory.build_for(cache)
    healthy.load_state_dict(cache['healthy'])
    healthy.eval().requires_grad_(False)
    fitting, validation = ref['fitting'], ref['validation']
    hfit, hval = cache['h'][:, fitting], cache['h'][:, validation]
    xfit = torch.cat((hfit, cache['hidden'][:, fitting]), -1).flatten(0, 1).double()
    phi_val = torch.cat((hval, cache['hidden'][:, validation]), -1)
    mean, sd = xfit.mean(0), xfit.std(0, unbiased=False).clamp_min(.001)
    design = (xfit - mean) / sd
    gram = design.T @ design / len(design)
    inputs_fit = cache['x'][:, fitting]
    inputs_val = cache['x'][:, validation]
    old = base.late.linear(hval[0], ref['heads']['coarse'])
    coefficients = COEFFICIENTS
    if args.stage == 'confirmation':
        choice = json.loads(args.choice.read_text())
        assert choice['plan_sha256'] == args.plan_sha256
        coefficients = (choice['coefficient'],)
    outputs, models, teacher_predictions, teacher_aux, rows = {}, {}, {}, {}, []
    identity = torch.eye(gram.shape[0], dtype=gram.dtype)
    solvers = {str(coefficient): torch.linalg.cholesky(gram + coefficient * identity) for coefficient in coefficients}
    for name, state in states.items():
        model = copy.deepcopy(healthy)
        model.load_state_dict(state['model'])
        model.eval().requires_grad_(False)
        fitting_features = target_features(model, inputs_fit)
        validation_features = target_features(model, inputs_val)
        reader, head = state['reader'], state['head']
        teacher_z = base.late.linear(validation_features[0], reader)
        torch.testing.assert_close(teacher_z, state['own_logits'], atol=2e-4, rtol=2e-4)
        teacher_predictions[name] = teacher_z.argmax(1)
        target = (fitting_features - hfit).flatten(0, 1).double()
        target_mean = target.mean(0)
        cross = design.T @ (target - target_mean) / len(design)
        teacher_aux[name] = auxiliary(validation_features, head['mean'], head['sd'])
        for coefficient in coefficients:
            key = f'{name}/{coefficient}'
            coeff = torch.cholesky_solve(cross, solvers[str(coefficient)])
            weight = (coeff / sd[:, None]).T.float()
            bias = (target_mean - (mean / sd) @ coeff).float()
            student_h = hval + F.linear(phi_val, weight, bias)
            z = base.late.linear(student_h[0], reader)
            folded_weight = reader['weight'] @ weight
            folded_weight[:, :hfit.shape[-1]] += reader['weight']
            folded_bias = reader['bias'] + reader['weight'] @ bias
            fused = F.linear(phi_val[0], folded_weight, folded_bias)
            torch.testing.assert_close(fused, z, atol=2e-3, rtol=2e-3)
            outputs[key] = z.argmax(1)
            aux = auxiliary(student_h, head['mean'], head['sd'])
            models[key] = dict(weight=folded_weight, bias=folded_bias)
            rows.append(dict(key=key, teacher=name, coefficient=coefficient,
                             accuracy=accuracy(z, cache['y'][validation]),
                             teacher_accuracy=accuracy(teacher_z, cache['y'][validation]), auxiliary=aux,
                             folded_changes=int((fused.argmax(1) != z.argmax(1)).sum()),
                             max_fold_error=float((fused - z).abs().max())))
        print(json.dumps(dict(teacher_exported=name, elapsed_s=round(time.perf_counter() - start))), flush=True)
        del model, fitting_features, validation_features
    rich = base.late.linear(phi_val[0], ref['rich_reader']).argmax(1)
    folder = args.out.parent / (args.out.stem + '_states')
    folder.mkdir(exist_ok=True, parents=True)
    replay = folder / 'replay.pt'
    program = folder / 'program.pt'
    torch.save(dict(labels=cache['y'][validation], predictions=outputs, teachers=teacher_predictions,
                    old=old, protected=base.late.linear(hval[0], ref['heads']['coarse']),
                    thresholds=list(ref['thresholds'].values()), rich=rich), replay)
    torch.save(dict(models=models, reference=ref, records={name:records[name]['state_sha256'] for name in TEACHERS},
                    healthy_sha256=base.paired.state_hash(cache['healthy'])), program)
    base.save(args.out, dict(complete=True, device='cpu', stage=args.stage,
        reference_seed=args.reference_seed, seed=args.seed, plan_sha256=args.plan_sha256,
        choice_sha256=base.digest(args.choice) if args.choice else None,
        official_test_loaded=False, fitting_examples=len(fitting), fitting_views=2,
        labeled_examples=int(ref['label_mask'].sum()), validation_examples=len(validation),
        outcomes=rows, teacher_auxiliary=teacher_aux, replay_sha256=base.digest(replay),
        program_sha256=base.digest(program), source_sha256={Path(__file__).name:base.digest(Path(__file__))},
        elapsed_s=time.perf_counter() - start, branches=[]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('pilot', 'confirmation'), required=True)
    parser.add_argument('--population', type=Path, required=True)
    parser.add_argument('--joint', type=Path, required=True)
    parser.add_argument('--two-view', type=Path, required=True)
    parser.add_argument('--double-epoch', type=Path, required=True)
    parser.add_argument('--reference-seed', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--choice', type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    run(parser.parse_args())
