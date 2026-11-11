"""Match SNR continuation and exact-student distillation work across arms."""
import argparse
import json
from pathlib import Path

import torch

from consumer_contracts import confidence_thresholds
from rml_anchored_one_pass_cpu import exact
from rml_one_pass_cpu import affine, folded, split_and_warm
from rml_trajectory_cpu import base, fit_stream, fresh_state, legacy_logits


def run(args):
    seed = args.seed
    healthy, moments, data, raw, _, generator, mean, scale = split_and_warm(seed)
    x, y, _, known = data
    h = base.embeds(healthy, x[known])
    reader = base.fit_affine(h, y[known])
    thresholds = list(confidence_thresholds(base.late.linear(h, reader)).values())
    readers = dict(modulation=reader)
    teachers, students = {}, {}
    for objective in ('detach', 'joint'):
        state = fresh_state(healthy, moments, objective, 1., .0001, 300)
        fit_stream(state, data, seed, 0, 24)
        teachers[objective] = state['model']
        torch.manual_seed(seed + 2000)
        students[objective] = exact(healthy, teachers[objective], reader, x, 'exact/16', seed)
    old_program = folded(healthy)
    old_program[-1] = affine(reader['weight'].float() @ healthy.encoder[-1].weight,
                              reader['bias'].float() + reader['weight'].float() @ healthy.encoder[-1].bias)
    print(json.dumps(dict(programs_frozen=True, stage='matched')), flush=True)
    indices = torch.randperm(len(raw['y_te']), generator=generator)[:4400]
    x_eval, labels = (raw['X_te'][indices].float() - mean) / scale, raw['y_te'][indices].long()
    with torch.no_grad():
        old = legacy_logits(healthy, x_eval, readers)['modulation']
        teacher_logits = {name: model.primary(base.embeds(model, x_eval)) for name, model in teachers.items()}
        student_logits = {name: program(x_eval) for name, program in students.items()}
    folder = args.out.parent / f'{args.out.stem}_states'
    folder.mkdir(exist_ok=True, parents=True)
    replay, programs = folder / 'replay.pt', folder / 'program.pt'
    torch.save(dict(labels=labels, snr=raw['aux_te'][indices].flatten(), thresholds=thresholds,
                    old=old, teachers=teacher_logits, students=student_logits), replay)
    torch.save(dict(reference=old_program.state_dict(),
                    **{name: model.state_dict() for name, model in students.items()}), programs)
    base.save(args.out, dict(complete=True, device='cpu', stage='matched', seed=seed, branches=[],
        plan_sha256=args.plan_sha256, test_loaded=True, labeled=int(known.sum()),
        replay_sha256=base.digest(replay), program_sha256=base.digest(programs),
        source_sha256={Path(__file__).name: base.digest(Path(__file__))}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run(parser.parse_args())
