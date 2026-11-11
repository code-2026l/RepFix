"""Raw-image held-out audit and paired forward latency of exact ViT students."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from contrastive_serving_cpu import trace
from suffix_serving import WholeNetwork
from vit_exact_student_cpu import ExactVisualStudent, base, factory
from consumer_routing import decisions
from vit_nonlinear_student_cpu import LogitStudent, NonlinearVisualStudent


class TeacherSuffix(nn.Module):
    def __init__(self, suffix, reader):
        super().__init__()
        self.suffix = suffix
        self.register_buffer('weight', reader['weight'])
        self.register_buffer('bias', reader['bias'])

    def forward(self, tokens):
        return F.linear(self.suffix(tokens), self.weight, self.bias)


@torch.no_grad()
def prepare(args):
    cache = torch.load(args.training_population, map_location='cpu', mmap=True, weights_only=True)
    images, labels, coarse, data_sha = base.late.images(False)
    assert len(images) == 10000
    prefix, healthy = factory.build_for(cache)
    healthy.load_state_dict(cache['healthy'])
    prefix.eval().requires_grad_(False)
    healthy.eval().requires_grad_(False)
    tokens, features, hidden = [], [], []
    for start in range(0, len(images), 128):
        raw = base.late.preprocess(images[start:start + 128], cache['metadata']['image_size'])
        xx = prefix(raw)
        hh, rr = healthy.features(xx)
        tokens.append(xx)
        features.append(hh)
        hidden.append(rr)
        if start % 1024 == 0:
            print(json.dumps(dict(prepared=start)), flush=True)
    folder = args.out.parent / (args.out.stem + '_states')
    folder.mkdir(exist_ok=True, parents=True)
    population = folder / 'population.pt'
    torch.save(dict(tokens=torch.cat(tokens), h=torch.cat(features), hidden=torch.cat(hidden),
                    labels=labels, coarse=coarse, data_sha256=data_sha,
                    healthy_sha256=base.paired.state_hash(cache['healthy'])), population)
    base.save(args.out, dict(complete=True, device='cpu', stage='prepare', plan_sha256=args.plan_sha256,
        official_test_loaded=True, population_sha256=base.digest(population), examples=len(labels),
        source_sha256={Path(__file__).name:base.digest(Path(__file__))}, branches=[]))


@torch.no_grad()
def run(args):
    population = torch.load(args.population, map_location='cpu', mmap=True, weights_only=True)
    cache = torch.load(args.training_population, map_location='cpu', mmap=True, weights_only=True)
    record = json.loads(args.student_record.read_text())
    assert record['stage'] == 'confirmation'
    source = args.student_record.parent / (args.student_record.stem + '_states') / 'program.pt'
    assert base.digest(source) == record['program_sha256']
    state = torch.load(source, map_location='cpu', weights_only=True)
    assert state['healthy_sha256'] == population['healthy_sha256']
    ref = state['reference']
    labels = population['labels']
    phi = torch.cat((population['h'], population['hidden']), 1)
    prefix, healthy = factory.build_for(cache)
    healthy.load_state_dict(cache['healthy'])
    healthy.eval().requires_grad_(False)
    old = base.late.linear(population['h'], ref['heads']['coarse'])
    fine_old = base.late.linear(population['h'], ref['heads']['fine'])
    teacher_predictions, student_predictions = {}, {}
    for name, path in zip(('joint', 'two_view', 'double_epoch'), (args.joint, args.two_view, args.double_epoch)):
        original = json.loads(path.read_text())
        artifact = path.parent / (path.stem + '_states') / 'endpoint.pt'
        assert base.digest(artifact) == state['records'][name] == original['state_sha256']
        endpoint = torch.load(artifact, map_location='cpu', weights_only=True)
        if args.stage == 'timing' and name != 'joint':
            continue
        teacher = copy.deepcopy(healthy)
        teacher.load_state_dict(endpoint['model'])
        teacher.eval().requires_grad_(False)
        student_key = next(key for key in state['models'] if key.startswith(name + '/'))
        params = state['models'][student_key]
        if args.stage == 'audit':
            logits = base.late.linear(base.embeds(teacher, population['tokens']), endpoint['reader'])
            teacher_predictions[name] = logits.argmax(1)
            if args.nonlinear:
                student_head = LogitStudent(len(params['mean']), params['hidden.weight'].shape[0],
                                           params['base_weight'], params['base_bias'], params['mean'], params['sd'])
                student_head.load_state_dict(params)
                student_head.eval().requires_grad_(False)
                student_predictions[name] = torch.cat([student_head(batch) for batch in phi.split(256)]).argmax(1)
            else:
                student_predictions[name] = F.linear(phi, params['weight'], params['bias']).argmax(1)
        if name == 'joint':
            joint_reader = endpoint['reader']
            joint_suffix = teacher
            selected = params
    if args.nonlinear:
        student = NonlinearVisualStudent(healthy, selected, ref['heads']['coarse'])
    else:
        student = ExactVisualStudent(healthy, selected['weight'], selected['bias'], ref['heads']['coarse'])
    current = WholeNetwork(prefix, TeacherSuffix(joint_suffix, joint_reader)).eval()
    both = WholeNetwork(prefix, student).eval()
    only = copy.deepcopy(student)
    only.both = False
    only_student = WholeNetwork(prefix, only).eval()
    images, _, _, data_sha = base.late.images(False)
    assert data_sha == population['data_sha256']
    sample = base.late.preprocess(images[:128], cache['metadata']['image_size'])
    output = {}
    if args.stage == 'audit':
        fine_thresholds = list(base.confidence_thresholds(
            base.late.linear(cache['h'][0,ref['fitting'][ref['label_mask']]], ref['heads']['fine'])).values())
        raw_current, raw_old = both(sample)
        compiled = trace(student, population['tokens'][:1])
        served_old, served_current = [], []
        native_old, native_current, native_fine = [], [], []
        for start in range(0, len(images), 128):
            tokens = population['tokens'][start:start + 128]
            h, hidden = healthy.features(tokens)
            features = torch.cat((h, hidden), 1)
            current_z = student.readout(features) if args.nonlinear else F.linear(
                features, selected['weight'], selected['bias'])
            native_old.append(base.late.linear(h, ref['heads']['coarse']))
            native_current.append(current_z)
            native_fine.append(base.late.linear(h, ref['heads']['fine']))
            z_compiled, old_compiled = compiled(tokens)
            served_old.append(old_compiled)
            served_current.append(z_compiled.argmax(1))
        served_old, served_current = torch.cat(served_old), torch.cat(served_current)
        native_old, native_current, native_fine = (torch.cat(parts) for parts in (native_old,native_current,native_fine))
        thresholds = list(ref['thresholds'].values())
        output = dict(labels=labels, teachers=teacher_predictions, students=student_predictions,
                      native_current=native_current.argmax(1), coarse=population['coarse'],
                      old=old, native_old=native_old, fine_old=fine_old, native_fine=native_fine,
                      coarse_thresholds=thresholds, fine_thresholds=fine_thresholds,
                      compiled_old=served_old, compiled_current=served_current,
                      compiled_old_changed=(decisions(old, thresholds) != decisions(served_old, thresholds)).sum(0).tolist(),
                      coarse_changed=(decisions(old, thresholds) != decisions(native_old, thresholds)).sum(0).tolist(),
                      fine_changed=(decisions(fine_old, fine_thresholds) != decisions(native_fine, fine_thresholds)).sum(0).tolist(),
                      current_changes=int((native_current.argmax(1) != student_predictions['joint']).sum()),
                      raw_sample_current_changes=int((raw_current.argmax(1) != native_current[:128].argmax(1)).sum()),
                      raw_sample_old_changes=(decisions(raw_old, thresholds) != decisions(old[:128], thresholds)).sum(0).tolist())
    else:
        fast = dict(teacher=current, student=only_student, protected=both)
        compiled = {name:trace(model, sample[:1]) for name,model in fast.items()}
        timings = {}
        for batch in (1,128):
            xx = sample[:batch]
            for fn in compiled.values():
                for _ in range(12):
                    fn(xx)
            blocks = {name:[] for name in compiled}
            for index in range(31):
                names = list(compiled)
                if index % 2:
                    names.reverse()
                for name in names:
                    start = time.perf_counter_ns()
                    compiled[name](xx)
                    blocks[name].append(time.perf_counter_ns() - start)
            timings[str(batch)] = blocks
        values = {name:fn(sample) for name,fn in compiled.items()}
        protected_current, protected_old = values['protected']
        thresholds = list(ref['thresholds'].values())
        output = dict(timings=timings,
                      current_changes=int((protected_current.argmax(1) != values['student'].argmax(1)).sum()),
                      old_changed=(decisions(protected_old,thresholds) != decisions(old[:128],thresholds)).sum(0).tolist())
    folder = args.out.parent / (args.out.stem + '_states')
    folder.mkdir(exist_ok=True,parents=True)
    replay = folder / 'replay.pt'
    torch.save(output,replay)
    base.save(args.out,dict(complete=True,device='cpu',stage=args.stage,seed=record['seed'],
        reference_seed=record['reference_seed'],plan_sha256=args.plan_sha256,
        official_test_loaded=True,student_program_sha256=record['program_sha256'],
        population_sha256=args.population_sha256,replay_sha256=base.digest(replay),
        source_sha256={Path(__file__).name:base.digest(Path(__file__))},branches=[]))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('prepare','audit','timing'),required=True)
    parser.add_argument('--training-population',type=Path,required=True)
    parser.add_argument('--population',type=Path)
    parser.add_argument('--population-sha256')
    parser.add_argument('--nonlinear',action='store_true')
    parser.add_argument('--student-record',type=Path)
    parser.add_argument('--joint',type=Path)
    parser.add_argument('--two-view',type=Path)
    parser.add_argument('--double-epoch',type=Path)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    args=parser.parse_args()
    prepare(args) if args.stage=='prepare' else run(args)
