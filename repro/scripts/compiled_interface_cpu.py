"""Predeclared numerical CPU study of a foldable compatible last affine map."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import copy
import hashlib
from pathlib import Path
import platform
import time

import numpy as np
import torch

import benefit_compatibility_cpu as previous
import paired_readout_continuation as paired
from compiled_readout import TrainableInterface, interface_shapes
from consumer_risk import certificate
from readout_nullspace import readout_projector
from readout_transfer_audit import fit_head

MODES = ['D', 'J', 'LD', 'LJ', 'CD', 'CJ']


def load(args):
    base, data, kind, splits = previous.load(args)
    return TrainableInterface(base), data, kind, splits


@torch.no_grad()
def score(model, part, base, head, mu, sd, kind, monitor=None):
    x, y, a = part
    h = paired.embed(model, x)
    z, z0 = head((h - mu) / sd), head((base - mu) / sd)
    hc, bc = h - h.mean(0), base - base.mean(0)
    result = dict(own=paired.metrics(previous.current_logits(model, h), y),
        frozen=paired.metrics(z, y), reference=paired.metrics(z0, y),
        prediction_changes=int((z.argmax(1) != z0.argmax(1)).sum()),
        disagreement=float((z.argmax(1) != z0.argmax(1)).double().mean()),
        max_absolute_healthy_logit_error=float((z - z0).abs().max()),
        relative_trace=float(hc.square().sum() / bc.square().sum().clamp_min(1e-30)),
        auxiliary_mse=float(paired.auxiliary_loss(model, h.float(), a, kind)))
    if monitor is not None:
        bound = certificate(z.argmax(1).numpy(), z0.argmax(1).numpy(), monitor, .05 / 120)
        remaining = np.setdiff1d(np.arange(len(y)), monitor)
        bound.update(budget=.05, certified=bound['upper_remaining_disagreement'] <= .05,
            remainder_loss_increase=float((z0.argmax(1)[remaining] == y[remaining]).double().mean()
                                         - (z.argmax(1)[remaining] == y[remaining]).double().mean()),
            remainder_disagreement=float((z.argmax(1)[remaining] != z0.argmax(1)[remaining]).double().mean()))
        result['consumer_audit'] = bound
    return result


@torch.no_grad()
def benchmark(ordinary, joint, compiled, reference, projector, inputs):
    from torch import nn
    timings = {}
    for size in [1, 1024]:
        sample = inputs[:min(size, len(inputs))]
        def two_encoders():
            h0 = reference(sample)
            return previous.current_logits(joint, h0 + (joint(sample) - h0) @ projector)
        operations = dict(ordinary=lambda: previous.current_logits(ordinary, ordinary(sample)),
                          compiled=lambda: previous.current_logits(compiled, compiled(sample)),
                          two_encoders=two_encoders)
        for operation in operations.values():
            for _ in range(5):
                operation()
        samples = {name: [] for name in operations}
        names = list(operations)
        for repeat in range(21):
            order = names if repeat % 2 == 0 else list(reversed(names))
            for name in order:
                start = time.perf_counter()
                operations[name]()
                samples[name].append(time.perf_counter() - start)
        timings[str(size)] = {name: dict(batch_size=len(sample), median_seconds=float(np.median(values)),
                                               repeats=len(values)) for name, values in samples.items()}
    calls = []
    modules = [*compiled.trunk.modules(), compiled.final]
    handles = [module.register_forward_hook(lambda *_: calls.append(1))
               for module in modules if isinstance(module, nn.Linear)]
    compiled(inputs[:1])
    for handle in handles:
        handle.remove()
    assert len(calls) == 3
    assert interface_shapes(compiled) == interface_shapes(ordinary)
    assert not any('projector' in key or 'anchor' in key for key in compiled.state_dict())
    return dict(timings=timings, deployed_encoder_affine_calls=len(calls),
        deployed_encoder_shapes=interface_shapes(compiled), inference_reference_required=False,
        inference_projection_required=False,
        encoder_parameters=sum(p.numel() for layer in [compiled.trunk, compiled.final] for p in layer.parameters()),
        encoder_parameter_bytes=sum(p.numel() * p.element_size()
                                    for layer in [compiled.trunk, compiled.final] for p in layer.parameters()))


def run(args):
    start = time.time()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, data, kind, splits = load(args)
    x, y, a, mask = [data[key] for key in ['x', 'y', 'a', 'labeled']]
    warm = args.warm_epochs or (60 if args.domain == 'har' else 25)
    cont = args.cont_epochs or (30 if args.domain == 'har' else 25)
    dose = 10. if args.domain == 'har' else 1.
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(args.seed + 10000)
    for epoch in range(warm):
        order = torch.randperm(len(x), generator=rng)
        for j in range(0, len(order), 256):
            idx = order[j:j+256]
            previous.step(model, optimizer, x[idx], y[idx], a[idx], mask[idx], kind, 1., True)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(dict(stage='warm', epoch=epoch+1, elapsed_s=time.time()-start), flush=True)
    model.eval()
    healthy_model = copy.deepcopy(model)
    initial = copy.deepcopy(model.state_dict())
    moments = copy.deepcopy(optimizer.state_dict())
    model_sha, adam_sha = paired.state_hash(initial), paired.state_hash(moments)
    hfit = paired.embed(model, x)
    head, mu, sd = fit_head(hfit[mask], y[mask])
    projector, projection_info = readout_projector(head.weight.detach(), sd)
    bases = {name: paired.embed(model, data[name][0]) for name in ['validation', 'test']}
    n = len(data['test'][1])
    monitor = np.random.default_rng(2870001 + args.seed).choice(n, min(1024, n-1), replace=False)
    states = args.out.parent / (args.out.stem + '_states')
    states.mkdir(exist_ok=True, parents=True)
    torch.save(dict(model=initial, optimizer=moments, head=head.state_dict(), mean=mu, sd=sd,
                    projector=projector, splits=splits), states / 'reference.pt')
    sources = ['compiled_interface_cpu.py', 'compiled_readout.py', 'benefit_compatibility_cpu.py',
               'paired_readout_continuation.py', 'readout_nullspace.py', 'consumer_risk.py']
    out = dict(protocol='predeclared folded compatible affine feature adaptation',
        plan_sha256=args.plan_sha256, seed=args.seed, domain=args.domain, auxiliary=kind,
        configuration=dict(width=64, batch=256, lr=.001, warm_epochs=warm,
                           continuation_epochs=cont, dose=dose, modes=MODES,
                           final_affine_dtype='float64', trunk_and_auxiliary_dtype='float32'),
        splits=splits, device='cpu', cuda_visibility=os.environ['CUDA_VISIBLE_DEVICES'],
        torch_threads=torch.get_num_threads(), torch_version=torch.__version__, python_version=platform.python_version(),
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in sources},
        reference_model_sha256=model_sha, reference_optimizer_sha256=adam_sha, projector=projection_info,
        consumer_bank=dict(candidates=120, delta=.05, budget=.05, monitor_inputs=len(monitor),
            monitor_membership_sha256=hashlib.sha256(monitor.astype(np.int64).tobytes()).hexdigest(),
            independent_of_training=True, separate_from_previous_banks=True), branches=[], complete=False)
    for mode in MODES:
        model = copy.deepcopy(healthy_model)
        assert paired.state_hash(model.state_dict()) == model_sha
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        optimizer.load_state_dict(copy.deepcopy(moments))
        assert paired.state_hash(optimizer.state_dict()) == adam_sha
        if mode.startswith('L'):
            model.freeze_trunk()
        elif mode.startswith('C'):
            model.constrain_last(projector)
        detached = mode.endswith('D')
        frozen_trunk_sha = paired.state_hash(model.trunk.state_dict())
        anchor_n = min(512, len(x))
        row = dict(mode=mode, initial_model_sha256=model_sha, initial_optimizer_sha256=adam_sha,
            guard_fixed_before_training=mode.startswith('C'), nonlinear_trunk_frozen=mode[0] in 'LC',
            diagnostic_trace=[dict(epoch=0, **previous.diagnose(model, x[:anchor_n], hfit[:anchor_n], head, mu, sd, None))])
        rng = torch.Generator().manual_seed(args.seed + 20000)
        order_hash = hashlib.sha256()
        begin = time.time()
        for epoch in range(cont):
            order = torch.randperm(len(x), generator=rng)
            order_hash.update(order.numpy().tobytes())
            for j in range(0, len(order), 256):
                idx = order[j:j+256]
                previous.step(model, optimizer, x[idx], y[idx], a[idx], mask[idx], kind, dose, detached)
            if epoch == 0 or (epoch+1) % 5 == 0 or epoch+1 == cont:
                trace = dict(epoch=epoch+1, **previous.diagnose(model, x[:anchor_n], hfit[:anchor_n], head, mu, sd, None))
                row['diagnostic_trace'].append(trace)
                print(dict(stage='continue', mode=mode, epoch=epoch+1, elapsed_s=time.time()-begin), flush=True)
        if mode[0] in 'LC':
            assert paired.state_hash(model.trunk.state_dict()) == frozen_trunk_sha
        deployed = model.fold()
        fold_error = float((paired.embed(model, data['test'][0]) - paired.embed(deployed, data['test'][0])).abs().max())
        assert fold_error < 1e-10
        row.update(validation=score(deployed, data['validation'], bases['validation'], head, mu, sd, kind),
            test=score(deployed, data['test'], bases['test'], head, mu, sd, kind, monitor),
            folding_max_absolute_feature_error=fold_error,
            minibatch_order_sha256=order_hash.hexdigest(), final_model_sha256=paired.state_hash(model.state_dict()),
            final_optimizer_sha256=paired.state_hash(optimizer.state_dict()),
            elapsed_s=time.time()-begin, trunk_sha256=frozen_trunk_sha)
        if mode.startswith('C'):
            assert row['test']['max_absolute_healthy_logit_error'] < 1e-8
            assert row['test']['prediction_changes'] == 0
            torch.save(dict(model=deployed.state_dict()), states / (mode + '_compiled.pt'))
            row['compiled_model_sha256'] = paired.state_hash(deployed.state_dict())
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), states / (mode + '.pt'))
        out['branches'].append(row)
        paired.save_json(args.out.with_suffix('.partial.json'), out)
    assert len({row['minibatch_order_sha256'] for row in out['branches']}) == 1
    ordinary = copy.deepcopy(healthy_model)
    ordinary.load_state_dict(torch.load(states / 'D.pt', weights_only=False)['model'])
    joint = copy.deepcopy(healthy_model)
    joint.load_state_dict(torch.load(states / 'J.pt', weights_only=False)['model'])
    out['deployment'] = benchmark(ordinary.fold(), joint.fold(), deployed, healthy_model.fold(),
                                  projector, data['validation'][0])
    out.update(complete=True, elapsed_s=time.time()-start)
    paired.save_json(args.out, out)
    print(dict(stage='complete', output=args.out.name, elapsed_s=out['elapsed_s']), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=['har', 'radioml'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--warm-epochs', type=int, default=0)
    parser.add_argument('--cont-epochs', type=int, default=0)
    parser.add_argument('--limit-fit', type=int, default=0)
    parser.add_argument('--limit-eval', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.phase = 'confirmation'
    if len(args.plan_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.plan_sha256):
        parser.error('The locally committed plan SHA-256 is required')
    run(args)


if __name__ == '__main__':
    main()
