"""Predeclared protected-head continuation: preserve the old function, not its weights.

Continuation minimises ``L_primary + alpha*L_aux + lam*||p(h(x)) - z0(x)||^2``,
where ``p`` is a linear head initialised to the healthy consumer map and trained
with the encoder, and ``z0`` is the frozen healthy consumer's logits on the same
example.  Deployment uses one encoder and one forward pass; the old prediction
is ``argmax p(h(x))``.  The fixed-weight consumer is scored alongside, so the
record separates function compatibility from weight compatibility.

Protocol: docs/plans/2026-11-11-protected-head.md
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import paired_readout_continuation as paired
import benefit_compatibility_cpu as bc
from consumer_risk import certificate
from readout_nullspace import masked_primary_loss
from readout_transfer_audit import fit_head


def consumer_map(head, mu, sd):
    mu = mu.detach().double().reshape(-1)
    sd = sd.detach().double().reshape(-1)
    A = head.weight.detach().double() / sd
    B = head.bias.detach().double() - A @ mu
    return A, B


def step(model, protected, opt, x, y, a, mask, kind, alpha, detached, lam, z0):
    h = model(x)
    lp = masked_primary_loss(bc.current_logits(model, h), y, mask)
    ha = h.detach() if detached else h
    la = paired.auxiliary_loss(model, ha.float(), a, kind)
    if lam > 0:
        lprot = F.mse_loss(protected(h.float()), z0)
    else:
        lprot = torch.zeros((), dtype=lp.dtype)
    loss = lp + alpha * la + lam * lprot
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite operating loss')
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(lp.detach()), float(la.detach()), float(lprot.detach())


@torch.no_grad()
def score(model, protected, part, base, A, B, kind, monitor=None):
    x, y, a = part
    h = paired.embed(model, x)
    z = protected(h.float()).double() if protected is not None else h @ A.T + B
    z0 = base @ A.T + B
    fixed = h @ A.T + B
    changes = int((z.argmax(1) != z0.argmax(1)).sum())
    fixed_changes = int((fixed.argmax(1) != z0.argmax(1)).sum())
    hcenter = h - h.mean(0)
    bcenter = base - base.mean(0)
    result = dict(
        frozen=paired.metrics(z, y), fixed_weights=paired.metrics(fixed, y),
        own=paired.metrics(bc.current_logits(model, h), y),
        reference=paired.metrics(z0, y),
        max_absolute_healthy_logit_error=float((z - z0).abs().max()),
        prediction_changes=changes, disagreement=changes / len(y),
        fixed_weight_disagreement=fixed_changes / len(y),
        relative_trace=float(hcenter.square().sum() / bcenter.square().sum().clamp_min(1e-30)),
        auxiliary_mse=float(paired.auxiliary_loss(model, h.float(), a, kind)))
    if monitor is not None:
        bound = certificate(z.argmax(1).numpy(), z0.argmax(1).numpy(), monitor, .05 / 40)
        remaining = np.setdiff1d(np.arange(len(y)), monitor)
        old_correct = z0.argmax(1).numpy() == y.numpy()
        new_correct = z.argmax(1).numpy() == y.numpy()
        bound.update(budget=.05, certified=bound['upper_remaining_disagreement'] <= .05,
                     remainder_loss_increase=float(old_correct[remaining].mean() - new_correct[remaining].mean()),
                     remainder_disagreement=float(np.mean(z.argmax(1).numpy()[remaining] != z0.argmax(1).numpy()[remaining])))
        result['consumer_audit'] = bound
    return result


def run(args):
    started = time.time()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, data, kind, splits = bc.load(args)
    x, y, a, mask = [data[k] for k in ('x', 'y', 'a', 'labeled')]
    warm = args.warm_epochs or (60 if args.domain == 'har' else 25)
    cont = args.cont_epochs or (30 if args.domain == 'har' else 25)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = torch.Generator().manual_seed(args.seed + 10000)
    for epoch in range(warm):
        order = torch.randperm(len(x), generator=rng)
        for j in range(0, len(order), 256):
            idx = order[j:j + 256]
            bc.step(model, opt, x[idx], y[idx], a[idx], mask[idx], kind, 1., True)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps(dict(stage='warm', epoch=epoch + 1, seed=args.seed,
                                  elapsed_s=time.time() - started)), flush=True)
    model.eval()
    reference = copy.deepcopy(model.state_dict())
    moments = copy.deepcopy(opt.state_dict())
    base = paired.embed(model, x)
    head, mu, sd = fit_head(base[mask], y[mask])
    A, B = consumer_map(head, mu, sd)
    z0 = (base @ A.T + B).float()
    target = z0.argmax(1)
    bases = {name: paired.embed(model, part[0]) for name, part in data.items()
             if name in ('validation', 'test')}
    model_sha, moment_sha = paired.state_hash(reference), paired.state_hash(moments)
    n_classes, width = A.shape
    monitor = None
    if args.phase == 'confirmation':
        n = len(data['test'][1])
        monitor = np.random.default_rng(1850001 + args.seed).choice(
            n, min(1024, n - 1), replace=False)
    out = dict(
        protocol='predeclared protected-head continuation, single encoder one pass',
        plan_sha256=args.plan_sha256, phase=args.phase, seed=args.seed,
        domain=args.domain, auxiliary=kind,
        device='cpu', cuda_visibility=os.environ['CUDA_VISIBLE_DEVICES'],
        torch_threads=torch.get_num_threads(), torch_version=torch.__version__,
        python_version=platform.python_version(), splits=splits,
        configuration=dict(warm_epochs=warm, continuation_epochs=cont, dose=args.dose,
                           width=width, batch=256, lr=1e-3, encoder_at_inference=1,
                           forward_passes_at_inference=1,
                           protected_head_parameters=int(n_classes * width + n_classes)),
        source_sha256={n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                       for n in ['protected_head_cpu.py', 'benefit_compatibility_cpu.py',
                                 'readout_nullspace.py', 'paired_readout_continuation.py',
                                 'consumer_risk.py', 'readout_transfer_audit.py']},
        reference_model_sha256=model_sha, reference_optimizer_sha256=moment_sha,
        reference_encoder_required_at_inference=False,
        validation=score(model, None, data['validation'], bases['validation'], A, B, kind),
        branches=[], complete=False)
    states = args.out.parent / (args.out.stem + '_states')
    states.mkdir(exist_ok=True, parents=True)
    torch.save(dict(model=reference, optimizer=moments, head=head.state_dict(),
                    mean=mu, sd=sd, splits=splits), states / 'reference.pt')
    arms = [('detach', 0., 0.), ('joint', 0., 0.)]
    arms += [('protected', lam, lam) for lam in args.lams]
    for mode, lam, _ in arms:
        model.load_state_dict(reference)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.load_state_dict(copy.deepcopy(moments))
        protected = None
        if mode == 'protected':
            protected = nn.Linear(width, n_classes)
            with torch.no_grad():
                protected.weight.copy_(A.float())
                protected.bias.copy_(B.float())
            opt.add_param_group(dict(params=protected.parameters(), lr=1e-3))
        rng = torch.Generator().manual_seed(args.seed + 20000)
        order_hash = hashlib.sha256()
        trace = []
        begin = time.time()
        for epoch in range(cont):
            order = torch.randperm(len(x), generator=rng)
            order_hash.update(order.numpy().tobytes())
            for j in range(0, len(order), 256):
                idx = order[j:j + 256]
                step(model, protected, opt, x[idx], y[idx], a[idx], mask[idx], kind,
                     args.dose, mode == 'detach', lam, z0[idx])
            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == cont:
                with torch.no_grad():
                    h = paired.embed(model, x[:2048]).float()
                    z = protected(h) if protected is not None else h @ A.float().T + B.float()
                    changed = int((z.argmax(1) != target[:2048]).sum())
                trace.append(dict(epoch=epoch + 1, fitting_disagreement=changed / 2048))
                print(json.dumps(dict(stage='continue', mode=mode, lam=lam, epoch=epoch + 1,
                                      seed=args.seed, fitting_disagreement=changed / 2048,
                                      elapsed_s=time.time() - begin)), flush=True)
        row = dict(mode=mode, lam=lam, initial_model_sha256=model_sha,
                   initial_optimizer_sha256=moment_sha, trace=trace,
                   minibatch_order_sha256=order_hash.hexdigest(),
                   final_model_sha256=paired.state_hash(model.state_dict()),
                   validation=score(model, protected, data['validation'],
                                    bases['validation'], A, B, kind),
                   elapsed_s=time.time() - begin)
        if protected is not None:
            row['protected_head_sha256'] = paired.state_hash(protected.state_dict())
        if args.phase == 'confirmation':
            row['test'] = score(model, protected, data['test'], bases['test'], A, B, kind,
                                monitor)
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                        protected=None if protected is None else protected.state_dict()),
                   states / ('%s_%g.pt' % (mode, lam)))
        out['branches'].append(row)
        paired.save_json(args.out.with_suffix('.partial.json'), out)
    assert len({b['minibatch_order_sha256'] for b in out['branches']}) == 1
    if args.phase == 'confirmation':
        sample = data['validation'][0][:min(1024, len(data['validation'][0]))]
        with torch.no_grad():
            probe = nn.Linear(width, n_classes)
            probe.weight.copy_(A.float())
            probe.bias.copy_(B.float())
            model.load_state_dict(reference)
            times = {}
            for name, operation in [('ordinary', lambda: bc.current_logits(model, model(sample))),
                                    ('protected', lambda: bc.current_logits(model, model(sample))),
                                    ('protected_head', lambda: probe(model(sample)))]:
                for _ in range(3):
                    operation()
                samples = []
                for _ in range(9):
                    clock = time.perf_counter()
                    operation()
                    samples.append(time.perf_counter() - clock)
                times[name] = dict(batch_size=len(sample),
                                   median_seconds=float(np.median(samples)), repeats=9)
        out['inference_latency'] = times
    out.update(complete=True, elapsed_s=time.time() - started)
    paired.save_json(args.out, out)
    print(json.dumps(dict(stage='complete', output=args.out.name,
                          elapsed_s=out['elapsed_s'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=['har', 'radioml'], required=True)
    parser.add_argument('--phase', choices=['development', 'confirmation'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--dose', type=float, default=1.)
    parser.add_argument('--lams', type=float, nargs='+', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--warm-epochs', type=int, default=0)
    parser.add_argument('--cont-epochs', type=int, default=0)
    parser.add_argument('--limit-fit', type=int, default=0)
    parser.add_argument('--limit-eval', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
