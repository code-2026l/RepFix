"""Predeclared decision-preserving continuation arms.

Two training-time constraints on the frozen consumer, tested separately:
a hinge on its top-1 margin (``--gamma``) and a squared-logit anchor on its
function (``--anchor-gamma``). The healthy encoder and readout enter fitting
only; inference is the single shared encoder, so the deployed cost is the
ordinary cost. Development uses validation only; confirmation scores the fixed
recipe on the test corpus.

Protocols: docs/plans/2026-11-09-decision-preserving.md and
docs/plans/2026-11-10-anchored-consumer.md
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


def margin_penalty(h, A, B, target):
    z = h @ A.float().T + B.float()
    chosen = z.gather(1, target[:, None]).squeeze(1)
    others = z.scatter(1, target[:, None], -1e9).max(1).values
    return F.relu(others - chosen).mean()


def anchor_penalty(h, h0, A, B):
    z = h @ A.float().T + B.float()
    z0 = h0 @ A.float().T + B.float()
    return F.mse_loss(z, z0)


def step(model, opt, x, y, a, mask, kind, alpha, detached, A, B, target, gamma,
         anchor_gamma=0., frozen=None):
    h = model(x)
    lp = masked_primary_loss(bc.current_logits(model, h), y, mask)
    ha = h.detach() if detached else h
    la = paired.auxiliary_loss(model, ha.float(), a, kind)
    loss = lp + alpha * la
    if gamma > 0:
        loss = loss + gamma * margin_penalty(h, A, B, target)
    if anchor_gamma > 0:
        with torch.no_grad():
            h0 = frozen(x)
        loss = loss + anchor_gamma * anchor_penalty(h, h0, A, B)
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite operating loss')
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(lp.detach()), float(la.detach())


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
            step(model, opt, x[idx], y[idx], a[idx], mask[idx], kind, 1., True,
                 None, None, None, 0.)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps(dict(stage='warm', epoch=epoch + 1, seed=args.seed,
                                  elapsed_s=time.time() - started)), flush=True)
    model.eval()
    reference = copy.deepcopy(model.state_dict())
    moments = copy.deepcopy(opt.state_dict())
    frozen = copy.deepcopy(model)
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    base = paired.embed(model, x)
    head, mu, sd = fit_head(base[mask], y[mask])
    A, B = consumer_map(head, mu, sd)
    with torch.no_grad():
        target = (base @ A.T + B).argmax(1)
    bases = {name: paired.embed(model, part[0]) for name, part in data.items()
             if name in ('validation', 'test')}
    model_sha, moment_sha = paired.state_hash(reference), paired.state_hash(moments)
    monitor = None
    if args.phase == 'confirmation':
        n = len(data['test'][1])
        monitor = np.random.default_rng(1850001 + args.seed).choice(
            n, min(1024, n - 1), replace=False)
    out = dict(
        protocol='predeclared decision-preserving continuation, single shared encoder',
        plan_sha256=args.plan_sha256, phase=args.phase, seed=args.seed,
        domain=args.domain, auxiliary=kind, gamma=args.gamma, margin=0.,
        anchor_gamma=args.anchor_gamma,
        device='cpu', cuda_visibility=os.environ['CUDA_VISIBLE_DEVICES'],
        torch_threads=torch.get_num_threads(), torch_version=torch.__version__,
        python_version=platform.python_version(), splits=splits,
        configuration=dict(warm_epochs=warm, continuation_epochs=cont, dose=args.dose,
                           width=64, batch=256, lr=1e-3, encoder_at_inference=1),
        source_sha256={n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                       for n in ['decision_preserving_cpu.py', 'benefit_compatibility_cpu.py',
                                 'readout_nullspace.py', 'paired_readout_continuation.py',
                                 'consumer_risk.py', 'readout_transfer_audit.py']},
        reference_model_sha256=model_sha, reference_optimizer_sha256=moment_sha,
        reference_encoder_required_at_inference=False,
        validation=bc.score(model, data['validation'], bases['validation'],
                            head, mu, sd, None, kind),
        branches=[], complete=False)
    states = args.out.parent / (args.out.stem + '_states')
    states.mkdir(exist_ok=True, parents=True)
    torch.save(dict(model=reference, optimizer=moments, head=head.state_dict(),
                    mean=mu, sd=sd, splits=splits), states / 'reference.pt')
    arms = [('detach', 0., True), ('joint', 0., False)]
    if args.gamma > 0:
        arms.append(('penalty', args.gamma, False))
    if args.anchor_gamma > 0:
        arms.append(('anchor', args.anchor_gamma, False))
    for mode, gamma, detached in arms:
        anchor_gamma = args.anchor_gamma if mode == 'anchor' else 0.
        model.load_state_dict(reference)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.load_state_dict(copy.deepcopy(moments))
        assert paired.state_hash(model.state_dict()) == model_sha
        assert paired.state_hash(opt.state_dict()) == moment_sha
        rng = torch.Generator().manual_seed(args.seed + 20000)
        order_hash = hashlib.sha256()
        trace = []
        begin = time.time()
        for epoch in range(cont):
            order = torch.randperm(len(x), generator=rng)
            order_hash.update(order.numpy().tobytes())
            for j in range(0, len(order), 256):
                idx = order[j:j + 256]
                step(model, opt, x[idx], y[idx], a[idx], mask[idx], kind, args.dose,
                     detached, A, B, target[idx], gamma,
                     anchor_gamma=anchor_gamma, frozen=frozen)
            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == cont:
                with torch.no_grad():
                    z = paired.embed(model, x[:2048]) @ A.T + B
                    p0 = target[:2048]
                    changed = int((z.argmax(1) != p0).sum())
                trace.append(dict(epoch=epoch + 1, fitting_disagreement=changed / 2048))
                print(json.dumps(dict(stage='continue', mode=mode, gamma=gamma, epoch=epoch + 1,
                                      seed=args.seed, fitting_disagreement=changed / 2048,
                                      elapsed_s=time.time() - begin)), flush=True)
        row = dict(mode=mode, gamma=gamma, detached=detached,
                   initial_model_sha256=model_sha, initial_optimizer_sha256=moment_sha,
                   trace=trace, minibatch_order_sha256=order_hash.hexdigest(),
                   final_model_sha256=paired.state_hash(model.state_dict()),
                   validation=bc.score(model, data['validation'], bases['validation'],
                                       head, mu, sd, None, kind),
                   elapsed_s=time.time() - begin)
        if args.phase == 'confirmation':
            row['test'] = bc.score(model, data['test'], bases['test'], head, mu, sd,
                                   None, kind, monitor)
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()),
                   states / ('%s.pt' % mode))
        out['branches'].append(row)
        paired.save_json(args.out.with_suffix('.partial.json'), out)
    assert len({b['minibatch_order_sha256'] for b in out['branches']}) == 1
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
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--anchor-gamma', type=float, default=0.)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--warm-epochs', type=int, default=0)
    parser.add_argument('--cont-epochs', type=int, default=0)
    parser.add_argument('--limit-fit', type=int, default=0)
    parser.add_argument('--limit-eval', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
