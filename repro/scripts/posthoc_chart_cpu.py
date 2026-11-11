"""Fixed CPU development/confirmation calculation for post-hoc affine charts."""
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
from affine_chart import fold_chart, raw_chart
from compiled_readout import TrainableInterface, interface_shapes
from consumer_risk import certificate
from readout_alignment import fit_repairs
from readout_transfer_audit import fit_head

CANDIDATES = ['diagonal_stats', 'paired_ridge']


@torch.no_grad()
def score(model, part, reference, head, mean, sd, kind, monitor=None):
    x, y, a = part
    h = paired.embed(model, x)
    logits, old = head((h-mean)/sd), head((reference-mean)/sd)
    ah = h.to(next(model.aux.parameters()).dtype)
    result = dict(own=paired.metrics(previous.current_logits(model, h), y),
                  frozen=paired.metrics(logits, y), reference=paired.metrics(old, y),
                  disagreement=float((logits.argmax(1) != old.argmax(1)).double().mean()),
                  prediction_changes=int((logits.argmax(1) != old.argmax(1)).sum()),
                  max_absolute_healthy_logit_error=float((logits-old).abs().max()),
                  auxiliary_mse=float(paired.auxiliary_loss(model, ah, a.to(ah.dtype), kind)))
    if monitor is not None:
        bound = certificate(logits.argmax(1).numpy(), old.argmax(1).numpy(), monitor, .05/40)
        bound.update(budget=.05, certified=bound['upper_remaining_disagreement'] <= .05)
        result['consumer_audit'] = bound
    return result


@torch.no_grad()
def benchmark(native, deployed, inputs):
    output = {}
    for size in [1, 1024]:
        sample = inputs[:min(size, len(inputs))]
        operations = dict(native=lambda: previous.current_logits(native, native(sample)),
                          repaired=lambda: previous.current_logits(deployed, deployed(sample)))
        for operation in operations.values():
            for _ in range(5):
                operation()
        times = {key: [] for key in operations}
        for repeat in range(31):
            for key in (list(operations) if repeat % 2 else list(reversed(operations))):
                start = time.perf_counter()
                operations[key]()
                times[key].append(time.perf_counter()-start)
        medians = {key: float(np.median(values)) for key, values in times.items()}
        output[str(size)] = dict(batch_size=len(sample), medians_s=medians,
                                 ratio=medians['repaired']/medians['native'], samples_s=times)
    return dict(timings=output, identical_encoder_shapes=interface_shapes(native)==interface_shapes(deployed),
                deployed_encoder_shapes=interface_shapes(deployed), inference_reference_required=False,
                inference_projection_required=False,
                encoder_parameter_bytes=sum(p.numel()*p.element_size()
                    for layer in [deployed.trunk, deployed.final] for p in layer.parameters()))


def run(args):
    start = time.time()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    base, data, kind, splits = previous.load(args)
    model = TrainableInterface(base)
    x, y, a, mask = [data[key] for key in ['x', 'y', 'a', 'labeled']]
    warm = args.warm_epochs or (60 if args.domain=='har' else 25)
    cont = args.cont_epochs or (30 if args.domain=='har' else 25)
    dose = 10. if args.domain=='har' else 1.
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(args.seed+10000)
    for epoch in range(warm):
        order = torch.randperm(len(x), generator=rng)
        for j in range(0, len(order), 256):
            idx = order[j:j+256]
            previous.step(model, optimizer, x[idx], y[idx], a[idx], mask[idx], kind, 1., True)
        if epoch==0 or (epoch+1)%10==0:
            print(dict(stage='warm', epoch=epoch+1, elapsed_s=time.time()-start), flush=True)
    model.eval()
    healthy = copy.deepcopy(model)
    initial, moments = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    model_sha, adam_sha = paired.state_hash(initial), paired.state_hash(moments)
    hfit = paired.embed(healthy, x)
    head, mean, sd = fit_head(hfit[mask], y[mask])
    parts = ['validation'] + (['test'] if args.phase=='confirmation' else [])
    references = {part: paired.embed(healthy, data[part][0]) for part in parts}
    anchors = torch.randperm(len(x), generator=torch.Generator().manual_seed(args.seed+32000))[:min(1024, len(x))]
    monitor = None
    if args.phase=='confirmation':
        n = len(data['test'][1])
        monitor = np.random.default_rng(3870001+args.seed).choice(n, min(1024, n-1), replace=False)
    states = args.out.parent/(args.out.stem+'_states')
    states.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=initial, optimizer=moments, head=head.state_dict(), mean=mean, sd=sd,
                    anchors=anchors, splits=splits), states/'reference.pt')
    sources = ['posthoc_chart_cpu.py', 'affine_chart.py', 'readout_alignment.py',
               'compiled_readout.py', 'benefit_compatibility_cpu.py', 'paired_readout_continuation.py']
    out = dict(protocol='post-hoc invertible affine coordinates after unrestricted nonlinear learning',
        plan_sha256=args.plan_sha256, phase=args.phase, seed=args.seed, domain=args.domain,
        device='cpu', cuda_visibility=os.environ['CUDA_VISIBLE_DEVICES'], torch_threads=1,
        torch_version=torch.__version__, python_version=platform.python_version(), splits=splits,
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources},
        configuration=dict(warm_epochs=warm, continuation_epochs=cont, dose=dose, batch=256, lr=.001,
                           candidates=CANDIDATES, anchor_examples=len(anchors), singular_value_relative_floor=.001),
        reference_model_sha256=model_sha, reference_optimizer_sha256=adam_sha,
        anchor_membership_sha256=paired.state_hash(anchors), labels_used_by_repair=False,
        old_anchor_features_required=True, branches=[], complete=False)
    for mode in ['D', 'J']:
        model = copy.deepcopy(healthy)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        optimizer.load_state_dict(copy.deepcopy(moments))
        assert paired.state_hash(model.state_dict())==model_sha
        assert paired.state_hash(optimizer.state_dict())==adam_sha
        rng = torch.Generator().manual_seed(args.seed+20000)
        order_hash = hashlib.sha256()
        for epoch in range(cont):
            order = torch.randperm(len(x), generator=rng)
            order_hash.update(order.numpy().tobytes())
            for j in range(0, len(order), 256):
                idx = order[j:j+256]
                previous.step(model, optimizer, x[idx], y[idx], a[idx], mask[idx], kind, dose, mode=='D')
            if epoch==0 or (epoch+1)%10==0 or epoch+1==cont:
                print(dict(stage='continue', mode=mode, epoch=epoch+1, elapsed_s=time.time()-start), flush=True)
        model.eval()
        native = model.fold()
        changed_trunk = paired.state_hash(model.trunk.state_dict())!=paired.state_hash(healthy.trunk.state_dict())
        new_fit = paired.embed(model, x)
        refit, refit_mu, refit_sd = fit_head(new_fit[mask], y[mask])
        repairs = fit_repairs(hfit[anchors], new_fit[anchors], mean, sd)
        row = dict(mode=mode, nonlinear_trunk_changed=changed_trunk,
                   initial_model_sha256=model_sha, initial_optimizer_sha256=adam_sha,
                   final_model_sha256=paired.state_hash(model.state_dict()),
                   final_optimizer_sha256=paired.state_hash(optimizer.state_dict()),
                   minibatch_order_sha256=order_hash.hexdigest(), repairs={})
        for part in parts:
            h = paired.embed(native, data[part][0])
            row[part] = score(native, data[part], references[part], head, mean, sd, kind)
            row[part]['supervised_refit'] = paired.metrics(refit((h-refit_mu)/refit_sd), data[part][1])
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), refit=refit.state_dict(),
                        refit_mean=refit_mu, refit_sd=refit_sd), states/(mode+'.pt'))
        for candidate in CANDIDATES:
            chart = raw_chart(repairs[candidate], mean, sd)
            deployed = fold_chart(model, chart)
            item = dict(condition_number=chart['condition_number'], clipped_directions=chart['clipped_directions'],
                        chart_sha256=paired.state_hash(chart), deployment=benchmark(native, deployed, data['validation'][0]))
            for part in parts:
                item[part] = score(deployed, data[part], references[part], head, mean, sd, kind,
                                   monitor if part=='test' else None)
                hn, hr = paired.embed(native, data[part][0]), paired.embed(deployed, data[part][0])
                zn, zr = previous.current_logits(native, hn), previous.current_logits(deployed, hr)
                item[part]['current_max_absolute_logit_error'] = float((zn-zr).abs().max())
                item[part]['current_prediction_changes'] = int((zn.argmax(1)!=zr.argmax(1)).sum())
                if item[part]['current_max_absolute_logit_error']>=1e-8:
                    raise ArithmeticError('Current-head rebasing failed its numerical identity')
            torch.save(dict(chart=chart, deployed=deployed.state_dict()), states/(mode+'_'+candidate+'.pt'))
            row['repairs'][candidate] = item
        out['branches'].append(row)
        paired.save_json(args.out.with_suffix('.partial.json'), out)
    assert out['branches'][0]['minibatch_order_sha256']==out['branches'][1]['minibatch_order_sha256']
    out.update(complete=True, elapsed_s=time.time()-start)
    paired.save_json(args.out, out)
    print(dict(stage='complete', elapsed_s=out['elapsed_s']), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=['har', 'radioml'], required=True)
    parser.add_argument('--phase', choices=['development', 'confirmation'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--warm-epochs', type=int, default=0)
    parser.add_argument('--cont-epochs', type=int, default=0)
    parser.add_argument('--limit-fit', type=int, default=0)
    parser.add_argument('--limit-eval', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
