"""CPU-only paired continuation from one healthy model and Adam checkpoint.

Author and analyze locally. This program performs numerical calculations only.
Calibration examples never enter encoder training or readout fitting. Every
branch starts from identical model/optimizer states and uses the same future
minibatch order. Test outcomes never select checkpoints or hyperparameters.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cross_domain_mtl as cdm
import real_aux_mtl as ram
from readout_transfer_audit import fit_head, stats, stratified_split
from scale_boundaries import affine_scale_boundaries, boundary_summary


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(safe_json(payload), indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def state_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            arr = item.detach().cpu().contiguous().numpy()
            digest.update(str((arr.dtype, arr.shape)).encode())
            digest.update(arr.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


@torch.no_grad()
def embed(model, x, batch=2048):
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)]).double()


def auxiliary_loss(model, h, target, kind):
    if kind == 'variance':
        return (h - h.mean(0, keepdim=True)).square().mean()
    if kind == 'projected':
        return cdm.var_loss(model.aux_proj(h))
    if kind == 'subject':
        return F.cross_entropy(model.aux(h), target)
    if kind == 'snr':
        return F.mse_loss(model.aux(h).squeeze(-1), target)
    if kind == 'recon':
        return F.mse_loss(model.aux(h), target)
    raise ValueError(kind)


def filtered_input(h, mode, gate):
    if mode == 'detach':
        return h.detach()
    if mode == 'stfcal' and gate:
        hc = h - h.mean(0, keepdim=True)
        u = hc / hc.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return h - (hc * u).sum(-1, keepdim=True) * u
    return h


def calibrate_gate(model, x, y, a, kind, alpha):
    h = model(x).detach().requires_grad_(True)
    gp = torch.autograd.grad(F.cross_entropy(model.primary(h), y), h, retain_graph=True)[0]
    ga = torch.autograd.grad(auxiliary_loss(model, h, a, kind), h)[0]
    hc = h.detach() - h.detach().mean(0, keepdim=True)
    u = hc / hc.norm(dim=1, keepdim=True).clamp_min(1e-8)
    proj = (ga * u).sum(-1, keepdim=True) * u
    rho = float((proj.norm(dim=1) / (gp.norm(dim=1) + 1e-8)).mean())
    return dict(rho=rho, margin=alpha * rho, gate=bool(alpha * rho >= 1))


def metrics(logits, y, temperature=1.0, q=None):
    z = logits.detach() / float(temperature)
    p = z.softmax(1)
    confidence, prediction = p.max(1)
    correct = prediction == y
    ece = 0.0
    for k in range(15):
        selected = (confidence >= k / 15) & ((confidence < (k + 1) / 15) if k < 14 else (confidence <= 1))
        if selected.any():
            ece += float(selected.double().mean()) * abs(float(correct[selected].double().mean()) - float(confidence[selected].mean()))
    result = dict(acc=float(correct.double().mean()), nll=float(F.cross_entropy(z, y)), ece=ece,
                  largest_prediction_fraction=float(torch.bincount(prediction, minlength=z.shape[1]).max() / len(y)))
    if q is not None:
        inclusion = (1 - p) <= q
        result.update(coverage=float(inclusion[torch.arange(len(y)), y].double().mean()),
                      mean_set_size=float(inclusion.sum(1).double().mean()))
    return result


def temperature_fit(logits, y):
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], max_iter=80, line_search_fn='strong_wolfe')
    detached = logits.detach()
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(detached / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.detach().exp())


def conformal_quantile(logits, y, target=0.9):
    scores = 1 - logits.detach().softmax(1)[torch.arange(len(y)), y]
    rank = math.ceil((len(scores) + 1) * target)
    return float(scores.kthvalue(rank).values) if rank <= len(scores) else float('inf')


def geometry(h, reference, train_h=None, reference_train=None):
    hc = h - h.mean(0, keepdim=True)
    rc = reference - reference.mean(0, keepdim=True)
    eig = torch.linalg.eigvalsh(hc.T @ hc / len(hc)).clamp_min(0)
    weights = eig / eig.sum().clamp_min(1e-30)
    scale = float((hc * rc).sum() / rc.square().sum().clamp_min(1e-30))
    residual = float((hc - scale * rc).norm() / hc.norm().clamp_min(1e-30))
    return dict(covariance_trace=float(eig.sum()),
                relative_trace=float(hc.square().sum() / rc.square().sum().clamp_min(1e-30)),
                relative_scatter=float(hc.std(1).mean() / rc.std(1).mean().clamp_min(1e-30)),
                effective_rank=float(torch.exp(-(weights * weights.clamp_min(1e-30).log()).sum())),
                participation_rank=float(eig.sum().square() / eig.square().sum().clamp_min(1e-30)),
                best_scalar=scale, scalar_fit_relative_residual=residual)


def snapshot_metrics(model, data, reference, endpoint=False):
    xfit, yfit, xcal, ycal, xtest, ytest = data
    head, mu0, sd0, href, temperature, q0 = reference
    hfit = embed(model, xfit)
    htest = embed(model, xtest)
    with torch.no_grad():
        frozen = head((htest - mu0) / sd0)
        mu, sd = stats(hfit)
        recal = head((htest - mu) / sd)
        own = model.primary(htest.float()).double()
    out = dict(geometry=geometry(htest, href),
               frozen=metrics(frozen, ytest, q=q0),
               frozen_temperature=metrics(frozen, ytest, temperature=temperature),
               recalibrated=metrics(recal, ytest), own=metrics(own, ytest))
    if endpoint:
        head_new, mu_new, sd_new = fit_head(hfit, yfit)
        with torch.no_grad():
            retrained = head_new((htest - mu_new) / sd_new)
            hcal = embed(model, xcal)
            recal_cal = head((hcal - mu) / sd)
        q_recal = conformal_quantile(recal_cal, ycal)
        out['retrained'] = metrics(retrained, ytest)
        out['recalibrated_refit_coverage'] = metrics(recal, ytest, q=q_recal)
    return out


def update(model, optimizer, x, y, a, kind, mode, alpha, gate,
           audit_x=None, counterfactual=False):
    parameters = list(model.parameters())
    h = model(x)
    lp = F.cross_entropy(model.primary(h), y)
    la = auxiliary_loss(model, filtered_input(h, mode, gate), a, kind)
    operating_alpha = 0.0 if mode == 'primary' else alpha
    audit = None
    if audit_x is not None:
        before = [p.detach().clone() for p in parameters]
        cf_model = copy.deepcopy(model) if counterfactual else None
        cf_state = copy.deepcopy(optimizer.state_dict()) if counterfactual else None
        gp = torch.autograd.grad(lp, parameters, retain_graph=True, allow_unused=True)
        ga = (torch.autograd.grad(la, parameters, retain_graph=True, allow_unused=True)
              if la.requires_grad else tuple(None for _ in parameters))
        hq = model(audit_x)
        v_before = (hq - hq.mean(0, keepdim=True)).square().sum() / (2 * len(hq))
        vg = torch.autograd.grad(v_before, parameters, allow_unused=True)
        r = -sum(float((v * p).sum()) for v, p in zip(vg, gp) if v is not None and p is not None)
        c = sum(float((v * a0).sum()) for v, a0 in zip(vg, ga) if v is not None and a0 is not None)
        audit = dict(V_before=float(v_before.detach()), r=r, c=c,
                     gradient_flow_dV=r - operating_alpha * c,
                     dose=operating_alpha, auxiliary_detached=(mode == 'detach'))
    optimizer.zero_grad(set_to_none=True)
    total = lp if mode == 'primary' else lp + operating_alpha * la
    if not torch.isfinite(total):
        raise FloatingPointError('Non-finite training loss')
    total.backward()
    optimizer.step()
    if audit is not None:
        with torch.no_grad():
            h_after = model(audit_x)
            after = float((h_after - h_after.mean(0, keepdim=True)).square().sum() / (2 * len(h_after)))
            first_order = sum(float((v * (p - old)).sum()) for v, p, old in zip(vg, parameters, before) if v is not None)
        audit.update(V_after=after, actual_delta_V=after - audit['V_before'],
                     actual_step_first_order=first_order,
                     finite_step_remainder=after - audit['V_before'] - first_order)
        if counterfactual:
            cf_opt = torch.optim.Adam(cf_model.parameters(), lr=optimizer.param_groups[0]['lr'])
            cf_opt.load_state_dict(cf_state)
            cf_opt.zero_grad(set_to_none=True)
            F.cross_entropy(cf_model.primary(cf_model(x)), y).backward()
            cf_opt.step()
            with torch.no_grad():
                hc = cf_model(audit_x)
                vp = float((hc - hc.mean(0, keepdim=True)).square().sum() / (2 * len(hc)))
            audit.update(primary_counterfactual_V_after=vp,
                         auxiliary_counterfactual_delta_V=after - vp)
    return float(lp.detach()), float(la.detach()), audit


def make_data(args):
    data = cdm.load_domain(args.domain)
    x = torch.from_numpy(data['X_tr']).float()
    y = torch.from_numpy(data['y_tr']).long()
    fit, cal = stratified_split(y, 0.2, args.seed)
    if args.aux == 'subject':
        base = Path(os.environ['REPFIX_DATA_X']) / 'uci_har/extracted/UCI HAR Dataset'
        subjects = np.loadtxt(base / 'train/subject_train.txt', dtype=int)
        ids = sorted(set(subjects[fit.numpy()].tolist()))
        mapping = {s: i for i, s in enumerate(ids)}
        aux = torch.tensor([mapping.get(s, len(ids)) for s in subjects], dtype=torch.long)
        aux_dim = len(ids)
    elif args.aux == 'snr':
        raw = torch.from_numpy(np.asarray(data['aux_tr'])).float()
        aux = (raw - raw[fit].mean()) / raw[fit].std(unbiased=False).clamp_min(1e-8)
        aux_dim = 1
    elif args.aux == 'recon':
        aux = x.clone()
        aux_dim = data['d_in']
    else:
        aux = torch.zeros(len(x))
        aux_dim = 0
    if args.limit_train:
        shuffle = torch.randperm(len(fit), generator=torch.Generator().manual_seed(args.seed + 301))
        fit = fit[shuffle[:args.limit_train]]
    if args.limit_test:
        xtest = torch.from_numpy(data['X_te'][:args.limit_test]).float()
        ytest = torch.from_numpy(data['y_te'][:args.limit_test]).long()
    else:
        xtest = torch.from_numpy(data['X_te']).float()
        ytest = torch.from_numpy(data['y_te']).long()
    if args.aux in ('variance', 'projected'):
        model = cdm.MTLModel(data['d_in'], data['n_classes'], m=args.m,
                             k_aux=3, seed=args.seed, arch='mlp')
    else:
        kind = dict(subject='clf', snr='reg', recon='recon')[args.aux]
        model = ram.RealAuxModel(data['d_in'], data['n_classes'], args.m, kind, aux_dim)
    return model, (x[fit], y[fit], x[cal], y[cal], xtest, ytest), aux[fit], dict(
        fitting_examples=len(fit), calibration_examples=len(cal), test_examples=len(ytest),
        fitting_indices_hash=hashlib.sha256(fit.numpy().tobytes()).hexdigest(),
        calibration_indices_hash=hashlib.sha256(cal.numpy().tobytes()).hexdigest(),
        calibration_used_for_encoder=False, calibration_used_for_head=False,
        calibration_test_exchangeability_assumed=False)


def scale_sweep(head, mu, sd, hfit, htest, ytest):
    with torch.no_grad():
        w, bias = head.weight.detach(), head.bias.detach()
        intercept = bias - (w @ (mu / sd).squeeze(0))
        signal = (htest / sd) @ w.T
        boundary = affine_scale_boundaries(intercept.numpy(), signal.numpy())
        original = (intercept + signal).argmax(1)
        curve = []
        for c in [1., .8, .6, .4, .3, .2, .15, .1, .07, .05, .03, .02, .01, .005, .001]:
            logits = intercept + c * signal
            mu_c, sd_c = stats(c * hfit)
            recal = head((c * htest - mu_c) / sd_c)
            curve.append(dict(c=c, frozen=metrics(logits, ytest), recalibrated=metrics(recal, ytest),
                              changed_from_healthy=float((logits.argmax(1) != original).double().mean()),
                              departed_bias_class=float((logits.argmax(1) != boundary['bias_class']).double().mean())))
    return dict(bias_exit=boundary_summary(boundary['bias_exit']),
                shrink_first=boundary_summary(boundary['shrink_first']),
                no_positive_shrink_flip_fraction=float(np.mean(boundary['shrink_first'] == 0)), curve=curve)


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start = time.time()
    model, data, aux, splits = make_data(args)
    assert all(p.device.type == 'cpu' for p in model.parameters())
    xfit, yfit, xcal, ycal, xtest, ytest = data
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    warm_rng = torch.Generator().manual_seed(args.seed + 10000)
    for epoch in range(args.warm_epochs):
        order = torch.randperm(len(xfit), generator=warm_rng)
        for j in range(0, len(order), args.batch):
            idx = order[j:j + args.batch]
            update(model, opt, xfit[idx], yfit[idx], aux[idx], args.aux, 'detach', 1.0, False)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps(dict(stage='warm', seed=args.seed, epoch=epoch + 1,
                                  elapsed_s=round(time.time() - start, 2))), flush=True)
    model.eval()
    hfit, htest, hcal = embed(model, xfit), embed(model, xtest), embed(model, xcal)
    head, mu, sd = fit_head(hfit, yfit)
    with torch.no_grad():
        logits_cal = head((hcal - mu) / sd)
    temperature = temperature_fit(logits_cal, ycal)
    q0 = conformal_quantile(logits_cal, ycal)
    reference = (head, mu, sd, htest, temperature, q0)
    baseline = snapshot_metrics(model, data, reference)
    model_state = copy.deepcopy(model.state_dict())
    opt_state = copy.deepcopy(opt.state_dict())
    model_sha = state_hash(model_state)
    opt_sha = state_hash(opt_state)
    out = dict(protocol='paired checkpoint continuation with independent calibration',
               config=vars(args).copy(), seed=args.seed, splits=splits,
               reference_model_sha256=model_sha, reference_optimizer_sha256=opt_sha,
               torch_version=torch.__version__, python_version=platform.python_version(),
               device='cpu', torch_threads=torch.get_num_threads(),
               cuda_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               healthy=baseline, healthy_temperature=temperature,
               healthy_raw_conformal_q=q0, scale_sweep=scale_sweep(head, mu, sd, hfit, htest, ytest),
               branches=[], complete=False)
    out['config']['out'] = str(args.out.name)
    out['config'].pop('data_root', None)
    artifacts = args.out.parent / (args.out.stem + '_states')
    artifacts.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model_state, optimizer=opt_state, splits=splits), artifacts / 'reference.pt')
    audit_x = xfit[:min(256, len(xfit))]
    primary_shared_hash = None
    doses = [float(v) for v in args.doses.split(',')]
    branches = [('primary', 0.0)] + [(mode, dose) for dose in doses for mode in args.modes.split(',')]
    for mode, alpha in branches:
        model.load_state_dict(model_state)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        opt.load_state_dict(copy.deepcopy(opt_state))
        assert state_hash(model.state_dict()) == model_sha
        assert state_hash(opt.state_dict()) == opt_sha
        gate = calibrate_gate(model, audit_x, yfit[:len(audit_x)], aux[:len(audit_x)], args.aux, alpha)
        branch = dict(mode=mode, alpha=alpha, gate=gate, initial_model_sha256=model_sha,
                      initial_optimizer_sha256=opt_sha, trajectory=[dict(epoch=0, **copy.deepcopy(baseline))],
                      update_audits=[])
        future_rng = torch.Generator().manual_seed(args.seed + 20000)
        order_digest = hashlib.sha256()
        branch_start = time.time()
        for epoch in range(args.cont_epochs):
            order = torch.randperm(len(xfit), generator=future_rng)
            order_digest.update(order.numpy().tobytes())
            lp_sum = la_sum = 0.0
            for j in range(0, len(order), args.batch):
                idx = order[j:j + args.batch]
                selected_audit = j == 0 and (epoch in (0, 1, 4) or (epoch + 1) % args.audit_every == 0)
                lp, la, audit = update(model, opt, xfit[idx], yfit[idx], aux[idx], args.aux,
                                       mode, alpha, gate['gate'], audit_x=audit_x if selected_audit else None,
                                       counterfactual=selected_audit)
                lp_sum += lp * len(idx)
                la_sum += la * len(idx)
                if audit is not None:
                    branch['update_audits'].append(dict(epoch=epoch + 1, batch_start=j, **audit))
            if epoch == 0 or (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.cont_epochs:
                record = snapshot_metrics(model, data, reference, endpoint=epoch + 1 == args.cont_epochs)
                record.update(epoch=epoch + 1, train_primary_loss=lp_sum / len(xfit),
                              train_auxiliary_loss=la_sum / len(xfit))
                branch['trajectory'].append(record)
                print(json.dumps(dict(stage='continuation', seed=args.seed, mode=mode, alpha=alpha,
                                      epoch=epoch + 1, frozen=record['frozen']['acc'],
                                      recal=record['recalibrated']['acc'],
                                      relative_trace=record['geometry']['relative_trace'],
                                      elapsed_s=round(time.time() - branch_start, 2))), flush=True)
        branch['minibatch_order_sha256'] = order_digest.hexdigest()
        shared = {k: v for k, v in model.state_dict().items() if k.startswith(('encoder.', 'primary.'))}
        branch['final_shared_sha256'] = state_hash(shared)
        if mode == 'primary':
            primary_shared_hash = branch['final_shared_sha256']
        if mode == 'detach':
            branch['matches_primary_only_exactly'] = branch['final_shared_sha256'] == primary_shared_hash
            if not branch['matches_primary_only_exactly']:
                raise AssertionError('Detached branch must match primary-only encoder and head')
        branch['elapsed_s'] = time.time() - branch_start
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()),
                   artifacts / ('%s_%g.pt' % (mode, alpha)))
        out['branches'].append(branch)
        out['elapsed_s'] = time.time() - start
        save_json(args.out.with_suffix('.partial.json'), out)
    hashes = {b['minibatch_order_sha256'] for b in out['branches']}
    if len(hashes) != 1:
        raise AssertionError('All branches must use the same minibatch order')
    out.update(complete=True, elapsed_s=time.time() - start)
    try:
        import psutil
        out['final_rss_bytes'] = psutil.Process().memory_info().rss
    except ImportError:
        pass
    save_json(args.out, out)
    print(json.dumps(dict(stage='complete', output=args.out.name, elapsed_s=out['elapsed_s'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=['har', 'radioml'], default='har')
    parser.add_argument('--aux', choices=['variance', 'projected', 'subject', 'snr', 'recon'], default='variance')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warm-epochs', type=int, default=60)
    parser.add_argument('--cont-epochs', type=int, default=20)
    parser.add_argument('--m', type=int, default=64)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--doses', default='0.1,3,100')
    parser.add_argument('--modes', default='joint,detach,stfcal')
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--audit-every', type=int, default=5)
    parser.add_argument('--limit-train', type=int, default=0)
    parser.add_argument('--limit-test', type=int, default=0)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.data_root:
        os.environ['REPFIX_DATA_X'] = str(args.data_root)
    if args.aux == 'subject' and args.domain != 'har':
        parser.error('Subject auxiliary requires HAR')
    if args.aux == 'snr' and args.domain != 'radioml':
        parser.error('SNR auxiliary requires RadioML')
    if args.warm_epochs < 1 or args.cont_epochs < 1:
        parser.error('Positive training lengths are required')
    if set(args.modes.split(',')) - {'joint', 'detach', 'stfcal'}:
        parser.error('Unknown continuation mode')
    run(args)


if __name__ == '__main__':
    main()
