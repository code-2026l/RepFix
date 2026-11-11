"""Matched low-label HAR continuation and sealed early consumer forecasts."""

import argparse
import copy
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
from torch.nn import functional as F

import contrastive_benefit_cpu as contrastive
import late_vision_cpu as vision
import paired_readout_continuation as paired
import vision_auxiliary_cpu as base
from consumer_contracts import confidence_thresholds
from consumer_routing import decisions
from readout_nullspace import masked_primary_loss

SCREEN = (('control', 0.), ('reconstruction', .03), ('reconstruction', .3),
          ('reconstruction', 3.), ('contrastive', .03), ('contrastive', .1),
          ('contrastive', .3), ('neighborhood', .1), ('neighborhood', 1.))
RATES = (.0001, .001)


def load_split(name):
    root = Path(paired.cdm._path('uci_har', 'extracted', 'UCI HAR Dataset')) / name
    x = torch.from_numpy(np.loadtxt(root / f'X_{name}.txt', dtype=np.float32))
    y = torch.from_numpy(np.loadtxt(root / f'y_{name}.txt', dtype=np.int64)) - 1
    subjects = torch.from_numpy(np.loadtxt(root / f'subject_{name}.txt', dtype=np.int64))
    return x, y, subjects


def partitions(seed, phase):
    x, y, subject = load_split('train')
    rng = torch.Generator().manual_seed(seed + 10000)
    unique = torch.unique(subject)
    order = unique[torch.randperm(len(unique), generator=rng)]
    n_eval = 0 if phase == 'confirmation' else 4
    n_monitor = 4 if phase == 'confirmation' else 3
    evaluation = torch.isin(subject, order[:n_eval])
    monitor = torch.isin(subject, order[n_eval:n_eval + n_monitor])
    fit = ~(evaluation | monitor)
    assert not (set(subject[fit].tolist()) & set(subject[monitor].tolist()))
    mean = x[fit].mean(0)
    xx, yy = x[fit] - mean, y[fit]
    mask = torch.zeros(len(yy), dtype=torch.bool)
    for label in range(6):
        indices = (yy == label).nonzero().flatten()
        sample = indices[torch.randperm(len(indices), generator=rng)[:max(1, round(.03 * len(indices)))]]
        mask[sample] = True
    data = dict(x=xx, y=yy, mask=mask, mean=mean, monitor=x[monitor] - mean,
                evaluation=x[evaluation] - mean, evaluation_labels=y[evaluation],
                fit_subjects=sorted(set(subject[fit].tolist())),
                monitor_subjects=sorted(set(subject[monitor].tolist())),
                evaluation_subjects=sorted(set(subject[evaluation].tolist())))
    return data


def masked(x, rng):
    return x.masked_fill(torch.rand(x.shape, generator=rng) < .25, 0.)


def neighborhood(h0, h1, x):
    clean = F.normalize(x)
    target = (clean @ clean.T / .05).softmax(1)
    predicted = (F.normalize(h0) @ F.normalize(h1).T / .2).log_softmax(1)
    return -(target * predicted).sum(1).mean()


def auxiliary_losses(model, h0, h1, x, variance, reconstruction_dose=0.):
    # The reconstruction decoder sees the same gradient convention in every arm.
    z = h1.detach() + reconstruction_dose * (h1 - h1.detach())
    recon = F.mse_loss(model.aux(z), x) / variance
    return dict(reconstruction=recon,
                contrastive=contrastive.contrastive(h0, h1, h0.new_zeros(h0.shape[1]),
                                                    h0.new_ones(h0.shape[1])),
                neighborhood=neighborhood(h0, h1, x))


def train(state, data, seed, start, stop):
    model, optimizer = state['model'], state['optimizer']
    x, y, mask = data['x'], data['y'], data['mask']
    objective, dose = state['objective'], state['dose']
    variance = x.var(unbiased=False).clamp_min(1e-6)
    for epoch in range(start, stop):
        rng = torch.Generator().manual_seed(seed * 10000 + epoch + state['stream'])
        for indices in torch.randperm(len(x), generator=rng).split(256):
            clean = x[indices]
            h0, h1 = model(clean), model(masked(clean, rng))
            ce = (masked_primary_loss(model.primary(h0), y[indices], mask[indices]) +
                  masked_primary_loss(model.primary(h1), y[indices], mask[indices])) / 2
            dose_recon = dose if objective == 'reconstruction' else 0.
            z = h1.detach() + dose_recon * (h1 - h1.detach())
            loss = ce + F.mse_loss(model.aux(z), clean) / variance
            if objective == 'contrastive':
                loss += dose * contrastive.contrastive(h0, h1, h0.new_zeros(h0.shape[1]),
                                                       h0.new_ones(h0.shape[1]))
            if objective == 'neighborhood':
                loss += dose * neighborhood(h0, h1, clean)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def fresh_state(model, moments, objective, dose, rate, stream):
    cloned = copy.deepcopy(model)
    optimizer = torch.optim.Adam(cloned.parameters(), lr=rate)
    optimizer.load_state_dict(copy.deepcopy(moments))
    optimizer.param_groups[0]['lr'] = rate
    return dict(model=cloned, optimizer=optimizer, objective=objective, dose=dose, rate=rate, stream=stream)


@torch.no_grad()
def score(model, x, labels, seed, variance):
    rng = torch.Generator().manual_seed(seed + 92000)
    total = dict.fromkeys(('reconstruction', 'contrastive', 'neighborhood'), 0.)
    outputs = []
    for clean in x.split(256):
        h0, h1 = model(clean), model(masked(clean, rng))
        outputs.append(model.primary(h0))
        for key, value in auxiliary_losses(model, h0, h1, clean, variance).items():
            total[key] += float(value) * len(clean)
    logits = torch.cat(outputs)
    return logits, float((logits.argmax(1) == labels).double().mean()), {
        name: value / len(x) for name, value in total.items()}


@torch.no_grad()
def legacy_logits(model, x, readers):
    return {name: vision.linear(base.embeds(model, x), reader) for name, reader in readers.items()}


def model_and_readers(data, seed):
    torch.manual_seed(seed)
    model = paired.ram.RealAuxModel(data['x'].shape[1], 6, 64, 'recon', data['x'].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    train(dict(model=model, optimizer=optimizer, objective='control', dose=0., stream=20),
          data, seed, 0, 8)
    features = base.embeds(model, data['x'][data['mask']])
    labels = data['y'][data['mask']]
    readers = dict(activity=base.fit_affine(features, labels),
                   movement=base.fit_affine(features, (labels < 3).long()))
    thresholds = {name: list(confidence_thresholds(vision.linear(features, reader)).values())
                  for name, reader in readers.items()}
    return model, optimizer.state_dict(), readers, thresholds


def program_key(objective, rate):
    return f'{objective}/{rate}'


def run(args):
    seed = args.seed
    selection = json.loads(args.choice.read_text()) if args.choice else None
    if args.stage != 'screen':
        assert (selection['passed'] or selection['fallback']) and selection['plan_sha256'] == args.plan_sha256
        chosen = selection['selected']
    if args.stage == 'confirmation':
        from har_early_forecast import predict
        predictor = json.loads(args.forecaster.read_text())
        assert predictor['choice_sha256'] == base.digest(args.choice)
        assert predictor['plan_sha256'] == args.plan_sha256
    data = partitions(seed, args.stage)
    healthy, moments, readers, thresholds = model_and_readers(data, seed)
    variance = data['x'].var(unbiased=False).clamp_min(1e-6)
    outputs, summaries, sealed = {}, [], None
    if args.stage == 'screen':
        x, labels = data['evaluation'], data['evaluation_labels']
        for objective, dose in SCREEN:
            state = fresh_state(healthy, moments, objective, dose, .001, 300)
            train(state, data, seed, 0, 24)
            logits, accuracy, losses = score(state['model'], x, labels, seed, variance)
            outputs[f'{objective}/{dose}'] = logits
            summaries.append(dict(objective=objective, dose=dose, accuracy=accuracy, auxiliary=losses))
    else:
        arms = [fresh_state(healthy, moments, objective, dose, rate, 300)
                for rate in RATES for objective, dose in (('control', 0.), (chosen['objective'], chosen['dose']))]
        x = data['monitor'] if args.stage == 'confirmation' else data['evaluation']
        old_early = legacy_logits(healthy, x, readers)
        early_data, forecasts = {}, {}
        for state in arms:
            key = program_key(state['objective'], state['rate'])
            train(state, data, seed, 0, 1)
            first = legacy_logits(state['model'], x, readers)
            train(state, data, seed, 1, 3)
            third = legacy_logits(state['model'], x, readers)
            early_data[key] = (first, third)
            if args.stage == 'confirmation':
                forecasts[key] = predict(predictor, key, old_early, first, third, thresholds)
        receipt = args.out.with_suffix('.forecast.json')
        base.save(receipt, dict(seed=seed, plan_sha256=args.plan_sha256, early_epoch=3,
            test_loaded=False, choice_sha256=base.digest(args.choice),
            forecaster_sha256=base.digest(args.forecaster) if args.forecaster else None,
            early_sha256=paired.state_hash((old_early, early_data, thresholds)), forecasts=forecasts))
        sealed = base.digest(receipt)
        print(json.dumps(dict(forecast_fixed=True, epoch=3)), flush=True)
        for state in arms:
            train(state, data, seed, 3, 24)
        assert base.digest(receipt) == sealed
        if args.stage == 'confirmation':
            x, labels, subject = load_split('test')
            assert not set(subject.tolist()) & set(data['fit_subjects'] + data['monitor_subjects'])
            data['evaluation_subjects'] = sorted(set(subject.tolist()))
            x = x - data['mean']
        else:
            x, labels = data['evaluation'], data['evaluation_labels']
        old = legacy_logits(healthy, x, readers)
        retained = legacy_logits(healthy, x, readers)
        assert all(torch.equal(old[name], retained[name]) for name in old)
        terminal = {}
        for state in arms:
            key = program_key(state['objective'], state['rate'])
            logits, accuracy, losses = score(state['model'], x, labels, seed, variance)
            outputs[key] = logits
            terminal[key] = legacy_logits(state['model'], x, readers)
            summaries.append(dict(key=key, objective=state['objective'], dose=state['dose'],
                                  rate=state['rate'], accuracy=accuracy, auxiliary=losses))
        outputs = dict(primary=outputs, old=old, retained=retained, terminal=terminal,
                       old_early=old_early, early=early_data, thresholds=thresholds)
    folder = args.out.parent / f'{args.out.stem}_states'
    folder.mkdir(exist_ok=True, parents=True)
    replay = folder / 'replay.pt'
    torch.save(dict(labels=labels, outputs=outputs), replay)
    base.save(args.out, dict(complete=True, device='cpu', stage=args.stage, seed=seed,
        plan_sha256=args.plan_sha256, branches=[], test_loaded=args.stage == 'confirmation',
        source_sha256={Path(__file__).name: base.digest(Path(__file__))},
        fit_subjects=data['fit_subjects'], monitor_subjects=data['monitor_subjects'],
        evaluation_subjects=data['evaluation_subjects'], fitting_examples=len(data['x']),
        labeled_examples=int(data['mask'].sum()), evaluation_examples=len(labels),
        forecast_sha256=sealed, replay_sha256=base.digest(replay), outcomes=summaries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('screen', 'calibration', 'confirmation'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--choice', type=Path)
    parser.add_argument('--forecaster', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run(parser.parse_args())
