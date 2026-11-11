"""CPU-only RadioML release forecasts sealed after the sixth update epoch."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from consumer_contracts import confidence_thresholds
from har_early_forecast import predict
from har_transfer_cpu import base, fresh_state, legacy_logits, paired
from readout_nullspace import masked_primary_loss

RATES = (.00001, .0001, .001)


def fit_stream(state, data, seed, first, last):
    x, y, snr, mask = data
    model, optimizer = state['model'], state['optimizer']
    for epoch in range(first, last):
        generator = torch.Generator().manual_seed(seed * 1000 + epoch + state['stream'])
        for ix in torch.randperm(len(x), generator=generator).split(256):
            h = model(x[ix])
            ce = masked_primary_loss(model.primary(h), y[ix], mask[ix])
            aux_input = h if state['objective'] == 'joint' else h.detach()
            loss = ce + F.mse_loss(model.aux(aux_input).flatten(), snr[ix])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def run(args):
    confirmation = args.stage == 'confirmation'
    config = json.loads(args.forecaster.read_text()) if confirmation else None
    if confirmation:
        assert config['plan_sha256'] == args.plan_sha256
    cache = Path(paired.cdm._path('RML2016.10a_pre.pt'))
    raw = torch.load(cache, map_location='cpu', mmap=True, weights_only=True)
    generator = torch.Generator().manual_seed(args.seed + 10000)
    order = torch.randperm(len(raw['y_tr']), generator=generator)
    fit, monitor, evaluation = order[:12000], order[12000:13600], order[13600:15200]
    x = raw['X_tr'][fit].float()
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-8)
    x = (x - mean) / scale
    y, snr = raw['y_tr'][fit].long(), raw['aux_tr'][fit].float()
    snr = (snr - snr.mean()) / snr.std(unbiased=False)
    known = torch.zeros(len(y), dtype=torch.bool)
    for label in range(11):
        ids = (y == label).nonzero().flatten()
        known[ids[torch.randperm(len(ids), generator=generator)[:round(.03 * len(ids))]]] = True
    data = x, y, snr, known
    torch.manual_seed(args.seed)
    healthy = paired.ram.RealAuxModel(x.shape[1], 11, 64, 'reg', 1)
    optimizer = torch.optim.Adam(healthy.parameters(), lr=.001)
    fit_stream(dict(model=healthy, optimizer=optimizer, objective='detach', stream=100),
               data, args.seed, 0, 12)
    h = base.embeds(healthy, x[known])
    reader = base.fit_affine(h, y[known])
    readers = dict(modulation=reader)
    thresholds = dict(modulation=list(confidence_thresholds(base.late.linear(h, reader)).values()))
    states = {f'{objective}/{rate}': fresh_state(healthy, optimizer.state_dict(), objective, 1., rate, 300)
              for objective in ('detach', 'joint') for rate in RATES}
    probe = monitor if confirmation else evaluation
    x_probe = (raw['X_tr'][probe].float() - mean) / scale
    old_early = legacy_logits(healthy, x_probe, readers)
    early, forecasts = {}, {}
    for program, state in states.items():
        fit_stream(state, data, args.seed, 0, 2)
        first = legacy_logits(state['model'], x_probe, readers)
        fit_stream(state, data, args.seed, 2, 6)
        sixth = legacy_logits(state['model'], x_probe, readers)
        early[program] = first, sixth
        if confirmation:
            forecasts[program] = predict(config, program, old_early, first, sixth, thresholds)
    receipt = args.out.with_suffix('.forecast.json')
    base.save(receipt, dict(seed=args.seed, plan_sha256=args.plan_sha256, early_epoch=6,
        test_loaded=False, forecaster_sha256=base.digest(args.forecaster) if confirmation else None,
        early_sha256=paired.state_hash((old_early, early, thresholds)), forecasts=forecasts))
    seal = base.digest(receipt)
    print(json.dumps(dict(forecast_fixed=True, epoch=6)), flush=True)
    for state in states.values():
        fit_stream(state, data, args.seed, 6, 24)
    assert base.digest(receipt) == seal
    if confirmation:
        test = torch.randperm(len(raw['y_te']), generator=generator)[:4400]
        x_end, labels = (raw['X_te'][test].float() - mean) / scale, raw['y_te'][test].long()
    else:
        x_end, labels = x_probe, raw['y_tr'][evaluation].long()
    old, retained = (legacy_logits(healthy, x_end, readers) for _ in range(2))
    assert torch.equal(old['modulation'], retained['modulation'])
    terminal, primary = {}, {}
    for program, state in states.items():
        model = state['model']
        terminal[program] = legacy_logits(model, x_end, readers)
        with torch.no_grad():
            primary[program] = model.primary(base.embeds(model, x_end))
    folder = args.out.parent / f'{args.out.stem}_states'
    folder.mkdir(exist_ok=True, parents=True)
    replay = folder / 'replay.pt'
    outputs = dict(old_early=old_early, early=early, thresholds=thresholds, old=old,
                   retained=retained, terminal=terminal, primary=primary)
    torch.save(dict(outputs=outputs, labels=labels), replay)
    base.save(args.out, dict(complete=True, device='cpu', stage=args.stage, seed=args.seed,
        plan_sha256=args.plan_sha256, test_loaded=confirmation, branches=[],
        labeled_examples=int(known.sum()), forecast_sha256=seal, replay_sha256=base.digest(replay),
        source_sha256={Path(__file__).name: base.digest(Path(__file__))}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('calibration', 'confirmation'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--forecaster', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run(parser.parse_args())
