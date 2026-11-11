"""Label-free forecasts of terminal changes in affine readout decisions."""

import numpy as np
import torch
from torch.nn import functional as F

from consumer_routing import decisions

SCALES = (0., .25, .5, .75, 1.)
BASELINES = ('schedule_only', 'persistence')


def winner_odds(z, winner):
    other = z.scatter(1, winner[:, None], -torch.inf)
    chosen = z.gather(1, winner[:, None]).squeeze(1)
    return chosen - other.max(1).values, chosen - other.logsumexp(1)


def signed_margins(z, reference, thresholds):
    z = z.double()
    class_margin, old_odds = winner_odds(z, reference[:, 0])
    _, current_odds = winner_odds(z, z.argmax(1))
    cutoffs = z.new_tensor(thresholds).logit()
    accepted = torch.minimum(class_margin[:, None], old_odds[:, None] - cutoffs)
    gated = torch.where(reference[:, 1:] >= 0, accepted, cutoffs - current_odds[:, None])
    return torch.cat((class_margin[:, None], gated), dim=1)


def features(old, first, third, thresholds):
    reference = decisions(old, thresholds)
    m0, m1, m3 = (signed_margins(z, reference, thresholds) for z in (old, first, third))
    d1 = decisions(first, thresholds) != reference
    d3 = decisions(third, thresholds) != reference
    return torch.stack((m0, m1, m3, m3 - m0, m3 - m1, d1, d3), dim=-1)


def cell(program, head, rule):
    return f'{program}|{head}|{rule}'


def examples(outputs):
    for program, (first, third) in outputs['early'].items():
        for head, old in outputs['old_early'].items():
            thresholds = outputs['thresholds'][head]
            x = features(old, first[head], third[head], thresholds)
            y = decisions(old, thresholds) != decisions(outputs['terminal'][program][head], thresholds)
            yield program, head, x, y


def fit(rows):
    by_cell = {}
    for _, outputs, _ in rows:
        for program, head, x, y in examples(outputs):
            for rule in range(x.shape[1]):
                by_cell.setdefault(cell(program, head, rule), []).append((x[:, rule], y[:, rule]))
    groups = [(torch.cat([x for x, _ in pairs]), torch.cat([y for _, y in pairs]).double())
              for pairs in by_cell.values()]
    x = torch.cat([x for x, _ in groups])
    y = torch.cat([y for _, y in groups])
    mean, sd = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-8)
    x = (x - mean) / sd
    indices = torch.repeat_interleave(torch.arange(len(groups)), torch.tensor([len(y) for _, y in groups]))
    prior = torch.stack([torch.stack([y.double().mean() for _, y in pairs]).mean()
                         for pairs in by_cell.values()])
    bias = prior.clamp(.001, .999).logit().requires_grad_()
    weights = torch.zeros(x.shape[1], dtype=torch.double, requires_grad=True)
    optimizer = torch.optim.LBFGS([bias, weights], max_iter=50, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ weights + bias[indices], y) + .01 * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        fitted = (x @ weights + bias[indices]).sigmoid()
        centers = []
        for pairs, predictions in zip(by_cell.values(), fitted.split([len(y) for _, y in groups])):
            centers.append(torch.stack([p.mean() for p in predictions.split([len(y) for _, y in pairs])]).mean().item())
    return dict(mean=mean.tolist(), sd=sd.tolist(), weights=weights.tolist(),
                intercept=dict(zip(by_cell, bias.tolist())), prior=dict(zip(by_cell, prior.tolist())),
                center=dict(zip(by_cell, centers)))


def scores(model, program, head, x):
    keys = [cell(program, head, rule) for rule in range(x.shape[1])]
    xstd = (x - x.new_tensor(model['mean'])) / x.new_tensor(model['sd'])
    bias, prior, center = (x.new_tensor([model[field][key] for key in keys])
                           for field in ('intercept', 'prior', 'center'))
    prediction = (xstd @ x.new_tensor(model['weights']) + bias).sigmoid().mean(0)
    return prior, prediction - center


def predict(config, program, old, first, third, thresholds):
    result = {method: {} for method in ('forecast', 'unshrunk', *BASELINES)}
    for head, logits in old.items():
        x = features(logits, first[head], third[head], thresholds[head])
        prior, delta = scores(config['model'], program, head, x)
        result['forecast'][head] = (prior + config['scale'] * delta).clamp(0, 1).tolist()
        result['unshrunk'][head] = (prior + delta).clamp(0, 1).tolist()
        result['schedule_only'][head] = prior.tolist()
        result['persistence'][head] = x[..., -1].mean(0).tolist()
    return result


def calibrate(rows):
    from analyze_har_transfer import ADDENDUM, CHOICE, PLAN, digest

    tune = fit(rows[:8])
    errors = {scale: [] for scale in SCALES}
    for _, outputs, _ in rows[8:]:
        for program, head, x, y in examples(outputs):
            prior, delta = scores(tune, program, head, x)
            target = y.double().mean(0).max()
            for scale in SCALES:
                errors[scale].append(float(abs((prior + scale * delta).clamp(0, 1).max() - target)))
    means = {str(scale): float(np.mean(errors[scale])) for scale in SCALES}
    scale = min(SCALES, key=lambda s: means[str(s)])
    return dict(scale=scale, tuning_mae=means, model=fit(rows), plan_sha256=digest(PLAN),
                addendum_sha256=digest(ADDENDUM), choice_sha256=digest(CHOICE))


def confirm(rows):
    import json
    from analyze_har_transfer import CHOICE, FORECASTER, digest, interval

    choice = json.loads(CHOICE.read_text())
    objective = choice['selected']['objective']
    config = json.loads(FORECASTER.read_text())
    gain, loss, references = {rate: [] for rate in (.0001, .001)}, {rate: [] for rate in (.0001, .001)}, []
    for row, outputs, receipt in rows:
        by_key = {(a['rate'], a['objective']): a for a in row['outcomes']}
        for rate in gain:
            control, treated = (by_key[rate, name] for name in ('control', objective))
            gain[rate].append(100 * (treated['accuracy'] - control['accuracy']))
            loss[rate].append(treated['auxiliary'][objective] / control['auxiliary'][objective])
        episodes = []
        for program, (first, third) in outputs['early'].items():
            forecast = predict(config, program, outputs['old_early'], first, third, outputs['thresholds'])
            for head, old in outputs['old'].items():
                thresholds = outputs['thresholds'][head]
                actual = (decisions(outputs['terminal'][program][head], thresholds) !=
                          decisions(old, thresholds)).double().mean(0).tolist()
                for method, values in forecast.items():
                    np.testing.assert_allclose(values[head], receipt['forecasts'][program][method][head],
                                               atol=1e-9, rtol=0)
                episodes.append(dict(program=program, head=head, actual=actual,
                                     **{m: values[head] for m, values in forecast.items()}))
        references.append(dict(seed=row['seed'], labeled=row['labeled_examples'],
                               tested=row['evaluation_examples'], episodes=episodes))
    methods = ('forecast', 'unshrunk', *BASELINES)
    mae = {method: np.array([np.mean([abs(max(e[method]) - max(e['actual'])) for e in r['episodes']])
                            for r in references]) * 100 for method in methods}
    difference = {method: interval(mae['forecast'] - mae[method]) for method in BASELINES}
    alarms = {method: dict(missed=0, false=0, flagged=0) for method in methods}
    exceedances = sum(max(e['actual']) > .05 for r in references for e in r['episodes'])
    for ref in references:
        for e in ref['episodes']:
            for method in methods:
                flags, exceeds = max(e[method]) > .05, max(e['actual']) > .05
                alarms[method]['missed'] += int(exceeds and not flags)
                alarms[method]['false'] += int(flags and not exceeds)
                alarms[method]['flagged'] += int(flags)
    accuracy = {str(rate): interval(values) for rate, values in gain.items()}
    auxiliary = {str(rate): interval(values) for rate, values in loss.items()}
    return dict(choice_sha256=digest(CHOICE), forecaster_sha256=digest(FORECASTER),
                calibration_scale=config['scale'], screen_passed=choice['passed'],
                utility_passed=accuracy['0.001']['ci95'][0] > 0 and auxiliary['0.001']['ci95'][1] < 1,
                forecast_passed=all(d['ci95'][1] < 0 for d in difference.values()),
                accuracy_pp=accuracy, auxiliary_loss_ratio=auxiliary,
                exact_old_readers=2 * len(references), exact_old_decision_rules=6 * len(references),
                mae_pp={method: interval(values) for method, values in mae.items()},
                paired_mae_difference_pp=difference, alarms=alarms, exceedances=exceedances,
                references=references)
