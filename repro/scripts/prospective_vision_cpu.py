"""Early forecasts and paired legacy-output interventions for new CPU updates."""

import argparse
import copy
import hashlib
import json
import os
import tarfile
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

import contrastive_benefit_cpu as contrastive
import late_vision_cpu as vision
import paired_readout_continuation as paired
import vision_auxiliary_cpu as base
from conditional_reference_cpu import ridge
from consumer_contracts import confidence_thresholds
from consumer_routing import decisions
from nonlinear_suffix import build_full
from shared_reference_vision import NormalizedHead

ARCHIVE_MD5 = 'c32a1d4ab5d03f1284b67883e8d87530'
ARCHIVE_SHA256 = 'c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd'
MULTIPLIERS = (1., 1.5, 2., 3., 4.)
RATES = {'mild': 3e-6, 'nominal': 3e-5, 'strong': 3e-4}
STABLE_RATE = 3e-7
FAMILIES = ('direct', 'moments', 'ridge', 'reference')
EVALUATION_BUDGET = .05
ACTION_BUDGET = .04


def load_training_archive(path):
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'md5').hexdigest() == ARCHIVE_MD5
        stream.seek(0)
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == ARCHIVE_SHA256
    rows = []
    with tarfile.open(path, 'r:gz') as archive:
        for index in range(1, 6):
            member = archive.getmember(f'cifar-10-batches-bin/data_batch_{index}.bin')
            assert member.isfile() and member.size == 10000 * 3073
            with archive.extractfile(member) as stream:
                rows.append(np.frombuffer(stream.read(), dtype=np.uint8).reshape(10000, 3073))
    population = np.concatenate(rows)
    labels = torch.from_numpy(population[:, 0].astype(np.int64))
    images = population[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    assert torch.bincount(labels).tolist() == [5000] * 10
    return torch.from_numpy(images), labels


@torch.no_grad()
def prepare(args):
    images, labels = load_training_archive(args.archive)
    augmented, views = base.augment(images)
    prefix, healthy = build_full('resnet18')
    features = []
    for view, raw in enumerate((images, augmented)):
        blocks = []
        for start in range(0, len(raw), 128):
            blocks.append(prefix(vision.preprocess(raw[start:start + 128], 64)).contiguous())
            if start % 8192 == 0:
                print(json.dumps(dict(view=view, prepared=start)), flush=True)
        features.append(torch.cat(blocks))
    folder = args.out.parent / f'{args.out.stem}_states'
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / 'population.pt'
    torch.save(dict(x=torch.stack(features), y=labels, healthy=healthy.state_dict(),
                    dataset_md5=ARCHIVE_MD5, augmentation=views), path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    base.save(args.out, dict(complete=True, phase='prepare', device='cpu',
        population_sha256=digest, archive_md5=ARCHIVE_MD5, archive_sha256=ARCHIVE_SHA256, augmentation=views,
        source_sha256={Path(__file__).name: base.digest(Path(__file__))},
        plan_sha256=args.plan_sha256, official_test_loaded=False, branches=[]))


def split(reference_seed):
    population = torch.randperm(50000, generator=torch.Generator().manual_seed(7400))
    dev_fit, dev_policy, dev_eval, con_fit, con_policy, con_eval = population.split(
        [8000, 2000, 4000, 14000, 6000, 16000])
    if 7410 <= reference_seed <= 7411:
        fit, policy, evaluation, offset = dev_fit, dev_policy, dev_eval, reference_seed - 7410
    else:
        assert 7420 <= reference_seed <= 7425
        fit, policy, evaluation, offset = con_fit, con_policy, con_eval, reference_seed - 7420
    rng = torch.Generator().manual_seed(reference_seed)
    fitting = fit[torch.randperm(len(fit), generator=rng)[:4096]]
    order = policy[torch.randperm(len(policy), generator=rng)[:1536]]
    mask = torch.rand(len(fitting), generator=torch.Generator().manual_seed(reference_seed + 10000)) < .2
    return fitting, mask, order[:1024], order[1024:], evaluation[2000*offset:2000*(offset+1)]


def disagreement(logits, legacy, thresholds):
    changed = decisions(logits, thresholds) != decisions(legacy, thresholds)
    return changed.double().mean(0).tolist()


def choose(rates):
    return next(family for family in FAMILIES
                if family == 'reference' or max(rates[family]) <= ACTION_BUDGET)


@torch.no_grad()
def candidate_logits(old_anchors, new_anchors, features, legacy):
    old_mu, new_mu = old_anchors.mean(0), new_anchors.mean(0)
    old_sd, new_sd = old_anchors.std(0), new_anchors.std(0)
    ratio = old_sd / new_sd.clamp_min(1e-6)
    direct = vision.linear(features, legacy)
    moments = vision.linear((features - new_mu) * ratio + old_mu, legacy)
    residual = vision.linear(old_anchors, legacy) - vision.linear(new_anchors, legacy)
    residual -= residual.mean(1, keepdim=True)
    correction, _, _ = ridge(new_anchors, residual, .001)
    return dict(direct=direct, moments=moments, ridge=direct + vision.linear(features, correction))


@torch.no_grad()
def forecast(model, anchor_tokens, monitor_tokens, old_anchors, old_monitor, legacy, thresholds):
    early_anchors = base.embeds(model, anchor_tokens)
    early_monitor = base.embeds(model, monitor_tokens)
    target = vision.linear(old_monitor, legacy)
    arrays = {}
    for k in MULTIPLIERS:
        anchor_prediction = (old_anchors + k * (early_anchors - old_anchors)).clamp_min(0)
        monitor_prediction = (old_monitor + k * (early_monitor - old_monitor)).clamp_min(0)
        arrays[str(k)] = candidate_logits(old_anchors, anchor_prediction, monitor_prediction, legacy)
    risks = {key: {family: disagreement(logits, target, thresholds) for family, logits in rows.items()}
             for key, rows in arrays.items()}
    return target, arrays, risks


def train_epochs(model, head, optimizer, xx, yy, mask, alpha, rng, views, start, stop):
    trace = []
    for epoch in range(start, stop):
        for indices in torch.randperm(len(yy), generator=rng).split(128):
            view = torch.randint(2, (len(indices),), generator=views)
            features = model(xx[view, indices])
            loss = base.primary_loss(head, features, yy[indices], mask[indices])
            if alpha:
                counterpart = model(xx[1 - view, indices])
                loss = loss + alpha * contrastive.contrastive(features, counterpart, head.mean, head.sd)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        trace.append(epoch + 1)
        print(json.dumps(dict(epoch=epoch + 1)), flush=True)
    return trace


@torch.no_grad()
def current_output(model, fitting_views, labels, mask, evaluation):
    features = base.embeds(model, fitting_views[:, mask].flatten(0, 1))
    reader = base.fit_affine(features, labels[mask].repeat(2))
    return vision.linear(base.embeds(model, evaluation), reader)


def run(args):
    assert args.alpha in (0., .1) and (args.alpha or args.rate == 'nominal')
    assert (args.stage == 'development') == (args.reference_seed in (7410, 7411))
    assert (args.stage == 'stability') == (args.rate == 'stable')
    choice = json.loads(args.choice.read_text()) if args.choice else None
    if args.stage == 'confirmation':
        assert choice and choice['plan_sha256'] == args.plan_sha256
        assert choice['multiplier'] in MULTIPLIERS
    cache = torch.load(args.population, map_location='cpu', weights_only=True, mmap=True)
    assert cache['dataset_md5'] == ARCHIVE_MD5
    fitting, mask, anchors, monitor, validation = split(args.reference_seed)
    xx, yy = cache['x'][:, fitting], cache['y'][fitting]
    anchor_tokens, monitor_tokens = cache['x'][0, anchors], cache['x'][0, monitor]
    _, healthy = build_full('resnet18')
    healthy.load_state_dict(cache['healthy'])
    healthy.requires_grad_(False)
    torch.manual_seed(args.reference_seed)
    initial = base.embeds(healthy, xx[:, mask].flatten(0, 1))
    first, _ = initial.split(int(mask.sum()))
    legacy = base.fit_affine(first, yy[mask])
    thresholds = list(confidence_thresholds(vision.linear(first, legacy)).values())
    mean, sd = initial.mean(0), initial.std(0, unbiased=False)
    sd = sd.clamp_min(.1 * sd[sd > 0].median())
    head = NormalizedHead(legacy['weight'] * sd, legacy['bias'] + legacy['weight'] @ mean, mean, sd)
    model = copy.deepcopy(healthy).requires_grad_(True)
    rate = STABLE_RATE if args.rate == 'stable' else RATES[args.rate]
    optimizer = torch.optim.AdamW([dict(params=model.parameters(), lr=rate),
                                  dict(params=head.parameters(), lr=1e-4)], weight_decay=1e-4)
    seed = args.reference_seed + 1000
    rng = torch.Generator().manual_seed(seed + 20000)
    views = torch.Generator().manual_seed(seed + 71000)
    trace = train_epochs(model, head, optimizer, xx, yy, mask, args.alpha, rng, views, 0, 2)
    early_model = copy.deepcopy(model).requires_grad_(False)
    old_anchors = base.embeds(healthy, anchor_tokens)
    old_monitor = base.embeds(healthy, monitor_tokens)
    target, forecasts, risks = forecast(model, anchor_tokens, monitor_tokens, old_anchors,
                                        old_monitor, legacy, thresholds)
    frozen = dict(reference_seed=args.reference_seed, stage=args.stage, alpha=args.alpha, rate=args.rate,
                  plan_sha256=args.plan_sha256, choice_sha256=base.digest(args.choice) if choice else None,
                  early_epoch=2, evaluated_examples=0, risks=risks,
                  actions={key: choose(rows) for key, rows in risks.items()},
                  monitor_logits_sha256=paired.state_hash((target, forecasts, thresholds)),
                  early_state_sha256=paired.state_hash(early_model.state_dict()))
    if choice:
        frozen['multiplier'] = choice['multiplier']
        frozen['schedule_only_risks'] = choice['schedule_only_risks'][args.rate]
        frozen['schedule_only_action'] = choose(frozen['schedule_only_risks'])
    receipt = args.out.with_suffix('.forecast.json')
    base.save(receipt, frozen)
    receipt_sha = base.digest(receipt)
    print(json.dumps(dict(forecast_fixed=True, stage=args.stage)), flush=True)
    trace += train_epochs(model, head, optimizer, xx, yy, mask, args.alpha, rng, views, 2, 8)
    assert base.digest(receipt) == receipt_sha

    with torch.no_grad():
        # The evaluation indices are first used after the forecast and the continuation.
        evaluation = cache['x'][:, validation]
        labels = cache['y'][validation]
        old = vision.linear(base.embeds(healthy, evaluation[0]), legacy)
        new = base.embeds(model, evaluation[0])
        candidates = candidate_logits(old_anchors, base.embeds(model, anchor_tokens), new, legacy)
        candidates['reference'] = vision.linear(base.embeds(healthy, evaluation[0]), legacy)
        outcomes = {family: disagreement(logits, old, thresholds) for family, logits in candidates.items()}
        torch.manual_seed(args.reference_seed + 40000)
        current = current_output(model, xx, yy, mask, evaluation[0])
        torch.manual_seed(args.reference_seed + 40000)
        stop = current_output(early_model, xx, yy, mask, evaluation[0])
        auxiliary = contrastive.evaluate_aux(model, evaluation, head)
    folder = args.out.parent / f'{args.out.stem}_states'
    folder.mkdir(exist_ok=True, parents=True)
    replay = folder / 'replay.pt'
    torch.save(dict(forecasts=forecasts, monitor_old=target, outcomes=candidates,
                    old=old, current=current, stop=stop, labels=labels,
                    validation_indices=validation, thresholds=thresholds), replay)
    base.save(args.out, dict(complete=True, device='cpu', official_test_loaded=False,
        stage=args.stage, rate=args.rate, alpha=args.alpha, reference_seed=args.reference_seed,
        plan_sha256=args.plan_sha256, forecast_sha256=receipt_sha, replay_sha256=base.digest(replay),
        source_sha256={Path(__file__).name: base.digest(Path(__file__))},
        epochs=trace, validation_examples=len(labels), monitor_examples=len(monitor),
        outcomes=outcomes, current_accuracy=float((current.argmax(1) == labels).double().mean()),
        stop_accuracy=float((stop.argmax(1) == labels).double().mean()),
        auxiliary_loss=auxiliary, branches=[]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'development', 'confirmation', 'stability'), required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--population', type=Path)
    parser.add_argument('--reference-seed', type=int)
    parser.add_argument('--rate', choices=(*RATES, 'stable'))
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--choice', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    if args.stage == 'prepare':
        prepare(args)
    else:
        run(args)
