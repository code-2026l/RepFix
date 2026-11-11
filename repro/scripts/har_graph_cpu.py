"""Paired HAR continuation with fixed graph and warm-model teachers."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from har_transfer_cpu import base, fresh_state, legacy_logits, load_split, masked, model_and_readers, partitions
from readout_nullspace import masked_primary_loss

DOSES = (.1, .3, 1.)


def graph_targets(x, labels, known):
    x = F.normalize(x)
    weights, neighbors = [], []
    for start in range(0, len(x), 256):
        similarity = x[start:start + 256] @ x.T
        row = torch.arange(len(similarity))
        similarity[row, start + row] = -torch.inf
        values, columns = similarity.topk(30, dim=1)
        weights.append(((values - 1) / .08).exp())
        neighbors.append(columns)
    graph = sparse.csr_matrix((torch.cat(weights).flatten().numpy(),
                               (np.repeat(np.arange(len(x)), 30), torch.cat(neighbors).flatten().numpy())),
                              shape=(len(x), len(x)))
    graph = graph.maximum(graph.T)
    graph = graph.multiply(1. / np.asarray(graph.sum(1))).tocsr()
    labeled = F.one_hot(labels[known], 6).float().numpy()
    targets = np.broadcast_to(labeled.mean(0), (len(x), 6)).copy()
    targets[known.numpy()] = labeled
    for _ in range(100):
        targets = graph @ targets
        targets[known.numpy()] = labeled
    return torch.from_numpy(targets)


@torch.no_grad()
def primary(model, x):
    return torch.cat([model.primary(model(chunk)) for chunk in x.split(256)])


def continuation(state, data, seed, targets):
    x, y, known = (data[key] for key in ('x', 'y', 'mask'))
    model, optimizer = state['model'], state['optimizer']
    variance = x.var(unbiased=False).clamp_min(1e-6)
    for epoch in range(24):
        rng = torch.Generator().manual_seed(seed * 10000 + epoch + 300)
        for ids in torch.randperm(len(x), generator=rng).split(256):
            clean = x[ids]
            h0, h1 = model(clean), model(masked(clean, rng))
            z0, z1 = model.primary(h0), model.primary(h1)
            ce = (masked_primary_loss(z0, y[ids], known[ids]) +
                  masked_primary_loss(z1, y[ids], known[ids])) / 2
            loss = ce + F.mse_loss(model.aux(h1.detach()), clean) / variance
            if targets is not None:
                unlabeled = ~known[ids]
                soft_ce = -(targets[ids] * z1.log_softmax(1)).sum(1)
                loss += state['dose'] * (unlabeled * soft_ce).sum() / unlabeled.sum().clamp_min(1)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def run(args):
    confirm = args.stage == 'confirmation'
    choice = json.loads(args.choice.read_text()) if confirm else None
    if confirm:
        assert choice['passed'] and choice['plan_sha256'] == args.plan_sha256
    data = partitions(args.seed, args.stage)
    healthy, moments, readers, _ = model_and_readers(data, args.seed)
    start = time.perf_counter()
    graph = graph_targets(data['x'], data['y'], data['mask'])
    graph_seconds = time.perf_counter() - start
    teachers = dict(control=None, warm=primary(healthy, data['x']).softmax(1), graph=graph)
    doses = (choice['dose'],) if confirm else DOSES
    recipes = [('control', 0.)] + [(name, dose) for name in ('warm', 'graph') for dose in doses]
    models = {}
    for name, dose in recipes:
        state = fresh_state(healthy, moments, name, dose, .001, 300)
        models[f'{name}/{dose}'] = continuation(state, data, args.seed, teachers[name])
    x, labels, subjects = data['evaluation'], data['evaluation_labels'], None
    if confirm:
        x, labels, subjects = load_split('test')
        assert not set(subjects.tolist()) & set(data['fit_subjects'] + data['monitor_subjects'])
        data['evaluation_subjects'] = sorted(set(subjects.tolist()))
        x = x - data['mean']
    outputs = {name: primary(model, x) for name, model in models.items()}
    scores = {name: float((logits.argmax(1) == labels).double().mean()) for name, logits in outputs.items()}
    old, retained = (legacy_logits(healthy, x, readers) for _ in range(2))
    assert all(torch.equal(old[name], retained[name]) for name in old)
    folder = args.out.parent / f'{args.out.stem}_states'
    folder.mkdir(exist_ok=True, parents=True)
    replay = folder / 'replay.pt'
    torch.save(dict(labels=labels, subjects=subjects, outputs=outputs, old=old, retained=retained), replay)
    base.save(args.out, dict(complete=True, device='cpu', stage=args.stage, seed=args.seed,
        plan_sha256=args.plan_sha256, choice_sha256=base.digest(args.choice) if confirm else None,
        test_loaded=confirm, branches=[], fitting_examples=len(data['x']),
        labeled_examples=int(data['mask'].sum()), evaluation_examples=len(labels),
        fit_subjects=data['fit_subjects'], monitor_subjects=data['monitor_subjects'],
        evaluation_subjects=data['evaluation_subjects'], graph_setup_seconds=graph_seconds,
        scores=scores, replay_sha256=base.digest(replay),
        source_sha256={Path(__file__).name: base.digest(Path(__file__))}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('screen', 'confirmation'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--choice', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run(parser.parse_args())
