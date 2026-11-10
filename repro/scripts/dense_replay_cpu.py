"""Predeclared dense low-rank shared replay: development and confirmation, CPU only.

Same data, splits, healthy checkpoints, minibatch order and monitoring as the
released sparse-row study.  Only the first-layer parameterization changes:
every row may move through a rank-s factor, and the exact replay correction
costs one m*m product instead of k*m.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from consumer_contracts import confidence_thresholds, evaluate_contracts
from dense_replay import DenseReferenceMLP, serve_dense_mlp, merged_serving, serve_merged_mlp
from readout_nullspace import masked_primary_loss

CONFIGURATIONS = [(16, 12), (24, 16), (32, 24)]
SOURCES = ['dense_replay_cpu.py', 'dense_replay.py', 'consumer_contracts.py',
           'benefit_compatibility_cpu.py', 'paired_readout_continuation.py', 'readout_nullspace.py']


@torch.no_grad()
def cache(healthy, x):
    first, second = [], []
    for j in range(0, len(x), 1024):
        p = healthy.encoder[0](x[j:j + 1024])
        first.append(p)
        second.append(healthy.encoder[2](F.relu(p)))
    return torch.cat(first), torch.cat(second)


def calibration(healthy, data, kind, dose, seed):
    idx = torch.randperm(len(data['x']), generator=torch.Generator().manual_seed(seed + 51000))[:1024]
    h = healthy(data['x'][idx])
    loss = (masked_primary_loss(bc.current_logits(healthy, h), data['y'][idx], data['labeled'][idx])
            + dose * paired.auxiliary_loss(healthy, h.float(), data['a'][idx], kind))
    g1, gb1, g2 = torch.autograd.grad(loss, [healthy.encoder[0].weight, healthy.encoder[0].bias,
                                             healthy.encoder[2].weight])
    q0 = torch.linalg.svd(g1.double(), full_matrices=False).Vh
    v0 = torch.linalg.svd(g2.double(), full_matrices=False).Vh
    return q0, v0, dict(indices_sha256=paired.state_hash(idx),
                        first_gradient_sha256=paired.state_hash(g1), second_gradient_sha256=paired.state_hash(g2),
                        q0_sha256=paired.state_hash(q0), v0_sha256=paired.state_hash(v0))


def train(healthy, moments, data, kind, caches, A, B, q0, v0, s, r, mode, seed, epochs, dose):
    model = DenseReferenceMLP(healthy, q0[:s], v0[:r], A, B)
    opt = torch.optim.Adam(model.old_parameters(), lr=.001)
    opt.load_state_dict(copy.deepcopy(moments))
    original = paired.state_hash(opt.state_dict())
    opt.add_param_group(dict(params=model.update_parameters(), lr=.001))
    rng = torch.Generator().manual_seed(seed + 20000)
    order_sha = hashlib.sha256()
    trace = []
    start = time.time()
    for epoch in range(epochs):
        order = torch.randperm(len(data['x']), generator=rng)
        order_sha.update(order.numpy().tobytes())
        for j in range(0, len(order), 256):
            idx = order[j:j + 256]
            h = model(data['x'][idx], tuple(v[idx] for v in caches))
            primary = masked_primary_loss(bc.current_logits(model, h), data['y'][idx], data['labeled'][idx])
            auxiliary = paired.auxiliary_loss(model, h.detach().float() if mode == 'detach' else h.float(),
                                              data['a'][idx], kind)
            loss = primary + dose * auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite dense-replay objective')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            trace.append(dict(epoch=epoch + 1, primary=float(primary.detach()), auxiliary=float(auxiliary.detach())))
    row = dict(mode=mode, rank1=s, rank2=r, trace=trace, initial_optimizer_sha256=original,
               minibatch_order_sha256=order_sha.hexdigest(), elapsed_s=time.time() - start,
               final_state_sha256=paired.state_hash(model.state_dict()),
               serving_sha256=paired.state_hash(model.serving()),
               first_delta_norm=float((model.p.detach() @ model.q.detach()).norm()),
               second_delta_norm=float((model.u @ model.v).detach().norm()))
    return model, opt, row


@torch.no_grad()
def outputs(x, p, dense=True):
    values = [serve_dense_mlp(x[j:j + 1024], p) for j in range(0, len(x), 1024)]
    return torch.cat([v[0] for v in values]), torch.cat([v[1] for v in values])


@torch.no_grad()
def outputs_merged(x, p):
    m = merged_serving(p)
    return torch.cat([serve_merged_mlp(x[j:j + 1024], m) for j in range(0, len(x), 1024)])


@torch.no_grad()
def score(model, data, kind, cached, old, thresholds):
    x, y, a = data
    p = model.serving()
    own, legacy = outputs(x, p)
    merged = outputs_merged(x, p)
    _, old2 = model.forward_states(x, cached)
    full = model.legacy_logits(old2, False)
    algebraic = model.legacy_logits(old2, True)
    features = model(x, cached)
    delta_pre1 = F.linear(x, model.p @ model.q) + model.delta1_bias
    return dict(own=paired.metrics(own, y),
                auxiliary_mse=float(paired.auxiliary_loss(model, features.float(), a, kind)),
                contracts=evaluate_contracts(legacy.double(), old, y, thresholds),
                full_reference_maximum_logit_error=float((full - old).abs().max()),
                compressed_maximum_logit_error=float((legacy.double() - old).abs().max()),
                affine_compression_float64_error=float((algebraic - old).abs().max()),
                current_merge_maximum_logit_error=float((own - merged).abs().max()),
                current_merge_prediction_changes=int((own.argmax(1) != merged.argmax(1)).sum()),
                changed_first_relu_gates=int(((cached[0] + delta_pre1 > 0) != (cached[0] > 0)).sum()))
