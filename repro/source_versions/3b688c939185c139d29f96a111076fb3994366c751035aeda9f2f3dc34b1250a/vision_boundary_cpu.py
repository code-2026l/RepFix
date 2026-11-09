"""Prespecified ResNet-18/CIFAR-100 boundary-crossing screen (CPU, fp32).

Question: does the affine first-flip boundary predict when a frozen readout
survives an auxiliary continuation, on a real vision backbone?  The earlier
rotation probe (bf16) contracted the covariance trace to 0.321 of healthy,
which is a scale ratio of 0.567 and therefore above the median first-flip scale
0.432; the frozen readout survived, as the boundary predicts.  This screen
crosses the boundary on the same backbone with two fixed instruments:

  D      primary cross-entropy only (reference continuation)
  J<d>   cross-entropy + d * rotation cross-entropy (natural auxiliary)
  S<l>   cross-entropy + l * mean(h^2) (controlled scale instrument)

Training is float32 to avoid the bf16 non-finite loss seen at rotation dose 10.
All branches clone the healthy model and share minibatch order.  Batch
normalization running statistics stay frozen.  No official test image is loaded.

Reported per branch: covariance-trace ratio, implied scale ratio, median
first-flip scale of the healthy readout, frozen / current / refit / recalibrated
accuracy, NLL and ECE.  Outcomes are not used to retune doses or epochs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '8'

import argparse
import copy
import hashlib
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18, ResNet18_Weights

import paired_readout_continuation as paired
from readout_alignment import apply_map, fit_repairs
from readout_transfer_audit import fit_head, stratified_split


def preprocess(images):
    x = images.permute(0, 3, 1, 2).float() / 255.
    x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False, antialias=True)
    mean = torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1)
    sd = torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1)
    return ((x - mean) / sd).contiguous(memory_format=torch.channels_last)


def rotated_batch(x, labels):
    result = torch.empty_like(x)
    for k in range(4):
        mask = labels == k
        result[mask] = torch.rot90(x[mask], k, (-2, -1))
    return result


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def bn_state(model):
    return {key: value for key, value in model.state_dict().items()
            if key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


@torch.no_grad()
def embed(model, images, indices):
    model.eval()
    return torch.cat([model(preprocess(images[indices[j:j + 128]])).double()
                      for j in range(0, len(indices), 128)])


def trace_V(h):
    hc = h - h.mean(0, keepdim=True)
    return float((hc ** 2).sum() / (2 * h.shape[0]))


def scatter_S(h):
    hc = h - h.mean(0, keepdim=True)
    return float(hc.std(0).mean())


def first_flip_scale(features, weight, bias, mean, sd):
    """Median c>0 where the frozen readout's top-2 logits cross under h -> c h."""
    A = weight / sd
    B = bias - (weight * mean / sd).sum(1)
    logits = features @ A.T + B
    order = torch.argsort(logits, dim=1)
    top, second = order[:, -1], order[:, -2]
    slope = ((A[top] - A[second]) * features).sum(1)
    offset = B[top] - B[second]
    valid = slope > 0
    c = (-offset[valid] / slope[valid])
    c = c[c > 0]
    return float(torch.median(c)), int(valid.sum()), int(c.numel())


def run(args):
    started = time.time()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cache = Path(os.environ['REPFIX_DATA_X']) / 'v037'
    cache.mkdir(exist_ok=True)
    dataset = CIFAR100(str(cache), train=True, download=True)
    images = torch.from_numpy(dataset.data)
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    fitting_pool, validation = stratified_split(labels, .1, args.seed)
    permutation = torch.randperm(len(fitting_pool), generator=torch.Generator().manual_seed(args.seed + 31000))
    anchors, fitting = fitting_pool[permutation[:2048]], fitting_pool[permutation[2048:]]
    assert len(validation) == 5000 and len(anchors) == 2048
    assert len(set(fitting.tolist()) & set(anchors.tolist())) == 0
    assert len(set(fitting.tolist()) & set(validation.tolist())) == 0

    encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    encoder.fc = nn.Identity()
    encoder.to(memory_format=torch.channels_last).eval()
    print(dict(stage='healthy_features', elapsed_s=time.time() - started), flush=True)
    healthy_fit = embed(encoder, images, fitting)
    healthy_val = embed(encoder, images, validation)
    healthy_anchor = embed(encoder, images, anchors)
    head, mean, sd = fit_head(healthy_fit, labels[fitting])
    primary = nn.Linear(512, 100)
    with torch.no_grad():
        primary.weight.copy_((head.weight / sd).float())
        primary.bias.copy_((head.bias - ((mean / sd) @ head.weight.T).reshape(-1)).float())
    rotation_head = nn.Linear(512, 4)
    initial = copy.deepcopy(encoder.state_dict())
    initial_primary = copy.deepcopy(primary.state_dict())
    initial_rotation = copy.deepcopy(rotation_head.state_dict())
    initial_bn_sha = paired.state_hash(bn_state(encoder))

    monotone, n_valid, n_flip = first_flip_scale(healthy_val, head.weight.double(), head.bias.double(),
                                                 mean.double(), sd.double())
    out = dict(protocol='prespecified ResNet-18 CIFAR-100 boundary crossing, fp32, development',
               phase='development', seed=args.seed, plan_sha256=args.plan_sha256, device='cpu',
               cuda_visibility=os.environ['CUDA_VISIBLE_DEVICES'], torch_threads=8,
               torch_version=torch.__version__, python_version=platform.python_version(),
               source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ['vision_boundary_cpu.py', 'readout_alignment.py',
                                           'readout_transfer_audit.py', 'paired_readout_continuation.py']},
               configuration=dict(pretrained_weights='torchvision ResNet18_Weights.IMAGENET1K_V1',
                                  image_size=64, batch=128, epochs=args.epochs, encoder_lr=.001,
                                  heads_lr=.01, momentum=.9, weight_decay=.0005, train_dtype='float32',
                                  eval_dtype='float64', bn_running_statistics_frozen=True),
               splits=dict(fitting=len(fitting), anchors=len(anchors), validation=len(validation),
                           official_test_loaded=False, test_metrics_computed=False,
                           fitting_sha256=paired.state_hash(fitting),
                           anchor_sha256=paired.state_hash(anchors),
                           validation_sha256=paired.state_hash(validation),
                           anchors_excluded_from_training=True),
               reference_encoder_sha256=paired.state_hash(initial),
               reference_bn_sha256=initial_bn_sha,
               healthy=dict(metrics=paired.metrics(head((healthy_val - mean) / sd), labels[validation]),
                            trace_V=trace_V(healthy_val), scatter_S=scatter_S(healthy_val),
                            median_first_flip_c=monotone, first_flip_defined=n_flip,
                            first_flip_positive=n_valid),
               branches=[], complete=False)
    paired.save_json(args.out.with_suffix('.partial.json'), out)
    del healthy_fit

    modes = args.modes.split(',')
    for mode in modes:
        encoder.load_state_dict(copy.deepcopy(initial))
        primary.load_state_dict(copy.deepcopy(initial_primary))
        rotation_head.load_state_dict(copy.deepcopy(initial_rotation))
        optimizer = torch.optim.SGD(
            [dict(params=encoder.parameters(), lr=.001),
             dict(params=primary.parameters(), lr=.01),
             dict(params=rotation_head.parameters(), lr=.01)],
            momentum=.9, weight_decay=.0005)
        rng = torch.Generator().manual_seed(args.seed + 20000)
        rot_rng = torch.Generator().manual_seed(args.seed + 21000)
        order_hash, rotation_hash = hashlib.sha256(), hashlib.sha256()
        trace = []
        epoch_traces = []
        for epoch in range(args.epochs):
            encoder.train()
            freeze_bn(encoder)
            order = torch.randperm(len(fitting), generator=rng)
            order_hash.update(order.numpy().tobytes())
            total_primary, total_aux = 0., 0.
            limit = len(order) if args.limit_batches <= 0 else min(len(order), args.limit_batches * 128)
            for j in range(0, limit, 128):
                idx = fitting[order[j:j + 128]]
                x, y = preprocess(images[idx]), labels[idx]
                rotations = torch.randint(4, (len(idx),), generator=rot_rng)
                rotation_hash.update(rotations.numpy().tobytes())
                h = encoder(x)
                lp = F.cross_entropy(primary(h), y)
                if mode == 'D':
                    loss = lp
                    aux = torch.zeros((), dtype=lp.dtype)
                elif mode.startswith('J'):
                    dose = float(mode[1:])
                    rotated = encoder(rotated_batch(x, rotations))
                    lr = F.cross_entropy(rotation_head(rotated), rotations)
                    loss = lp + dose * lr
                    aux = lr.detach()
                elif mode.startswith('S'):
                    lam = float(mode[1:])
                    lscale = (h ** 2).mean()
                    loss = lp + lam * lscale
                    aux = lscale.detach()
                else:
                    raise ValueError(f'unknown mode {mode}')
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite continuation loss in {mode} epoch {epoch + 1}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_primary += float(lp.detach()) * len(idx)
                total_aux += float(aux) * len(idx)
            assert paired.state_hash(bn_state(encoder)) == initial_bn_sha
            trace.append(dict(epoch=epoch + 1, primary_loss=total_primary / limit,
                              aux_loss=total_aux / limit))
            if epoch + 1 < args.epochs:
                subset = validation[:1024]
                epoch_traces.append(dict(epoch=epoch + 1,
                                         trace_ratio=trace_V(embed(encoder, images, subset)) /
                                         max(trace_V(healthy_val[:1024]), 1e-12)))
            print(dict(stage='continue', mode=mode, **trace[-1],
                       elapsed_s=time.time() - started), flush=True)
        encoder.eval()
        native_fit = embed(encoder, images, fitting)
        native_val = embed(encoder, images, validation)
        native_anchor = embed(encoder, images, anchors)
        refit, refit_mean, refit_sd = fit_head(native_fit, labels[fitting])
        maps = fit_repairs(healthy_anchor, native_anchor, mean, sd)
        recal = paired.metrics(head(apply_map(native_val, maps['diagonal_stats'])), labels[validation])
        v_new = trace_V(native_val)
        scale = float(np.sqrt(v_new / max(trace_V(healthy_val), 1e-12)))
        row = dict(mode=mode, dose=(float(mode[1:]) if mode[0] in ('J', 'S') else 0.0),
                   trace=trace, epoch_trace_ratio=epoch_traces,
                   initial_encoder_sha256=out['reference_encoder_sha256'],
                   bn_buffers_unchanged=True,
                   minibatch_order_sha256=order_hash.hexdigest(),
                   rotation_order_sha256=rotation_hash.hexdigest(),
                   final_encoder_sha256=paired.state_hash(encoder.state_dict()),
                   trace_V=v_new, trace_ratio=v_new / max(trace_V(healthy_val), 1e-12),
                   scale_ratio=scale, scatter_S=scatter_S(native_val),
                   frozen=paired.metrics(head((native_val - mean) / sd), labels[validation]),
                   current=paired.metrics(F.linear(native_val, primary.weight.double(), primary.bias.double()),
                                          labels[validation]),
                   refit=paired.metrics(refit((native_val - refit_mean) / refit_sd), labels[validation]),
                   recalibrated=recal)
        torch.save(dict(encoder=encoder.state_dict(), primary=primary.state_dict(),
                        rotation=rotation_head.state_dict(), optimizer=optimizer.state_dict(),
                        refit=refit.state_dict(), refit_mean=refit_mean, refit_sd=refit_sd,
                        maps=maps, validation_features=native_val, anchor_features=native_anchor),
                   args.out.parent / (args.out.stem + '_' + mode + '.pt'))
        out['branches'].append(row)
        paired.save_json(args.out.with_suffix('.partial.json'), out)
        del native_fit
    out.update(complete=True, elapsed_s=time.time() - started)
    paired.save_json(args.out, out)
    print(dict(stage='complete', elapsed_s=out['elapsed_s']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--modes', default='D')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--limit-batches', type=int, default=0)
    run(parser.parse_args())
