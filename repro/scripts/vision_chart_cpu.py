"""Predeclared CPU ResNet-18/CIFAR-100 development probe; no test evaluation.

Uses official pretrained weights and a frozen BN running-statistics protocol.
This is a transfer-learning probe, not a from-scratch CIFAR accuracy benchmark.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'

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
    x = images.permute(0, 3, 1, 2).float()/255.
    x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False, antialias=True)
    mean = torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1)
    sd = torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1)
    return ((x-mean)/sd).contiguous(memory_format=torch.channels_last)


def rotated_batch(x, labels):
    result = torch.empty_like(x)
    for k in range(4):
        mask = labels==k
        result[mask] = torch.rot90(x[mask], k, (-2, -1))
    return result


def freeze_bn(model):
    # eval changes running statistics only; affine weights retain gradients.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def bn_state(model):
    return {key:value for key, value in model.state_dict().items()
            if key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


@torch.no_grad()
def embed(model, images, indices):
    model.eval()
    return torch.cat([model(preprocess(images[indices[j:j+128]])).double()
                      for j in range(0, len(indices), 128)])


def repair_metrics(ref_anchor, new_anchor, new_eval, labels, head, mean, sd):
    maps = fit_repairs(ref_anchor, new_anchor, mean, sd)
    result = {}
    for candidate in ['diagonal_stats', 'paired_ridge']:
        result[candidate] = paired.metrics(head(apply_map(new_eval, maps[candidate])), labels)
    return result, {name:maps[name] for name in ['diagonal_stats', 'paired_ridge']}


def run(args):
    started = time.time()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cache = Path(os.environ['REPFIX_DATA_X'])/'v037'
    cache.mkdir(exist_ok=True)
    print(dict(stage='download_inputs', elapsed_s=0), flush=True)
    dataset = CIFAR100(str(cache), train=True, download=True)
    images = torch.from_numpy(dataset.data)
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    fitting_pool, validation = stratified_split(labels, .1, args.seed)
    permutation = torch.randperm(len(fitting_pool), generator=torch.Generator().manual_seed(args.seed+31000))
    anchors, fitting = fitting_pool[permutation[:2048]], fitting_pool[permutation[2048:]]
    assert len(validation)==5000 and len(anchors)==2048
    assert len(set(fitting.tolist()) & set(anchors.tolist()))==0
    assert len(set(fitting.tolist()) & set(validation.tolist()))==0
    encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    encoder.fc = nn.Identity()
    encoder.to(memory_format=torch.channels_last).eval()
    print(dict(stage='healthy_features', fitting=len(fitting), elapsed_s=time.time()-started), flush=True)
    healthy_fit = embed(encoder, images, fitting)
    healthy_val = embed(encoder, images, validation)
    healthy_anchor = embed(encoder, images, anchors)
    head, mean, sd = fit_head(healthy_fit, labels[fitting])
    primary = nn.Linear(512, 100)
    with torch.no_grad():
        primary.weight.copy_((head.weight/sd).float())
        primary.bias.copy_((head.bias - ((mean/sd) @ head.weight.T).reshape(-1)).float())
    rotation_head = nn.Linear(512, 4)
    initial = copy.deepcopy(encoder.state_dict())
    initial_primary = copy.deepcopy(primary.state_dict())
    initial_rotation = copy.deepcopy(rotation_head.state_dict())
    initial_bn_sha = paired.state_hash(bn_state(encoder))
    states = args.out.parent/(args.out.stem+'_states')
    states.mkdir(parents=True, exist_ok=True)
    torch.save(dict(encoder=initial, primary=initial_primary, rotation=initial_rotation,
                    head=head.state_dict(), mean=mean, sd=sd, fitting=fitting, validation=validation,
                    anchors=anchors, healthy_validation_features=healthy_val,
                    healthy_anchor_features=healthy_anchor), states/'reference.pt')
    out = dict(protocol='predeclared ImageNet-V1 ResNet-18 CIFAR-100 rotation development',
        phase='development', seed=args.seed, plan_sha256=args.plan_sha256, device='cpu',
        cuda_visibility=os.environ['CUDA_VISIBLE_DEVICES'], torch_threads=4,
        torch_version=torch.__version__, python_version=platform.python_version(),
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ['vision_chart_cpu.py', 'readout_alignment.py', 'readout_transfer_audit.py']},
        configuration=dict(pretrained_weights='torchvision ResNet18_Weights.IMAGENET1K_V1',
            image_size=64, original_stem=True, batch=128, epochs=3, encoder_lr=.001, heads_lr=.01,
            momentum=.9, weight_decay=.0005, train_dtype='cpu_bfloat16', eval_dtype='float32',
            bn_running_statistics_frozen=True, anchor_budgets=[128, 512, 1024, 2048]),
        splits=dict(fitting=len(fitting), anchors=len(anchors), validation=len(validation),
            official_test_loaded=False, test_metrics_computed=False,
            fitting_sha256=paired.state_hash(fitting), anchor_sha256=paired.state_hash(anchors),
            validation_sha256=paired.state_hash(validation), anchors_excluded_from_training=True),
        reference_encoder_sha256=paired.state_hash(initial), reference_bn_sha256=initial_bn_sha,
        healthy=paired.metrics(head((healthy_val-mean)/sd), labels[validation]), branches=[], complete=False)
    paired.save_json(args.out.with_suffix('.partial.json'), out)
    del healthy_fit
    for mode, dose in [('D', 1.), ('J1', 1.), ('J10', 10.)]:
        encoder.load_state_dict(copy.deepcopy(initial))
        primary.load_state_dict(copy.deepcopy(initial_primary))
        rotation_head.load_state_dict(copy.deepcopy(initial_rotation))
        optimizer = torch.optim.SGD([dict(params=encoder.parameters(), lr=.001),
            dict(params=primary.parameters(), lr=.01), dict(params=rotation_head.parameters(), lr=.01)],
            momentum=.9, weight_decay=.0005)
        rng = torch.Generator().manual_seed(args.seed+20000)
        rot_rng = torch.Generator().manual_seed(args.seed+21000)
        order_hash, rotation_hash = hashlib.sha256(), hashlib.sha256()
        trace = []
        for epoch in range(3):
            encoder.train()
            freeze_bn(encoder)
            order = torch.randperm(len(fitting), generator=rng)
            order_hash.update(order.numpy().tobytes())
            total_primary, total_rotation = 0., 0.
            for j in range(0, len(order), 128):
                idx = fitting[order[j:j+128]]
                x, y = preprocess(images[idx]), labels[idx]
                rotations = torch.randint(4, (len(idx),), generator=rot_rng)
                rotation_hash.update(rotations.numpy().tobytes())
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    h = encoder(x)
                    lp = F.cross_entropy(primary(h), y)
                    rotated = encoder(rotated_batch(x, rotations))
                    lr = F.cross_entropy(rotation_head(rotated.detach() if mode=='D' else rotated), rotations)
                    loss = lp + dose*lr
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite vision continuation loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_primary += float(lp.detach())*len(idx)
                total_rotation += float(lr.detach())*len(idx)
            assert paired.state_hash(bn_state(encoder))==initial_bn_sha
            trace.append(dict(epoch=epoch+1, primary_loss=total_primary/len(fitting),
                              rotation_loss=total_rotation/len(fitting)))
            print(dict(stage='continue', mode=mode, **trace[-1], elapsed_s=time.time()-started), flush=True)
        encoder.eval()
        native_fit = embed(encoder, images, fitting)
        native_val = embed(encoder, images, validation)
        native_anchor = embed(encoder, images, anchors)
        refit, refit_mean, refit_sd = fit_head(native_fit, labels[fitting])
        row = dict(mode=mode, dose=dose, trace=trace,
            initial_encoder_sha256=out['reference_encoder_sha256'], bn_buffers_unchanged=True,
            minibatch_order_sha256=order_hash.hexdigest(), rotation_order_sha256=rotation_hash.hexdigest(),
            final_encoder_sha256=paired.state_hash(encoder.state_dict()),
            frozen=paired.metrics(head((native_val-mean)/sd), labels[validation]),
            own=paired.metrics(F.linear(native_val, primary.weight.double(), primary.bias.double()), labels[validation]),
            supervised_refit=paired.metrics(refit((native_val-refit_mean)/refit_sd), labels[validation]), repairs={})
        saved_maps = {}
        for budget in [128, 512, 1024, 2048]:
            row['repairs'][str(budget)], saved_maps[str(budget)] = repair_metrics(healthy_anchor[:budget],
                native_anchor[:budget], native_val, labels[validation], head, mean, sd)
        # This stress condition deliberately uses labels to create a changed
        # anchor prior. The repair estimators themselves still receive no labels.
        restricted = torch.where(labels[anchors] < 50)[0][:1024]
        row['prior_shift_stress'], saved_maps['prior_shift'] = repair_metrics(healthy_anchor[restricted],
            native_anchor[restricted], native_val, labels[validation], head, mean, sd)
        row['prior_shift_anchor_examples'] = len(restricted)
        torch.save(dict(encoder=encoder.state_dict(), primary=primary.state_dict(), rotation=rotation_head.state_dict(),
            optimizer=optimizer.state_dict(), refit=refit.state_dict(), refit_mean=refit_mean, refit_sd=refit_sd,
            maps=saved_maps, validation_features=native_val, anchor_features=native_anchor), states/(mode+'.pt'))
        out['branches'].append(row)
        paired.save_json(args.out.with_suffix('.partial.json'), out)
        del native_fit
    assert len({row['minibatch_order_sha256'] for row in out['branches']})==1
    assert len({row['rotation_order_sha256'] for row in out['branches']})==1
    # An imposed scaling control is explicitly separate from learned updates.
    imposed = healthy_val*.25
    control, maps = repair_metrics(healthy_anchor, healthy_anchor*.25, imposed,
                                  labels[validation], head, mean, sd)
    out['imposed_scale_control'] = dict(scale=.25, artificial=True,
        frozen=paired.metrics(head((imposed-mean)/sd), labels[validation]), repairs=control)
    # Preserve the anchor mean and covariance while rotating whitened features.
    # Marginal matching cannot determine the orientation of their semantics.
    center = healthy_anchor.mean(0, keepdim=True)
    covariance = (healthy_anchor-center).T @ (healthy_anchor-center)/(len(healthy_anchor)-1)
    values, vectors = torch.linalg.eigh(covariance)
    values = values.clamp_min(1e-8*values.max())
    white = (vectors*values.rsqrt())@vectors.T
    color = (vectors*values.sqrt())@vectors.T
    generator = torch.Generator().manual_seed(args.seed+41000)
    rotation = torch.linalg.qr(torch.randn(512, 512, generator=generator, dtype=torch.float64)).Q
    matrix = white@rotation@color
    shifted_anchor = (healthy_anchor-center)@matrix+center
    shifted_val = (healthy_val-center)@matrix+center
    control, _ = repair_metrics(healthy_anchor, shifted_anchor, shifted_val,
                                labels[validation], head, mean, sd)
    shifted_covariance = (shifted_anchor-center).T@(shifted_anchor-center)/(len(shifted_anchor)-1)
    out['covariance_rotation_control'] = dict(artificial=True,
        relative_anchor_covariance_error=float((shifted_covariance-covariance).norm()/covariance.norm()),
        frozen=paired.metrics(head((shifted_val-mean)/sd), labels[validation]), repairs=control,
        known_inverse=paired.metrics(head((((shifted_val-center)@torch.linalg.inv(matrix)+center)-mean)/sd),
                                    labels[validation]))
    out.update(complete=True, elapsed_s=time.time()-started)
    paired.save_json(args.out, out)
    print(dict(stage='complete', elapsed_s=out['elapsed_s']), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
