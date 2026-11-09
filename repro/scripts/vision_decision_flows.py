"""Exact correct/wrong decision flows behind a vision continuation branch.

Recomputes the healthy readout exactly as ``vision_scale_replay.py`` does and
decomposes every changed prediction into correct->wrong, wrong->correct and
wrong->wrong.  The headline disagreement counts both directions together and
can hide a large traffic between two wrong classes; this reader separates them.
Read-only: no training, no official test image, no new hyperparameter.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '8'

import argparse
import hashlib
from pathlib import Path

import torch
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18, ResNet18_Weights

import paired_readout_continuation as paired
from readout_transfer_audit import fit_head, stratified_split
from vision_boundary_cpu import preprocess


@torch.no_grad()
def embed(model, images, indices):
    model.eval()
    return torch.cat([model(preprocess(images[indices[j:j + 128]]))
                      for j in range(0, len(indices), 128)]).double()


def flows(reference, prediction, labels):
    old_right = reference == labels
    new_right = prediction == labels
    changed = prediction != reference
    return dict(
        examples=int(len(labels)),
        changed=int(changed.sum()),
        correct_old=int(old_right.sum()),
        correct_new=int(new_right.sum()),
        correct_to_wrong=int((old_right & ~new_right & changed).sum()),
        wrong_to_correct=int((~old_right & new_right & changed).sum()),
        wrong_to_wrong=int((~old_right & ~new_right & changed).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--features', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cache = Path(os.environ['REPFIX_DATA_X']) / 'v037'
    dataset = CIFAR100(str(cache), train=True, download=True)
    images = torch.from_numpy(dataset.data)
    labels = torch.tensor(dataset.targets, dtype=torch.long)
    pool, validation = stratified_split(labels, .1, args.seed)
    permutation = torch.randperm(len(pool), generator=torch.Generator().manual_seed(args.seed + 31000))
    anchors, fitting = pool[permutation[:2048]], pool[permutation[2048:]]
    encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    encoder.fc = torch.nn.Identity()
    encoder.to(memory_format=torch.channels_last).eval()
    healthy_fit = embed(encoder, images, fitting)
    healthy_val = embed(encoder, images, validation)
    head, mean, sd = fit_head(healthy_fit, labels[fitting])
    y = labels[validation]
    z = (healthy_val - mean) / sd
    reference = head(z).argmax(1)
    record = dict(
        protocol='exact decision flows behind the vision boundary replay',
        seed=args.seed,
        source=dict(script='vision_decision_flows.py',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    features={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in args.features}),
        splits=dict(fitting=len(fitting), anchors=len(anchors),
                    validation=len(validation), official_test_loaded=False),
        healthy_acc=float((reference == y).double().mean()),
        scale_replay=[dict(c=c, **flows(reference, (c * z @ head.weight.double().T
                                                    + head.bias.double()).argmax(1), y))
                      for c in (1.0, 0.432, 0.318)],
        branches=[])
    for path in args.features:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        prediction = head((saved['validation_features'].double() - mean) / sd).argmax(1)
        record['branches'].append(dict(file=path.name, **flows(reference, prediction, y)))
    paired.save_json(args.out, record)
    print(__import__('json').dumps(record))


if __name__ == '__main__':
    main()
