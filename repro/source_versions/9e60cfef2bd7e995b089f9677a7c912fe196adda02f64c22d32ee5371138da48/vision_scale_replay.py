"""Post-hoc pure-scale replay of the frozen vision readout.

Rebuilds the deterministic validation split of ``vision_boundary_cpu.py``,
embeds it with the same pretrained encoder, and evaluates the healthy readout
on ``c*h`` for a fixed grid of ``c``.  This isolates the scale component from
the direction change learned in continuation.  No training, no official test
image, no new hyperparameter.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '8'

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18, ResNet18_Weights

import paired_readout_continuation as paired
from readout_transfer_audit import fit_head, stratified_split
from vision_boundary_cpu import preprocess

GRID = [1.0, 0.9, 0.7, 0.6, 0.5, 0.432, 0.367, 0.318, 0.25, 0.15, 0.067, 0.03]


@torch.no_grad()
def embed(model, images, indices):
    model.eval()
    return torch.cat([model(preprocess(images[indices[j:j + 128]]))
                      for j in range(0, len(indices), 128)]).double()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=900)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--features', type=Path, nargs='*', default=[],
                        help='branch *_<mode>.pt files with validation_features')
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
    assert len(validation) == 5000 and len(anchors) == 2048
    encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    healthy_fit = embed(encoder, images, fitting)
    healthy_val = embed(encoder, images, validation)
    head, mean, sd = fit_head(healthy_fit, labels[fitting])
    y = labels[validation]
    base = head((healthy_val - mean) / sd)
    z = (healthy_val - mean) / sd
    reference_pred = base.argmax(1)
    rows = []
    for c in GRID:
        logits = c * z @ head.weight.double().T + head.bias.double()
        rows.append(dict(c=c,
                         acc=float((logits.argmax(1) == y).double().mean()),
                         retained=float((logits.argmax(1) == reference_pred).double().mean())))
        print(rows[-1], flush=True)
    branches = []
    for path in args.features:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        native = saved['validation_features'].double()
        logits = head((native - mean) / sd).argmax(1)
        record = dict(file=path.name,
                      acc=float((logits == y).double().mean()),
                      disagreement=float((logits != reference_pred).double().mean()))
        branches.append(record)
        print(record, flush=True)
    paired.save_json(args.out, dict(
        protocol='post-hoc pure-scale replay of the frozen vision readout',
        note='Isolates the scale component; no training and no dose selection.',
        seed=args.seed, grid=GRID, device='cpu',
        source_sha256={'vision_scale_replay.py': paired.state_hash(Path(__file__).read_bytes()),
                       'vision_boundary_cpu.py': paired.state_hash(Path(__file__).with_name('vision_boundary_cpu.py').read_bytes())},
        splits=dict(fitting=len(fitting), anchors=len(anchors), validation=len(validation),
                    validation_sha256=paired.state_hash(validation),
                    official_test_loaded=False),
        healthy=paired.metrics(base, y), rows=rows, branches=branches))


if __name__ == '__main__':
    main()
