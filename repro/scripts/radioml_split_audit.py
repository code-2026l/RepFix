"""CPU verification of the prepared RadioML cache against the source split.

Read-only: reconstruct the data loader's seeded per-modulation split in memory,
compare every tensor, and never replace or rewrite the prepared input cache.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def run(args):
    torch.set_num_threads(1)
    root = Path(os.environ['REPFIX_DATA_X'])
    cache_path = root / 'RML2016.10a_pre.pt'
    raw_path = root / 'RML2016.10a_dict_optimized.pkl'
    cached = torch.load(cache_path, map_location='cpu', mmap=True, weights_only=True)
    with raw_path.open('rb') as stream:
        raw = pickle.load(stream)
    mods = sorted({k[0] for k in raw})
    x, y, auxiliary = [], [], []
    for (modulation, snr), values in raw.items():
        x.append(values)
        y.append(np.full(len(values), mods.index(modulation)))
        auxiliary.append(np.full(len(values), snr))
    x = np.concatenate(x).astype(np.float32).reshape(-1, 256)
    y = np.concatenate(y).astype(np.int64)
    auxiliary = np.concatenate(auxiliary).astype(np.float32)
    generator = np.random.RandomState(0)
    train, test = [], []
    for modulation in range(len(mods)):
        indices = np.where(y == modulation)[0]
        generator.shuffle(indices)
        half = len(indices) // 2
        train.append(indices[:half])
        test.append(indices[half:])
    tr, te = np.concatenate(train), np.concatenate(test)
    expected = dict(X_tr=x[tr], y_tr=y[tr], aux_tr=auxiliary[tr],
                    X_te=x[te], y_te=y[te], aux_te=auxiliary[te])
    equality = {key: bool(np.array_equal(value, cached[key].numpy())) for key, value in expected.items()}
    if not all(equality.values()):
        raise AssertionError('Prepared cache is not the documented seeded split: %s' % equality)
    result = dict(complete=True, device='cpu', torch_threads=torch.get_num_threads(),
        cuda_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'), branches=[],
        source_sha256=digest(Path(__file__)), raw_sha256=digest(raw_path), cache_sha256=digest(cache_path),
        split='50/50 per modulation; numpy RandomState(0); raw dictionary iteration order',
        train_examples=len(tr), test_examples=len(te), modulations=len(mods),
        train_indices_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),
        test_indices_sha256=hashlib.sha256(te.tobytes()).hexdigest(), cache_tensor_equality=equality)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
