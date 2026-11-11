"""Isolated native-float32 serving cost of the dense low-rank replay; CPU only.

The current head, the unpacked exact replay, the packed exact replay and an
independent second network are traced and timed on identical inputs at batch 1
and 1024.  Timings use the released protocol: warm-up, median of 71 alternating
repetitions, four threads.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dense_replay import serve_dense_mlp, merged_serving, serve_merged_mlp, packed_serving, serve_packed_mlp
from paired_readout_continuation import save_json, state_hash


@torch.no_grad()
def main(a):
    choice = json.loads(a.choice.read_text())
    out = dict(complete=False, device='cpu', threads=4, dtype='float32',
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               choice_sha256=hashlib.sha256(a.choice.read_bytes()).hexdigest(), branches=[])
    for path in sorted(p for p in a.records.glob('*.json') if not p.name.endswith('.partial.json')):
        record = json.loads(path.read_text())
        sel = choice['domains'][record['domain']]['selected']
        s, r = sel['rank1'], sel['rank2']
        folder = path.parent / (path.stem + '_states')
        for mode in ['detach', 'joint']:
            p = torch.load(folder / ('%d_%d_%s.pt' % (s, r, mode)), map_location='cpu', weights_only=True)['serving']
            merged = merged_serving(p)
            packed = packed_serving(p)

            def current(x):
                return serve_merged_mlp(x, merged)

            def replay(x):
                return serve_dense_mlp(x, p)

            def optimized(x):
                return serve_packed_mlp(x, packed)

            def independent(x):
                old = F.relu(F.linear(x, p['first_weight'], p['first_bias']))
                old = F.relu(F.linear(old, p['second_weight'], p['second_bias']))
                return current(x), F.linear(old, p['old_weight'], p['old_bias'])

            operations = dict(current=current, replay=replay, packed_replay=optimized, two_independent=independent)
            sample = torch.randn(1024, p['first_weight'].shape[1],
                                 generator=torch.Generator().manual_seed(record['seed'] + 90000))
            target = replay(sample)
            errors = {name: float((op(sample)[0] - target[0]).abs().max())
                      for name, op in operations.items()}
            errors['old_output'] = float((target[1] - F.linear(
                F.relu(F.linear(F.relu(F.linear(sample, p['first_weight'], p['first_bias'])),
                                p['second_weight'], p['second_bias'])), p['old_weight'], p['old_bias'])).abs().max())
            row = dict(domain=record['domain'], seed=record['seed'], mode=mode, rank1=s, rank2=r,
                       serving_sha256=state_hash(p),
                       packed_parameters=sum(v.numel() for v in packed.values() if isinstance(v, torch.Tensor)),
                       current_parameters=sum(v.numel() for v in merged.values()),
                       random_input_maximum_errors=errors, backends={})
            for backend in ['eager', 'torchscript']:
                start = time.perf_counter()
                if backend == 'eager':
                    ops = operations
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', torch.jit.TracerWarning)
                        ops = {name: torch.jit.trace(op, sample[:1], check_trace=True) for name, op in operations.items()}
                timing = {}
                setup = time.perf_counter() - start
                for batch in [1, 1024]:
                    x = sample[:batch]
                    for op in ops.values():
                        for _ in range(12):
                            op(x)
                    durations = {name: [] for name in ops}
                    for rep in range(71):
                        for name in (list(ops) if rep % 2 else list(reversed(ops))):
                            begin = time.perf_counter()
                            ops[name](x)
                            durations[name].append(time.perf_counter() - begin)
                    medians = {name: float(np.median(v)) for name, v in durations.items()}
                    timing[str(batch)] = dict(medians_seconds=medians, samples_seconds=durations,
                                              ratios_to_current={name: value / medians['current']
                                                                 for name, value in medians.items()})
                row['backends'][backend] = dict(setup_seconds=setup, timings=timing)
            out['branches'].append(row)
            save_json(a.out.with_suffix('.partial.json'), out)
            ratios = {name: {batch: entry['timings'][batch]['ratios_to_current']['packed_replay']
                              for batch in entry['timings']}
                      for name, entry in row['backends'].items()}
            print(json.dumps(dict(domain=record['domain'], seed=record['seed'], mode=mode, ratios=ratios)),
                  flush=True)
    out['complete'] = True
    save_json(a.out, out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--choice', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    main(a)
