"""Independent CPU replay of the fixed compiled-interface endpoint family."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import compiled_interface_cpu as experiment
import paired_readout_continuation as paired
from consumer_risk import certificate


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    record = json.loads(args.record.read_text())
    assert record['complete'] and len(record['branches']) == 6
    torch.manual_seed(record['seed'])
    np.random.seed(record['seed'])
    cfg = SimpleNamespace(domain=record['domain'], seed=record['seed'], phase='confirmation',
                          limit_fit=0, limit_eval=0)
    reference, data, kind, splits = experiment.load(cfg)
    assert splits == record['splits']
    states = args.record.parent / (args.record.stem + '_states')
    ref = torch.load(states / 'reference.pt', map_location='cpu', weights_only=True)
    assert paired.state_hash(ref['model']) == record['reference_model_sha256']
    assert paired.state_hash(ref['optimizer']) == record['reference_optimizer_sha256']
    reference.load_state_dict(ref['model'])
    reference.eval()
    x, y, aux = data['test']
    h0 = paired.embed(reference, x)
    weight, bias = ref['head']['weight'], ref['head']['bias']
    mu, sd = ref['mean'], ref['sd']
    z0 = F.linear((h0 - mu) / sd, weight, bias)
    monitor = np.random.default_rng(2870001 + record['seed']).choice(len(y), 1024, replace=False)
    assert hashlib.sha256(monitor.astype(np.int64).tobytes()).hexdigest() == record['consumer_bank']['monitor_membership_sha256']
    result = dict(complete=False, device='cpu', cuda_visibility='', torch_threads=1,
        seed=record['seed'], domain=record['domain'],
        record_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), branches=[])
    for row in record['branches']:
        mode = row['mode']
        state = torch.load(states / (mode + '.pt'), map_location='cpu', weights_only=True)
        assert paired.state_hash(state['model']) == row['final_model_sha256']
        assert paired.state_hash(state['optimizer']) == row['final_optimizer_sha256']
        model = copy.deepcopy(reference)
        if mode.startswith('C'):
            model.constrain_last(ref['projector'])
        model.load_state_dict(state['model'])
        model.eval()
        # Reconstruct the final affine map from saved tensors, independently
        # of both the train-time effective_affine() and deployment fold().
        raw = state['model']
        wf, bf = raw['final.weight'], raw['final.bias']
        if mode.startswith('C'):
            wf = raw['anchor_weight'] + raw['projector'].matmul(wf - raw['anchor_weight'])
            bf = raw['anchor_bias'] + raw['projector'].matmul(bf - raw['anchor_bias'])
            compiled = torch.load(states / (mode + '_compiled.pt'), map_location='cpu', weights_only=True)['model']
            assert paired.state_hash(compiled) == row['compiled_model_sha256']
            assert torch.equal(compiled['final.weight'], wf)
            assert torch.equal(compiled['final.bias'], bf)
            assert not any('anchor' in key or 'projector' in key for key in compiled)
        h = torch.cat([F.linear(model.trunk(x[j:j+2048]).double(), wf, bf)
                       for j in range(0, len(x), 2048)])
        z = F.linear((h-mu)/sd, weight, bias)
        own = F.linear(h, model.primary.weight.double(), model.primary.bias.double())
        actual = dict(frozen=paired.metrics(z, y), own=paired.metrics(own, y))
        maximum = 0.
        for head in ['frozen', 'own']:
            for metric in ['acc', 'nll', 'ece']:
                error = abs(actual[head][metric] - row['test'][head][metric])
                assert error < 1e-7, (mode, head, metric, error)
                maximum = max(maximum, error)
        assert abs(float(paired.auxiliary_loss(model, h.float(), aux, kind)) - row['test']['auxiliary_mse']) < 1e-7
        changes = int((z.argmax(1) != z0.argmax(1)).sum())
        assert changes == row['test']['prediction_changes']
        logit_error = float((z-z0).abs().max())
        assert abs(logit_error-row['test']['max_absolute_healthy_logit_error']) < 1e-10
        bound = certificate(z.argmax(1).numpy(), z0.argmax(1).numpy(), monitor, .05/120)
        for key, value in bound.items():
            assert abs(value-row['test']['consumer_audit'][key]) < 1e-12
        result['branches'].append(dict(mode=mode, all_scores_replayed=True,
            maximum_metric_error=maximum, prediction_changes=changes,
            max_absolute_healthy_logit_error=logit_error, bounds_replayed=True,
            saved_compiled_map_verified=mode.startswith('C')))
    result['complete'] = True
    paired.save_json(args.out, result)
    print(json.dumps(dict(seed=record['seed'], domain=record['domain'], replayed_consumers=6)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
