"""CPU auxiliary-weight/Adam-state factorial from a fixed healthy reference.

All four predeclared cells retain shared parameters, shared Adam states, the
consumer, dose 30, and future minibatches. Outside the consumer-certificate bank.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import paired_readout_continuation as paired
from checkpoint_repair_audit import assert_metrics

CELLS = [('warm', 'retained'), ('warm', 'reset'), ('initial', 'retained'), ('initial', 'reset')]


def shared_optimizer_hash(model, optimizer):
    return paired.state_hash({name: optimizer.state.get(parameter, {})
        for name, parameter in model.named_parameters() if name.startswith(('encoder.', 'primary.'))})


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    old = json.loads(args.record.read_text())
    config = argparse.Namespace(**old['config'])
    if not old['complete'] or config.domain != 'radioml' or config.aux != 'snr':
        raise ValueError('A completed healthy paired SNR reference is required')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model, data, aux, splits = paired.make_data(config)
    initial_aux = copy.deepcopy(model.aux.state_dict())
    xfit, yfit, _, _, xtest, _ = data
    source_states = args.record.parent / (args.record.stem + '_states')
    source_path = source_states / 'reference.pt'
    state = torch.load(source_path, map_location='cpu', weights_only=True)
    assert paired.state_hash(state['model']) == old['reference_model_sha256']
    assert paired.state_hash(state['optimizer']) == old['reference_optimizer_sha256']
    assert splits == old['splits']
    model.load_state_dict(state['model'])
    model.eval()
    hfit, htest = paired.embed(model, xfit), paired.embed(model, xtest)
    head, mu, sd = paired.fit_head(hfit, yfit)
    reference = (head, mu, sd, htest, old['healthy_temperature'], old['healthy_raw_conformal_q'])
    healthy = paired.snapshot_metrics(model, data, reference)
    healthy_replay = assert_metrics(healthy['frozen'], old['healthy']['frozen'], 'healthy')
    source_data = paired.cdm.load_domain('radioml')
    raw = torch.from_numpy(np.asarray(source_data['aux_tr'])).float()
    fit, _ = paired.stratified_split(torch.from_numpy(source_data['y_tr']).long(), .2, config.seed)
    aux_test = (torch.from_numpy(np.asarray(source_data['aux_te'])).float() - raw[fit].mean()) / raw[fit].std(unbiased=False)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    opt.load_state_dict(copy.deepcopy(state['optimizer']))
    shared = {k: v for k, v in model.state_dict().items() if k.startswith(('encoder.', 'primary.'))}
    shared_hash, shared_opt_hash = paired.state_hash(shared), shared_optimizer_hash(model, opt)
    result = dict(protocol='fixed dose-30 auxiliary weights by auxiliary Adam-state factorial',
        source_record=args.record.name, source_record_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        reference_checkpoint_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        paired_source_sha256=hashlib.sha256(Path(paired.__file__).read_bytes()).hexdigest(),
        seed=config.seed, dose=30., continuation_epochs=15, device='cpu', torch_threads=torch.get_num_threads(),
        cuda_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=torch.__version__, splits=splits,
        healthy=healthy, healthy_reconstruction_error=healthy_replay,
        initial_shared_sha256=shared_hash, initial_shared_optimizer_sha256=shared_opt_hash,
        initial_auxiliary_sha256=paired.state_hash(initial_aux), outside_consumer_certificate_bank=True,
        branches=[], complete=False)
    destination = args.out.parent / (args.out.stem + '_states')
    destination.mkdir(parents=True, exist_ok=True)
    original = next(b for b in old['branches'] if b['mode'] == 'joint' and b['alpha'] == 30)
    for weight_state, moment_state in CELLS:
        model.load_state_dict(state['model'])
        opt = torch.optim.Adam(model.parameters(), lr=config.lr)
        opt.load_state_dict(copy.deepcopy(state['optimizer']))
        if weight_state == 'initial':
            model.aux.load_state_dict(initial_aux)
        if moment_state == 'reset':
            for parameter in model.aux.parameters():
                opt.state.pop(parameter, None)
        shared = {k: v for k, v in model.state_dict().items() if k.startswith(('encoder.', 'primary.'))}
        assert paired.state_hash(shared) == shared_hash
        assert shared_optimizer_hash(model, opt) == shared_opt_hash
        branch = dict(weight_state=weight_state, moment_state=moment_state,
            initial_model_sha256=paired.state_hash(model.state_dict()), initial_optimizer_sha256=paired.state_hash(opt.state_dict()),
            initial_shared_sha256=shared_hash, initial_shared_optimizer_sha256=shared_opt_hash,
            initial_auxiliary_sha256=paired.state_hash(model.aux.state_dict()), trajectory=[], update_audits=[])
        future = torch.Generator().manual_seed(config.seed + 20000)
        order_hash = hashlib.sha256()
        for epoch in range(15):
            order = torch.randperm(len(xfit), generator=future)
            order_hash.update(order.numpy().tobytes())
            for start in range(0, len(order), config.batch):
                indices = order[start:start + config.batch]
                batch_id = start // config.batch
                dense = epoch < 2 and (batch_id in (0, 1, 4, 9) or batch_id % 32 == 0)
                late = start == 0 and epoch in (4, 9, 14)
                _, _, audit = paired.update(model, opt, xfit[indices], yfit[indices], aux[indices],
                    'snr', 'joint', 30., False, audit_x=xfit[:256] if dense or late else None,
                    counterfactual=dense or late)
                if audit is not None:
                    branch['update_audits'].append(dict(epoch=epoch + 1, batch_start=start, **audit))
            if epoch in (0, 1, 4, 9, 14):
                torch.manual_seed(config.seed + 333333)
                metrics = paired.snapshot_metrics(model, data, reference, endpoint=epoch == 14)
                with torch.no_grad():
                    h = paired.embed(model, xtest)
                    metrics['auxiliary_test_mse'] = float((model.aux(h.float()).squeeze(-1) - aux_test).square().mean())
                branch['trajectory'].append(dict(epoch=epoch + 1, **metrics))
        branch['minibatch_order_sha256'] = order_hash.hexdigest()
        assert branch['minibatch_order_sha256'] == original['minibatch_order_sha256']
        branch['final_model_sha256'] = paired.state_hash(model.state_dict())
        branch['final_optimizer_sha256'] = paired.state_hash(opt.state_dict())
        if (weight_state, moment_state) == ('warm', 'retained'):
            original_final = torch.load(source_states / 'joint_30.pt', map_location='cpu', weights_only=True)
            assert branch['final_model_sha256'] == paired.state_hash(original_final['model'])
            assert branch['final_optimizer_sha256'] == paired.state_hash(original_final['optimizer'])
            branch['archived_endpoint_exact_replay'] = True
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()),
                   destination / (weight_state + '_' + moment_state + '.pt'))
        result['branches'].append(branch)
        paired.save_json(args.out.with_suffix('.partial.json'), result)
        print(json.dumps(dict(weights=weight_state, moments=moment_state,
            frozen=branch['trajectory'][-1]['frozen']['acc'], trace=branch['trajectory'][-1]['geometry']['relative_trace'],
            elapsed_s=round(time.time() - started, 2))), flush=True)
    result.update(complete=True, elapsed_s=time.time() - started)
    paired.save_json(args.out, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
