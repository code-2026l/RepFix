"""Independent tensor-only CPU replay of the nonlinear-channel confirmation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import nonlinear_interface_cpu as experiment
import paired_readout_continuation as paired
from consumer_risk import certificate


def dense(state, x, trunk='trunk.', final='final.', weight=None, bias=None):
    h = F.relu(F.linear(x.double(),state[trunk+'0.weight'],state[trunk+'0.bias']))
    h = F.relu(F.linear(h,state[trunk+'2.weight'],state[trunk+'2.bias']))
    return F.linear(h,state[final+'weight'] if weight is None else weight,
                    state[final+'bias'] if bias is None else bias)


def batched(operation, x):
    return torch.cat([operation(x[j:j+2048]) for j in range(0,len(x),2048)])


@torch.no_grad()
def run(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    record = json.loads(args.record.read_text())
    assert record['complete'] and record['phase'] in ['development','confirmation']
    torch.manual_seed(record['seed']);np.random.seed(record['seed'])
    cfg = SimpleNamespace(domain=record['domain'],seed=record['seed'],phase=record['phase'],
                          labels=record['splits']['primary_label_fraction'],limit_fit=0,limit_eval=0)
    _,data,kind,splits = experiment.load(cfg)
    assert splits==record['splits']
    states = args.record.parent/(args.record.stem+'_states')
    ref = torch.load(states/'reference.pt',map_location='cpu',weights_only=True)
    assert paired.state_hash(ref['model'])==record['reference_model_sha256']
    assert paired.state_hash(ref['optimizer'])==record['reference_optimizer_sha256']
    part = 'test' if record['phase']=='confirmation' else 'validation'
    x,y,auxiliary = data[part]
    h0 = batched(lambda values:dense(ref['model'],values),x)
    w,b,mu,sd = ref['head']['weight'],ref['head']['bias'],ref['mean'],ref['sd']
    z0 = F.linear((h0-mu)/sd,w,b)
    monitor = None
    if record['phase']=='confirmation':
        monitor = np.random.default_rng(3890001+record['seed']).choice(len(y),1024,replace=False)
        assert hashlib.sha256(monitor.astype(np.int64).tobytes()).hexdigest()==record['consumer_bank']['monitor_membership_sha256']
    result = dict(complete=False,device='cpu',cuda_visibility='',torch_threads=1,
        seed=record['seed'],domain=record['domain'],scored_corpus=part,
        record_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),branches=[])
    for row in record['branches']:
        mode,key = row['mode'],row['key']
        saved = torch.load(states/(key+'.pt'),map_location='cpu',weights_only=True)
        raw = saved['model']
        assert paired.state_hash(raw)==row['final_model_sha256']
        assert paired.state_hash(saved['optimizer'])==row['final_optimizer_sha256']
        if mode[0] in 'TU':
            raw_h = batched(lambda values:dense(raw,values,'base_trunk.','base_final.')+
                dense(raw,values,'side_trunk.','side_final.') @ raw['projector'].T,x)
        elif mode[0]=='C':
            wf = raw['anchor_weight']+raw['projector']@(raw['final.weight']-raw['anchor_weight'])
            bf = raw['anchor_bias']+raw['projector']@(raw['final.bias']-raw['anchor_bias'])
            raw_h = batched(lambda values:dense(raw,values,weight=wf,bias=bf),x)
        else:
            raw_h = batched(lambda values:dense(raw,values),x)
            if mode[0]=='G':raw_h=h0+(raw_h-h0)@ref['projector']
        h = raw_h
        fusion_error = 0.
        if mode[0] in 'TUC':
            deployed = torch.load(states/(key+'_deployed.pt'),map_location='cpu',weights_only=True)['model']
            assert paired.state_hash(deployed)==row['deployed_model_sha256']
            assert not any('base_' in name or 'side_' in name or 'projector' in name for name in deployed)
            if mode[0] in 'TU':
                base_width = raw['base_final.weight'].shape[1]
                assert torch.equal(deployed['trunk.0.weight'],torch.cat([raw['base_trunk.0.weight'],raw['side_trunk.0.weight']]))
                assert torch.equal(deployed['trunk.2.weight'],torch.block_diag(raw['base_trunk.2.weight'],raw['side_trunk.2.weight']))
                assert torch.equal(deployed['final.weight'],torch.cat([raw['base_final.weight'],raw['projector']@raw['side_final.weight']],dim=1))
                assert torch.equal(deployed['final.bias'],raw['base_final.bias']+raw['projector']@raw['side_final.bias'])
                assert torch.count_nonzero(deployed['trunk.2.weight'][:base_width,base_width:])==0
                assert torch.count_nonzero(deployed['trunk.2.weight'][base_width:,:base_width])==0
            h = batched(lambda values:dense(deployed,values),x)
            fusion_error = float((h-raw_h).abs().max())
            assert fusion_error<1e-10
        z = F.linear((h-mu)/sd,w,b)
        own = F.linear(h,raw['primary.weight'].double(),raw['primary.bias'].double())
        actual = dict(frozen=paired.metrics(z,y),own=paired.metrics(own,y))
        maximum = 0.
        for head in ['frozen','own']:
            for metric in ['acc','nll','ece']:
                error = abs(actual[head][metric]-row[part][head][metric])
                assert error<1e-7,(mode,head,metric,error)
                maximum=max(maximum,error)
        if kind=='snr':
            prediction = F.linear(h.float(),raw['aux.weight'],raw['aux.bias']).squeeze(-1)
        else:
            prediction = F.linear(F.relu(F.linear(h.float(),raw['aux.0.weight'],raw['aux.0.bias'])),
                                  raw['aux.2.weight'],raw['aux.2.bias'])
        mse = float(F.mse_loss(prediction,auxiliary))
        assert abs(mse-row[part]['auxiliary_mse'])<1e-7
        changes = int((z.argmax(1)!=z0.argmax(1)).sum())
        assert changes==row[part]['prediction_changes']
        if monitor is not None:
            bound = certificate(z.argmax(1).numpy(),z0.argmax(1).numpy(),monitor,.05/200)
            for name,value in bound.items():
                assert abs(value-row[part]['consumer_audit'][name])<1e-12
        result['branches'].append(dict(mode=mode,key=key,all_scores_replayed=True,bounds_replayed=monitor is not None,
            maximum_metric_error=maximum,fusion_max_absolute_feature_error=fusion_error,
            prediction_changes=changes,max_absolute_healthy_logit_error=float((z-z0).abs().max()),
            tensor_only_feature_and_fusion_reconstruction=True))
    result['complete']=True
    paired.save_json(args.out,result)
    print(json.dumps(dict(seed=record['seed'],domain=record['domain'],saved_consumers_replayed=len(record['branches']),corpus=part)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);run(parser.parse_args())
