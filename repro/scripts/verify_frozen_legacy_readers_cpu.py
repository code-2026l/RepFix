"""Replay fixed teacher heads and their convex stationarity without refitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit_protected_head_cpu import load_record
from consumer_risk import certificate
import paired_readout_continuation as paired


@torch.no_grad()
def run(args):
    audit=json.loads(args.audit.read_text())
    record,model,data,kind,states,A,B=load_record(args.record)
    assert audit['original_record_sha256']==hashlib.sha256(args.record.read_bytes()).hexdigest()
    x,y,_=data['test']
    teacher=paired.embed(model,x)@A.T+B
    train_teacher=(paired.embed(model,data['x'])@A.T+B).float().double()
    monitor=np.random.default_rng(6290001+record['seed']+(100000 if record['domain']=='har' else 0)).choice(len(y),min(1024,len(y)-1),replace=False)
    assert hashlib.sha256(monitor.astype(np.int64).tobytes()).hexdigest()==audit['new_monitor_sha256']
    result=dict(complete=False,device='cpu',cuda_visibility='',domain=record['domain'],seed=record['seed'],
        audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),branches=[])
    maximum=0.;gradient_max=0.
    def check(logits,expected):
        nonlocal maximum
        actual=paired.metrics(logits,y)
        for key,value in actual.items():
            difference=abs(value-expected['metrics'][key]);maximum=max(maximum,difference)
            assert difference<1e-7
        assert abs(float((logits.argmax(1)!=teacher.argmax(1)).double().mean())-expected['disagreement'])<1e-12
        bound=certificate(logits.argmax(1).numpy(),teacher.argmax(1).numpy(),monitor,.05/140)
        for key,value in bound.items():assert abs(value-expected['consumer_audit'][key])<1e-12
    directory=args.audit.parent/(args.audit.stem+'_readers')
    for source,donor in zip(record['branches'],audit['branches']):
        mode=source['mode'];assert mode==donor['mode']
        saved=torch.load(states/('%s_%g.pt'%(mode,source['lam'])),map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model']);model.eval()
        features=paired.embed(model,x)
        train=paired.embed(model,data['x'])
        mu=train.mean(0,keepdim=True);sd=train.std(0,keepdim=True).clamp_min(1e-6)
        standardized=(train-mu)/sd
        for method,item in donor['fitted_readers'].items():
            head=torch.load(directory/(mode+'_'+method+'.pt'),map_location='cpu',weights_only=True)
            assert paired.state_hash(head)==item['head_sha256']
            logits=F.linear(features,head['weight'],head['bias'])
            check(logits,item['test'])
            residual=F.linear(train,head['weight'],head['bias'])-train_teacher
            assert abs(float(residual.square().mean())-head['fitting_logit_mse'])<1e-8
            coefficient=head['weight'].T*sd.reshape(-1,1)
            gradient=standardized.T@residual/len(train)+head['ridge_coefficient']*coefficient
            error=max(float(gradient.abs().max()),float(residual.mean(0).abs().max()))
            assert error<1e-6,(mode,method,error)
            gradient_max=max(gradient_max,error)
            result['branches'].append(dict(mode=mode,method=method,metrics_and_bounds_replayed=True,
                stationarity_residual=error))
        if mode=='protected':
            p=saved['protected'];online=F.linear(features.float(),p['weight'],p['bias']).double()
            check(online,donor['online_head_fresh_audit'])
            result['branches'].append(dict(mode=mode,method='online',metrics_and_bounds_replayed=True))
    result.update(complete=True,maximum_metric_error=maximum,maximum_stationarity_residual=gradient_max,
                  refitting_performed=False)
    paired.save_json(args.out,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);run(a)
