"""Matched development: known feature alignment versus consumer-logit alignment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='4'

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from readout_nullspace import masked_primary_loss
from legacy_head_fit import fit_legacy_heads
from alignment_objectives import alignment_loss,folded_reader,reader_logits
from consumer_risk import certificate

GRIDS={'feature_map':[.01,.1,1.,10.,100.],'logit_map':[.1,1.,10.]}
SOURCES=['joint_alignment_cpu.py','alignment_objectives.py','legacy_head_fit.py',
         'benefit_compatibility_cpu.py','paired_readout_continuation.py',
         'readout_nullspace.py','readout_transfer_audit.py','consumer_risk.py']


def state_metadata(model,reference):
    return {prefix:paired.state_hash({k:v for k,v in model.state_dict().items() if k.startswith(prefix)})!=
            paired.state_hash({k:v for k,v in reference.items() if k.startswith(prefix)})
            for prefix in ['encoder.0.','encoder.2.']}


@torch.no_grad()
def evaluate(model,data,bases,A,B,kind,reader,phase,seed,safety_candidate=True):
    out={}
    for part in ['validation']+(['test'] if phase=='confirmation' else []):
        x,y,a=data[part];h=paired.embed(model,x)
        z=reader_logits(h,reader).double();old=bases[part]@A.T+B
        row=dict(own=paired.metrics(bc.current_logits(model,h),y),
                 legacy=paired.metrics(z,y),reference=paired.metrics(old,y),
                 fixed=paired.metrics(h@A.T+B,y),
                 disagreement=float((z.argmax(1)!=old.argmax(1)).double().mean()),
                 prediction_changes=int((z.argmax(1)!=old.argmax(1)).sum()),
                 max_healthy_logit_error=float((z-old).abs().max()),
                 auxiliary_mse=float(paired.auxiliary_loss(model,h.float(),a,kind)))
        if part=='test' and safety_candidate:
            n=len(y)
            monitor=np.random.default_rng(7390001+seed+(100000 if data['domain']=='har' else 0)).permutation(n)
            row['certificates']={}
            for count in [1024,2048]:
                indices=monitor[:min(count,n-1)]
                bound=certificate(z.argmax(1).numpy(),old.argmax(1).numpy(),indices,.05/160)
                bound['certified']=bound['upper_remaining_disagreement']<=.05
                bound['monitor_sha256']=hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest()
                row['certificates'][str(count)]=bound
        out[part]=row
    return out


def train_branch(model,data,kind,reference,moments,A,B,healthy_features,mode,lam,seed,epochs,dose):
    model.load_state_dict(reference)
    opt=torch.optim.Adam(model.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(moments))
    before=paired.state_hash(model.state_dict());before_moments=paired.state_hash(opt.state_dict())
    mapper=protected=None
    if mode in GRIDS:
        mapper=nn.Linear(A.shape[1],A.shape[1],bias=False)
        with torch.no_grad():mapper.weight.copy_(torch.eye(A.shape[1]))
        opt.add_param_group(dict(params=mapper.parameters(),lr=.001))
    elif mode=='protected':
        protected=nn.Linear(A.shape[1],A.shape[0])
        with torch.no_grad():protected.weight.copy_(A.float());protected.bias.copy_(B.float())
        opt.add_param_group(dict(params=protected.parameters(),lr=.001))
    elif mode not in ['detach','joint']:raise ValueError(mode)
    teacher=(healthy_features.double()@A.T+B).float()
    rng=torch.Generator().manual_seed(seed+20000);order_hash=hashlib.sha256()
    x,y,a,mask=[data[k] for k in ['x','y','a','labeled']]
    start=time.time();trace=[]
    for epoch in range(epochs):
        order=torch.randperm(len(x),generator=rng);order_hash.update(order.numpy().tobytes())
        model.train()
        for j in range(0,len(order),256):
            idx=order[j:j+256];h=model(x[idx])
            primary=masked_primary_loss(bc.current_logits(model,h),y[idx],mask[idx])
            aux=paired.auxiliary_loss(model,h.detach().float() if mode=='detach' else h.float(),a[idx],kind)
            if mapper is not None:
                penalty=alignment_loss(mapper(h.float()),healthy_features[idx].float(),A.float(),B.float(),mode,teacher[idx])
            elif protected is not None:penalty=F.mse_loss(protected(h.float()),teacher[idx])
            else:penalty=h.sum()*0
            loss=primary+dose*aux+lam*penalty
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint alignment loss')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if epoch==0 or (epoch+1)%5==0 or epoch+1==epochs:
            trace.append(dict(epoch=epoch+1,primary=float(primary.detach()),auxiliary=float(aux.detach()),alignment=float(penalty.detach())))
            print(json.dumps(dict(mode=mode,lam=lam,seed=seed,**trace[-1],elapsed_s=time.time()-start)),flush=True)
    model.eval()
    if mapper is not None:reader=folded_reader(mapper.weight.detach(),A,B)
    elif protected is not None:reader={k:v.detach().clone() for k,v in protected.state_dict().items()}
    else:reader=dict(weight=A.clone(),bias=B.clone())
    state=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
        mapper=None if mapper is None else copy.deepcopy(mapper.state_dict()),reader=reader)
    row=dict(mode=mode,lam=lam,initial_model_sha256=before,initial_optimizer_sha256=before_moments,
        final_model_sha256=paired.state_hash(state['model']),reader_sha256=paired.state_hash(reader),
        minibatch_order_sha256=order_hash.hexdigest(),nonlinear_layers_changed=state_metadata(model,reference),
        trace=trace,elapsed_s=time.time()-start,
        alignment_map_parameters=0 if mapper is None else mapper.weight.numel(),
        serving_reader_parameters=sum(v.numel() for v in reader.values()))
    return row,state


@torch.no_grad()
def add_ols(model,data,healthy_features,A,B):
    features=paired.embed(model,data['x'])
    target=(healthy_features.double()@A.T+B).float().double()
    fit=fit_legacy_heads(features,target)['ols']
    reader={k:fit[k].float() for k in ['weight','bias']}
    differences={}
    for part in ['validation']+(['test'] if 'test' in data else []):
        h=paired.embed(model,data[part][0]);z64=reader_logits(h,fit);z32=reader_logits(h,reader).double()
        differences[part]=dict(maximum_logit_error=float((z64-z32).abs().max()),
            float32_prediction_changes=int((z64.argmax(1)!=z32.argmax(1)).sum()))
    return reader,fit,differences


def development(args):
    start=time.time();record=json.loads(args.record.read_text())
    assert record['complete'] and record['phase']=='development'
    for name,expected in record['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==expected
    domain,seed=record['domain'],record['seed']
    torch.manual_seed(seed);np.random.seed(seed)
    cfg=SimpleNamespace(domain=domain,seed=seed,phase='development',limit_fit=0,limit_eval=0)
    model,data,kind,splits=bc.load(cfg);data['domain']=domain
    assert splits==record['splits'] and 'test' not in data
    states=args.record.parent/(args.record.stem+'_states')
    ref=torch.load(states/'reference.pt',map_location='cpu',weights_only=True)
    assert paired.state_hash(ref['model'])==record['reference_model_sha256']
    assert paired.state_hash(ref['optimizer'])==record['reference_optimizer_sha256']
    model.load_state_dict(ref['model']);model.eval()
    A=ref['head']['weight']/ref['sd'].reshape(-1)
    B=ref['head']['bias']-A@ref['mean'].reshape(-1)
    healthy=paired.embed(model,data['x'])
    bases={'validation':paired.embed(model,data['validation'][0])}
    out=dict(complete=False,device='cpu',cuda_visibility='',phase='development',domain=domain,seed=seed,
        plan_sha256=args.plan_sha256,source_record_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in SOURCES},
        splits=splits,original_controls={b['mode']:b['validation'] for b in record['branches'] if b['mode'] in ['detach','joint']},
        configuration=dict(grids=GRIDS,feature_objective_divisor=64,map_has_bias=False,epochs=record['configuration']['continuation_epochs'],
                           dose=record['configuration']['dose']),branches=[])
    directory=args.out.parent/(args.out.stem+'_states');directory.mkdir(exist_ok=True,parents=True)
    families=GRIDS if not args.smoke else {'feature_map':[.1],'logit_map':[.1]}
    if args.smoke:
        for key in ['x','y','a','labeled']:data[key]=data[key][:512]
        healthy=healthy[:512]
    for mode,lams in families.items():
        for lam in lams:
            row,state=train_branch(model,data,kind,ref['model'],ref['optimizer'],A,B,healthy,mode,lam,seed,
                1 if args.smoke else record['configuration']['continuation_epochs'],record['configuration']['dose'])
            row.update(evaluate(model,data,bases,A,B,kind,state['reader'],'development',seed))
            torch.save(state,directory/(mode+'_%g.pt'%lam));out['branches'].append(row)
            paired.save_json(args.out.with_suffix('.partial.json'),out)
    direct_lam=.1 if domain=='har' else 10.
    saved=torch.load(states/('protected_%g.pt'%direct_lam),map_location='cpu',weights_only=True)
    model.load_state_dict(saved['model']);model.eval()
    reader,fit,difference=add_ols(model,data,healthy,A,B)
    row=dict(mode='direct_ols',lam=direct_lam,precision_comparison=difference,
             reader_sha256=paired.state_hash(reader),final_model_sha256=paired.state_hash(model.state_dict()))
    row.update(evaluate(model,data,bases,A,B,kind,reader,'development',seed))
    torch.save(dict(model=model.state_dict(),reader=reader,ols64=fit),directory/'direct_ols.pt')
    out['branches'].append(row);out.update(complete=True,elapsed_s=time.time()-start,smoke=args.smoke)
    paired.save_json(args.out,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);development(a)
