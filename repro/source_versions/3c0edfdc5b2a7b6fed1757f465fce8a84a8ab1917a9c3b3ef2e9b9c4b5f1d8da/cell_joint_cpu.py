"""Matched joint guidance with fixed consumer cells and strong point controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from readout_nullspace import masked_primary_loss
from consumer_cells import cell_spec,slice_spec
from consumer_contracts import confidence_thresholds
from recovery_readers import RecoveryReader,export_reader
from cell_recovery_cpu import reader_loss,score_reader

FAMILIES=['kl1','centered_mse','cell_kl','cell_logodds']
LAMBDAS=[.1,1.,10.,100.]
SOURCES=['cell_joint_cpu.py','cell_recovery_cpu.py','consumer_cells.py','recovery_readers.py',
    'consumer_contracts.py','benefit_compatibility_cpu.py','paired_readout_continuation.py','readout_nullspace.py']


def train_branch(model,data,kind,reference,moments,A,B,healthy,target,thresholds,mode,architecture,objective,lam,seed,epochs,dose):
    model.load_state_dict(reference);torch.manual_seed(seed+42000)
    opt=torch.optim.Adam(model.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(moments))
    initial_moments=paired.state_hash(opt.state_dict());extra=None;mapper=None
    if mode=='protected':
        extra=RecoveryReader(healthy.shape[1],A.shape[0],architecture,healthy.mean(0),healthy.std(0).clamp_min(1e-6),dict(weight=A,bias=B))
        opt.add_param_group(dict(params=extra.parameters(),lr=.001))
    elif mode=='feature_map':
        mapper=nn.Linear(A.shape[1],A.shape[1],bias=False)
        with torch.no_grad():mapper.weight.copy_(torch.eye(A.shape[1]))
        opt.add_param_group(dict(params=mapper.parameters(),lr=.001))
    elif mode not in ['detach','joint']:raise ValueError(mode)
    spec=cell_spec(target,list(thresholds.values()));rng=torch.Generator().manual_seed(seed+20000)
    order_hash=hashlib.sha256();trace=[];maximum_norm=0.;clipped=0;start=time.time()
    parameters=list(model.parameters())+(list(extra.parameters()) if extra is not None else [])+(list(mapper.parameters()) if mapper is not None else [])
    x,y,aux,mask=[data[k] for k in ['x','y','a','labeled']]
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(x),generator=rng);order_hash.update(order.numpy().tobytes())
        for j in range(0,len(order),256):
            idx=order[j:j+256];h=model(x[idx]);lp=masked_primary_loss(bc.current_logits(model,h),y[idx],mask[idx])
            la=paired.auxiliary_loss(model,h.detach().float() if mode=='detach' else h.float(),aux[idx],kind)
            if extra is not None:
                z=extra(h)
                penalty=F.mse_loss(z.double(),target[idx]) if objective=='raw_mse' else reader_loss(z,target[idx],slice_spec(spec,idx),objective)
            elif mapper is not None:penalty=F.mse_loss(mapper(h),healthy[idx].float())
            else:penalty=h.sum()*0
            loss=lp+dose*la+lam*penalty
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint consumer-cell objective')
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,5.));maximum_norm=max(maximum_norm,norm);clipped+=norm>5
            if not np.isfinite(norm):raise FloatingPointError('Nonfinite joint gradient norm')
            opt.step()
        if epoch==0 or (epoch+1)%10==0 or epoch+1==epochs:
            trace.append(dict(epoch=epoch+1,primary=float(lp.detach()),auxiliary=float(la.detach()),preservation=float(penalty.detach())))
    model.eval()
    if extra is not None:native=export_reader(extra)
    elif mapper is not None:native=dict(kind='affine',weight=A.float()@mapper.weight.detach(),bias=B.float())
    else:native=dict(kind='affine',weight=A.float(),bias=B.float())
    saved=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),reader=native,
        unfolded=None if extra is None else copy.deepcopy(extra.state_dict()),mapper=None if mapper is None else copy.deepcopy(mapper.state_dict()))
    row=dict(mode=mode,architecture=architecture,objective=objective,lam=lam,
        initial_model_sha256=paired.state_hash(reference),initial_optimizer_sha256=initial_moments,
        final_model_sha256=paired.state_hash(saved['model']),serving_reader_sha256=paired.state_hash(native),
        nonlinear_layers_changed={prefix:paired.state_hash({k:v for k,v in saved['model'].items() if k.startswith(prefix)})!=
            paired.state_hash({k:v for k,v in reference.items() if k.startswith(prefix)}) for prefix in ['encoder.0.','encoder.2.']},
        minibatch_order_sha256=order_hash.hexdigest(),trace=trace,maximum_unclipped_gradient_norm=maximum_norm,
        clipped_batches=clipped,elapsed_s=time.time()-start)
    return row,saved


def run(a):
    start=time.time();original=json.loads(a.record.read_text());assert original['complete'] and original['phase']=='development'
    domain,seed=original['domain'],original['seed'];torch.manual_seed(seed);np.random.seed(seed)
    model,data,kind,splits=bc.load(SimpleNamespace(domain=domain,seed=seed,phase='development',limit_fit=0,limit_eval=0))
    assert splits==original['splits'] and 'test' not in data
    folder=a.record.parent/(a.record.stem+'_states');ref=torch.load(folder/'reference.pt',map_location='cpu',weights_only=True)
    assert paired.state_hash(ref['model'])==original['reference_model_sha256']
    A=ref['head']['weight'].double()/ref['sd'].reshape(-1);B=ref['head']['bias'].double()-A@ref['mean'].reshape(-1)
    model.load_state_dict(ref['model']);model.eval();healthy=paired.embed(model,data['x'],batch=1024)
    target=healthy@A.T+B;oldval=paired.embed(model,data['validation'][0],batch=1024)@A.T+B;thresholds=confidence_thresholds(target)
    epochs=30 if domain=='har' else 25;dose=100. if domain=='har' else 1.
    configurations=[('detach','affine','none',0.),('joint','affine','none',0.),
        ('protected','affine','raw_mse',.1 if domain=='har' else 10.),('feature_map','affine','features',1. if domain=='har' else 100.)]
    configurations += [('protected',arch,obj,lam) for arch in ['affine','residual32'] for obj in FAMILIES for lam in LAMBDAS]
    if a.smoke:configurations=[('protected','residual32','cell_logodds',1.)];epochs=1
    out=dict(complete=False,device='cpu',cuda_visibility='',phase='development',route='joint_guidance',domain=domain,seed=seed,smoke=a.smoke,
        plan_sha256=a.plan_sha256,metric_addendum_plan_sha256=a.addendum_sha256,source_record_sha256=hashlib.sha256(a.record.read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in SOURCES},splits=splits,thresholds=thresholds,
        configuration=dict(epochs=epochs,dose=dose,batch=256,lr=.001,gradient_clip=5.,head_weight_decay=0.,
            reader_input_normalizer='fixed healthy fitting mean/std',reader_runtime_dtype='float32'),branches=[])
    destination=a.out.parent/(a.out.stem+'_states');destination.mkdir(exist_ok=True,parents=True)
    torch.save(dict(reference=ref,A=A,B=B,thresholds=thresholds),destination/'reference.pt')
    for mode,arch,obj,lam in configurations:
        row,saved=train_branch(model,data,kind,ref['model'],ref['optimizer'],A,B,healthy,target,thresholds,mode,arch,obj,lam,seed,epochs,dose)
        assert row['initial_optimizer_sha256']==paired.state_hash(ref['optimizer'])
        h=paired.embed(model,data['validation'][0],batch=1024)
        row.update(score_reader(model,h,data['validation'],oldval,thresholds,saved['reader']))
        row['auxiliary_mse']=float(paired.auxiliary_loss(model,h.float(),data['validation'][2],kind))
        name=mode+'_'+arch+'_'+obj+'_%g.pt'%lam;row['state_file']=name;torch.save(saved,destination/name)
        out['branches'].append(row);paired.save_json(a.out.with_suffix('.partial.json'),out)
        print(json.dumps(dict(domain=domain,seed=seed,mode=mode,architecture=arch,objective=obj,lam=lam,
            own=row['own']['acc'],changes={c:r['disagreement'] for c,r in row['contracts'].items()},elapsed_s=time.time()-start)),flush=True)
    assert len({r['minibatch_order_sha256'] for r in out['branches']})==1
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--addendum-sha256',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);run(a)
