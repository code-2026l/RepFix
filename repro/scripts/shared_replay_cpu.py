"""Prospective shared-reference replay: numerical development, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from readout_nullspace import masked_primary_loss
from consumer_contracts import confidence_thresholds,evaluate_contracts
from shared_reference_replay import SharedReferenceMLP,serve_mlp,merged_serving,serve_merged_mlp

CONFIGURATIONS=[(0,4),(4,4),(8,8),(16,8)]
SOURCES=['shared_replay_cpu.py','shared_reference_replay.py','consumer_contracts.py',
         'benefit_compatibility_cpu.py','paired_readout_continuation.py','readout_nullspace.py']


@torch.no_grad()
def cache(healthy,x):
    first=[];second=[]
    for j in range(0,len(x),1024):
        p=healthy.encoder[0](x[j:j+1024]);first.append(p);second.append(healthy.encoder[2](F.relu(p)))
    return torch.cat(first),torch.cat(second)


def calibration(healthy,data,kind,dose,seed):
    idx=torch.randperm(len(data['x']),generator=torch.Generator().manual_seed(seed+51000))[:1024]
    h=healthy(data['x'][idx]);lp=masked_primary_loss(bc.current_logits(healthy,h),data['y'][idx],data['labeled'][idx])
    loss=lp+dose*paired.auxiliary_loss(healthy,h.float(),data['a'][idx],kind)
    g1,gb,g2=torch.autograd.grad(loss,[healthy.encoder[0].weight,healthy.encoder[0].bias,healthy.encoder[2].weight])
    importance=g1.square().sum(1)+gb.square();support=importance.argsort(descending=True,stable=True)
    right=torch.linalg.svd(g2.double(),full_matrices=False).Vh
    return support,right,dict(indices_sha256=paired.state_hash(idx),importance=importance.detach().tolist(),
        support_order=support.tolist(),second_gradient_sha256=paired.state_hash(g2),right_vectors_sha256=paired.state_hash(right))


@torch.no_grad()
def outputs(x,p,merged=False):
    if merged:return torch.cat([serve_merged_mlp(x[j:j+1024],p) for j in range(0,len(x),1024)])
    values=[serve_mlp(x[j:j+1024],p) for j in range(0,len(x),1024)]
    return torch.cat([v[0] for v in values]),torch.cat([v[1] for v in values])


@torch.no_grad()
def score(model,data,kind,cached,old,thresholds):
    x,y,a=data;p=model.serving();own,legacy=outputs(x,p);merged=outputs(x,merged_serving(p),True)
    _,old2=model.forward_states(x,cached)
    full=model.legacy_logits(old2,False);algebraic=model.legacy_logits(old2,True)
    features=model(x,cached)
    return dict(own=paired.metrics(own,y),auxiliary_mse=float(paired.auxiliary_loss(model,features.float(),a,kind)),
        contracts=evaluate_contracts(legacy.double(),old,y,thresholds),
        full_reference_contracts=evaluate_contracts(full,old,y,thresholds),
        full_reference_maximum_logit_error=float((full-old).abs().max()),
        compressed_maximum_logit_error=float((legacy.double()-old).abs().max()),
        affine_compression_float64_error=float((algebraic-old).abs().max()),
        current_merge_maximum_logit_error=float((own-merged).abs().max()),
        current_merge_prediction_changes=int((own.argmax(1)!=merged.argmax(1)).sum()))


def train(healthy,moments,data,kind,caches,A,B,support,right,k,r,mode,seed,epochs,dose):
    model=SharedReferenceMLP(healthy,support[:k],right[:r],A,B)
    opt=torch.optim.Adam(model.old_parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(moments))
    original_moment_sha=paired.state_hash(opt.state_dict());opt.add_param_group(dict(params=model.update_parameters(),lr=.001))
    rng=torch.Generator().manual_seed(seed+20000);order_sha=hashlib.sha256();trace=[];start=time.time()
    for epoch in range(epochs):
        order=torch.randperm(len(data['x']),generator=rng);order_sha.update(order.numpy().tobytes())
        for j in range(0,len(order),256):
            idx=order[j:j+256];h=model(data['x'][idx],tuple(v[idx] for v in caches))
            primary=masked_primary_loss(bc.current_logits(model,h),data['y'][idx],data['labeled'][idx])
            auxiliary=paired.auxiliary_loss(model,h.detach().float() if mode=='detach' else h.float(),data['a'][idx],kind)
            loss=primary+dose*auxiliary
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite shared-replay objective')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if epoch==0 or (epoch+1)%10==0 or epoch+1==epochs:
            trace.append(dict(epoch=epoch+1,primary=float(primary.detach()),auxiliary=float(auxiliary.detach())))
    row=dict(mode=mode,k=k,rank=r,trace=trace,initial_optimizer_sha256=original_moment_sha,
        minibatch_order_sha256=order_sha.hexdigest(),elapsed_s=time.time()-start,
        final_state_sha256=paired.state_hash(model.state_dict()),serving_sha256=paired.state_hash(model.serving()),
        first_delta_norm=float(model.delta1.detach().norm()),second_delta_norm=float((model.u@model.v).detach().norm()))
    return model,opt,row


def run(a):
    start=time.time();original=json.loads(a.record.read_text());assert original['complete'] and original['phase']=='development'
    domain,seed=original['domain'],original['seed'];torch.manual_seed(seed);np.random.seed(seed)
    healthy,data,kind,splits=bc.load(SimpleNamespace(domain=domain,seed=seed,phase='development',limit_fit=0,limit_eval=0))
    assert splits==original['splits'] and 'test' not in data
    folder=a.record.parent/(a.record.stem+'_states');ref=torch.load(folder/'reference.pt',map_location='cpu',weights_only=True)
    assert paired.state_hash(ref['model'])==original['reference_model_sha256']
    healthy.load_state_dict(ref['model']);healthy.eval()
    A=ref['head']['weight'].double()/ref['sd'].reshape(-1);B=ref['head']['bias'].double()-A@ref['mean'].reshape(-1)
    fitting_cache=cache(healthy,data['x']);validation_cache=cache(healthy,data['validation'][0])
    oldfit=F.linear(healthy.encoder[4](F.relu(fitting_cache[1])).double(),A,B).detach()
    oldval=F.linear(healthy.encoder[4](F.relu(validation_cache[1])).double(),A,B).detach()
    thresholds=confidence_thresholds(oldfit);epochs=30 if domain=='har' else 25;dose=100. if domain=='har' else 1.
    support,right,selection=calibration(healthy,data,kind,dose,seed)
    out=dict(complete=False,device='cpu',cuda_visibility='',phase='development',domain=domain,seed=seed,smoke=a.smoke,
        plan_sha256=a.plan_sha256,source_record_sha256=hashlib.sha256(a.record.read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in SOURCES},
        splits=splits,thresholds=thresholds,support_calibration=selection,reference_model_sha256=paired.state_hash(ref['model']),
        configuration=dict(epochs=epochs,dose=dose,batch=256,lr=.001,gradient_clip=None,threads=4,
            configurations=CONFIGURATIONS,legacy_reference_dtype='float64',compressed_serving_dtype='float32'),full_controls={},branches=[])
    for mode in ['detach','joint']:
        donor=torch.load(folder/(mode+'_0.pt'),map_location='cpu',weights_only=True)
        full=copy.deepcopy(healthy);full.load_state_dict(donor['model']);full.eval()
        h2=F.relu(cache(full,data['validation'][0])[1]);w=(full.primary.weight.double()@full.encoder[4].weight.double()).float()
        b=(full.primary.bias.double()+full.primary.weight.double()@full.encoder[4].bias.double()).float()
        out['full_controls'][mode]=dict(own=paired.metrics(F.linear(h2,w,b),data['validation'][1]),
            original_diagnostic=next(r['validation']['own'] for r in original['branches'] if r['mode']==mode),
            model_sha256=paired.state_hash(donor['model']))
    destination=a.out.parent/(a.out.stem+'_states');destination.mkdir(exist_ok=True,parents=True)
    torch.save(dict(reference=ref,A=A,B=B,thresholds=thresholds,support=support,right_vectors=right),destination/'reference.pt')
    for k,r in ([(4,4)] if a.smoke else CONFIGURATIONS):
        for mode in ['detach','joint']:
            model,opt,row=train(healthy,ref['optimizer'],data,kind,fitting_cache,A,B,support,right,k,r,mode,seed,1 if a.smoke else epochs,dose)
            assert row['initial_optimizer_sha256']==paired.state_hash(ref['optimizer'])
            model.eval();row.update(score(model,data['validation'],kind,validation_cache,oldval,thresholds))
            row['extra_macs']=k*healthy.encoder[0].in_features+k*64+2*r*64+len(B)*64
            name='%s_k%d_r%d.pt'%(mode,k,r);row['state_file']=name
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),serving=model.serving()),destination/name)
            out['branches'].append(row);paired.save_json(a.out.with_suffix('.partial.json'),out)
            print(json.dumps(dict(domain=domain,seed=seed,k=k,rank=r,mode=mode,own=row['own']['acc'],mse=row['auxiliary_mse'],
                changes={c:x['disagreement'] for c,x in row['contracts'].items()},elapsed_s=time.time()-start)),flush=True)
    assert len({r['minibatch_order_sha256'] for r in out['branches']})==1
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);run(a)
