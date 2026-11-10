"""Fixed standard distillation baseline for the selected visual suffixes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import paired_readout_continuation as paired
import late_vision_cpu as late
import nonlinear_suffix_cpu as study
from consumer_contracts import evaluate_contracts

@torch.no_grad()
def evaluate(update,reader,protected,part,ref):
    features=study.embed(update,part['x']);own=late.linear(features,reader);legacy=protected(features)
    reference=late.linear(part['h'],ref['heads']['coarse'])
    return dict(own=paired.metrics(own,part['y']),contracts=evaluate_contracts(legacy.double(),reference.double(),part['coarse'],ref['thresholds']),
        maximum_legacy_logit_error=float((legacy-reference).abs().max()))

def main(a):
    start=time.time();ref=torch.load(a.reference,map_location='cpu',weights_only=True)
    cache=torch.load(a.reference.with_name('cache.pt'),map_location='cpu',weights_only=True);train=cache['train'];part=cache['validation']
    baseline=json.loads(a.baseline.read_text());assert baseline['phase']=='development' and baseline['seed']==a.seed
    assert paired.state_hash(ref)==baseline['reference_state_sha256']
    _,healthy=study.build_for(ref);healthy.load_state_dict(ref['healthy']);healthy.requires_grad_(False)
    out=dict(complete=False,device='cpu',phase='development',arch=ref['arch'],seed=a.seed,weight=a.weight,plan_sha256=a.plan_sha256,
        reference_state_sha256=paired.state_hash(ref),baseline_record_sha256=hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in [Path(__file__).name,*study.NAMES]},
        splits=ref['splits'],healthy=baseline['healthy'],branches=[])
    update,head,groups=study.make_model(ref,healthy,'U',a.seed)
    old=ref['heads']['coarse'];protected=nn.Linear(old['weight'].shape[1],old['weight'].shape[0])
    with torch.no_grad():protected.weight.copy_(old['weight']);protected.bias.copy_(old['bias'])
    groups.append(dict(params=protected.parameters(),lr=.0001,weight_decay=0.))
    optimizer=torch.optim.AdamW(groups,weight_decay=.0001)
    rng=torch.Generator().manual_seed(a.seed+20000);vrng=torch.Generator().manual_seed(a.seed+71000)
    mask=ref['labeled'];lookup=torch.full((len(mask),),-1,dtype=torch.long);lookup[mask]=torch.arange(int(mask.sum()));trace=[]
    for epoch in range(10):
        for jdx in torch.randperm(len(mask),generator=rng).split(128):
            idx=lookup[jdx];idx=idx[idx>=0]
            if not len(idx):continue
            views=torch.randint(2,(len(idx),),generator=vrng);h=update(train['x'][views,idx]);z=head(h)
            target=late.linear(train['h'][views,idx],old).detach()
            kl=F.kl_div(protected(h).log_softmax(1),target.softmax(1),reduction='batchmean')
            loss=F.cross_entropy(z,train['y'][idx])+.0001*sum(p.square().sum() for p in head.parameters())+a.weight*kl
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite protected-head objective')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in groups for p in g['params']):raise FloatingPointError('Nonfinite protected-head gradient')
            optimizer.step()
        trace.append(dict(epoch=epoch+1,loss=float(loss.detach()),kl=float(kl.detach())))
        print(json.dumps(dict(arch=ref['arch'],seed=a.seed,weight=a.weight,epoch=epoch+1)),flush=True)
    with torch.no_grad():
        online=paired.metrics(head(study.embed(update,part['x'])),part['y'])
        reader=study.fit_reader(study.embed(update,train['x'].flatten(0,1)),train['y'].repeat(2))
        result=evaluate(update,reader,protected,part,ref)
    controls={b['mode']:b['validation']['own']['acc'] for b in baseline['branches']};own=result['own']['acc']
    utility=own>=baseline['healthy']['acc']+.02-1e-12 and own>=controls['F']+.02-1e-12 and all(own>=controls[m]-.005-1e-12 for m in ['U','N','H'])
    interface=all(v['disagreement']<=.01+1e-12 for v in result['contracts'].values())
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    state=dict(mode='protected',update=update.state_dict(),head=head.state_dict(),reader=reader,protected=protected.state_dict())
    torch.save(state,folder/'protected.pt')
    out['branches']=[dict(mode='protected',trace=trace,validation=result,online_own=online,utility_gate=utility,consumer_gate=interface,passes=utility and interface,
        extra_parameters=sum(p.numel() for p in protected.parameters()),state_sha256=paired.state_hash(state),state_file='protected.pt')]
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)
    print(json.dumps(dict(arch=ref['arch'],weight=a.weight,accuracy=own,utility=utility,interface=interface)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--weight',type=float,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
