"""Fixed cross-view auxiliary continuation of a saved healthy visual reference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
import vision_auxiliary_cpu as base
import nonlinear_suffix_cpu as factory
import late_vision_cpu as late
import paired_readout_continuation as paired


def contrastive(h0,h1,mean,sd):
    z=F.normalize((torch.cat([h0,h1])-mean)/sd,dim=1);n=len(h0)
    logits=z@z.T/.2;logits=logits.masked_fill(torch.eye(2*n,dtype=torch.bool),-torch.inf)
    targets=(torch.arange(2*n)+n)%(2*n)
    return F.cross_entropy(logits,targets)


@torch.no_grad()
def evaluate_aux(model,x,head):
    losses=[]
    for start in range(0,x.shape[1],128):
        n=min(128,x.shape[1]-start)
        if n<2:continue
        losses.append((n,float(contrastive(model(x[0,start:start+n]),model(x[1,start:start+n]),head.mean,head.sd))))
    return sum(n*v for n,v in losses)/sum(n for n,_ in losses)


def train(cache,ref,healthy,a):
    torch.manual_seed(a.seed);model=copy.deepcopy(healthy).requires_grad_(True)
    for layer in model.modules():
        if isinstance(layer,torch.nn.LayerNorm):layer.requires_grad_(False)
    head=base.initial_head(ref);opt=torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.requires_grad],lr=.00003),dict(params=head.parameters(),lr=.0001)],weight_decay=.0001)
    fit=ref['fitting'];val=ref['validation'];mask=ref['label_mask'];xx=cache['x'][:,fit];y=cache['y'][fit]
    rng=torch.Generator().manual_seed(a.seed+20000);vrng=torch.Generator().manual_seed(a.seed+71000);trace=[]
    for epoch in range(a.epochs):
        ce_sum=aux_sum=0.;batches=0
        for idx in torch.randperm(len(fit),generator=rng).split(128):
            view=torch.randint(2,(len(idx),),generator=vrng);h=model(xx[view,idx]);ce=base.primary_loss(head,h,y[idx],mask[idx])
            if a.primary_two_view:
                assert a.alpha==0
                other=model(xx[1-view,idx]);ce=(ce+base.primary_loss(head,other,y[idx],mask[idx]))/2
            aux=ce*0
            if a.alpha:
                other=model(xx[1-view,idx]);aux=contrastive(h,other,head.mean,head.sd)
            loss=ce+a.alpha*aux;assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();ce_sum+=float(ce.detach());aux_sum+=float(aux.detach());batches+=1
        trace.append(dict(epoch=epoch+1,primary=ce_sum/batches,auxiliary=aux_sum/batches))
        print(json.dumps(dict(epoch=epoch+1,alpha=a.alpha,reference=a.reference_seed,seed=a.seed)),flush=True)
    with torch.no_grad():
        train_h=base.embeds(model,xx[:,mask].flatten(0,1));reader=base.fit_affine(train_h,y[mask].repeat(2));validation_h=base.embeds(model,cache['x'][0,val]);z=late.linear(validation_h,reader)
        own=paired.metrics(z,cache['y'][val]);aux=evaluate_aux(model,cache['x'][:,val],head)
        old=late.linear(cache['h'][0,val],ref['heads']['coarse'])
    return model,head,reader,z,old,own,aux,trace


def main(a):
    start=time.time();cache=torch.load(a.population,map_location='cpu',weights_only=True,mmap=True)
    torch.manual_seed(a.reference_seed);ref=base.make_reference(cache,a.reference_seed)
    _,healthy=factory.build_for(cache);healthy.load_state_dict(cache['healthy']);healthy.eval().requires_grad_(False)
    model,head,reader,z,old,own,aux,trace=train(cache,ref,healthy,a)
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True);state=folder/'endpoint.pt'
    torch.save(dict(model=model.state_dict(),head=head.state_dict(),reader=reader,own_logits=z,old_logits=old,reference=ref),state)
    baseline=None
    if a.baseline_record:
        control=json.loads(a.baseline_record.read_text());assert (control['reference_seed'],control['seed'])==(a.reference_seed,a.seed)
        info=next(v for v in control['branches'] if v['mode']=='D')
        source=a.baseline_record.parent/(a.baseline_record.stem+'_states')/'D.pt';assert base.digest(source)==info['state_sha256']
        stored=torch.load(source,map_location='cpu',weights_only=True)
        _,detached=factory.build_for(cache);detached.load_state_dict(stored['model']);detached.eval().requires_grad_(False)
        dhead=base.initial_head(ref);dhead.load_state_dict(stored['head'])
        val=ref['validation']
        baseline=dict(state_sha256=base.digest(source),record_sha256=base.digest(a.baseline_record),
            accuracy=info['validation']['own']['acc'],contrastive_loss=evaluate_aux(detached,cache['x'][:,val],dhead))
    value=dict(complete=True,device='cpu',phase='contrastive_'+a.phase,alpha=a.alpha,reference_seed=a.reference_seed,seed=a.seed,
        plan_sha256=a.plan_sha256,source_sha256={Path(__file__).name:base.digest(Path(__file__)),**base.sources()},official_test_loaded=False,
        validation=own,contrastive_loss=aux,baseline=baseline,epochs=a.epochs,primary_two_view=a.primary_two_view,healthy_path='retained frozen prefix/suffix; cached output pending independent replay',trace=trace,
        state_sha256=base.digest(state),elapsed_s=time.time()-start,branches=[])
    base.save(a.out,value);print(json.dumps(dict(complete=True,alpha=a.alpha,accuracy=own['acc'],auxiliary_loss=aux)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--population',type=Path,required=True);p.add_argument('--alpha',type=float,required=True)
    p.add_argument('--reference-seed',type=int,default=6200);p.add_argument('--seed',type=int,default=6300);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=['development','confirmation','compute_control'],default='development');p.add_argument('--baseline-record',type=Path)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--primary-two-view',action='store_true')
    a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
