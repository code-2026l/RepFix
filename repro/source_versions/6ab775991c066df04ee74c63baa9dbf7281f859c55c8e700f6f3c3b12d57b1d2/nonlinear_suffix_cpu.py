"""Matched fixed transfer recipe for a complete nonlinear suffix."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import paired_readout_continuation as paired
import late_vision_cpu as late
from nonlinear_suffix import build_full
from shared_reference_vision import NormalizedHead
from late_replay import ResidualHead
from readout_transfer_audit import fit_head
from consumer_contracts import evaluate_contracts

NAMES=['nonlinear_suffix_cpu.py','nonlinear_suffix.py','late_vision_cpu.py','late_replay.py','shared_reference_vision.py','readout_transfer_audit.py','consumer_contracts.py']
def sources():return {n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in NAMES}

def augment(images):
    generator=torch.Generator().manual_seed(3901);n=len(images);offsets=torch.randint(0,9,(n,2),generator=generator);flips=torch.rand(n,generator=generator)<.5
    padded=F.pad(images.permute(0,3,1,2).float(),(4,4,4,4),mode='reflect');out=torch.empty_like(images)
    for i in range(n):
        y,x=offsets[i].tolist();crop=padded[i,:,y:y+32,x:x+32]
        if flips[i]:crop=crop.flip(-1)
        out[i]=crop.permute(1,2,0).to(torch.uint8)
    return out,dict(offsets_sha256=paired.state_hash(offsets),flips_sha256=paired.state_hash(flips))

@torch.no_grad()
def collect(prefix,images,size):
    values=[]
    for j in range(0,len(images),128):values.append(prefix(late.preprocess(images[j:j+128],size)).contiguous())
    return torch.cat(values)

@torch.no_grad()
def embed(model,x):return torch.cat([model(x[j:j+256]) for j in range(0,len(x),256)])

@torch.no_grad()
def richer(model,x):
    return torch.cat([torch.cat([model(x[j:j+256]),late.pooled(model.hidden(x[j:j+256]))],1) for j in range(0,len(x),256)])

def fit_reader(h,y):
    clf,mean,sd=fit_head(h.double(),y);return NormalizedHead(clf.weight.detach(),clf.bias.detach(),mean,sd).affine()

def prepare(a):
    start=time.time();torch.manual_seed(3800);old=torch.load(a.reference,map_location='cpu',weights_only=True);prior=torch.load(a.reference.with_name('cache.pt'),map_location='cpu',weights_only=True)
    prefix,healthy=build_full(old['arch']);healthy.eval();images,y,coarse,sha=late.images(True);assert sha==old['data_sha256']
    idx=old['fitting'][old['labeled']];aug,aug_info=augment(images[idx]);x0=prior['train']['residual'];x1=collect(prefix,aug,old['metadata']['image_size'])
    xx=torch.stack([x0,x1]);h0=embed(healthy,x0);h1=embed(healthy,x1);hh=torch.stack([h0,h1]);labels=prior['train']['y'];n=len(labels)
    values=hh.flatten(0,1);mean=values.mean(0);sd=values.std(0,unbiased=False);floor=.1*sd[sd>0].median();sd=sd.clamp_min(floor)
    affine=old['heads']['fine'];head=NormalizedHead(affine['weight']*sd,affine['bias']+affine['weight']@mean,mean,sd)
    train=dict(x=xx,h=hh,y=labels);part=dict(x=prior['validation']['residual'],h=prior['validation']['h'],y=prior['validation']['y'],coarse=prior['validation']['coarse'])
    probe=fit_reader(torch.cat([richer(healthy,x0),richer(healthy,x1)]),labels.repeat(2))
    subset=torch.randperm(n,generator=torch.Generator().manual_seed(54800))[:1024]
    loss=F.cross_entropy(head(healthy(x0[subset])),labels[subset]);gradients=torch.autograd.grad(loss,[m.weight for m in healthy.maps])
    right=[torch.linalg.svd(g.flatten(1).double(),full_matrices=False).Vh[:64].float() for g in gradients]
    healthy.requires_grad_(False)
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    reference=dict(arch=old['arch'],healthy=healthy.state_dict(),head=head.state_dict(),heads=old['heads'],thresholds=old['thresholds'],
        hidden_probe=probe,right=right,labeled=old['labeled'],fitting=old['fitting'],validation_indices=old['validation_indices'],metadata=old['metadata'],
        splits=old['splits'],plan_sha256=a.plan_sha256,source_reference_sha256=paired.state_hash(old))
    torch.save(reference,folder/'reference.pt');torch.save(dict(train=train,validation=part),folder/'cache.pt')
    out=dict(complete=True,device='cpu',phase='nonlinear_prepare',arch=old['arch'],plan_sha256=a.plan_sha256,source_sha256=sources(),
        reference_state_sha256=paired.state_hash(reference),source_reference_sha256=paired.state_hash(old),splits=old['splits'],augmentation=aug_info,
        normalization_floor=float(floor),initial_current_function_error=float((late.linear(h0,old['heads']['fine'])-head(h0)).abs().max()),
        healthy=paired.metrics(late.linear(part['h'],old['heads']['fine']),part['y']),elapsed_s=time.time()-start,branches=[])
    paired.save_json(a.out,out);print(json.dumps(dict(arch=old['arch'],stage='prepared_full_suffix',elapsed_s=time.time()-start)),flush=True)

def init_head(ref):
    h=ref['head'];head=NormalizedHead(h['linear.weight'],h['linear.bias'],h['mean'],h['sd']);head.load_state_dict(h);return head

def make_model(ref,healthy,mode,seed):
    torch.manual_seed(seed);head=init_head(ref);update=None
    if mode=='N':head=ResidualHead(head)
    elif mode in ['L','U']:
        update=copy.deepcopy(healthy).requires_grad_(True)
        update.norm1.requires_grad_(False);update.norm2.requires_grad_(False)
        if mode=='L':
            update.requires_grad_(False);update.maps[-1].requires_grad_(True)
    elif mode=='R64':update=healthy.lowrank(ref['right'],64)
    groups=[dict(params=head.parameters(),lr=.0001)]
    if update is not None:groups.append(dict(params=[p for p in update.parameters() if p.requires_grad],lr=.00003))
    return update,head,groups

@torch.no_grad()
def current_features(update,head,mode,x,h):
    if mode=='N':return torch.cat([h,F.gelu(head.hidden((h-head.base.mean)/head.base.sd))],1)
    return h if update is None else embed(update,x)

@torch.no_grad()
def score(update,head,mode,part,healthy,ref,reader,monitor=None):
    old=embed(healthy,part['x']);features=current_features(update,head,mode,part['x'],old)
    z=late.linear(features,reader);legacy=late.linear(old,ref['heads']['coarse']);reference=late.linear(part['h'],ref['heads']['coarse'])
    return dict(own=paired.metrics(z,part['y']),contracts=evaluate_contracts(legacy.double(),reference.double(),part['coarse'],ref['thresholds'],monitor,.05/2000),
        maximum_legacy_logit_error=float((legacy-reference).abs().max()),maximum_feature_change=float((features[:,:old.shape[1]]-old).abs().max()))

def branch(a):
    start=time.time();prepared=a.reference.parent.parent/(a.reference.parent.name.removesuffix('_states')+'.json')
    while not prepared.exists():
        if time.time()-start>1800:raise TimeoutError('Nonlinear suffix preparation did not complete')
        time.sleep(2)
    assert json.loads(prepared.read_text())['complete']
    ref=torch.load(a.reference,map_location='cpu',weights_only=True);cached=torch.load(a.reference.with_name('cache.pt'),map_location='cpu',weights_only=True)
    assert ref['plan_sha256']==a.plan_sha256
    _,healthy=build_full(ref['arch']);healthy.load_state_dict(ref['healthy']);healthy.requires_grad_(False);train=cached['train'];part=cached['validation']
    modes=['F','N','H','L','U','R64'];phase='development';monitor=None
    if a.choice is not None:
        selected=json.loads(a.choice.read_text())['backbones'][ref['arch']]['selected'];assert selected in ['U','R64'];phase='confirmation'
        test=torch.load(a.test_cache,map_location='cpu',weights_only=True);assert test['reference_state_sha256']==paired.state_hash(ref);part=test['test']
        monitor=np.random.default_rng(13700001+a.seed).choice(len(part['y']),1024,replace=False)
        modes=list(dict.fromkeys(['F','N','H','L','U',selected]))
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    out=dict(complete=False,device='cpu',phase=phase,arch=ref['arch'],seed=a.seed,plan_sha256=a.plan_sha256,source_sha256=sources(),
        reference_state_sha256=paired.state_hash(ref),selection_sha256=None if a.choice is None else hashlib.sha256(a.choice.read_bytes()).hexdigest(),
        splits=dict(ref['splits'],official_test_loaded=phase=='confirmation',test_metrics_computed=phase=='confirmation'),
        configuration=dict(epochs=10,batch=128,head_lr=.0001,update_lr=.00003,head_l2=.0001,adamw_decay=.0001,views=2,threads=12),
        healthy=paired.metrics(late.linear(part['h'],ref['heads']['fine']),part['y']),branches=[])
    for mode in modes:
        if mode=='H':
            features=richer(healthy,part['x']);reader=ref['hidden_probe'];row=dict(mode=mode,validation=dict(own=paired.metrics(late.linear(features,reader),part['y'])))
            state=dict(mode=mode,reader=reader)
        else:
            update,head,groups=make_model(ref,healthy,mode,a.seed);optimizer=torch.optim.AdamW(groups,weight_decay=.0001)
            rng=torch.Generator().manual_seed(a.seed+20000);vrng=torch.Generator().manual_seed(a.seed+71000);order_sha=hashlib.sha256();view_sha=hashlib.sha256()
            mask=ref['labeled'];lookup=torch.full((len(mask),),-1,dtype=torch.long);lookup[mask]=torch.arange(int(mask.sum()));trace=[]
            for epoch in range(10):
                order=torch.randperm(len(mask),generator=rng);order_sha.update(order.numpy().tobytes())
                for j in range(0,len(order),128):
                    idx=lookup[order[j:j+128]];idx=idx[idx>=0]
                    if not len(idx):continue
                    views=torch.randint(2,(len(idx),),generator=vrng);view_sha.update(views.numpy().tobytes())
                    h=train['h'][views,idx] if update is None else update(train['x'][views,idx]);z=head(h)
                    loss=F.cross_entropy(z,train['y'][idx])+.0001*sum(p.square().sum() for p in head.parameters())
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-suffix objective')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in groups for p in g['params']):raise FloatingPointError('Nonfinite full-suffix gradient')
                    optimizer.step()
                trace.append(dict(epoch=epoch+1,loss=float(loss.detach())));print(json.dumps(dict(arch=ref['arch'],seed=a.seed,mode=mode,epoch=epoch+1)),flush=True)
            with torch.no_grad():
                online_features=embed(healthy if update is None else update,part['x'])
                online_own=paired.metrics(head(online_features),part['y'])
            all_x=train['x'].flatten(0,1);all_h=train['h'].flatten(0,1)
            fit_features=current_features(update,head,mode,all_x,all_h);reader=fit_reader(fit_features,train['y'].repeat(2))
            row=dict(mode=mode,trace=trace,minibatch_order_sha256=order_sha.hexdigest(),view_order_sha256=view_sha.hexdigest(),
                validation=score(update,head,mode,part,healthy,ref,reader,monitor),online_own=online_own,
                trainable_parameters=sum(p.numel() for g in groups for p in g['params']))
            state=dict(mode=mode,head=head.state_dict(),reader=reader)
            if update is not None:
                if mode=='R64':state['update']={k:v for k,v in update.state_dict().items() if k.endswith(('.u','.v','.delta_bias'))}
                elif mode=='L':state['update']={k:v for k,v in update.state_dict().items() if k.startswith('maps.%d.'%(len(update.maps)-1))}
                else:state['update']=update.state_dict()
        if phase=='confirmation':row['test']=row.pop('validation')
        row['state_file']=mode+'.pt';row['state_sha256']=paired.state_hash(state);torch.save(state,folder/row['state_file']);out['branches'].append(row)
        paired.save_json(a.out.with_suffix('.partial.json'),out)
        print(json.dumps(dict(stage='branch_complete',arch=ref['arch'],mode=mode,acc=row.get('test',row.get('validation'))['own']['acc'],elapsed_s=time.time()-start)),flush=True)
    assert len({b['minibatch_order_sha256'] for b in out['branches'] if 'minibatch_order_sha256' in b})==1
    assert len({b['view_order_sha256'] for b in out['branches'] if 'view_order_sha256' in b})==1
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)

@torch.no_grad()
def prepare_test(a):
    ref=torch.load(a.reference,map_location='cpu',weights_only=True);choice=json.loads(a.choice.read_text())
    assert ref['plan_sha256']==a.plan_sha256 and choice['backbones'][ref['arch']]['selected'] in ['U','R64']
    original=torch.load(a.original_reference,map_location='cpu',weights_only=True)
    assert paired.state_hash(original)==ref['source_reference_sha256']
    images,y,coarse,sha=late.images(False)
    # Keep the pre-existing reference arithmetic as the comparison, rather than
    # comparing the folded suffix with itself.
    prefix,tail,norm,_=late.build_backbone(ref['arch'],True)
    tail.load_state_dict(original['tail']);norm.load_state_dict(original['norm'])
    values=late.collect(prefix,tail,norm,images,torch.arange(len(images)),ref['metadata']['image_size'],torch.ones(len(images),dtype=torch.bool))
    part=dict(x=values['residual'],h=values['h'],y=y,coarse=coarse)
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    cache=dict(test=part,reference_state_sha256=paired.state_hash(ref),data_sha256=sha)
    torch.save(cache,folder/'test_cache.pt')
    paired.save_json(a.out,dict(complete=True,device='cpu',phase='nonlinear_test_after_selection',arch=ref['arch'],
        plan_sha256=a.plan_sha256,selection_sha256=hashlib.sha256(a.choice.read_bytes()).hexdigest(),source_sha256=sources(),
        reference_state_sha256=paired.state_hash(ref),cache_state_sha256=paired.state_hash(cache),data_sha256=sha,branches=[]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','branch','test'],required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--original-reference',type=Path)
    p.add_argument('--seed',type=int);p.add_argument('--choice',type=Path);p.add_argument('--test-cache',type=Path);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(12);torch.set_num_interop_threads(1);{'prepare':prepare,'branch':branch,'test':prepare_test}[a.stage](a)
