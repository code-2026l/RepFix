"""Fixed-reference late suffix adaptation; CPU, no test access in development."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import paired_readout_continuation as paired
from consumer_contracts import confidence_thresholds,evaluate_contracts
from readout_transfer_audit import fit_head,stratified_split
from shared_reference_vision import NormalizedHead
from late_replay import build_backbone,finish,LateDelta,DenseSuffix,ResidualHead

RANKS=[16,64,128]
SOURCE_NAMES=['late_vision_cpu.py','late_replay.py','shared_reference_vision.py','consumer_contracts.py',
    'readout_transfer_audit.py','paired_readout_continuation.py']

def source_hashes():return {n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in SOURCE_NAMES}

def images(training=True):
    path=Path(os.environ['REPFIX_DATA_X'])/'v037/cifar-100-python'/('train' if training else 'test')
    with path.open('rb') as f:d=pickle.load(f,encoding='bytes')
    x=torch.from_numpy(d[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1).copy())
    return x,torch.tensor(d[b'fine_labels']),torch.tensor(d[b'coarse_labels']),hashlib.sha256(path.read_bytes()).hexdigest()

def preprocess(x,size):
    x=F.interpolate(x.permute(0,3,1,2).float()/255.,size=(size,size),mode='bilinear',align_corners=False,antialias=True)
    mean=torch.tensor([.485,.456,.406]).reshape(1,3,1,1);sd=torch.tensor([.229,.224,.225]).reshape(1,3,1,1)
    return ((x-mean)/sd).contiguous(memory_format=torch.channels_last)

def pooled(x):return x.mean((-2,-1)) if x.ndim==4 else x
def linear(h,p):return F.linear(h,p['weight'],p['bias'])

@torch.no_grad()
def collect(prefix,tail,norm,x,indices,size,keep=None):
    hs=[];xs=[];rs=[]
    for start in range(0,len(indices),128):
        idx=indices[start:start+128];features,residual=prefix(preprocess(x[idx],size));healthy=finish(tail(features),residual,norm);hs.append(healthy)
        if keep is not None:
            mask=keep[start:start+128];xs.append(features[mask]);rs.append(residual[mask])
        if start%8192==0:print(json.dumps(dict(stage='frozen_prefix',processed=start,total=len(indices))),flush=True)
    result=dict(h=torch.cat(hs))
    if keep is not None:result.update(x=torch.cat(xs),residual=torch.cat(rs))
    return result


def prepare(a):
    start=time.time();torch.manual_seed(3800);np.random.seed(3800);x,y,coarse,data_sha=images(True)
    pool,val=stratified_split(y,.1,3800);order=torch.randperm(len(pool),generator=torch.Generator().manual_seed(34800));anchors,fit=pool[order[:2048]],pool[order[2048:]]
    mask=torch.rand(len(fit),generator=torch.Generator().manual_seed(4301))<.2;assert len(fit)==42952 and len(val)==5000
    prefix,tail,norm,meta=build_backbone(a.arch,True)
    xf=collect(prefix,tail,norm,x,fit,meta['image_size'],mask)
    validation=collect(prefix,tail,norm,x,val,meta['image_size'],torch.ones(len(val),dtype=torch.bool))
    heads={};head=None
    for name,labels in [('fine',y),('coarse',coarse)]:
        model,mean,sd=fit_head(xf['h'][mask].double(),labels[fit][mask]);obj=NormalizedHead(model.weight.detach(),model.bias.detach(),mean,sd)
        heads[name]=obj.affine()
        if name=='fine':head=obj
    thresholds=confidence_thresholds(linear(xf['h'],heads['coarse']));train=dict(x=xf['x'],residual=xf['residual'],h=xf['h'][mask],y=y[fit][mask])
    validation.update(y=y[val],coarse=coarse[val])
    hprobe,mu,sd=fit_head(torch.cat([train['h'],pooled(train['x'])],1).double(),train['y'])
    hidden_probe=NormalizedHead(hprobe.weight.detach(),hprobe.bias.detach(),mu,sd).affine()
    # One training-only calibration is shared by every continuation seed.
    idx=torch.randperm(len(train['y']),generator=torch.Generator().manual_seed(54800))[:1024]
    temporary=copy.deepcopy(tail).requires_grad_(True)
    z=head(finish(temporary(train['x'][idx]),train['residual'][idx],norm));loss=F.cross_entropy(z,train['y'][idx])
    gradient=torch.autograd.grad(loss,temporary.weight)[0].flatten(1)
    right=torch.linalg.svd(gradient.double(),full_matrices=False).Vh[:128].float();del temporary,loss,z,gradient,xf
    checkpoint=Path(torch.hub.get_dir())/'checkpoints'/meta['weights_url'].rsplit('/',1)[1]
    meta.update(weights_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),prefix_sha256=paired.state_hash(prefix.state_dict()))
    splits=dict(fitting=len(fit),validation=len(val),anchors=len(anchors),labeled=int(mask.sum()),fitting_sha256=paired.state_hash(fit),
        validation_sha256=paired.state_hash(val),anchor_sha256=paired.state_hash(anchors),mask_sha256=paired.state_hash(mask),
        official_test_loaded=False,test_metrics_computed=False,reference_seed=3800,reference_fixed_across_continuation_seeds=True)
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    reference=dict(arch=a.arch,tail=tail.state_dict(),norm=norm.state_dict(),fine=head.state_dict(),heads=heads,hidden_probe=hidden_probe,
        thresholds=thresholds,right=right,metadata=meta,splits=splits,fitting=fit,validation_indices=val,anchors=anchors,labeled=mask,plan_sha256=a.plan_sha256,data_sha256=data_sha)
    torch.save(reference,folder/'reference.pt');torch.save(dict(train=train,validation=validation),folder/'cache.pt')
    out=dict(complete=True,device='cpu',phase='prepare',arch=a.arch,plan_sha256=a.plan_sha256,source_sha256=source_hashes(),splits=splits,metadata=meta,
        reference_state_sha256=paired.state_hash(reference),cache_state_sha256=None,thresholds=thresholds,
        healthy=paired.metrics(linear(validation['h'],heads['fine']),validation['y']),
        hidden_probe=paired.metrics(linear(torch.cat([validation['h'],pooled(validation['x'])],1),hidden_probe),validation['y']),
        elapsed_s=time.time()-start,branches=[])
    paired.save_json(a.out,out)


def construct(ref,tail,norm,mode,seed):
    # Initialization is independent of preceding arms and their RNG use.
    torch.manual_seed(seed)
    saved=ref['fine'];head=NormalizedHead(saved['linear.weight'],saved['linear.bias'],saved['mean'],saved['sd']);head.load_state_dict(saved)
    if mode=='F':return None,head,[dict(params=head.parameters(),lr=.001)]
    if mode=='N':
        head=ResidualHead(head);return None,head,[dict(params=head.parameters(),lr=.001)]
    update=DenseSuffix(tail,norm) if mode=='U' else LateDelta(tail,norm,ref['right'][:int(mode[1:])])
    parameters=[p for p in update.parameters() if p.requires_grad]
    return update,head,[dict(params=parameters,lr=.0001),dict(params=head.parameters(),lr=.001)]


@torch.no_grad()
def evaluate(update,head,mode,part,tail,norm,ref,monitor=None):
    current=[];legacy=[];difference=0.;changes=0;feature_change=0.;base_head=None
    if mode!='N' and mode!='H':base_head=head.affine()
    if mode.startswith('r'):merged=update.merged()
    for j in range(0,len(part['h']),256):
        old=part['h'][j:j+256];xx=part['x'][j:j+256];rr=part['residual'][j:j+256]
        if mode=='H':z=linear(torch.cat([old,pooled(xx)],1),ref['hidden_probe'])
        elif mode=='N':z=head(old)
        elif mode=='F':z=linear(old,base_head)
        else:
            h=update(xx,rr);z=linear(h,base_head);feature_change=max(feature_change,float((h-old).abs().max()))
            if mode.startswith('r'):
                dense=linear(finish(merged(xx),rr,norm),base_head);difference=max(difference,float((z-dense).abs().max()));changes+=int((z.argmax(1)!=dense.argmax(1)).sum())
        current.append(z);legacy.append(linear(old,ref['heads']['coarse']))
    z=torch.cat(current);old=torch.cat(legacy)
    contracts=evaluate_contracts(old,old,part['coarse'],ref['thresholds'],monitor,.05/2000)
    return dict(own=paired.metrics(z,part['y']),contracts=contracts,maximum_feature_change=feature_change,
        current_merge_maximum_logit_error=difference,current_merge_prediction_changes=changes)


def branch(a):
    start=time.time();ref=torch.load(a.reference,map_location='cpu',weights_only=True);cached=torch.load(a.reference.with_name('cache.pt'),map_location='cpu',weights_only=True)
    assert ref['plan_sha256']==a.plan_sha256
    _,tail,norm,_=build_backbone(ref['arch'],True);tail.load_state_dict(ref['tail']);norm.load_state_dict(ref['norm'])
    train=cached['train'];part=cached['validation'];modes=['F','U','N','H','r16','r64','r128']
    phase='development';monitor=None
    if a.choice is not None:
        choice=json.loads(a.choice.read_text());selected=choice['backbones'][ref['arch']]['selected'];assert selected is not None
        modes=list(dict.fromkeys(['F','U','N','H',selected]));phase='confirmation'
        test=torch.load(a.test_cache,map_location='cpu',weights_only=True);part=test['test'];assert test['reference_state_sha256']==paired.state_hash(ref)
        monitor=np.random.default_rng(13700001+a.seed).choice(len(part['y']),1024,replace=False)
    out=dict(complete=False,device='cpu',phase=phase,arch=ref['arch'],seed=a.seed,plan_sha256=a.plan_sha256,
        source_sha256=source_hashes(),reference_state_sha256=paired.state_hash(ref),reference_file_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        selection_sha256=None if a.choice is None else hashlib.sha256(a.choice.read_bytes()).hexdigest(),
        splits=dict(ref['splits'],official_test_loaded=phase=='confirmation',test_metrics_computed=phase=='confirmation'),
        configuration=dict(epochs=3,batch=128,head_lr=.001,update_lr=.0001,weight_decay=.0001,threads=12),
        healthy=paired.metrics(linear(part['h'],ref['heads']['fine']),part['y']),branches=[])
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    for mode in modes:
        if mode=='H':
            row=dict(mode=mode,validation=evaluate(None,None,mode,part,tail,norm,ref,monitor),state_file='H.pt')
            state=dict(mode=mode,head=ref['hidden_probe'])
        else:
            update,head,groups=construct(ref,tail,norm,mode,a.seed);optimizer=torch.optim.AdamW(groups,weight_decay=.0001)
            order_sha=hashlib.sha256();rng=torch.Generator().manual_seed(a.seed+20000);fit_mask=ref['labeled']
            lookup=torch.full((len(fit_mask),),-1,dtype=torch.long);lookup[fit_mask]=torch.arange(int(fit_mask.sum()))
            trace=[]
            for epoch in range(3):
                order=torch.randperm(len(fit_mask),generator=rng);order_sha.update(order.numpy().tobytes())
                for j in range(0,len(order),128):
                    idx=lookup[order[j:j+128]];idx=idx[idx>=0]
                    if not len(idx):continue
                    h=train['h'][idx] if update is None else update(train['x'][idx],train['residual'][idx])
                    loss=F.cross_entropy(head(h),train['y'][idx])
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite late-update loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for g in groups for p in g['params']):raise FloatingPointError('Nonfinite late-update gradient')
                    optimizer.step()
                trace.append(dict(epoch=epoch+1,loss=float(loss.detach())))
            row=dict(mode=mode,trace=trace,minibatch_order_sha256=order_sha.hexdigest(),
                validation=evaluate(update,head,mode,part,tail,norm,ref,monitor),state_file=mode+'.pt',
                trainable_parameters=sum(p.numel() for g in groups for p in g['params']))
            state=dict(mode=mode,head=head.state_dict())
            if mode=='U':state['tail']=update.tail.state_dict()
            elif mode.startswith('r'):state['update']={k:v for k,v in update.state_dict().items() if k in ['u','v','bias']}
        if phase=='confirmation':row['test']=row.pop('validation')
        row['state_sha256']=paired.state_hash(state);torch.save(state,folder/row['state_file']);out['branches'].append(row)
        paired.save_json(a.out.with_suffix('.partial.json'),out)
        score=row['test' if phase=='confirmation' else 'validation']['own']['acc']
        print(json.dumps(dict(arch=ref['arch'],seed=a.seed,mode=mode,accuracy=score,elapsed_s=time.time()-start)),flush=True)
    assert len({v['minibatch_order_sha256'] for v in out['branches'] if 'minibatch_order_sha256' in v})==1
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)


@torch.no_grad()
def prepare_test(a):
    ref=torch.load(a.reference,map_location='cpu',weights_only=True);choice=json.loads(a.choice.read_text());assert choice['backbones'][ref['arch']]['selected'] is not None
    x,y,coarse,sha=images(False);prefix,tail,norm,_=build_backbone(ref['arch'],True);tail.load_state_dict(ref['tail']);norm.load_state_dict(ref['norm'])
    idx=torch.arange(len(x));part=collect(prefix,tail,norm,x,idx,ref['metadata']['image_size'],torch.ones(len(x),dtype=torch.bool));part.update(y=y,coarse=coarse)
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    torch.save(dict(test=part,reference_state_sha256=paired.state_hash(ref),data_sha256=sha),folder/'test_cache.pt')
    paired.save_json(a.out,dict(complete=True,device='cpu',phase='test_cache_after_selection',arch=ref['arch'],plan_sha256=a.plan_sha256,
        selection_sha256=hashlib.sha256(a.choice.read_bytes()).hexdigest(),source_sha256=source_hashes(),data_sha256=sha,branches=[]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','branch','test'],required=True);p.add_argument('--arch',choices=['resnet18','resnet50','vit_b32'])
    p.add_argument('--reference',type=Path);p.add_argument('--seed',type=int);p.add_argument('--choice',type=Path);p.add_argument('--test-cache',type=Path)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1)
    {'prepare':prepare,'branch':branch,'test':prepare_test}[a.stage](a)
