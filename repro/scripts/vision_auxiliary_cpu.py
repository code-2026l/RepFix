"""Independent fitted references and a real image-reconstruction auxiliary.

CPU numerical execution only. Official test data are never loaded by this driver.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import paired_readout_continuation as paired
import late_vision_cpu as late
import nonlinear_suffix_cpu as oldstudy
from consumer_contracts import confidence_thresholds,evaluate_contracts
from shared_reference_vision import NormalizedHead
from late_replay import ResidualHead
from readout_transfer_audit import fit_head

DOSES=[.01,.1,1.]
NAMES=['vision_auxiliary_cpu.py',*oldstudy.NAMES]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources():return {n:digest(Path(__file__).with_name(n)) for n in NAMES}
def save(out,value):paired.save_json(out,value)

def augment(images):
    g=torch.Generator().manual_seed(6200);n=len(images);offsets=torch.randint(0,9,(n,2),generator=g);flips=torch.rand(n,generator=g)<.5
    padded=F.pad(images.permute(0,3,1,2).float(),(4,4,4,4),mode='reflect');out=torch.empty_like(images)
    for i in range(n):
        y,x=offsets[i].tolist();part=padded[i,:,y:y+32,x:x+32]
        if flips[i]:part=part.flip(-1)
        out[i]=part.permute(1,2,0).to(torch.uint8)
    return out,dict(offset_sha256=paired.state_hash(offsets),flip_sha256=paired.state_hash(flips))

@torch.no_grad()
def prepare(a):
    start=time.time();source=torch.load(a.reference,map_location='cpu',weights_only=True)
    prefix,healthy=oldstudy.build_for(source);healthy.load_state_dict(source['healthy']);healthy.requires_grad_(False)
    images,y,coarse,data_sha=late.images(True);aug,augmentation=augment(images)
    values=[];h=[];hidden=[];targets=[]
    for view,inputs in enumerate([images,aug]):
        vx=[];vh=[];vt=[];vr=[]
        for j in range(0,len(inputs),128):
            batch=inputs[j:j+128];xx=prefix(late.preprocess(batch,source['metadata']['image_size'])).contiguous()
            hh,rr=healthy.features(xx);vx.append(xx);vh.append(hh);vr.append(rr)
            vt.append(F.interpolate(batch.permute(0,3,1,2).float()/255,size=(8,8),mode='bilinear',align_corners=False,antialias=True).flatten(1))
            if j%4096==0:print(json.dumps(dict(stage='cache',arch=source['arch'],view=view,processed=j,total=len(inputs))),flush=True)
        values.append(torch.cat(vx));h.append(torch.cat(vh));hidden.append(torch.cat(vr));targets.append(torch.cat(vt))
        del vx,vh,vt,vr
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    artifact=dict(x=torch.stack(values),h=torch.stack(h),hidden=torch.stack(hidden),targets=torch.stack(targets),y=y,coarse=coarse,
        source_reference_sha256=paired.state_hash(source),data_sha256=data_sha,augmentation=augmentation,
        healthy=healthy.state_dict(),metadata=source['metadata'],arch=source['arch'],family=source.get('family'),attention_blocks=source.get('attention_blocks',1))
    torch.save(artifact,folder/'population.pt')
    save(a.out,dict(complete=True,device='cpu',phase='training_population_preparation',arch=source['arch'],plan_sha256=a.plan_sha256,
        source_sha256=sources(),data_sha256=data_sha,source_reference_sha256=artifact['source_reference_sha256'],
        population_file='population.pt',population_sha256=digest(folder/'population.pt'),bytes=(folder/'population.pt').stat().st_size,
        splits=dict(official_test_loaded=False,images=50000,views=2),elapsed_s=time.time()-start,branches=[]))

def fit_affine(features,labels):
    clf,mean,sd=fit_head(features.double(),labels)
    return NormalizedHead(clf.weight.detach(),clf.bias.detach(),mean,sd).affine()

def make_reference(cache,seed):
    order=torch.randperm(50000,generator=torch.Generator().manual_seed(seed));fit=order[:10000];val=order[10000:15000];anchors=order[15000:17048]
    mask=torch.rand(10000,generator=torch.Generator().manual_seed(seed+10000))<.2
    heads={name:fit_affine(cache['h'][0,fit[mask]],cache[label][fit[mask]]) for name,label in [('fine','y'),('coarse','coarse')]}
    labeled_features=cache['h'][:,fit[mask]].flatten(0,1);mean=labeled_features.mean(0);sd=labeled_features.std(0,unbiased=False);sd=sd.clamp_min(.1*sd[sd>0].median())
    primary=NormalizedHead(heads['fine']['weight']*sd,heads['fine']['bias']+heads['fine']['weight']@mean,mean,sd)
    thresholds=confidence_thresholds(late.linear(cache['h'][0,fit[mask]],heads['coarse']))
    hidden=torch.cat([cache['h'][:,fit[mask]],cache['hidden'][:,fit[mask]]],-1).flatten(0,1)
    reader_h=fit_affine(hidden,cache['y'][fit[mask]].repeat(2))
    return dict(reference_seed=seed,fitting=fit,validation=val,anchors=anchors,label_mask=mask,heads=heads,primary=primary.state_dict(),thresholds=thresholds,rich_reader=reader_h)

def initial_head(reference):
    h=reference['primary'];head=NormalizedHead(h['linear.weight'],h['linear.bias'],h['mean'],h['sd']);head.load_state_dict(h);return head

@torch.no_grad()
def embeds(model,x):return torch.cat([model(batch) for batch in x.split(256)])

@torch.no_grad()
def score(model,reader,decoder,xx,target,labels,old_logits,old_labels,thresholds,scale):
    hs=embeds(model,xx);z=late.linear(hs,reader);mse=0.
    if decoder is not None:mse=float((torch.cat([decoder(v) for v in hs.split(256)])-target).square().mean()/scale)
    return dict(own=paired.metrics(z,labels),reconstruction_mse=mse,
        contracts=evaluate_contracts(old_logits,old_logits,old_labels,thresholds),old_decisions_unchanged=True),z

def run(a):
    start=time.time();cache=torch.load(a.population,map_location='cpu',weights_only=True,mmap=True)
    torch.manual_seed(a.reference_seed);reference=make_reference(cache,a.reference_seed)
    _,healthy=oldstudy.build_for(cache);healthy.load_state_dict(cache['healthy']);healthy.requires_grad_(False)
    fit=reference['fitting'];val=reference['validation'];mask=reference['label_mask']
    xx=cache['x'][:,fit];hh=cache['h'][:,fit];targets=cache['targets'][:,fit];labels=cache['y'][fit]
    scale=targets.flatten(0,1).var(0,unbiased=False).mean().clamp_min(1e-6)
    vxx=cache['x'][0,val];vt=cache['targets'][0,val];vy=cache['y'][val];vc=cache['coarse'][val]
    old_logits=late.linear(cache['h'][0,val],reference['heads']['coarse'])
    ref=dict(arch=cache['arch'],family=cache.get('family'),attention_blocks=cache.get('attention_blocks',1))
    out=dict(complete=False,device='cpu',phase='development',arch=cache['arch'],reference_seed=a.reference_seed,seed=a.seed,plan_sha256=a.plan_sha256,
        source_sha256=sources(),population_sha256=digest(a.population),source_reference_sha256=cache['source_reference_sha256'],
        splits=dict(fitting=10000,validation=5000,anchors=2048,labeled=int(mask.sum()),official_test_loaded=False,
            fitting_sha256=paired.state_hash(fit),validation_sha256=paired.state_hash(val),mask_sha256=paired.state_hash(mask)),
        input_reconstruction_shape=[3,8,8],target_variance=float(scale),branches=[])
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True);torch.save(reference,folder/'reference.pt')
    # Frozen controls use exactly the same final two-view reader fitting pool.
    for mode in ['F','N','H']:
        head=initial_head(reference);htrain=hh[:,mask].flatten(0,1);features=cache['h'][0,val]
        if mode=='H':features=torch.cat([features,cache['hidden'][0,val]],1);reader=reference['rich_reader']
        elif mode=='F':reader=fit_affine(htrain,labels[mask].repeat(2))
        else:
            torch.manual_seed(a.seed);head=ResidualHead(head);opt=torch.optim.AdamW(head.parameters(),lr=.0001,weight_decay=.0001)
            rng=torch.Generator().manual_seed(a.seed+20000);vrng=torch.Generator().manual_seed(a.seed+71000)
            for epoch in range(10):
                for idx in torch.randperm(len(fit),generator=rng).split(128):
                    view=torch.randint(2,(len(idx),),generator=vrng);idx,view=idx[mask[idx]],view[mask[idx]]
                    if not len(idx):continue
                    loss=F.cross_entropy(head(hh[view,idx]),labels[idx])+.0001*sum(p.square().sum() for p in head.parameters())
                    opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                hidden=F.gelu(head.hidden((htrain-head.base.mean)/head.base.sd));htrain=torch.cat([htrain,hidden],1)
                hidden=F.gelu(head.hidden((features-head.base.mean)/head.base.sd));features=torch.cat([features,hidden],1)
            reader=fit_affine(htrain,labels[mask].repeat(2))
        own=paired.metrics(late.linear(features,reader),vy);out['branches'].append(dict(mode=mode,validation=dict(own=own)))
        torch.save(dict(reader=reader,head=head.state_dict()),folder/(mode+'.pt'));save(a.out.with_suffix('.partial.json'),out)
        print(json.dumps(dict(stage='frozen_control',arch=cache['arch'],mode=mode,accuracy=own['acc'])),flush=True)
    torch.manual_seed(a.seed);warm=nn.Sequential(nn.Linear(hh.shape[-1],256),nn.ReLU(),nn.Linear(256,192))
    warmopt=torch.optim.AdamW(warm.parameters(),lr=.001,weight_decay=.0001);g=torch.Generator().manual_seed(a.seed+90000)
    for epoch in range(5):
        for idx in torch.randperm(len(fit),generator=g).split(128):
            views=torch.randint(2,(len(idx),),generator=g);loss=(warm(hh[views,idx])-targets[views,idx]).square().mean()/scale
            warmopt.zero_grad();loss.backward();warmopt.step()
    torch.save(dict(decoder=warm.state_dict(),optimizer=warmopt.state_dict()),folder/'warm_decoder.pt')
    for mode,alpha in [('D',0.),*[(f'J{dose:g}',dose) for dose in DOSES]]:
        torch.manual_seed(a.seed);model=copy.deepcopy(healthy).requires_grad_(True)
        for layer in model.modules():
            if isinstance(layer,nn.LayerNorm):layer.requires_grad_(False)
        head=initial_head(reference);primary_opt=torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.requires_grad],lr=.00003),dict(params=head.parameters(),lr=.0001)],weight_decay=.0001)
        doses=DOSES if mode=='D' else [alpha];decoders=[];opts=[]
        for _ in doses:
            dec=copy.deepcopy(warm);opt=torch.optim.AdamW(dec.parameters(),lr=.001,weight_decay=.0001);opt.load_state_dict(warmopt.state_dict());decoders.append(dec);opts.append(opt)
        rng=torch.Generator().manual_seed(a.seed+20000);vrng=torch.Generator().manual_seed(a.seed+71000);trace=[]
        for epoch in range(10):
            losses=[]
            for idx in torch.randperm(len(fit),generator=rng).split(128):
                view=torch.randint(2,(len(idx),),generator=vrng);h=model(xx[view,idx]);selected=mask[idx]
                ce=F.cross_entropy(head(h[selected]),labels[idx[selected]])+.0001*sum(p.square().sum() for p in head.parameters())
                primary_opt.zero_grad(set_to_none=True)
                if mode=='D':
                    ce.backward();primary_opt.step()
                    for dose,decoder,opt in zip(doses,decoders,opts):
                        aux=(decoder(h.detach())-targets[view,idx]).square().mean()/scale;opt.zero_grad();(dose*aux).backward();opt.step()
                else:
                    aux=(decoders[0](h)-targets[view,idx]).square().mean()/scale;loss=ce+alpha*aux
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint reconstruction objective')
                    opts[0].zero_grad(set_to_none=True);loss.backward();primary_opt.step();opts[0].step()
                losses.append([float(ce.detach()),float(aux.detach())])
            trace.append(dict(epoch=epoch+1,primary=float(np.mean(losses,0)[0]),auxiliary=float(np.mean(losses,0)[1])))
            print(json.dumps(dict(stage='train',arch=cache['arch'],reference=a.reference_seed,seed=a.seed,mode=mode,epoch=epoch+1)),flush=True)
        with torch.no_grad():
            online=paired.metrics(head(embeds(model,vxx)),vy);train_h=embeds(model,xx[:,mask].flatten(0,1))
        reader=fit_affine(train_h,labels[mask].repeat(2))
        result,z=score(model,reader,decoders[0],vxx,vt,vy,old_logits,vc,reference['thresholds'],scale)
        detached_errors={}
        if mode=='D':
            for dose,decoder in zip(doses,decoders):detached_errors[str(dose)]=score(model,reader,decoder,vxx,vt,vy,old_logits,vc,reference['thresholds'],scale)[0]['reconstruction_mse']
        stored=dict(model=model.state_dict(),reader=reader,head=head.state_dict(),decoders=[d.state_dict() for d in decoders],own_logits=z,old_logits=old_logits,labels=vy,coarse=vc)
        torch.save(stored,folder/(mode+'.pt'))
        out['branches'].append(dict(mode=mode,alpha=alpha,validation=result,online_own=online,trace=trace,detached_mse_by_dose=detached_errors,
            state_file=mode+'.pt',state_sha256=digest(folder/(mode+'.pt'))))
        save(a.out.with_suffix('.partial.json'),out);print(json.dumps(dict(stage='endpoint',mode=mode,arch=cache['arch'],accuracy=result['own']['acc'],mse=result['reconstruction_mse'])),flush=True)
    out.update(complete=True,elapsed_s=time.time()-start);save(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','run'],required=True);p.add_argument('--reference',type=Path);p.add_argument('--population',type=Path)
    p.add_argument('--reference-seed',type=int,default=6200);p.add_argument('--seed',type=int,default=6300);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1)
    if a.stage=='prepare':prepare(a)
    else:run(a)
