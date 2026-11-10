"""Fixed ResNet-18/CIFAR-100 last-block screen. Training pickle only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.models import resnet18,ResNet18_Weights
import paired_readout_continuation as paired
from readout_transfer_audit import fit_head,stratified_split
from consumer_contracts import confidence_thresholds,evaluate_contracts
from vision_consumer_cpu import load_train
from vision_chart_cpu import preprocess
from shared_reference_vision import SharedReferenceBlock,NormalizedHead,dense_block

CONFIGS=[(0,8),(8,8),(16,16)]


@torch.no_grad()
def healthy_features(prefix,first,second,images,indices):
    x=[];h=[]
    for j in range(0,len(indices),128):
        value=prefix(preprocess(images[indices[j:j+128]])).contiguous();x.append(value);h.append(dense_block(value,first,second).mean((-2,-1)))
    return torch.cat(x),torch.cat(h)


@torch.no_grad()
def cached(first,second,x):
    p1=[];p2=[]
    for j in range(0,len(x),256):
        value=first(x[j:j+256]);p1.append(value);p2.append(second(F.relu(value)))
    return torch.cat(p1),torch.cat(p2)


def linear(h,head):return F.linear(h,head['weight'],head['bias'])


@torch.no_grad()
def evaluate(model,mode,x,cache,labels,coarse,healthy,legacy,thresholds):
    current=[];old=[];merge_error=0.;merge_changes=0;gate_changes=0
    if mode not in ['F','U']:f,s=model.merged()
    for j in range(0,len(x),256):
        value=x[j:j+256];c=tuple(t[j:j+256] for t in cache)
        if mode=='F':h=healthy[j:j+256];old_h=h;own=model(h)
        elif mode=='U':
            f,s,head=model;h=dense_block(value,f,s).mean((-2,-1));old_h=F.relu(c[1]+value).mean((-2,-1));own=head(h)
        else:
            h,h0=model.forward_states(value,c);own=model.head(h.mean((-2,-1)));old_h=h0.mean((-2,-1))
            merged=model.head(dense_block(value,f,s).mean((-2,-1)))
            merge_error=max(merge_error,float((own-merged).abs().max()));merge_changes+=int((own.argmax(1)!=merged.argmax(1)).sum())
            gate_changes+=int(((h>0)!=(h0>0)).sum())
        current.append(own);old.append(linear(old_h,legacy))
    z=torch.cat(current);old=torch.cat(old);reference=linear(healthy,legacy)
    return dict(own=paired.metrics(z,labels),contracts=evaluate_contracts(old,reference,coarse,thresholds),
        current_merge_maximum_logit_error=merge_error,current_merge_prediction_changes=merge_changes,changed_second_relu_gates=gate_changes,
        maximum_legacy_logit_error=float((old-reference).abs().max()))


def run(a):
    start=time.time();torch.manual_seed(a.seed);np.random.seed(a.seed);images,fine,coarse,data_sha=load_train()
    pool,val=stratified_split(fine,.1,a.seed);perm=torch.randperm(len(pool),generator=torch.Generator().manual_seed(a.seed+31000))
    anchors,fit=pool[perm[:2048]],pool[perm[2048:]];mask=torch.rand(len(fit),generator=torch.Generator().manual_seed(a.seed+501))<.2
    labeled=torch.where(mask)[0];assert len(val)==5000 and len(fit)==42952
    encoder=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(memory_format=torch.channels_last)
    block=encoder.layer4[-1];assert block.downsample is None
    first=fuse_conv_bn_eval(block.conv1,block.bn1);second=fuse_conv_bn_eval(block.conv2,block.bn2)
    prefix=nn.Sequential(encoder.conv1,encoder.bn1,encoder.relu,encoder.maxpool,encoder.layer1,encoder.layer2,encoder.layer3,encoder.layer4[0]).eval().requires_grad_(False)
    xfit,hfit=healthy_features(prefix,first,second,images,fit);xval,hval=healthy_features(prefix,first,second,images,val)
    print(json.dumps(dict(stage='cache',seed=a.seed,elapsed_s=time.time()-start)),flush=True)
    heads={};normalized=None
    for name,labels in [('fine',fine),('coarse',coarse)]:
        head,mean,sd=fit_head(hfit[mask].double(),labels[fit][mask]);w=head.weight.detach();b=head.bias.detach()
        obj=NormalizedHead(w,b,mean,sd);heads[name]=obj.affine()
        if name=='fine':normalized=obj
    legacy=heads['coarse'];thresholds=confidence_thresholds(linear(hfit,legacy));fit_cache=cached(first,second,xfit);val_cache=cached(first,second,xval)
    # Gradient support/basis is fixed from a training-only labeled calibration batch.
    idx=labeled[torch.randperm(len(labeled),generator=torch.Generator().manual_seed(a.seed+51000))[:1024]]
    f=copy.deepcopy(first);s=copy.deepcopy(second);loss=F.cross_entropy(normalized(dense_block(xfit[idx],f,s).mean((-2,-1))),fine[fit[idx]])
    g1,gb,g2=torch.autograd.grad(loss,[f.weight,f.bias,s.weight]);support=(g1.square().flatten(1).sum(1)+gb.square()).argsort(descending=True,stable=True)
    right=torch.linalg.svd(g2.flatten(1).double(),full_matrices=False).Vh;del f,s,loss,g1,gb,g2
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    torch.save(dict(prefix=prefix.state_dict(),first=first.state_dict(),second=second.state_dict(),normalized=normalized.state_dict(),
        heads=heads,thresholds=thresholds,fitting=fit,validation=val,anchors=anchors,labeled=mask,support=support,right_vectors=right[:16]),folder/'reference.pt')
    source=['vision_shared_replay_cpu.py','shared_reference_vision.py','shared_reference_replay.py','consumer_contracts.py',
        'readout_transfer_audit.py','vision_consumer_cpu.py','vision_chart_cpu.py','paired_readout_continuation.py']
    out=dict(complete=False,device='cpu',phase='development',seed=a.seed,smoke=a.smoke,plan_sha256=a.plan_sha256,
        data_sha256=data_sha,source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in source},
        splits=dict(fitting=len(fit),validation=len(val),anchors=len(anchors),labeled=int(mask.sum()),fitting_sha256=paired.state_hash(fit),
            validation_sha256=paired.state_hash(val),anchors_sha256=paired.state_hash(anchors),mask_sha256=paired.state_hash(mask),official_test_loaded=False,test_metrics_computed=False),
        configuration=dict(epochs=3,batch=128,threads=8,head_lr=.001,sparse_lr=.001,dense_lr=.0001,weight_decay=.0001,configs=CONFIGS),
        support_calibration=dict(indices_sha256=paired.state_hash(idx),support_order=support.tolist(),right_vectors_sha256=paired.state_hash(right)),
        healthy=paired.metrics(linear(hval,heads['fine']),fine[val]),branches=[])
    for mode,k,r in [('F',0,0),('U',0,0)]+[('replay',k,r) for k,r in ([(8,8)] if a.smoke else CONFIGS)]:
        head=copy.deepcopy(normalized)
        if mode=='F':model=head;groups=[dict(params=head.parameters(),lr=.001)]
        elif mode=='U':
            f,s=copy.deepcopy(first),copy.deepcopy(second);model=(f,s,head)
            groups=[dict(params=[*f.parameters(),*s.parameters()],lr=.0001),dict(params=head.parameters(),lr=.001)]
        else:
            model=SharedReferenceBlock(first,second,support[:k],right[:r],head)
            groups=[dict(params=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('head.')],lr=.001),dict(params=model.head.parameters(),lr=.001)]
        opt=torch.optim.AdamW(groups,weight_decay=.0001);rng=torch.Generator().manual_seed(a.seed+20000);order_hash=hashlib.sha256();trace=[]
        for epoch in range(1 if a.smoke else 3):
            order=torch.randperm(len(fit),generator=rng);order_hash.update(order.numpy().tobytes())
            for j in range(0,256 if a.smoke else len(order),128):
                local=order[j:j+128];idx=local[mask[local]]
                if not len(idx):continue
                if mode=='F':z=model(hfit[idx])
                elif mode=='U':z=head(dense_block(xfit[idx],f,s).mean((-2,-1)))
                else:z=model(xfit[idx],tuple(t[idx] for t in fit_cache))
                loss=F.cross_entropy(z,fine[fit[idx]])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite ResNet replay objective')
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            trace.append(dict(epoch=epoch+1,loss=float(loss.detach())));print(json.dumps(dict(stage=mode,k=k,rank=r,epoch=epoch+1,elapsed_s=time.time()-start)),flush=True)
        row=dict(mode=mode,k=k,rank=r,trace=trace,minibatch_order_sha256=order_hash.hexdigest(),
            validation=evaluate(model,mode,xval,val_cache,fine[val],coarse[val],hval,legacy,thresholds))
        if mode=='F':state=dict(head=model.state_dict())
        elif mode=='U':state=dict(first=f.state_dict(),second=s.state_dict(),head=head.state_dict())
        else:
            state=dict(updates={n:t for n,t in model.state_dict().items() if not n.startswith(('first.','second.'))})
            row.update(first_delta_norm=float(model.delta1.detach().norm()),second_delta_norm=float(model.u.detach().norm()))
        name='%s_k%d_r%d.pt'%(mode,k,r);row['state_file']=name;row['state_sha256']=paired.state_hash(state);torch.save(state,folder/name)
        out['branches'].append(row);paired.save_json(a.out.with_suffix('.partial.json'),out)
    assert len({row['minibatch_order_sha256'] for row in out['branches']})==1
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);run(a)
