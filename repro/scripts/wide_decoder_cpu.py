"""Ordinary widened residual decoders on frozen healthy visual features."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import vision_auxiliary_cpu as base
import late_vision_cpu as late
import paired_readout_continuation as paired


def main(a):
    started=time.time();cache=torch.load(a.population,map_location='cpu',weights_only=True,mmap=True)
    torch.manual_seed(a.reference_seed);ref=base.make_reference(cache,a.reference_seed);fit=ref['fitting'];val=ref['validation'];mask=ref['label_mask']
    features=torch.cat([cache['h'],cache['hidden']],-1);xx=features[:,fit];yy=cache['y'][fit]
    train=xx[:,mask].flatten(0,1);mean=train.mean(0);sd=train.std(0,unbiased=False);sd=sd.clamp_min(.1*sd[sd>0].median())
    torch.manual_seed(a.seed);hidden=nn.Linear(train.shape[1],a.width);output=nn.Linear(a.width,100);nn.init.zeros_(output.weight);nn.init.zeros_(output.bias)
    primary=base.initial_head(ref);parameters=[*primary.parameters(),*hidden.parameters(),*output.parameters()]
    opt=torch.optim.AdamW(parameters,lr=.0001,weight_decay=.0001);rng=torch.Generator().manual_seed(a.seed+20000);vrng=torch.Generator().manual_seed(a.seed+71000)
    for epoch in range(10):
        for idx in torch.randperm(len(fit),generator=rng).split(128):
            view=torch.randint(2,(len(idx),),generator=vrng);idx,view=idx[mask[idx]],view[mask[idx]]
            if not len(idx):continue
            h=xx[view,idx];learned=F.gelu(hidden((h-mean)/sd));z=primary(h[:,:cache['h'].shape[-1]])+output(learned)
            loss=F.cross_entropy(z,yy[idx])+.0001*sum(p.square().sum() for p in parameters)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        print(json.dumps(dict(epoch=epoch+1,width=a.width)),flush=True)
    with torch.no_grad():
        learned=torch.cat([F.gelu(hidden((h-mean)/sd)) for h in train.split(256)])
        reader=base.fit_affine(torch.cat([train,learned],1),yy[mask].repeat(2))
        vx=features[0,val];vh=torch.cat([F.gelu(hidden((h-mean)/sd)) for h in vx.split(256)])
        z=late.linear(torch.cat([vx,vh],1),reader);metrics=paired.metrics(z,cache['y'][val])
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True);state=folder/'endpoint.pt'
    torch.save(dict(hidden=hidden.state_dict(),reader=reader,mean=mean,sd=sd,predictions=z.argmax(1),labels=cache['y'][val]),state)
    base.save(a.out,dict(complete=True,device='cpu',phase='wide_decoder_development',reference_seed=a.reference_seed,seed=a.seed,width=a.width,
        validation=metrics,extends=metrics['acc']>=.7164,trainable_parameters=sum(p.numel() for p in parameters),
        official_test_loaded=False,source_sha256={Path(__file__).name:base.digest(Path(__file__)),**base.sources()},
        plan_sha256=a.plan_sha256,state_sha256=base.digest(state),elapsed_s=time.time()-started,branches=[]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--population',type=Path,required=True);p.add_argument('--width',type=int,required=True)
    p.add_argument('--reference-seed',type=int,default=6200);p.add_argument('--seed',type=int,default=6300);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
