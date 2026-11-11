"""Unlabeled compression of legacy decisions from an unchanged prefix; CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import vision_auxiliary_cpu as study
import paired_readout_continuation as paired
from consumer_contracts import evaluate_contracts


def features(x,arch,rich):
    if arch=='resnet18':return x.flatten(1) if rich else x.mean((-2,-1))
    return torch.cat([x[:,0],x[:,1:].mean(1)],1) if rich else x.mean(1)


def main(a):
    start=time.time();cache=torch.load(a.population,map_location='cpu',weights_only=True,mmap=True)
    torch.manual_seed(6200);ref=study.make_reference(cache,6200);fit=ref['fitting'];val=ref['validation'];old=ref['heads']['coarse']
    target=F.linear(cache['h'][:,fit].flatten(0,1),old['weight'],old['bias']);target-=target.mean(1,keepdim=True)
    reference=F.linear(cache['h'][0,val],old['weight'],old['bias']);scale=target.var(0,unbiased=False).mean().clamp_min(1e-6)
    result=dict(complete=False,device='cpu',phase='development',arch=cache['arch'],reference_seed=6200,seed=6400,
        plan_sha256=a.plan_sha256,population_sha256=study.digest(a.population),official_test_loaded=False,
        source_sha256={Path(__file__).name:study.digest(Path(__file__)),**study.sources()},branches=[])
    saved=dict(reference=reference,labels=cache['coarse'][val],thresholds=ref['thresholds'],readers={})
    for rich in [False,True]:
        name='rich' if rich else 'pooled';x=features(cache['x'][:,fit].flatten(0,1),cache['arch'],rich);vx=features(cache['x'][0,val],cache['arch'],rich)
        mean=x.mean(0);sd=x.std(0,unbiased=False).clamp_min(1e-4);x=(x-mean)/sd;vx=(vx-mean)/sd
        xd=x.double();yt=target.double();ym=yt.mean(0)
        weight=torch.linalg.solve(xd.T@xd/len(xd)+.001*torch.eye(xd.shape[1],dtype=torch.double),xd.T@(yt-ym)/len(xd)).float().T
        torch.manual_seed(6400);model=nn.Sequential(nn.Linear(x.shape[1],256),nn.GELU(),nn.Linear(256,20));nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
        base=F.linear(vx,weight,ym.float())
        for mode in ['affine','residual']:
            if mode=='residual':
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);g=torch.Generator().manual_seed(6400)
                for epoch in range(100):
                    for idx in torch.randperm(len(x),generator=g).split(256):
                        z=F.linear(x[idx],weight,ym.float())+model(x[idx]);loss=(z-target[idx]).square().mean()/scale
                        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
                    if (epoch+1)%20==0:print(json.dumps(dict(feature=name,epoch=epoch+1,loss=float(loss.detach()))),flush=True)
            with torch.no_grad():z=base if mode=='affine' else base+model(vx)
            contracts=evaluate_contracts(z,reference,cache['coarse'][val],ref['thresholds']);maximum=max(c['disagreement'] for c in contracts.values());key=name+'_'+mode
            result['branches'].append(dict(mode=key,feature_dimension=x.shape[1],contracts=contracts,maximum_disagreement=maximum,passes=maximum<=.01))
            saved['readers'][key]=dict(mean=mean,sd=sd,weight=weight,bias=ym.float(),residual=None if mode=='affine' else {k:v.clone() for k,v in model.state_dict().items()},logits=z)
            print(json.dumps(dict(mode=key,maximum_disagreement=maximum,passes=maximum<=.01)),flush=True)
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True);torch.save(saved,folder/'readers.pt')
    result.update(complete=True,elapsed_s=time.time()-start,state_sha256=study.digest(folder/'readers.pt'))
    paired.save_json(a.out,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--population',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
