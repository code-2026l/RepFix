"""Fixed exploratory Gaussian propagation vs direct consumer-error sampling."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from conditional_reference_cpu import ridge
from consumer_routing import decisions


def forecast(logits,residual,thresholds,draws,isotropic=False):
    residual=residual.double();mean=residual.mean(0);r=residual-mean
    cov=r.T@r/(len(r)-1);d=cov.shape[0];eye=torch.eye(d,dtype=torch.double);trace=cov.trace()/d
    cov=trace*eye if isotropic else .9*cov+.1*trace*eye
    factor=torch.linalg.cholesky(cov+1e-10*eye)
    simulated=logits.double()[None,:,:]+mean+(draws@factor.T)[:,None,:]
    reference=decisions(logits,thresholds)
    changed=decisions(simulated.flatten(0,1),thresholds).reshape(len(draws),len(logits),-1)!=reference[None,:,:]
    return float(changed.double().mean((0,1)).max())


def main(a):
    torch.set_num_threads(2);paths=sorted(a.states.rglob('seed_*.pt'));assert len(paths)==6
    order=torch.randperm(2048,generator=torch.Generator().manual_seed(6500));fit,monitor=order[:1024],order[1024:]
    draws=torch.randn(64,20,generator=torch.Generator().manual_seed(6700),dtype=torch.double)
    output=[]
    for path in paths:
        raw=torch.load(path,map_location='cpu',weights_only=True);vit=raw['anchor_current'].shape[1]==768
        refpath=a.references/('next_visiontwoprepare' if vit else 'next_visionfullprepare')/('p002_states' if vit else 'p000_states')/'reference.pt'
        ref=torch.load(refpath,map_location='cpu',weights_only=True);old=ref['heads']['coarse'];thresholds=list(ref['thresholds'].values())
        x=raw['anchor_current'].float();target=F.linear(raw['anchor_target'].float(),old['weight'],old['bias']);target-=target.mean(1,keepdim=True)
        reader,_,_=ridge(x[fit],target[fit],.001);z=F.linear(x,reader['weight'],reader['bias']);vz=F.linear(raw['validation_features'].float(),reader['weight'],reader['bias'])
        actual=float((decisions(vz,thresholds)!=decisions(raw['reference_logits'],thresholds)).double().mean(0).max())
        errors=(decisions(z,thresholds)!=decisions(target,thresholds)).double();residual=target-z
        sample=torch.randperm(len(vz),generator=torch.Generator().manual_seed(6700))[:512];rows=[]
        for n in [32,64,128,256]:
            for rep in range(20):
                idx=monitor[torch.randperm(len(monitor),generator=torch.Generator().manual_seed(6700+rep))[:n]]
                pred=dict(direct=float(errors[idx].mean(0).max()),covariance=forecast(vz[sample],residual[idx],thresholds,draws),
                    isotropic=forecast(vz[sample],residual[idx],thresholds,draws,True))
                rows.append(dict(budget=n,replicate=rep,predictions=pred))
        output.append(dict(arch='vit_b32' if vit else 'resnet18',seed=int(path.stem.split('_')[-1]),actual=actual,rows=rows,
            state_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    scores={}
    for n in [32,64,128,256]:
        scores[str(n)]={k:float(np.mean([abs(row['predictions'][k]-endpoint['actual']) for endpoint in output for row in endpoint['rows'] if row['budget']==n])) for k in ['direct','covariance','isotropic']}
    advance=all(scores[str(n)]['covariance']<=.8*scores[str(n)]['direct'] for n in [32,64]) and np.mean([scores[str(n)]['covariance'] for n in [128,256]])<=np.mean([scores[str(n)]['direct'] for n in [128,256]])
    result=dict(complete=True,phase='exploratory_retrospective_development',official_test_loaded=False,trained_endpoints=6,
        monitor_resamples_are_not_independent_models=True,advance=bool(advance),mae=scores,endpoints=output,
        source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest())
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(dict(advance=bool(advance),mae=scores),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--states',type=Path,required=True);p.add_argument('--references',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a)
