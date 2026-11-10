"""Independent rate-level replay, including a check against maximum-only bias."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from consumer_routing import decisions
from analyze_small_budget_confirmation import interval


def propagated(z,error,thresholds,normals,isotropic):
    e=error.double();mu=e.mean(0);e=e-mu;cov=e.T@e/(len(e)-1);eye=torch.eye(e.shape[1],dtype=torch.double);v=cov.trace()/len(eye)
    cov=v*eye if isotropic else .9*cov+.1*v*eye
    noise=normals@torch.linalg.cholesky(cov+1e-10*eye).T
    counterfactual=z.double()[None,:,:]+mu+noise[:,None,:]
    transformed=decisions(counterfactual.reshape(-1,z.shape[1]),thresholds).reshape(len(normals),len(z),3)
    return (transformed!=decisions(z,thresholds)[None,:,:]).double().mean((0,1))


def main(a):
    torch.set_num_threads(2);study=json.loads(a.summary.read_text());assert study['complete'];records=[]
    order=torch.randperm(2048,generator=torch.Generator().manual_seed(6500))[1024:];draw=torch.randn(64,20,generator=torch.Generator().manual_seed(6700),dtype=torch.double)
    for path in sorted(a.records.glob('c???.json')):
        meta=json.loads(path.read_text());state=path.parent/(path.stem+'_states')/'risk.pt';assert hashlib.sha256(state.read_bytes()).hexdigest()==meta['state_sha256']
        raw=torch.load(state,map_location='cpu',weights_only=True);ref=raw['reference_seed'];seed=raw['seed'];thresholds=list(raw['thresholds'].values());endpoint=next(e for e in study['endpoints'] if e['reference_seed']==ref and e['seed']==seed)
        err=(decisions(raw['anchor_cheap'],thresholds)!=decisions(raw['anchor_target'],thresholds)).double();residual=raw['anchor_target']-raw['anchor_cheap'];views={}
        for view,value in raw['views'].items():
            actual=(decisions(value['cheap'],thresholds)!=decisions(value['reference'],thresholds)).double().mean(0)
            idx=torch.randperm(len(value['cheap']),generator=torch.Generator().manual_seed(6700))[:512];rows=[]
            for original in endpoint['views'][view]['rows']:
                n=original['budget'];rep=original['replicate'];sample=order[torch.randperm(1024,generator=torch.Generator().manual_seed(6700+rep))[:n]]
                rates=dict(direct=err[sample].mean(0),covariance=propagated(value['cheap'][idx],residual[sample],thresholds,draw,False),isotropic=propagated(value['cheap'][idx],residual[sample],thresholds,draw,True))
                for k,r in rates.items():assert abs(float(r.max())-original['predictions'][k])<1e-12
                rows.append(dict(budget=n,replicate=rep,rates={k:r.tolist() for k,r in rates.items()},mean_consumer_absolute_error={k:float((r-actual).abs().mean()) for k,r in rates.items()}))
            views[view]=dict(actual_by_consumer=actual.tolist(),rows=rows)
        records.append(dict(reference_seed=ref,seed=seed,state_sha256=meta['state_sha256'],views=views))
    stats={}
    for view in ['clean','augmented']:
        stats[view]={}
        for n in [32,64,128,256]:
            means={k:[np.mean([row['mean_consumer_absolute_error'][k] for e in records if e['reference_seed']==ref for row in e['views'][view]['rows'] if row['budget']==n]) for ref in range(6201,6205)] for k in ['direct','covariance','isotropic']}
            stats[view][str(n)]=dict(mae={k:interval(v) for k,v in means.items()},improvement=interval(np.array(means['direct'])-np.array(means['covariance'])))
    result=dict(complete=True,all_primary_forecasts_reproduced=True,independent_implementation=True,
        primary_summary_sha256=hashlib.sha256(a.summary.read_bytes()).hexdigest(),
        source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},statistics=stats,records=records)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(dict(complete=True,statistics=stats),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a)
