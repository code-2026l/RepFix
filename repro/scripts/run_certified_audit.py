"""Local exact audit and survey/PPI controls on immutable saved-model outputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from consumer_routing import decisions
from certified_audit import allocate,bounds,decide
from analyze_small_budget_confirmation import interval

QUERIES=[64,128,256,512,1024]


def predictor(z,residual,thresholds):
    e=residual.double();mu=e.mean(0);sigma=(e-mu).square().sum()/((len(e)-1)*e.shape[1])
    normals=torch.randn(64,e.shape[1],generator=torch.Generator().manual_seed(7100),dtype=torch.double)
    noise=normals*float(sigma.sqrt())+mu
    altered=z.double()[None,:,:]+noise[:,None,:]
    return (decisions(altered.flatten(0,1),thresholds).reshape(64,len(z),3)!=decisions(z,thresholds)[None,:,:]).double().mean(0).numpy()


def main(a):
    torch.set_num_threads(2);results=[]
    first=torch.randperm(2048,generator=torch.Generator().manual_seed(6500))[:1024]
    for record in sorted(a.records.glob('c???.json')):
        meta=json.loads(record.read_text());path=record.parent/(record.stem+'_states')/'risk.pt'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['state_sha256']
        raw=torch.load(path,map_location='cpu',weights_only=True);thresholds=list(raw['thresholds'].values());residual=(raw['anchor_target']-raw['anchor_cheap'])[first]
        views={}
        for view,data in raw['views'].items():
            q=predictor(data['cheap'],residual,thresholds);N=len(q)
            errors=(decisions(data['cheap'],thresholds)!=decisions(data['reference'],thresholds)).numpy().astype(int);actual=errors.mean(0);rows=[]
            groupings={};allocations={}
            for method in ['uniform','stratified','reversed']:
                if method=='uniform':groups=[np.arange(N)];scores=[0.]
                else:
                    score=q.max(1);score=1-score if method=='reversed' else score
                    rank=np.argsort(score,kind='stable');groups=np.split(rank,[int(N*.5),int(N*.8),int(N*.95)])
                    scores=[float((score[g]*(1-score[g])).mean()) for g in groups]
                groupings[method]=groups
                allocations[method]={str(n):[n] if method=='uniform' else allocate([len(g) for g in groups],scores,n).tolist() for n in QUERIES}
            for rep in range(20):
                for method,groups in groupings.items():
                    rng=np.random.default_rng(7100+rep);orders=[rng.permutation(len(g)) for g in groups];history=[]
                    for n in QUERIES:
                        alpha=.05/(3*5*3*2*len(groups));b=bounds(errors,groups,orders,allocations[method][str(n)],alpha)
                        status={str(t):decide(b,t) for t in [.01,.05,.10]}
                        wrong=sum((v=='accept' and max(actual)>float(t)) or (v=='reject' and max(actual)<=float(t)) for t,v in status.items())
                        covered=bool(np.all(np.array(b['lower'])<=actual+1e-12) and np.all(np.array(b['upper'])>=actual-1e-12))
                        history.append(dict(queries=n,bounds=b,status=status,wrong_decisions=int(wrong),covers_all_consumers=covered))
                    stop={str(t):next((r['queries'] for r in history if r['status'][str(t)]!='abstain'),2048) for t in [.01,.05,.10]}
                    rows.append(dict(replicate=rep,method=method,history=history,first_decision_queries=stop))
            # Point-estimation controls share exactly the same uniform sample.
            estimates=[];qmean=q.mean(0)
            for rep in range(20):
                indices=np.random.default_rng(7100+rep).permutation(N)
                for n in QUERIES:
                    idx=indices[:n];y=errors[idx];f=q[idx];direct=y.mean(0)
                    fixed=qmean+(y-f).mean(0)
                    cov=((y-y.mean(0))*(f-f.mean(0))).mean(0);var=f.var(0)
                    power=np.clip(np.divide(cov,var,out=np.zeros_like(var),where=var>1e-12),0,1)
                    tuned=direct+power*(qmean-f.mean(0))
                    values=dict(direct=direct,ppi_fixed=fixed,power_tuned=tuned)
                    estimates.append(dict(replicate=rep,queries=n,power=power.tolist(),
                        rates={k:v.tolist() for k,v in values.items()},mae={k:float(np.abs(v-actual).mean()) for k,v in values.items()}))
            views[view]=dict(actual_by_consumer=actual.tolist(),predictor_mean=qmean.tolist(),allocations=allocations,audits=rows,point_estimates=estimates)
        results.append(dict(reference_seed=raw['reference_seed'],seed=raw['seed'],state_sha256=meta['state_sha256'],views=views))
        print(json.dumps(dict(reference=raw['reference_seed'],seed=raw['seed'],audited=True)),flush=True)
    stats={}
    for view in ['clean','augmented']:
        stats[view]={}
        for budget in ['0.01','0.05','0.1']:
            groups={m:[float(np.mean([r['first_decision_queries'][budget] for e in results if e['reference_seed']==ref for r in e['views'][view]['audits'] if r['method']==m])) for ref in range(6201,6205)] for m in ['uniform','stratified','reversed']}
            stats[view][budget]=dict(censored_query_score={k:interval(v) for k,v in groups.items()},
                saving=interval(np.array(groups['uniform'])-np.array(groups['stratified'])),ratio=float(np.mean(groups['stratified'])/np.mean(groups['uniform'])))
    primary=stats['clean']['0.01'];passed=primary['ratio']<=.8 and primary['saving']['ci95'][0]>0
    result=dict(complete=True,phase='saved_model_certified_audit',plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('certified_audit.py')]},
        initial_map_reference_cost=1024,query_grid=QUERIES,primary_utility_passed=bool(passed),statistics=stats,endpoints=results,
        wrong_certified_decisions=sum(r['wrong_decisions'] for e in results for v in e['views'].values() for run in v['audits'] for r in run['history']),
        empirical_noncoverage=sum(not r['covers_all_consumers'] for e in results for v in e['views'].values() for run in v['audits'] for r in run['history']),
        poisson_or_gaussian_validity_assumed_for_certificates=False,official_test_loaded=False)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(primary_passed=bool(passed),statistics=stats,wrong=result['wrong_certified_decisions'],noncoverage=result['empirical_noncoverage']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a)
