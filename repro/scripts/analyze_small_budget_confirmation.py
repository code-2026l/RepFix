"""Local nested-reference confirmation of a frozen risk estimator."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from small_budget_risk import forecast
from consumer_routing import decisions


def interval(values):
    values=np.asarray(values);assert values.shape==(4,)
    mean=float(values.mean());rad=float(t.ppf(.975,3)*values.std(ddof=1)/2)
    return dict(mean=mean,ci95=[mean-rad,mean+rad],reference_values=values.tolist(),df=3)


def main(a):
    torch.set_num_threads(2);paths=sorted(a.records.glob('c???.json'));assert len(paths)==12
    monitor=torch.randperm(2048,generator=torch.Generator().manual_seed(6500))[1024:]
    draws=torch.randn(64,20,generator=torch.Generator().manual_seed(6700),dtype=torch.double);endpoints=[]
    for path in paths:
        record=json.loads(path.read_text());assert record['complete'] and not record['official_test_loaded']
        state=path.parent/(path.stem+'_states')/'risk.pt';assert hashlib.sha256(state.read_bytes()).hexdigest()==record['state_sha256']
        raw=torch.load(state,map_location='cpu',weights_only=True);thresholds=list(raw['thresholds'].values())
        errors=(decisions(raw['anchor_cheap'],thresholds)!=decisions(raw['anchor_target'],thresholds)).double()
        residual=raw['anchor_target']-raw['anchor_cheap'];views={}
        for name,view in raw['views'].items():
            actual=float((decisions(view['cheap'],thresholds)!=decisions(view['reference'],thresholds)).double().mean(0).max())
            sample=torch.randperm(len(view['cheap']),generator=torch.Generator().manual_seed(6700))[:512];rows=[]
            for n in [32,64,128,256]:
                for rep in range(20):
                    idx=monitor[torch.randperm(len(monitor),generator=torch.Generator().manual_seed(6700+rep))[:n]]
                    pred=dict(direct=float(errors[idx].mean(0).max()),covariance=forecast(view['cheap'][sample],residual[idx],thresholds,draws),
                        isotropic=forecast(view['cheap'][sample],residual[idx],thresholds,draws,True))
                    rows.append(dict(budget=n,replicate=rep,predictions=pred))
            views[name]=dict(actual=actual,rows=rows)
        endpoints.append(dict(reference_seed=raw['reference_seed'],seed=raw['seed'],state_sha256=record['state_sha256'],views=views))
        print(json.dumps(dict(reference=raw['reference_seed'],seed=raw['seed'],scored=True)),flush=True)
    assert {(e['reference_seed'],e['seed']) for e in endpoints}=={(r,s) for r in range(6201,6205) for s in range(6310,6313)}
    stats={}
    for view in ['clean','augmented']:
        stats[view]={}
        for n in [32,64,128,256]:
            groups={k:[] for k in ['direct','covariance','isotropic']}
            for ref in range(6201,6205):
                own=[e['views'][view] for e in endpoints if e['reference_seed']==ref]
                for k in groups:groups[k].append(float(np.mean([abs(row['predictions'][k]-e['actual']) for e in own for row in e['rows'] if row['budget']==n])))
            comparisons={k:interval(np.array(groups[k])-np.array(groups['covariance'])) for k in ['direct','isotropic']}
            stats[view][str(n)]=dict(mae={k:interval(v) for k,v in groups.items()},improvement=comparisons,
                relative_reduction=1-np.mean(groups['covariance'])/np.mean(groups['direct']))
    passed=all(stats['clean'][str(n)]['relative_reduction']>=.2 and stats['clean'][str(n)]['improvement']['direct']['ci95'][0]>0 for n in [32,64])
    value=dict(complete=True,phase='saved_model_reference_confirmation',official_test_loaded=False,
        confirm_primary=bool(passed),statistics=stats,endpoints=endpoints,
        source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        fixed_estimator_source_sha256=hashlib.sha256(Path(__file__).with_name('small_budget_risk.py').read_bytes()).hexdigest(),
        access=dict(paired_map_fit=1024,extra_monitor_budgets=[32,64,128,256],unlabeled_current_sample=512),
        reference_units=4,continuations_per_reference=3,resamples_are_not_independent_models=True)
    a.out.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(confirm=bool(passed),statistics=stats),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a)
