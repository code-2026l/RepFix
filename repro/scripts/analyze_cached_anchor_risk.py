"""Zero-new-query controls using the mapping's existing paired fitting anchors."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from consumer_routing import decisions
from small_budget_risk import forecast
from analyze_small_budget_confirmation import interval


def main(a):
    torch.set_num_threads(2);rows=[];first=torch.randperm(2048,generator=torch.Generator().manual_seed(6500))[:1024]
    prior=json.loads(a.prior.read_text());draws=torch.randn(64,20,generator=torch.Generator().manual_seed(6700),dtype=torch.double)
    for path in sorted(a.records.glob('l???.json')):
        meta=json.loads(path.read_text());state=a.arrays/(path.stem+'_states')/'loo.pt';assert hashlib.sha256(state.read_bytes()).hexdigest()==meta['state_sha256']
        loo=torch.load(state,map_location='cpu',weights_only=True);cname='c'+path.stem[1:];rawpath=a.original/(cname+'_states')/'risk.pt'
        assert hashlib.sha256(rawpath.read_bytes()).hexdigest()==meta['raw_state_sha256'];raw=torch.load(rawpath,map_location='cpu',weights_only=True)
        assert torch.equal(first,loo['fit_indices']);thresholds=list(raw['thresholds'].values());y=loo['target'];sampled=raw['anchor_cheap'][first]
        residuals=dict(in_sample=y-sampled,loo=y-loo['loo'])
        predictions={name+'_count':float((decisions(pred,thresholds)!=decisions(y,thresholds)).double().mean(0).max())
            for name,pred in [('in_sample',sampled),('loo',loo['loo'])]}
        previous=next(r for r in prior['endpoints'] if (r['reference_seed'],r['seed'])==(meta['reference_seed'],meta['seed']));views={}
        for view,values in raw['views'].items():
            actual=float((decisions(values['cheap'],thresholds)!=decisions(values['reference'],thresholds)).double().mean(0).max())
            subset=torch.randperm(len(values['cheap']),generator=torch.Generator().manual_seed(6700))[:512]
            pred={**predictions,**{name+'_isotropic':forecast(values['cheap'][subset],residual,thresholds,draws,True) for name,residual in residuals.items()}}
            errors={k:abs(v-actual) for k,v in pred.items()}
            for n in [32,64]:
                errors['new_queries_'+str(n)]=float(np.mean([abs(r['predictions']['isotropic']-actual) for r in previous['views'][view]['rows'] if r['budget']==n]))
            views[view]=dict(actual=actual,predictions=pred,absolute_errors=errors)
        rows.append(dict(reference_seed=meta['reference_seed'],seed=meta['seed'],views=views,
            leverage_mean=float(loo['leverage'].mean()),leverage_max=float(loo['leverage'].max()),
            loo_residual_scale_ratio=float(residuals['loo'].double().square().mean().sqrt()/residuals['in_sample'].double().square().mean().sqrt())))
    assert len(rows)==12;stats={}
    for view in ['clean','augmented']:
        values={method:[float(np.mean([r['views'][view]['absolute_errors'][method] for r in rows if r['reference_seed']==ref])) for ref in range(6201,6205)]
            for method in rows[0]['views'][view]['absolute_errors']}
        stats[view]=dict(mae={k:interval(v) for k,v in values.items()},
            improvement_over_32={k:interval(np.array(values['new_queries_32'])-np.array(values[k])) for k in ['in_sample_count','in_sample_isotropic','loo_count','loo_isotropic']})
    result=dict(complete=True,phase='retrospective_cached_anchor_access_control',initial_paired_anchors=1024,additional_reference_queries=0,
        finite_corpus_certificate=False,fresh_model_confirmation=False,official_test_loaded=False,source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        statistics=stats,endpoints=rows,mean_leverage=float(np.mean([r['leverage_mean'] for r in rows])),
        mean_residual_scale_inflation=float(np.mean([r['loo_residual_scale_ratio'] for r in rows])))
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(complete=True,mean_leverage=result['mean_leverage'],statistics=stats)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--arrays',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
