"""Frozen cached-anchor estimator on previously unexamined auxiliary-update risk."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from consumer_routing import decisions
from small_budget_risk import forecast
from analyze_small_budget_confirmation import interval


def main(a):
    torch.set_num_threads(2);rows=[];plan_sha=hashlib.sha256(a.plan.read_bytes()).hexdigest()
    order=torch.randperm(2048,generator=torch.Generator().manual_seed(6500));fitting,monitor=order[:1024],order[1024:]
    draws=torch.randn(64,20,generator=torch.Generator().manual_seed(6700),dtype=torch.double)
    for path in sorted(a.records.glob('n???.json')):
        meta=json.loads(path.read_text());assert meta['phase']=='contrastive_native_audit' and meta['secondary_risk_plan_sha256']==plan_sha
        source=a.arrays/(path.stem+'_states')/'risk.pt';assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['secondary_risk_sha256']
        raw=torch.load(source,map_location='cpu',weights_only=True);loo=raw['loo'];assert torch.equal(loo['fit_indices'],fitting)
        thresholds=list(raw['thresholds'].values());target=raw['anchor_target'];residual=target-raw['anchor_cheap'];views={}
        zero_residuals=dict(in_sample=residual[fitting],loo=loo['target']-loo['loo'])
        zero_counts={name+'_count':float((decisions(z,thresholds)!=decisions(loo['target'],thresholds)).double().mean(0).max())
            for name,z in [('in_sample',raw['anchor_cheap'][fitting]),('loo',loo['loo'])]}
        anchor_errors=(decisions(raw['anchor_cheap'],thresholds)!=decisions(target,thresholds)).double()
        for view,values in raw['views'].items():
            actual=float((decisions(values['cheap'],thresholds)!=decisions(values['reference'],thresholds)).double().mean(0).max())
            subset=torch.randperm(len(values['cheap']),generator=torch.Generator().manual_seed(6700))[:512]
            zero={**zero_counts,**{name+'_isotropic':forecast(values['cheap'][subset],r,thresholds,draws,True) for name,r in zero_residuals.items()}}
            repeated=[]
            for n in [32,64]:
                for rep in range(20):
                    idx=monitor[torch.randperm(len(monitor),generator=torch.Generator().manual_seed(6700+rep))[:n]]
                    pred=dict(direct=float(anchor_errors[idx].mean(0).max()),isotropic=forecast(values['cheap'][subset],residual[idx],thresholds,draws,True))
                    repeated.append(dict(queries=n,replicate=rep,predictions=pred))
            errors={name:abs(value-actual) for name,value in zero.items()}
            for n in [32,64]:
                for method in ['direct','isotropic']:
                    errors[method+'_'+str(n)]=float(np.mean([abs(r['predictions'][method]-actual) for r in repeated if r['queries']==n]))
            views[view]=dict(actual=actual,zero_query_predictions=zero,monitor_estimates=repeated,absolute_errors=errors)
        rows.append(dict(reference_seed=raw['reference_seed'],seed=raw['seed'],state_sha256=meta['secondary_risk_sha256'],views=views,
            mean_leverage=float(loo['leverage'].mean())))
    assert {(r['reference_seed'],r['seed']) for r in rows}=={(ref,s) for ref in range(6201,6205) for s in range(6310,6313)}
    stats={}
    for view in ['clean','augmented']:
        groups={method:[float(np.mean([r['views'][view]['absolute_errors'][method] for r in rows if r['reference_seed']==ref])) for ref in range(6201,6205)]
            for method in rows[0]['views'][view]['absolute_errors']}
        difference=interval(np.array(groups['loo_isotropic'])-np.array(groups['isotropic_32']))
        stats[view]=dict(mae={k:interval(v) for k,v in groups.items()},zero_minus_32_isotropic=difference)
    primary=stats['clean'];passed=primary['mae']['loo_isotropic']['mean']<=.01 and primary['zero_minus_32_isotropic']['ci95'][1]<=.0025
    result=dict(complete=True,phase='outcome_blind_audit_transfer_to_new_auxiliary_updates',plan_sha256=plan_sha,transfer_gate_passed=passed,
        statistics=stats,endpoints=rows,initial_paired_anchors=1024,additional_reference_queries=0,noninferiority_margin=.0025,
        finite_corpus_certificate=False,reference_fits_reused=True,independently_pretrained_backbones=False,official_test_loaded=False,
        source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(complete=True,transfer_gate_passed=passed,statistics=stats)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--arrays',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
