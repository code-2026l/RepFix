"""Sharp finite empirical coupling controls, not real network-update outcomes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from consumer_routing import decisions
from run_certified_audit import predictor


def extrema(cost):
    a,b=linear_sum_assignment(cost);c,d=linear_sum_assignment(cost,maximize=True)
    return float(cost[a,b].mean()),float(cost[c,d].mean()),b,d


def main(a):
    torch.set_num_threads(2);audits=json.loads(a.audit.read_text());rows=[]
    fit=torch.randperm(2048,generator=torch.Generator().manual_seed(6500))[:1024]
    for record in sorted(a.records.glob('c???.json')):
        meta=json.loads(record.read_text());state=record.parent/(record.stem+'_states')/'risk.pt';assert hashlib.sha256(state.read_bytes()).hexdigest()==meta['state_sha256']
        raw=torch.load(state,map_location='cpu',weights_only=True);thresholds=list(raw['thresholds'].values());views={}
        for view,values in raw['views'].items():
            z=values['cheap'].double();old=values['reference'].double();z-=z.mean(1,keepdim=True);old-=old.mean(1,keepdim=True)
            observed=(decisions(z,thresholds)!=decisions(old,thresholds)).double().numpy()
            q=predictor(z,(raw['anchor_target']-raw['anchor_cheap'])[fit],thresholds)
            correlations=[]
            for j in range(3):
                cov=float(np.mean((q[:,j]-q[:,j].mean())*(observed[:,j]-observed[:,j].mean())));var=float(q[:,j].var());vy=float(observed[:,j].var())
                r2=0. if min(var,vy)<1e-15 else cov*cov/(var*vy)
                correlations.append(dict(consumer=j,squared_correlation=r2,oracle_linear_residual_variance_fraction=1-r2))
            index=torch.randperm(len(z),generator=torch.Generator().manual_seed(7200))[:256];zz=z[index];e=old[index]-zz
            reference=decisions(zz,thresholds);cost=(decisions((zz[:,None,:]+e[None,:,:]).reshape(-1,zz.shape[1]),thresholds).reshape(256,256,3)!=reference[:,None,:]).numpy().astype(int)
            bounds=[];base_prediction=predictor(zz,e,thresholds).mean(0)
            for j in range(3):
                lo,hi,minimum,maximum=extrema(cost[:,:,j]);actual=float(observed[index,j].mean())
                assert lo-1e-12<=actual<=hi+1e-12
                for assignment in [minimum,maximum]:
                    perm=e[assignment]
                    assert torch.allclose(perm.mean(0),e.mean(0),atol=1e-12,rtol=0)
                    assert torch.allclose(perm.T@perm,e.T@e,atol=1e-9,rtol=1e-12)
                    assert np.max(np.abs(predictor(zz,perm,thresholds).mean(0)-base_prediction))<1e-12
                bounds.append(dict(consumer=j,paired_error=actual,minimum_error=lo,maximum_error=hi,width=hi-lo,unchanged_gaussian_prediction=float(base_prediction[j])))
            views[view]=dict(predictor_error_correlations=correlations,coupling_bounds=bounds,controlled_examples=256,
                marginal_first_second_moments_preserved=True,gaussian_forecasts_unchanged=True)
        rows.append(dict(reference_seed=raw['reference_seed'],seed=raw['seed'],state_sha256=meta['state_sha256'],views=views))
    summary={view:dict(mean_squared_correlation=float(np.mean([c['squared_correlation'] for e in rows for c in e['views'][view]['predictor_error_correlations']])),
        mean_coupling_width=float(np.mean([c['width'] for e in rows for c in e['views'][view]['coupling_bounds']])),
        minimum_coupling_width=min(c['width'] for e in rows for c in e['views'][view]['coupling_bounds'])) for view in ['clean','augmented']}
    result=dict(complete=True,analysis='retrospective_mechanism_and_constructed_permutation_control',official_test_loaded=False,
        constructed_couplings_are_real_trained_networks=False,plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
        source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},summary=summary,endpoints=rows)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a)
