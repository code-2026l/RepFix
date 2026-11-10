"""Retrospective full-reference residual-law control; never a practical predictor."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from consumer_routing import decisions
from run_certified_audit import predictor


def main(a):
    torch.set_num_threads(2);rows=[]
    for path in sorted(a.records.glob('c???.json')):
        meta=json.loads(path.read_text());state=path.parent/(path.stem+'_states')/'risk.pt';assert hashlib.sha256(state.read_bytes()).hexdigest()==meta['state_sha256']
        raw=torch.load(state,map_location='cpu',weights_only=True);thresholds=list(raw['thresholds'].values());views={}
        for view,d in raw['views'].items():
            z=d['cheap'].double();old=d['reference'].double();z-=z.mean(1,keepdim=True);old-=old.mean(1,keepdim=True)
            error=(decisions(z,thresholds)!=decisions(old,thresholds)).double().numpy();q=predictor(z,old-z,thresholds)
            r2=[]
            for j in range(3):
                covariance=float(np.mean((q[:,j]-q[:,j].mean())*(error[:,j]-error[:,j].mean())));den=float(q[:,j].var()*error[:,j].var())
                r2.append(covariance*covariance/den if den>1e-15 else 0.)
            views[view]=dict(squared_correlations=r2,mean_error_by_consumer=error.mean(0).tolist(),predicted_mean_by_consumer=q.mean(0).tolist())
        rows.append(dict(reference_seed=raw['reference_seed'],seed=raw['seed'],views=views))
    result=dict(complete=True,analysis='retrospective_oracle_residual_law',all_validation_reference_outputs_used=True,
        operational_query_savings_claim=False,source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},endpoints=rows,
        mean_squared_correlations={view:float(np.mean([r for e in rows for r in e['views'][view]['squared_correlations']])) for view in ['clean','augmented']})
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result['mean_squared_correlations']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a)
