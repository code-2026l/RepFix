"""Apply the fixed disagreement choice before checking development utility."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[2]


def interval(values):
    x=np.asarray(values,dtype=float);mean=float(x.mean())
    radius=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return dict(mean=mean,ci95=[mean-radius,mean+radius],values=x.tolist())


def main(args):
    rows=[json.loads(p.read_text()) for p in sorted(args.records.glob('a???.json'))]
    assert len(rows)==6 and all(r['complete'] and not r['smoke'] and r['phase']=='development' for r in rows)
    plan=hashlib.sha256((ROOT/'docs/plans/2026-11-09-alignment-baseline-confirmation.md').read_bytes()).hexdigest()
    assert all(r['plan_sha256']==plan and not r['splits']['test_metrics_computed'] for r in rows)
    result=dict(plan_sha256=plan,choices_frozen_before_confirmation=True,selection_uses_accuracy=False,
        source_record_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.records.glob('a???.json'))},domains={})
    for domain in ['har','radioml']:
        group=sorted([r for r in rows if r['domain']==domain],key=lambda r:r['seed'])
        assert [r['seed'] for r in group]==[800,801,802]
        decisions={}
        for mode in ['feature_map','logit_map','direct_ols']:
            lams=sorted(set(b['lam'] for b in group[0]['branches'] if b['mode']==mode))
            all_candidates={}
            for lam in lams:
                selected=[next(b for b in r['branches'] if b['mode']==mode and b['lam']==lam) for r in group]
                all_candidates[str(lam)]=interval([b['validation']['disagreement'] for b in selected])
            lam=next((v for v in lams if all_candidates[str(v)]['mean']<=.05),lams[-1])
            selected=[next(b for b in r['branches'] if b['mode']==mode and b['lam']==lam) for r in group]
            gain=interval([b['validation']['own']['acc']-r['original_controls']['detach']['own']['acc'] for b,r in zip(selected,group)])
            vs_joint=interval([b['validation']['own']['acc']-r['original_controls']['joint']['own']['acc'] for b,r in zip(selected,group)])
            mse=interval([b['validation']['auxiliary_mse']/r['original_controls']['detach']['auxiliary_mse'] for b,r in zip(selected,group)])
            checks=dict(all_disagreements_within_budget=all(b['validation']['disagreement']<=.05 for b in selected))
            if domain=='har':checks.update(reconstruction_progress=mse['mean']<=.95,primary_within_half_point=abs(gain['mean'])<=.005)
            else:checks.update(positive_primary_gain=gain['ci95'][0]>0,within_half_point_of_joint=vs_joint['mean']>=-.005)
            decisions[mode]=dict(selected_lambda=lam,confirm=all(checks.values()),checks=checks,
                primary_gain=gain,primary_difference_to_joint=vs_joint,auxiliary_mse_ratio=mse,
                disagreement=all_candidates[str(lam)],all_candidate_disagreements=all_candidates)
        result['domains'][domain]=decisions
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({d:{m:dict(lam=v['selected_lambda'],confirm=v['confirm'],changed=v['disagreement']['mean'],gain=v['primary_gain']['mean']) for m,v in item.items()} for d,item in result['domains'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
