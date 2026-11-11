"""Apply frozen cross-view screening and reference-level confirmation gates."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from analyze_small_budget_confirmation import interval


def load(folder):
    return [json.loads(p.read_text()) for p in sorted(folder.glob('*.json')) if p.name[0] in 'pdc' and p.stem[1:].isdigit()]


def compare(records,weights):
    output=[]
    for seed in sorted({r['seed'] for r in records}):
        group=[r for r in records if r['seed']==seed];d=next(r for r in group if r['alpha']==0)
        for alpha in weights:
            j=next(r for r in group if r['alpha']==alpha)
            output.append(dict(seed=seed,alpha=alpha,baseline_accuracy=d['validation']['acc'],joint_accuracy=j['validation']['acc'],
                gain_correct=round(5000*j['validation']['acc'])-round(5000*d['validation']['acc']),
                gain_pp=100*(j['validation']['acc']-d['validation']['acc']),auxiliary_ratio=j['contrastive_loss']/d['contrastive_loss']))
    return output


def main(a):
    records=load(a.records);assert records and all(r['complete'] for r in records)
    result=dict(complete=True,stage=a.stage,official_test_loaded=False,
        source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if a.stage=='pilot':
        assert len(records)==4
        replay=json.loads(a.replay.read_text());assert replay['complete'] and not any(replay['old_changed_by_consumer'])
        d=next(r for r in replay['branches'] if r['alpha']==0);assert d['prior_D_model_exact'] and d['prior_D_logit_max_error']==0
        rows=compare(records,[.01,.1,1.]);chosen=[r['alpha'] for r in rows if r['gain_correct']>=10 and r['auxiliary_ratio']<=.95]
        result.update(comparisons=rows,extend_weights=chosen,replay_sha256=hashlib.sha256(a.replay.read_bytes()).hexdigest())
    elif a.stage=='development':
        pilot=json.loads(a.pilot_choice.read_text());weights=pilot['extend_weights'];records+=load(a.pilot_records)
        assert {r['seed'] for r in records}=={6300,6301,6302}
        rows=compare(records,weights);summaries={}
        for alpha in weights:
            group=[r for r in rows if r['alpha']==alpha];gains=[r['gain_correct']/50 for r in group];ratios=[r['auxiliary_ratio'] for r in group]
            summaries[str(alpha)]=dict(gain_pp=float(np.mean(gains)),auxiliary_ratio=float(np.mean(ratios)),all_seed_gains_positive=all(g>0 for g in gains),
                passes=all(g>0 for g in gains) and np.mean(gains)>=.20 and np.mean(ratios)<=.95)
        passing=[alpha for alpha in weights if summaries[str(alpha)]['passes']]
        result.update(comparisons=rows,weights=summaries,selected_weight=min(passing) if passing else None)
    else:
        assert {(r['reference_seed'],r['seed']) for r in records}=={(ref,s) for ref in range(6201,6205) for s in range(6310,6313)}
        assert all(r.get('baseline') for r in records)
        groups=[]
        for ref in range(6201,6205):
            rr=[r for r in records if r['reference_seed']==ref]
            groups.append(dict(reference_seed=ref,gain_pp=float(np.mean([100*(r['validation']['acc']-r['baseline']['accuracy']) for r in rr])),
                auxiliary_ratio=float(np.mean([r['contrastive_loss']/r['baseline']['contrastive_loss'] for r in rr]))))
        gain=interval([r['gain_pp'] for r in groups]);aux=interval([r['auxiliary_ratio'] for r in groups])
        result.update(reference_groups=groups,gain_pp=gain,auxiliary_ratio=aux,
            benefit_confirmed=gain['ci95'][0]>0 and aux['ci95'][1]<=.95,compatibility_and_serving_audit_required=True)
    a.out.write_text(json.dumps(result,indent=2,default=bool)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result,default=bool))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','development','confirmation'],required=True);p.add_argument('--records',type=Path,required=True)
    p.add_argument('--replay',type=Path);p.add_argument('--pilot-choice',type=Path);p.add_argument('--pilot-records',type=Path);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
