"""Fixed primary/secondary joint-development choices, derived locally."""
import argparse,hashlib,json
from pathlib import Path
from select_joint_alignment import interval

ARCH=['affine','residual32']

def summarize_candidate(rows,controls,domain,consumer_names):
    worst=[max(r['contracts'][c]['disagreement'] for c in consumer_names) for r in rows]
    primary=interval([r['own']['acc']-c['detach']['own']['acc'] for r,c in zip(rows,controls)])
    joint=interval([r['own']['acc']-c['joint']['own']['acc'] for r,c in zip(rows,controls)])
    mse=interval([r['auxiliary_mse']/c['detach']['auxiliary_mse'] for r,c in zip(rows,controls)])
    checks=dict(every_consumer_within_development_budget=all(x<=.025 for x in worst))
    if domain=='har':checks.update(primary_noninferiority=primary['mean']>=-.005,reconstruction_progress=mse['mean']<=.95)
    else:checks.update(positive_primary_gain=primary['ci95'][0]>0,noninferiority_to_joint=joint['mean']>=-.005)
    return dict(mode=rows[0]['mode'],architecture=rows[0]['architecture'],objective=rows[0]['objective'],lam=rows[0]['lam'],
        feasible=all(checks.values()),checks=checks,mean_worst_disagreement=sum(worst)/len(worst),worst_disagreements=worst,
        primary_gain=primary,primary_difference_to_joint=joint,auxiliary_mse_ratio=mse,
        contracts={c:interval([r['contracts'][c]['disagreement'] for r in rows]) for c in rows[0]['contracts']})

def choose(candidates,cell_names):
    eligible=[r for r in candidates if r['objective'] in cell_names and r['feasible']]
    selected=sorted(eligible,key=lambda r:(ARCH.index(r['architecture']),r['lam'],r['mean_worst_disagreement']))[0] if eligible else None
    comparator=None
    if selected:
        point=[r for r in candidates if r['architecture']==selected['architecture'] and r['objective'] in ['kl1','centered_mse','raw_mse']]
        comparator=sorted(point,key=lambda r:(not r['feasible'],r['mean_worst_disagreement']))[0]
    return dict(confirm=selected is not None,cell=selected,point_comparator=comparator,all_candidates=candidates)

def main(a):
    paths=sorted(a.records.glob('b???.json'));records=[json.loads(p.read_text()) for p in paths]
    assert len(records)==6 and all(r['complete'] and not r['smoke'] and r['route']=='joint_guidance' for r in records)
    secondary_paths=sorted(a.secondary.glob('s???.json')) if a.secondary else []
    secondary=[json.loads(p.read_text()) for p in secondary_paths]
    if a.secondary:assert len(secondary)==6 and all(r['complete'] and r['route']=='classification_secondary' for r in secondary)
    out=dict(choices_frozen_before_confirmation=True,selection_uses_accuracy_for_ranking=False,
        plan_sha256=records[0]['plan_sha256'],metric_addendum_plan_sha256=records[0]['metric_addendum_plan_sha256'],
        source_record_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+secondary_paths},domains={})
    for domain in ['har','radioml']:
        group=sorted([r for r in records if r['domain']==domain],key=lambda r:r['seed']);assert [r['seed'] for r in group]==[800,801,802]
        controls=[{b['mode']:b for b in r['branches'] if b['mode'] in ['detach','joint']} for r in group]
        full=[];class_point=[]
        for b in group[0]['branches']:
            if b['mode'] in ['detach','joint']:continue
            rows=[next(x for x in r['branches'] if (x['mode'],x['architecture'],x['objective'],x['lam'])==
                       (b['mode'],b['architecture'],b['objective'],b['lam'])) for r in group]
            full.append(summarize_candidate(rows,controls,domain,['classification','coverage80','coverage50']))
            class_point.append(summarize_candidate(rows,controls,domain,['classification']))
        primary=choose(full,['cell_kl','cell_logodds']);item=dict(primary=primary)
        if secondary:
            sg=sorted([r for r in secondary if r['domain']==domain],key=lambda r:r['seed'])
            for p,r in zip(group,sg):
                assert p['splits']==r['splits'] and p['thresholds']==r['thresholds']
                assert p['branches'][0]['initial_model_sha256']==r['branches'][0]['initial_model_sha256']
                assert p['branches'][0]['initial_optimizer_sha256']==r['branches'][0]['initial_optimizer_sha256']
                assert p['branches'][0]['minibatch_order_sha256']==r['branches'][0]['minibatch_order_sha256']
            for b in sg[0]['branches']:
                rows=[next(x for x in r['branches'] if (x['architecture'],x['objective'],x['lam'])==(b['architecture'],b['objective'],b['lam'])) for r in sg]
                class_point.append(summarize_candidate(rows,controls,domain,['classification']))
            item['classification_secondary']=choose(class_point,['class_cell_kl','class_cell_logodds'])
        out['domains'][domain]=item
    a.out.write_text(json.dumps(out,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({d:{k:{x:v[x] for x in ['confirm','cell','point_comparator']} for k,v in r.items()} for d,r in out['domains'].items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--secondary',type=Path)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
