"""Local fixed development gates; no testing or outcome-based grid expansion."""
import argparse,hashlib,json
from pathlib import Path
from select_joint_alignment import interval

ARCHITECTURES=['affine','quadratic','residual32']
CELLS=['cell_kl','cell_logodds'];POINTS=['kl1','kl2','kl4','centered_mse']

def main(a):
    paths=sorted(a.records.glob('a???.json'));rows=[json.loads(p.read_text()) for p in paths]
    assert len(rows)==6 and all(r['complete'] and not r['smoke'] and r['phase']=='development' for r in rows)
    out=dict(choices_frozen_before_confirmation=True,stage='fixed_features',selection_uses_accuracy_for_ranking=False,
        plan_sha256=rows[0]['plan_sha256'],metric_addendum_plan_sha256=rows[0]['metric_addendum_plan_sha256'],
        source_record_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},domains={})
    for domain in ['har','radioml']:
        group=sorted([r for r in rows if r['domain']==domain],key=lambda r:r['seed']);assert [r['seed'] for r in group]==[800,801,802]
        primary=interval([r['donor']['own']['acc']-r['controls']['detach']['own']['acc'] for r in group])
        joint=interval([r['donor']['own']['acc']-r['controls']['joint']['own']['acc'] for r in group])
        mse=interval([r['donor']['auxiliary_mse']/r['controls']['detach']['auxiliary_mse'] for r in group])
        utility=dict(primary_noninferior=primary['mean']>=-.005,reconstruction_progress=mse['mean']<=.95) if domain=='har' else dict(
            positive_primary_gain=primary['ci95'][0]>0,noninferiority_to_joint=joint['mean']>=-.005)
        candidates=[]
        for arch in ARCHITECTURES:
            for objective in CELLS+POINTS:
                selected=[next(b for b in r['branches'] if b['architecture']==arch and b['objective']==objective) for r in group]
                worst=[max(x['disagreement'] for x in b['contracts'].values()) for b in selected]
                feasible=all(x<=.025 for x in worst) and all(utility.values())
                candidates.append(dict(architecture=arch,objective=objective,feasible=feasible,
                    mean_worst_disagreement=sum(worst)/len(worst),worst_disagreements=worst,
                    contracts={c:interval([b['contracts'][c]['disagreement'] for b in selected]) for c in selected[0]['contracts']}))
        eligible=[r for r in candidates if r['objective'] in CELLS and r['feasible']]
        cell=sorted(eligible,key=lambda r:(ARCHITECTURES.index(r['architecture']),r['mean_worst_disagreement']))[0] if eligible else None
        point=None
        if cell:
            options=[r for r in candidates if r['architecture']==cell['architecture'] and r['objective'] in POINTS]
            point=sorted(options,key=lambda r:(not r['feasible'],r['mean_worst_disagreement']))[0]
        out['domains'][domain]=dict(confirm=cell is not None,route='fixed_features' if cell else 'joint_guidance_required',
            primary_gain=primary,primary_difference_to_joint=joint,auxiliary_mse_ratio=mse,utility_checks=utility,
            cell=cell,point_comparator=point,all_candidates=candidates)
    a.out.write_text(json.dumps(out,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({k:{x:v[x] for x in ['confirm','route','cell','point_comparator']} for k,v in out['domains'].items()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
