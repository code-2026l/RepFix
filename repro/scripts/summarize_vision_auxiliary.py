"""Apply the fixed auxiliary-development gates to completed, valid pilots."""
import argparse
import hashlib
import json
from pathlib import Path


def first_pilot(row):
    assert row['complete'] and row['phase']=='development'
    assert row['reference_seed']==6200 and row['seed']==6300
    assert not row['splits']['official_test_loaded']
    branches={b['mode']:b for b in row['branches']}
    assert set(branches)=={'F','N','H','D','J0.01','J0.1','J1'}
    moments={branches[k]['warm_optimizer_sha256'] for k in ['D','J0.01','J0.1','J1']}
    assert len(moments)==1, 'The paired decoder warm states differ'
    n=row['splits']['validation']
    correct=lambda b:round(b['validation']['own']['acc']*n)
    detached=branches['D'];candidates=[]
    for mode in ['J0.01','J0.1','J1']:
        branch=branches[mode];dose=branch['alpha']
        gain=correct(branch)-correct(detached)
        mse=branch['validation']['reconstruction_mse']/detached['detached_mse_by_dose'][str(dose)]
        preserved=branch['validation']['old_decisions_unchanged']
        candidates.append(dict(mode=mode,alpha=dose,gain_correct=gain,gain_pp=100*gain/n,
            dose_matched_mse_ratio=mse,old_decisions_unchanged=preserved,
            extend=gain*500>=n and mse<=.95 and preserved))
    return dict(reference_seed=6200,seed=6300,validation_images=n,
        accuracies={k:b['validation']['own']['acc'] for k,b in branches.items()},
        candidates=candidates,extend_doses=[v['alpha'] for v in candidates if v['extend']],
        detached_minus_rich_pp=100*(correct(detached)-correct(branches['H']))/n,
        ce_transfer_meets_pilot_gate=(correct(detached)-correct(branches['H']))*500>=n,
        confirm_ce_only_if_no_auxiliary=not any(v['extend'] for v in candidates)
            and (correct(detached)-correct(branches['H']))*500>=n)


def main(paths,out):
    rows=[json.loads(p.read_text()) for p in paths]
    assert len({r['arch'] for r in rows})==len(rows)
    result=dict(complete=len(rows)==2,phase='development_first_pilot',
        official_test_loaded=False,confirmation_runs=0,
        source_sha256={'summarize_vision_auxiliary.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        record_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        backbones={r['arch']:first_pilot(r) for r in rows})
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result['backbones'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.records,a.out)
