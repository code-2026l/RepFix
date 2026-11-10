"""Freeze auxiliary choices before any independent-reference outcomes."""
import argparse,hashlib,json
from pathlib import Path


def main(pilot,records,out):
    first=json.loads(pilot.read_text());assert first['complete']
    rows=[json.loads(p.read_text()) for p in records];result={}
    for arch,initial in first['backbones'].items():
        candidates=[]
        for alpha in initial['extend_doses']:
            data=[r for r in rows if r['arch']==arch]
            assert {r['seed'] for r in data}=={6300,6301,6302}
            assert all(r['complete'] and r['reference_seed']==6200 and not r['splits']['official_test_loaded'] for r in data)
            values=[]
            for row in data:
                branches={b['mode']:b for b in row['branches']};d=branches['D'];j=branches[f'J{alpha:g}'];n=row['splits']['validation']
                diff=round(j['validation']['own']['acc']*n)-round(d['validation']['own']['acc']*n)
                values.append(dict(gain_correct=diff,gain_pp=100*diff/n,
                    mse_ratio=j['validation']['reconstruction_mse']/d['detached_mse_by_dose'][str(alpha)],
                    unchanged=j['validation']['old_decisions_unchanged']))
            eligible=all(v['gain_correct']>0 and v['unchanged'] for v in values)
            eligible=eligible and sum(v['gain_correct'] for v in values)*500>=3*n and sum(v['mse_ratio'] for v in values)/3<=.95
            candidates.append(dict(alpha=alpha,values=values,eligible=eligible))
        passing=sorted(v['alpha'] for v in candidates if v['eligible'])
        selected=passing[:1]
        result[arch]=dict(candidates=candidates,selected_doses=selected,
            confirm_auxiliary=bool(selected),confirm_ce_only=not selected and initial['ce_transfer_meets_pilot_gate'])
    value=dict(complete=True,phase='development_choice',official_test_loaded=False,
        first_pilot_choice_sha256=hashlib.sha256(pilot.read_bytes()).hexdigest(),
        record_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in records},
        source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},backbones=result)
    out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--records',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.pilot,a.records,a.out)
