"""Apply the published ResNet cross-task screen gate locally, without retuning."""
import argparse,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main(a):
    paths=sorted(a.records.glob('v???.json'));rows=[json.loads(p.read_text()) for p in paths]
    assert rows and all(r['complete'] and not r['smoke'] and r['seed']==1800 for r in rows)
    queue=json.loads((a.records/'queue.json').read_text());assert queue['finished'] and not queue['running']
    if len(rows)!=8:assert queue['failed'],'An incomplete screen needs its stopped-queue evidence'
    assert not any(r['splits']['official_test_loaded'] or r['splits']['test_metrics_computed'] for r in rows)
    assert len({r['minibatch_order_sha256'] for r in rows})==1
    assert len({r['initial_encoder_sha256'] for r in rows})==1
    assert all(r['bn_buffers_unchanged'] for r in rows)
    frozen=next(r for r in rows if r['mode']=='F');unconstrained=next(r for r in rows if r['mode']=='U')
    f,u=frozen['fine']['acc'],unconstrained['fine']['acc']
    changed=next(b for b in unconstrained['branches'] if b['mode']=='fixed')['contracts']['classification']['disagreement']
    gate=dict(primary_gain=u-f>=.02,legacy_changes=changed>=.10)
    result=dict(phase='vision_cross_task_screen',plan_sha256=rows[0]['plan_sha256'],seed=1800,
        source_record_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        frozen_fine_accuracy=f,unconstrained_fine_accuracy=u,unconstrained_legacy_disagreement=changed,
        resource_scientific_gate=gate,families={},official_test_evaluated=False,full_grid_completed=len(rows)==8,
        execution=dict(completed=len(queue['completed']),failed=len(queue['failed']),unlaunched=len(queue['pending']),
            interrupted=8-len(queue['completed'])-len(queue['failed'])-len(queue['pending']),
            failed_job_ids=[r['id'] for r in queue['failed']],stop_reason=queue.get('error')))
    for family in ['P','B']:
        candidates=[]
        for r in rows:
            if r['mode']!=family:continue
            reader=next(b for b in r['branches'] if b['mode']==('protected' if family=='P' else 'feature_map'))
            d=reader['contracts']['classification']['disagreement']
            checks=dict(legacy_within_budget=d<=.05,within_half_point_of_U=r['fine']['acc']>=u-.005,
                at_least_two_points_above_F=r['fine']['acc']>=f+.02)
            candidates.append(dict(lam=r['lam'],fine_accuracy=r['fine']['acc'],disagreement=d,checks=checks,feasible=all(checks.values())))
        feasible=[r for r in candidates if r['feasible']]
        selected=sorted(feasible,key=lambda r:(-r['fine_accuracy'],r['lam']))[0] if feasible and all(gate.values()) and len(rows)==8 else None
        result['families'][family]=dict(candidates=candidates,selected=selected,continue_development=selected is not None)
    result['continue_development']=any(v['continue_development'] for v in result['families'].values())
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
