"""Validate nonlinear visual student selection and historical-reference confirmation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_har_transfer import ROOT, RESULTS, digest, interval, save
from consumer_routing import decisions

PLAN=ROOT/'docs/plans/2026-11-11-vit-nonlinear-student.md'
CHOICE=RESULTS/'vit_nonlinear_student_choice_20261111.json'


def load(stage,records,arrays):
    count,prefix=(1,'p') if stage=='pilot' else (12,'c')
    for i in range(count):
        key=f'{prefix}{i:03d}'
        row=json.loads((records/(key+'.json')).read_text())
        assert row['complete'] and row['device']=='cpu' and row['stage']==stage
        assert row['plan_sha256']==digest(PLAN) and not row['official_test_loaded']
        assert (row['reference_seed'],row['seed'])==((6200,6300) if stage=='pilot' else (6201+i//3,6310+i%3))
        if stage=='confirmation':
            assert row['choice_sha256']==digest(CHOICE)
        tensor=arrays/(key+'_states/replay.pt')
        assert digest(tensor)==row['replay_sha256']
        value=torch.load(tensor,map_location='cpu',weights_only=True)
        assert not (decisions(value['old'],value['thresholds'])!=decisions(value['protected'],value['thresholds'])).any()
        for arm in row['outcomes']:
            assert arm['accuracy']==(value['predictions'][arm['key']]==value['labels']).double().mean().item()
            assert arm['teacher_accuracy']==(value['teachers'][arm['teacher']]==value['labels']).double().mean().item()
        yield row


def select(rows):
    outcomes=rows[0]['outcomes']
    candidates=[]
    for width in (128,512):
        arms={row['teacher']:row for row in outcomes if row['width']==width}
        gain={name:100*(arms['joint']['accuracy']-arms[name]['accuracy']) for name in ('two_view','double_epoch')}
        retention=100*(arms['joint']['accuracy']-arms['joint']['teacher_accuracy'])
        candidates.append(dict(width=width,gains_pp=gain,vs_teacher_pp=retention,
                               qualifies=min(gain.values())>=.2 and retention>=-.25))
    valid=[c for c in candidates if c['qualifies']]
    chosen=valid[0] if valid else max(candidates,key=lambda c:min(c['gains_pp'].values()))
    return dict(plan_sha256=digest(PLAN),pilot_passed=bool(valid),width=chosen['width'],candidates=candidates)


def confirm(rows):
    refs=[]
    for row in rows:
        arms={arm['teacher']:arm for arm in row['outcomes']}
        refs.append(dict(reference_seed=row['reference_seed'],seed=row['seed'],
                         gains_pp={name:100*(arms['joint']['accuracy']-arms[name]['accuracy'])
                                   for name in ('two_view','double_epoch')},
                         vs_teacher_pp=100*(arms['joint']['accuracy']-arms['joint']['teacher_accuracy'])))
    groups=[]
    for seed in range(6201,6205):
        group=[row for row in refs if row['reference_seed']==seed]
        assert len(group)==3
        groups.append(dict(reference_seed=seed,
                           gains_pp={name:float(np.mean([row['gains_pp'][name] for row in group]))
                                     for name in ('two_view','double_epoch')},
                           vs_teacher_pp=float(np.mean([row['vs_teacher_pp'] for row in group]))))
    gains={name:interval([row['gains_pp'][name] for row in groups]) for name in ('two_view','double_epoch')}
    retention=interval([row['vs_teacher_pp'] for row in groups])
    return dict(plan_sha256=digest(PLAN),choice_sha256=digest(CHOICE),gains_pp=gains,
                vs_teacher_pp=retention,passed=all(v['ci95'][0]>0 for v in gains.values())
                and retention['ci95'][0]>-.25,references=refs,reference_groups=groups)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('pilot','confirmation'))
    parser.add_argument('--records',type=Path)
    parser.add_argument('--arrays',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    default=ROOT/'tmp'/('conditional_contrastivevitnonlinear'+args.stage)
    rows=list(load(args.stage,args.records or default,args.arrays or default))
    result=select(rows) if args.stage=='pilot' else confirm(rows)
    path=CHOICE if args.stage=='pilot' else RESULTS/'vit_nonlinear_student_confirmation_20261111.json'
    save(path,result)
    print(json.dumps(result,indent=2))
