"""Freeze fitting-margin rescue from the original pilot and replay its choice."""
import argparse
import json
from pathlib import Path

import torch

from analyze_har_transfer import ROOT,RESULTS,digest,save

PLAN=ROOT/'docs/plans/2026-11-11-vit-current-rescue.md'
CHOICE=RESULTS/'vit_rescue_choice_20261111.json'


def run(records,arrays):
    row=json.loads((records/'p000.json').read_text())
    assert row['complete'] and row['device']=='cpu' and row['plan_sha256']==digest(PLAN)
    assert row['reference_seed']==6200 and row['seed']==6300 and not row['official_test_loaded']
    path=arrays/'p000_states/replay.pt'
    assert digest(path)==row['replay_sha256']
    data=torch.load(path,map_location='cpu',weights_only=True)
    candidates=[]
    for percentile in (.1,.25,.5):
        arms={}
        for name in ('joint','two_view','double_epoch'):
            key=f'{name}/{percentile}'
            arm=row['outcomes'][key]
            cutoff=torch.quantile(data['fit_margins'][name],percentile).item()
            assert cutoff==arm['cutoff']
            selected=data['val_margins'][name]<=cutoff
            pred=data['predictions'][key]
            assert arm['accuracy']==(pred==data['labels']).double().mean().item()
            assert arm['fraction']==selected.double().mean().item()
            assert torch.equal(pred[selected],data['teachers'][name][selected])
            arms[name]=arm
        gain={name:100*(arms['joint']['accuracy']-arms[name]['accuracy']) for name in ('two_view','double_epoch')}
        retention=100*(arms['joint']['accuracy']-arms['joint']['teacher_accuracy'])
        candidates.append(dict(percentile=percentile,gains_pp=gain,vs_teacher_pp=retention,
                               fractions={name:arm['fraction'] for name,arm in arms.items()},
                               qualifies=min(gain.values())>=.2 and retention>=-.25))
    valid=[row for row in candidates if row['qualifies']]
    chosen=valid[0] if valid else max(candidates,key=lambda row:row['vs_teacher_pp'])
    return dict(plan_sha256=digest(PLAN),pilot_passed=bool(valid),percentile=chosen['percentile'],candidates=candidates)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--records',type=Path)
    parser.add_argument('--arrays',type=Path)
    args=parser.parse_args()
    local=ROOT/'tmp/conditional_contrastivevitrescuepilot'
    torch.set_num_threads(4)
    result=run(args.records or local,args.arrays or local)
    save(CHOICE,result)
    print(json.dumps(result,indent=2))
