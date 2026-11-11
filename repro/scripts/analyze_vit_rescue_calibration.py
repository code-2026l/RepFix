"""Freeze the routing fraction using four training-only validation reference means."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_har_transfer import ROOT,RESULTS,digest,interval,save

PLAN=ROOT/'docs/plans/2026-11-11-vit-rescue-calibration.md'
PILOT=RESULTS/'vit_rescue_choice_20261111.json'
OUT=RESULTS/'vit_rescue_calibration_20261111.json'


def run(records,arrays):
    rows=[]
    for i in range(12):
        key=f'd{i:03d}'
        record=json.loads((records/(key+'.json')).read_text())
        assert record['complete'] and record['device']=='cpu' and record['stage']=='calibration'
        assert record['plan_sha256']==digest(PLAN) and not record['official_test_loaded']
        assert record['reference_seed']==6201+i//3 and record['seed']==6310+i%3
        replay=arrays/(key+'_states/replay.pt')
        assert digest(replay)==record['replay_sha256']
        value=torch.load(replay,map_location='cpu',weights_only=True)
        for key,outcome in record['outcomes'].items():
            name,percentile=key.split('/')
            cutoff=torch.quantile(value['fit_margins'][name],float(percentile)).item()
            assert cutoff==outcome['cutoff']
            assert (value['predictions'][key]==value['labels']).double().mean().item()==outcome['accuracy']
        rows.append(record)
    candidates=[]
    for percentile in (.1,.25,.5):
        groups=[]
        for ref in range(6201,6205):
            group=[r for r in rows if r['reference_seed']==ref]
            arms=[{name:r['outcomes'][f'{name}/{percentile}'] for name in ('joint','two_view','double_epoch')} for r in group]
            groups.append(dict(reference_seed=ref,
                gains_pp={name:float(np.mean([100*(a['joint']['accuracy']-a[name]['accuracy']) for a in arms]))
                          for name in ('two_view','double_epoch')},
                vs_teacher_pp=float(np.mean([100*(a['joint']['accuracy']-a['joint']['teacher_accuracy']) for a in arms])),
                fraction=float(np.mean([a['joint']['fraction'] for a in arms]))))
        gains={name:interval([r['gains_pp'][name] for r in groups]) for name in ('two_view','double_epoch')}
        retention=interval([r['vs_teacher_pp'] for r in groups])
        candidates.append(dict(percentile=percentile,gains_pp=gains,versus_teacher_pp=retention,
                               fraction=interval([r['fraction'] for r in groups]),
                               qualifies=all(g['ci95'][0]>0 for g in gains.values()) and retention['ci95'][0]>-.25,
                               reference_groups=groups))
    valid=[r for r in candidates if r['qualifies']]
    chosen=valid[0] if valid else candidates[-1]
    return dict(plan_sha256=digest(PLAN),pilot_choice_sha256=digest(PILOT),passed=bool(valid),
                percentile=chosen['percentile'],candidates=candidates)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--records',type=Path)
    parser.add_argument('--arrays',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    local=ROOT/'tmp/conditional_contrastivevitrescuecalibration'
    result=run(args.records or local,args.arrays or local)
    save(OUT,result)
    print(json.dumps(result,indent=2))
