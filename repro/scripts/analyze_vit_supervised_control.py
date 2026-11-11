"""Replay the richer supervised frozen-feature control on validation and test."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_har_transfer import ROOT,RESULTS,digest,interval,save

PLAN=ROOT/'docs/plans/2026-11-11-vit-supervised-student.md'


def load(records,arrays,official):
    for i in range(12):
        key=('n' if official else 'd')+f'{i:03d}'
        row=json.loads((records/(key+'.json')).read_text())
        assert row['complete'] and row['device']=='cpu' and row['plan_sha256']==digest(PLAN)
        assert row['reference_seed']==6201+i//3 and row['seed']==6310+i%3
        assert row['official_test_loaded']==official
        path=arrays/(key+'_states/replay.pt')
        assert digest(path)==row['replay_sha256']
        value=torch.load(path,map_location='cpu',weights_only=True)
        assert len(value['labels'])==(10000 if official else 5000)
        yield row,value


def run(args):
    validation=list(load(args.validation_records,args.validation_arrays,False))
    official=list(load(args.official_records,args.official_arrays,True))
    from analyze_vit_student_native import load as load_native
    natives=list(load_native('audit',args.native_records,args.native_arrays))
    cal_dir=RESULTS/'contrastive_contrastivevitrescuecalibration_export'
    fraction=json.loads((RESULTS/'vit_rescue_calibration_20261111.json').read_text())['percentile']
    rows=[]
    for i,((vr,vv),(tr,tv),(_,native)) in enumerate(zip(validation,official,natives)):
        assert torch.equal(tv['labels'],native['labels'])
        calibration=json.loads((ROOT/cal_dir/f'd{i:03d}.json').read_text())
        val_student=calibration['outcomes'][f'joint/{fraction}']['accuracy']*100
        rows.append(dict(reference_seed=tr['reference_seed'],seed=tr['seed'],
            validation_gain_pp=val_student-100*(vv['predictions']==vv['labels']).double().mean().item(),
            official_gain_pp=100*((native['compiled_current']==tv['labels']).double().mean().item()-
                                 (tv['predictions']==tv['labels']).double().mean().item()),
            supervised_accuracy_pp=100*(tv['predictions']==tv['labels']).double().mean().item()))
    groups=[]
    for ref in range(6201,6205):
        group=[row for row in rows if row['reference_seed']==ref]
        groups.append(dict(reference_seed=ref,**{
            field:float(np.mean([r[field] for r in group]))
            for field in ('validation_gain_pp','official_gain_pp','supervised_accuracy_pp')}))
    validation_result=interval([r['validation_gain_pp'] for r in groups])
    test=interval([r['official_gain_pp'] for r in groups])
    return dict(plan_sha256=digest(PLAN),validation_gain_pp=validation_result,
                official_gain_pp=test,official_passed=test['ci95'][0]>0,
                reference_groups=groups,references=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name,default in (
        ('validation-records','repro/results/CPUCONT/contrastive_contrastivevitsupervised_export'),
        ('validation-arrays','tmp/conditional_contrastivevitsupervised'),
        ('official-records','repro/results/CPUCONT/contrastive_contrastivevitsupervisedaudit_export'),
        ('official-arrays','tmp/conditional_contrastivevitsupervisedaudit'),
        ('native-records','repro/results/CPUCONT/contrastive_contrastivevitnativeaudit_export'),
        ('native-arrays','tmp/conditional_contrastivevitnativeaudit')):
        parser.add_argument('--'+name,type=Path,default=ROOT/default)
    args=parser.parse_args()
    torch.set_num_threads(4)
    result=run(args)
    save(RESULTS/'vit_supervised_control_20261111.json',result)
    print(json.dumps(result,indent=2))
