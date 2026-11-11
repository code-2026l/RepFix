"""Verify the scripted policy against prior decisions and replay isolated timing."""
import argparse
import json
from pathlib import Path

import torch

from analyze_har_transfer import ROOT,RESULTS,digest,save
from analyze_vit_student_native import timing
from consumer_routing import decisions

PLAN=ROOT/'docs/plans/2026-11-11-vit-scripted-rescue.md'
CHOICE=RESULTS/'vit_rescue_calibration_20261111.json'


def load(stage,records,arrays,previous_records,previous_arrays):
    for index in (range(12) if stage=='audit' else (0,3,6,9)):
        key=('n' if stage=='audit' else 't')+f'{index:03d}'
        row=json.loads((records/(key+'.json')).read_text())
        assert row['complete'] and row['stage']==stage and row['device']=='cpu'
        assert row['plan_sha256']==digest(PLAN) and row['rescue_choice_sha256']==digest(CHOICE)
        assert row['reference_seed']==6201+index//3 and row['seed']==6310+index%3
        prior=json.loads((previous_records/f'n{index:03d}.json').read_text())
        assert row['previous_replay_sha256']==prior['replay_sha256']
        assert row['student_program_sha256']==prior['student_program_sha256']
        assert row['threads']==12
        path=arrays/(key+'_states/replay.pt')
        assert digest(path)==row['replay_sha256']
        value=torch.load(path,map_location='cpu',weights_only=True)
        if stage=='audit':
            previous_path=previous_arrays/f'n{index:03d}_states/replay.pt'
            assert digest(previous_path)==prior['replay_sha256']
            old=torch.load(previous_path,map_location='cpu',weights_only=True)
            reference=torch.cat((decisions(old['old'],old['coarse_thresholds']),
                                 decisions(old['fine_old'],old['fine_thresholds'])),1)
            assert torch.equal(value['reference_rules'],reference)
            assert torch.equal(value['previous_current'],old['compiled_current'])
            assert torch.equal(value['labels'],old['labels'])
        yield row,value


def audit(rows):
    references=[]
    for row,value in rows:
        current=int((value['current']!=value['previous_current']).sum())
        old=(value['served_rules']!=value['reference_rules']).sum(0).tolist()
        assert current==value['current_changes'] and old==value['old_changes']
        indices=value['indices']
        assert torch.equal(indices,torch.arange(0,128,4))
        single_current=int((value['single_current']!=value['current'][indices]).sum())
        single_old=(value['single_rules']!=value['reference_rules'][indices]).sum(0).tolist()
        assert single_current==value['single_current_changes'] and single_old==value['single_old_changes']
        references.append(dict(reference_seed=row['reference_seed'],seed=row['seed'],current_changes=current,
                               old_changes=old,single_current_changes=single_current,single_old_changes=single_old))
    changes=sum(r['current_changes']+r['single_current_changes']+sum(r['old_changes'])+
                sum(r['single_old_changes']) for r in references)
    return dict(plan_sha256=digest(PLAN),passed=changes==0,total_decision_changes=changes,
                full_current_inputs=10000*len(rows),full_old_rule_decisions=10000*6*len(rows),
                individual_inputs=32*len(rows),references=references)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('audit','timing'))
    parser.add_argument('--records',type=Path)
    parser.add_argument('--arrays',type=Path)
    parser.add_argument('--previous-records',type=Path,default=RESULTS/'contrastive_contrastivevitnativeaudit_export')
    parser.add_argument('--previous-arrays',type=Path,default=ROOT/'tmp/conditional_contrastivevitnativeaudit')
    args=parser.parse_args()
    torch.set_num_threads(4)
    default=ROOT/'tmp'/('conditional_contrastivevitscripted'+args.stage)
    rows=list(load(args.stage,args.records or default,args.arrays or default,args.previous_records,args.previous_arrays))
    if args.stage=='audit':
        result=audit(rows)
    else:
        affinity=rows[0][0]['affinity']
        assert len(affinity)==12 and all(row['affinity']==affinity for row,_ in rows)
        result=timing(rows)
        result.update(plan_sha256=digest(PLAN),affinity=affinity,individual_batch1_images=32,alternating_blocks=9)
    save(RESULTS/f'vit_scripted_{args.stage}_20261111.json',result)
    print(json.dumps({key:val for key,val in result.items() if key!='references'},indent=2))
