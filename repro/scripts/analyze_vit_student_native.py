"""Replay official-image accuracy, old contracts, routing and paired CPU timings."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_har_transfer import ROOT,RESULTS,digest,interval,save
from consumer_routing import decisions

PLAN=ROOT/'docs/plans/2026-11-11-vit-official-test.md'
RESCUE=RESULTS/'vit_rescue_calibration_20261111.json'


def load(stage,records,arrays):
    indices=range(12) if stage=='audit' else (0,3,6,9)
    for index in indices:
        key=('n' if stage=='audit' else 't')+f'{index:03d}'
        row=json.loads((records/(key+'.json')).read_text())
        assert row['complete'] and row['device']=='cpu' and row['stage']==stage
        assert row['plan_sha256']==digest(PLAN) and row['rescue_choice_sha256']==digest(RESCUE)
        assert row['reference_seed']==6201+index//3 and row['seed']==6310+index%3
        assert row['official_test_loaded']
        path=arrays/(key+'_states/replay.pt')
        assert digest(path)==row['replay_sha256']
        value=torch.load(path,map_location='cpu',weights_only=True)
        yield row,value


def audit(rows):
    refs=[]
    for row,value in rows:
        labels=value['labels']
        assert len(labels)==10000
        pure={name:(z==labels).double().mean().item()*100 for name,z in value['students'].items()}
        teachers={name:(z==labels).double().mean().item()*100 for name,z in value['teachers'].items()}
        rescue={name:(z==labels).double().mean().item()*100 for name,z in value['rescued'].items()}
        # The primary joint endpoint uses the compiled serving suffix.
        rescue['joint']=(value['compiled_current']==labels).double().mean().item()*100
        for name in ('joint','two_view','double_epoch'):
            selected=value['rescue_masks'][name]
            assert torch.equal(value['rescued'][name][selected],value['teachers'][name][selected])
            assert torch.equal(value['rescued'][name][~selected],value['students'][name][~selected])
        changed={}
        for name,original,current,cutoffs in (
            ('coarse',value['old'],value['native_old'],value['coarse_thresholds']),
            ('fine',value['fine_old'],value['native_fine'],value['fine_thresholds']),
            ('compiled_coarse',value['old'],value['compiled_old'],value['coarse_thresholds']),
            ('compiled_fine',value['fine_old'],value['compiled_fine_old'],value['fine_thresholds'])):
            changed[name]=(decisions(original,cutoffs)!=decisions(current,cutoffs)).sum(0).tolist()
            assert changed[name]==value[{'coarse':'coarse_changed','fine':'fine_changed','compiled_coarse':'compiled_old_changed','compiled_fine':'compiled_fine_changed'}[name]]
        refs.append(dict(reference_seed=row['reference_seed'],seed=row['seed'],
                         pure_accuracy_pp=pure,teacher_accuracy_pp=teachers,rescue_accuracy_pp=rescue,
                         gains_pp={name:rescue['joint']-rescue[name] for name in ('two_view','double_epoch')},
                         pure_gains_pp={name:pure['joint']-pure[name] for name in ('two_view','double_epoch')},
                         vs_teacher_pp=rescue['joint']-teachers['joint'],
                         pure_vs_teacher_pp=pure['joint']-teachers['joint'],
                         teacher_gains_pp={name:teachers['joint']-teachers[name] for name in ('two_view','double_epoch')},
                         fraction={name:mask.double().mean().item() for name,mask in value['rescue_masks'].items()},
                         changed=changed,raw_current_changes=value['raw_sample_current_changes'],
                         raw_old_changes=value['raw_sample_old_changes'],raw_fine_changes=value['raw_sample_fine_changes'],
                         compiled_current_changes=int((value['compiled_current']!=value['rescued']['joint']).sum())))
    groups=[]
    for seed in range(6201,6205):
        group=[r for r in refs if r['reference_seed']==seed]
        assert len(group)==3
        result=dict(reference_seed=seed)
        for field in ('gains_pp','pure_gains_pp','teacher_gains_pp','fraction'):
            result[field]={name:float(np.mean([r[field][name] for r in group])) for name in group[0][field]}
        for field in ('vs_teacher_pp','pure_vs_teacher_pp'):
            result[field]=float(np.mean([r[field] for r in group]))
        groups.append(result)
    gains={name:interval([r['gains_pp'][name] for r in groups]) for name in ('two_view','double_epoch')}
    pure={name:interval([r['pure_gains_pp'][name] for r in groups]) for name in gains}
    teachers={name:interval([r['teacher_gains_pp'][name] for r in groups]) for name in gains}
    retention=interval([r['vs_teacher_pp'] for r in groups])
    old_changes=sum(sum(v) for r in refs for v in r['changed'].values())+sum(sum(r['raw_old_changes'])+sum(r['raw_fine_changes']) for r in refs)
    return dict(plan_sha256=digest(PLAN),rescue_choice_sha256=digest(RESCUE),
                gains_pp=gains,pure_gains_pp=pure,teacher_gains_pp=teachers,
                versus_teacher_pp=retention,pure_versus_teacher_pp=interval([r['pure_vs_teacher_pp'] for r in groups]),
                referral_fraction=interval([r['fraction']['joint'] for r in groups]),old_decision_changes=old_changes,
                compiled_current_changes=sum(r['compiled_current_changes'] for r in refs),
                passed=all(v['ci95'][0]>0 for v in gains.values()) and retention['ci95'][0]>-.25 and old_changes==0,
                references=refs,reference_groups=groups)


def timing(rows):
    results={}
    for batch,limit in (('1',1.20),('128',1.10)):
        ratio={name:[] for name in ('teacher','student','rescue_current')}
        for row,value in rows:
            assert value['current_changes']==0 and not any(value['old_changed']) and not any(value['fine_old_changed'])
            blocks=value['timings'][batch]
            assert all(len(x)==9 for x in blocks.values())
            for name in ratio:
                ratio[name].append(float(np.median(np.array(blocks['protected'])/np.array(blocks[name]))))
        stats={name:interval(v) for name,v in ratio.items()}
        results[batch]=dict(ratios=stats,passed=stats['teacher']['ci95'][1]<=limit)
    return dict(plan_sha256=digest(PLAN),rescue_choice_sha256=digest(RESCUE),
                passed=all(r['passed'] for r in results.values()),batches=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('audit','timing'))
    parser.add_argument('--records',type=Path)
    parser.add_argument('--arrays',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    default=ROOT/'tmp'/('conditional_contrastivevitnative'+args.stage)
    rows=list(load(args.stage,args.records or default,args.arrays or default))
    result=(audit if args.stage=='audit' else timing)(rows)
    save(RESULTS/f'vit_native_{args.stage}_20261111.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='references'},indent=2))
