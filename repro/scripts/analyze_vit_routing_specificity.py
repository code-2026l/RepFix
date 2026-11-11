"""Retrospective equal-call routing control from fixed official predictions."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from analyze_har_transfer import ROOT, RESULTS, digest, interval, save
from analyze_vit_student_native import load


def random_expectation(student, teacher, labels, k):
    n = len(labels)
    assert 0 <= k <= n and n > 0
    p = k / n
    accuracy = ((1-p)*(student == labels).double().mean()
                + p*(teacher == labels).double().mean()).item()*100
    disagreement = (1-p)*(student != teacher).double().mean().item()*100
    return accuracy, disagreement


def check_expectation():
    s=torch.tensor([0,1,0,1]);t=torch.tensor([1,1,1,0]);y=torch.tensor([1,0,1,0])
    for k in range(5):
        values=[]
        for selected in itertools.combinations(range(4),k):
            z=s.clone();z[list(selected)]=t[list(selected)]
            values.append([100*(z==y).double().mean().item(),100*(z!=t).double().mean().item()])
        assert np.allclose(np.mean(values,axis=0),random_expectation(s,t,y,k),atol=1e-12)


def analyze(args):
    check_expectation()
    rows=[]
    for record,value in load('audit',args.records,args.arrays):
        result=dict(reference_seed=record['reference_seed'],seed=record['seed'],
                    replay_sha256=record['replay_sha256'],arms={})
        for arm in ('joint','two_view','double_epoch'):
            s=value['students'][arm];t=value['teachers'][arm];mask=value['rescue_masks'][arm]
            y=value['labels'];z=torch.where(mask,t,s)
            assert torch.equal(z,value['rescued'][arm])
            if arm=='joint':assert torch.equal(z,value['compiled_current'])
            random_acc,random_dis=random_expectation(s,t,y,int(mask.sum()))
            actual_acc=100*(z==y).double().mean().item()
            actual_dis=100*(z!=t).double().mean().item()
            total=int((s!=t).sum());caught=int(((s!=t)&mask).sum())
            result['arms'][arm]=dict(calls=int(mask.sum()),inputs=len(y),
                targeted_accuracy_pp=actual_acc,random_expected_accuracy_pp=random_acc,
                random_accuracy_vs_teacher_pp=random_acc-100*(t==y).double().mean().item(),
                accuracy_advantage_pp=actual_acc-random_acc,
                targeted_teacher_disagreement_pp=actual_dis,random_expected_teacher_disagreement_pp=random_dis,
                captured_teacher_differences_fraction=caught/total if total else 1.,
                total_teacher_differences=total,captured_teacher_differences=caught)
        rows.append(result)
    fields=('accuracy_advantage_pp','random_accuracy_vs_teacher_pp','targeted_teacher_disagreement_pp','random_expected_teacher_disagreement_pp',
            'captured_teacher_differences_fraction')
    groups=[]
    for ref in range(6201,6205):
        members=[r for r in rows if r['reference_seed']==ref];assert len(members)==3
        group=dict(reference_seed=ref,**{f:float(np.mean([r['arms']['joint'][f] for r in members])) for f in fields})
        for arm in ('two_view','double_epoch'):
            group['random_aux_gain_'+arm]=float(np.mean([r['arms']['joint']['random_expected_accuracy_pp']-
                r['arms'][arm]['random_expected_accuracy_pp'] for r in members]))
        groups.append(group)
    metrics={f:interval([g[f] for g in groups]) for f in groups[0] if f!='reference_seed'}
    return dict(analysis_type='retrospective fixed-prediction equal-call control',
                analysis_plan_sha256=digest(ROOT/'docs/plans/2026-11-11-routing-specificity.md'),
                source_sha256={Path(__file__).name:digest(Path(__file__))},
                random_baseline='Exact expectation over uniform subsets of the same per-arm size; no sampled baseline noise',
                metrics=metrics,reference_groups=groups,references=rows,expectation_exhaustive_check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--records',type=Path,default=RESULTS/'contrastive_contrastivevitnativeaudit_export')
    parser.add_argument('--arrays',type=Path,default=ROOT/'tmp/conditional_contrastivevitnativeaudit')
    args=parser.parse_args();torch.set_num_threads(4)
    result=analyze(args);save(RESULTS/'vit_routing_specificity_20261111.json',result)
    print(json.dumps(result['metrics'],indent=2))
