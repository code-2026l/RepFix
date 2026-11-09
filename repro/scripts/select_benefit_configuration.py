"""Local, predeclared validation-only choice before independent confirmation."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from summarize_cpu_evidence import describe

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'repro/results/CPUCONT'


def main():
    output=BASE/'benefit_development_choice.json'
    if output.exists():
        raise FileExistsError('The frozen development choice must not be overwritten')
    assert not (BASE/'benefit_confirmation').exists(),'Configuration cannot be selected after confirmation'
    paths=sorted((BASE/'benefit_development').glob('l???.json'))
    assert len(paths)==6
    rows=[json.loads(p.read_text()) for p in paths]
    plan=ROOT/'docs/plans/2026-11-09-prospective-benefit-compatibility.md'
    plan_sha=hashlib.sha256(plan.read_bytes()).hexdigest()
    for r in rows:
        assert r['complete'] and r['phase']=='development' and not r['splits']['test_metrics_computed']
        assert r['plan_sha256']==plan_sha
        assert len({b['minibatch_order_sha256'] for b in r['branches']})==1
        for name,digest in r['source_sha256'].items():
            assert hashlib.sha256((ROOT/'repro/scripts'/name).read_bytes()).hexdigest()==digest
        assert all('test' not in b for b in r['branches'])
        assert all(b['initial_model_sha256']==r['reference_model_sha256'] and
                   b['initial_optimizer_sha256']==r['reference_optimizer_sha256'] for b in r['branches'])
    candidates=[]
    for domain in ['har','radioml']:
        part=sorted([r for r in rows if r['domain']==domain],key=lambda r:r['seed'])
        assert [r['seed'] for r in part]==[200,201,202]
        doses=sorted({b['alpha'] for b in part[0]['branches'] if b['mode']=='joint'})
        def acc(r,mode,dose):
            b=next(b for b in r['branches'] if b['mode']==mode and (b['alpha']==dose or mode.endswith('detach')))
            return b['validation']['own']['acc']
        for dose in doses:
            comparisons={name:describe([acc(r,left,dose)-acc(r,right,dose) for r in part])
                for name,left,right in [('ordinary_auxiliary_gain','joint','detach'),
                    ('guarded_auxiliary_gain','guarded_joint','guarded_detach'),
                    ('guarded_gain_over_ordinary_detach','guarded_joint','detach'),
                    ('guard_cost_in_accuracy','guarded_joint','joint')]}
            value=min(comparisons[k]['mean'] for k in ['ordinary_auxiliary_gain','guarded_auxiliary_gain',
                                                       'guarded_gain_over_ordinary_detach'])
            candidates.append(dict(domain=domain,dose=dose,score=value,comparisons=comparisons))
    best=max(c['score'] for c in candidates)
    selected=next(c for c in candidates if c['score']>=best-1e-12)
    payload=dict(plan_sha256=plan_sha,selected_before_confirmation=True,
        selection_rule='maximize minimum of three predeclared mean validation gains; ties HAR then smaller dose',
        selected=selected,positive_development_score=selected['score']>0,
        candidates=candidates,created_utc_unix=time.time(),
        confirmation_seeds=list(range(300,310)),
        input_record_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    output.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(selected=selected,positive_development_score=payload['positive_development_score']),indent=2))


if __name__=='__main__':main()
