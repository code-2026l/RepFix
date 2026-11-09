"""Local validation-only selection; refuse post-confirmation reselection."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from summarize_cpu_evidence import describe

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'repro/results/CPUCONT'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint(record, mode, width, dose):
    return next(row for row in record['branches'] if row['mode']==mode and
                row['channel_width']==width and row['alpha']==dose)


def exact_accuracy_mean(pairs):
    """Discrete correct counts must not acquire a false positive from rounding."""
    return sum((Fraction(round(left*n)-round(right*n),n) for left,right,n in pairs),Fraction())/len(pairs)


def main():
    target = BASE/'nonlinear_development_choice.json'
    if target.exists() or list((BASE/'nonlinear_confirmation').glob('c???.json')):
        raise RuntimeError('The choice is immutable and cannot be selected after confirmation')
    paths = sorted((BASE/'nonlinear_development').glob('b???.json'))
    assert len(paths)==6
    records = [json.loads(path.read_text()) for path in paths]
    plan = ROOT/'docs/plans/2026-11-09-nonlinear-interface.md'
    for record in records:
        assert record['complete'] and record['phase']=='development'
        assert record['device']=='cpu' and record['cuda_visibility']=='' and record['torch_threads']==1
        assert record['plan_sha256']==sha(plan)
        assert not record['splits']['test_metrics_computed']
        assert all('test' not in row for row in record['branches'])
        assert len(record['branches'])==(19 if record['domain']=='har' else 10)
        for name,digest in record['source_sha256'].items():
            assert digest==sha(ROOT/'repro/scripts'/name)
    candidates = []
    selected = {}
    for domain in ['radioml','har']:
        rows = sorted([record for record in records if record['domain']==domain],key=lambda r:r['seed'])
        assert [record['seed'] for record in rows]==[500,501,502]
        options = []
        for dose in ([1.] if domain=='radioml' else [1.,10.]):
            for width in [16,32]:
                effects = {}
                exact_effects = {}
                for name,left,lwidth,right,rwidth in [
                    ('ordinary_auxiliary_gain','J',0,'D',0),
                    ('compatible_auxiliary_gain','TJ',width,'TD',width),
                    ('compatible_total_gain','TJ',width,'D',0),
                    ('compatible_vs_joint','TJ',width,'J',0)]:
                    pairs = [(endpoint(record,left,lwidth,dose)['validation']['own']['acc'],
                              endpoint(record,right,rwidth,1. if right=='D' else dose)['validation']['own']['acc'],
                              record['splits']['validation_examples']) for record in rows]
                    values = [left-right for left,right,_ in pairs]
                    effects[name] = describe(values)
                    exact_effects[name] = exact_accuracy_mean(pairs)
                    effects[name]['mean'] = float(exact_effects[name])
                score = float(min(exact_effects['ordinary_auxiliary_gain'],exact_effects['compatible_auxiliary_gain'],
                                  exact_effects['compatible_total_gain'],exact_effects['compatible_vs_joint']+Fraction(1,200)))
                options.append(dict(domain=domain,width=width,dose=dose,score=score,comparisons=effects))
        best = options[0]
        for option in options[1:]:
            if option['score']>best['score']+1e-12 or (
                abs(option['score']-best['score'])<=1e-12 and (option['width'],option['dose'])<(best['width'],best['dose'])):
                best = option
        selected[domain] = dict(best,confirm=best['score']>0)
        candidates.extend(options)
    result = dict(plan_sha256=sha(plan),selected_before_confirmation=True,
        selection_arithmetic='integer validation correct counts and exact rational mean differences',
        rule='maximize min validation means J-D, TJ-TD, TJ-D, TJ-J+0.005; ties smaller width then dose; confirm iff positive',
        selected=selected,candidates=candidates,
        development_record_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in paths})
    prior = BASE/'nonlinear_development_choice_float_bug.json'
    if prior.exists():
        result['arithmetic_correction'] = dict(archived_initial_float_choice_sha256=sha(prior),
            reason='Exact zero aggregate HAR J-D count was represented as a tiny positive floating mean.',
            confirmation_started_before_correction=False,grid_or_criterion_changed=False)
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(selected=selected,total_candidates=len(candidates)),indent=2))


if __name__=='__main__':main()
