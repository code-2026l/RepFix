"""Local four-reference native utility, fidelity and fixed-ABI cost summaries."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from analyze_small_budget_confirmation import interval


def main(a):
    native=[json.loads(p.read_text()) for p in sorted(a.native.glob('n???.json'))]
    timing=[json.loads(p.read_text()) for p in sorted(a.timing.glob('n???.json'))]
    assert len(native)==12 and len(timing)==4 and all(r['complete'] for r in native+timing)
    assert len({r['choice_sha256'] for r in native+timing})==1
    groups=[]
    for ref in range(6201,6205):
        rows=[r for r in native if r['reference_seed']==ref];assert len(rows)==3
        groups.append(dict(reference_seed=ref,gain_pp=float(np.mean([100*(r['native']['current']['acc']-r['native']['detached']['acc']) for r in rows])),
            paired_view_retrieval_gain_pp=float(np.mean([100*(r['native']['cached_prefix_pair_retrieval']['current']-r['native']['cached_prefix_pair_retrieval']['detached']) for r in rows]))))
    cost={}
    for batch,limit in [('1',1.20),('128',1.10)]:
        roles={}
        current=[]
        for role,key in [('chosen','protected'),('full_precision','full_precision_protected')]:
            ratios=[];durations=[]
            for ref in range(6201,6205):
                r=next(r for r in timing if r['reference_seed']==ref);t=r['timings'][batch];names=t['backends'];med={k:float(np.median(v)) for k,v in t['samples_seconds'].items()}
                assert med==t['medians_seconds'];baseline=med[names['current']];duration=med[names[key]]
                ratios.append(duration/baseline);durations.append(duration)
                if role=='chosen':current.append(baseline)
            stat=interval(ratios);roles[role]=dict(ratio=stat,seconds=interval(durations),upper_ci_within_budget=stat['ci95'][1]<=limit)
        cost[batch]=dict(budget=limit,current_seconds=interval(current),**roles)
    fidelity={}
    for role in ['chosen','full_precision']:
        contracts=[v for r in native for b in ['1','128'] for v in r['audits'][b][role]['contracts'].values()]
        fidelity[role]=dict(consumer_checks=len(contracts),maximum_disagreement=max(c['disagreement'] for c in contracts),
            certified=sum(c['certificate']['certified'] for c in contracts),maximum_remaining_bound=max(c['certificate']['upper_remaining_disagreement'] for c in contracts),
            current_class_changes=sum(r['audits'][b][role]['current_class_changes'] for r in native for b in ['1','128']),
            maximum_current_logit_error=max(r['audits'][b][role]['current_logit_max_error'] for r in native for b in ['1','128']),
            maximum_old_logit_error=max(r['audits'][b][role]['old_logit_max_error'] for r in native for b in ['1','128']),
            batch1_individual_examples_per_endpoint=128,
            batch1_individual_current_class_changes=sum(r['native']['individual_batch1'][role]['current_class_changes'] for r in native))
    gain=interval([r['gain_pp'] for r in groups]);selected=fidelity['chosen']
    passes=gain['ci95'][0]>0 and selected['maximum_disagreement']<=.01 and selected['certified']==selected['consumer_checks'] and selected['current_class_changes']==0 and selected['batch1_individual_current_class_changes']==0 and all(cost[b]['chosen']['upper_ci_within_budget'] for b in ['1','128'])
    result=dict(complete=True,reference_groups=groups,native_gain_pp=gain,cost=cost,fidelity=fidelity,combined_native_gate=passes,
        secondary_paired_view_retrieval_gain_pp=interval([r['paired_view_retrieval_gain_pp'] for r in groups]),
        unprotected_class_disagreement=interval([np.mean([r['native']['unprotected_contracts']['classification']['disagreement'] for r in native if r['reference_seed']==ref]) for ref in range(6201,6205)]),
        maximum_unprotected_consumer_disagreement=max(c['disagreement'] for r in native for c in r['native']['unprotected_contracts'].values()),
        feature_trace_ratio=interval([np.mean([r['native']['feature_trace_ratio'] for r in native if r['reference_seed']==ref]) for ref in range(6201,6205)]),
        source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        maximum_cached_primary_difference=max(abs(r['native']['difference_from_cached_primary']) for r in native),
        scope='four fitted references, one shared pretrained checkpoint, clean validation population; timing seed6310 per reference')
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--timing',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
