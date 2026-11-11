"""Reference-level summaries of the fixed audit and coupling controls."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from analyze_small_budget_confirmation import interval


def main(a):
    root=Path(__file__).resolve().parents[2];folder=root/'repro/results/CPUCONT'
    names=['certified_audit_20261111','coupling_control_20261111','oracle_risk_diagnostic_20261111']
    audit,coupling,oracle=[json.loads((folder/(n+'.json')).read_text()) for n in names]
    summary={}
    for view in ['clean','augmented']:
        points={}
        for n in audit['query_grid']:
            points[str(n)]={method:interval([np.mean([p['mae'][method] for e in audit['endpoints'] if e['reference_seed']==ref
                for p in e['views'][view]['point_estimates'] if p['queries']==n]) for ref in range(6201,6205)])
                for method in ['direct','ppi_fixed','power_tuned']}
        censor={budget:{method:float(np.mean([run['first_decision_queries'][budget]==2048
            for e in audit['endpoints'] for run in e['views'][view]['audits'] if run['method']==method]))
            for method in ['uniform','stratified','reversed']} for budget in ['0.01','0.05','0.1']}
        r2={}
        for label,record,key in [('fit_anchor',coupling,'predictor_error_correlations'),('oracle',oracle,'squared_correlations')]:
            vals=[]
            for ref in range(6201,6205):
                values=[e['views'][view][key] for e in record['endpoints'] if e['reference_seed']==ref]
                vals.append(float(np.mean([[x['squared_correlation'] for x in v] for v in values] if label=='fit_anchor' else values)))
            r2[label]=interval(vals)
        summary[view]=dict(point_mae=points,unresolved_by_1024=censor,r2=r2,query_score=audit['statistics'][view])
    histories=[run['history'] for e in audit['endpoints'] for v in e['views'].values() for run in v['audits']]
    result=dict(complete=True,source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        input_sha256={n:hashlib.sha256((folder/(n+'.json')).read_bytes()).hexdigest() for n in names},
        statistics=summary,pointwise_bounds=len(histories)*5,pointwise_noncoverage=audit['empirical_noncoverage'],
        audit_runs=len(histories),runs_with_any_noncoverage=sum(any(not r['covers_all_consumers'] for r in run) for run in histories),
        wrong_decisions=audit['wrong_certified_decisions'],primary_utility_passed=audit['primary_utility_passed'],
        coupling_summary=coupling['summary'],score_2048_is_censor_code_not_observed_queries=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:result[k] for k in ['complete','pointwise_bounds','pointwise_noncoverage','audit_runs','runs_with_any_noncoverage','wrong_decisions','primary_utility_passed']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
