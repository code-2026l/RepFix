"""Local audit verdicts and paired summaries; original outcomes remain unchanged."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'repro/results/CPUCONT'


def estimate(values):
    a=np.asarray(values,dtype=float);mean=float(a.mean())
    half=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return dict(mean=mean,ci95=[mean-half,mean+half],values=a.tolist())


def main(args):
    original=BASE/'protected_head'
    audits=[json.loads(p.read_text()) for p in sorted(args.audit_dir.glob('r???.json'))]
    assert len(audits)==20 and all(a['complete'] for a in audits)
    timing=json.loads(args.timing.read_text());assert timing['complete'] and len(timing['branches'])==20
    result=dict(phase='retrospective_control_and_independent_replay',original_gate_replaced=False,
        original_legacy_readouts_replayed=60,maximum_original_metric_error=max(a['maximum_original_metric_error'] for a in audits),
        frozen_feature_control_family=140,fresh_monitor=True,domains={})
    table=[r'\begin{table}[H]',r'\centering',
        r'\caption{Fixed retrospective teacher-head fits on saved encoders. All fits use training features and healthy logits without labels. D/J/P denote detached, unconstrained joint, and jointly protected encoders. Certificates use the same fresh monitor and 140-consumer family for every row; original 9/10 and 0/10 decisions remain unchanged. Means over ten saved seeds per corpus.}',
        r'\label{tab:protected_audit}',r'\footnotesize',r'\begin{tabular}{llrrr}',r'\toprule',
        r'Corpus & donor / legacy reader & accuracy & changed & certified \\',r'\midrule']
    for domain,prefix in [('har','ph_har_conf_'),('radioml','ph_conf_')]:
        paths=[original/(prefix+str(seed)+'.json') for seed in range(300,310)]
        records=[json.loads(p.read_text()) for p in paths]
        assert [r['seed'] for r in records]==list(range(300,310))
        rows=sorted([a for a in audits if a['domain']==domain],key=lambda a:a['seed'])
        for record,path,audit in zip(records,paths,rows):
            assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['original_record_sha256']
        by_mode=[{b['mode']:b for b in r['branches']} for r in records]
        gain=estimate([r['protected']['test']['own']['acc']-r['detach']['test']['own']['acc'] for r in by_mode])
        joint_gain=estimate([r['joint']['test']['own']['acc']-r['detach']['test']['own']['acc'] for r in by_mode])
        auxiliary=[r['protected']['test']['auxiliary_mse']/r['detach']['test']['auxiliary_mse'] for r in by_mode]
        cert=sum(r['protected']['test']['consumer_audit']['certified'] for r in by_mode)
        checks=dict(every_seed_certified=cert==10,every_seed_empirical_disagreement_within_budget=
            all(r['protected']['test']['disagreement']<=.05 for r in by_mode),one_encoder_one_pass=True)
        if domain=='har':checks.update(auxiliary_progress=float(np.mean(auxiliary))<=.95,
                                      mean_primary_within_half_point=abs(gain['mean'])<=.005)
        else:checks.update(primary_gain=gain['ci95'][0]>0,noninferior_to_joint=gain['mean']-joint_gain['mean']>=-.005)
        item=dict(original=dict(protected_primary_gain=gain,joint_primary_gain=joint_gain,
            auxiliary_mse_ratio=estimate(auxiliary),certified=cert,checks=checks,passes=all(checks.values()),
            maximum_logit_error=max(r['protected']['test']['max_absolute_healthy_logit_error'] for r in by_mode)),
            frozen_controls={})
        labels={'detach':'D','joint':'J','protected':'P'}
        for mode in ['detach','joint','protected']:
            methods=['ols','ridge']+(['online'] if mode=='protected' else [])
            for method in methods:
                donors=[next(b for b in a['branches'] if b['mode']==mode) for a in rows]
                values=[d['online_head_fresh_audit'] if method=='online' else d['fitted_readers'][method]['test'] for d in donors]
                block=dict(disagreement=estimate([v['disagreement'] for v in values]),
                    legacy_accuracy=estimate([v['metrics']['acc'] for v in values]),
                    certified=sum(v['consumer_audit']['certified'] for v in values),
                    max_upper_bound=max(v['consumer_audit']['upper_remaining_disagreement'] for v in values))
                item['frozen_controls'][mode+'_'+method]=block
                table.append(('HAR' if domain=='har' else 'SNR')+' & '+labels[mode]+' / '+method+
                    f" & {block['legacy_accuracy']['mean']:.3f} & ${100*block['disagreement']['mean']:.2f}\\%$ & ${block['certified']}/10$"+r' \\')
        timed=[r for r in timing['branches'] if r['domain']==domain]
        item['timing']={str(batch):dict(
            original_online_both_vs_current=estimate([r['timings'][str(batch)]['both_vs_current_ratio'] for r in timed]),
            posthoc_ols_both_vs_current=estimate([r['posthoc_ols_timings'][str(batch)]['both_vs_current_ratio'] for r in timed])) for batch in [1,1024]}
        item['retrospective_protected_ols_conditions']=dict(
            auxiliary_mse_unchanged=True,current_head_unchanged=True,
            all_ten_new_certificates=item['frozen_controls']['protected_ols']['certified']==10,
            original_study_gate_still_failed=not item['original']['passes'])
        result['domains'][domain]=item
        table.append(r'\midrule' if domain=='har' else r'\bottomrule')
    table.extend([r'\end{tabular}',r'\end{table}'])
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    (ROOT/'paper/protected_audit_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({domain:dict(original_passes=d['original']['passes'],new_ols_certified=d['frozen_controls']['protected_ols']['certified'],
        new_ols_disagreement=d['frozen_controls']['protected_ols']['disagreement']['mean']) for domain,d in result['domains'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--timing',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
