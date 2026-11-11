"""Local paired statistics and strict original-budget confirmation verdicts."""
import argparse
import hashlib
import json
from pathlib import Path

from select_joint_alignment import interval

ROOT=Path(__file__).resolve().parents[2]
LABELS={'detach':'Detach','joint':'Joint','protected':'Direct',
        'logit_map':'Logit map','feature_map':'Feature map','direct_ols':'OLS-fp32'}


def main(args):
    paths=sorted(args.records.glob('c???.json'))
    records=[json.loads(p.read_text()) for p in paths]
    assert records and all(r['complete'] and r['phase']=='confirmation' for r in records)
    choice_path=ROOT/'repro/results/CPUCONT/alignment_development_choice.json'
    choice=json.loads(choice_path.read_text())
    assert all(r['selection_sha256']==hashlib.sha256(choice_path.read_bytes()).hexdigest() for r in records)
    timing=json.loads(args.timings.read_text()) if args.timings else None
    if timing:assert timing['complete']
    native=json.loads(args.native_timings.read_text()) if args.native_timings else None
    if native:assert native['complete'] and native['primary_runtime_dtype']=='float32'
    result=dict(phase='fresh_training_confirmation',selection_sha256=hashlib.sha256(choice_path.read_bytes()).hexdigest(),
        prior_study_decisions_unchanged=True,primary_monitor=1024,secondary_monitor=2048,
        bank=160,budget=.05,domains={})
    table=[r'\begin{table}[H]',r'\centering',
        r'\caption{Fresh HAR confirmation, seeds $1300$--$1309$. Primary values are absolute accuracy for detachment and paired percentage-point differences for other rows; brackets are $95\%$ intervals. MSE is reconstruction error relative to detachment. Certified counts use the primary $1{,}024$-input monitor and fixed $160$-consumer tail allocation. Native float32 CPU ratios return both current and legacy outputs at batch $1{,}024$. OLS changes only the reader.}',
        r'\label{tab:alignment_confirmation}',r'\footnotesize',r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{lrrrrr}',r'\toprule',r'Arm & primary & MSE / D & changed & certified & CPU \\',r'\midrule']
    for domain in sorted(set(r['domain'] for r in records)):
        group=sorted([r for r in records if r['domain']==domain],key=lambda r:r['seed'])
        assert [r['seed'] for r in group]==list(range(1300,1310))
        branch_sets=[{b['mode']:b for b in r['branches']} for r in group]
        domain_result={}
        for mode in branch_sets[0]:
            rows=[b[mode] for b in branch_sets]
            primary=interval([r['test']['own']['acc']-b['detach']['test']['own']['acc'] for r,b in zip(rows,branch_sets)])
            vs_joint=interval([r['test']['own']['acc']-b['joint']['test']['own']['acc'] for r,b in zip(rows,branch_sets)])
            mse=interval([r['test']['auxiliary_mse']/b['detach']['test']['auxiliary_mse'] for r,b in zip(rows,branch_sets)])
            item=dict(primary_accuracy=interval([r['test']['own']['acc'] for r in rows]),primary_gain=primary,
                primary_difference_to_joint=vs_joint,auxiliary_mse_ratio=mse,
                disagreement=interval([r['test']['disagreement'] for r in rows]),
                legacy_accuracy=interval([r['test']['legacy']['acc'] for r in rows]),
                certificates={},timings={})
            protected=mode not in ['detach','joint']
            if protected:
                for count in ['1024','2048']:
                    item['certificates'][count]=dict(count=sum(r['test']['certificates'][count]['certified'] for r in rows),
                        upper_bounds=[r['test']['certificates'][count]['upper_remaining_disagreement'] for r in rows])
                checks=dict(every_empirical_disagreement_within_budget=all(r['test']['disagreement']<=.05 for r in rows),
                    all_ten_primary_monitor_certificates=item['certificates']['1024']['count']==10,
                    nonlinear_layers_updated=all(all(r['nonlinear_layers_changed'].values()) for r in rows))
                if domain=='har':checks.update(reconstruction_progress=mse['ci95'][1]<=.95,primary_noninferiority=primary['ci95'][0]>=-.005)
                else:checks.update(positive_primary_gain=primary['ci95'][0]>0,noninferiority_to_joint=vs_joint['ci95'][0]>=-.005)
                if timing:
                    times=sorted([r for r in timing['branches'] if r['domain']==domain and r['mode']==mode],key=lambda r:r['seed'])
                    assert [r['seed'] for r in times]==list(range(1300,1310))
                    item['timings']={count:interval([r['timings'][count]['ratio'] for r in times]) for count in ['1','1024']}
                    checks.update(one_encoder_call=all(r['encoder_calls']==1 for r in times),
                        batch_one_cost=item['timings']['1']['mean']<=1.20,batch_1024_cost=item['timings']['1024']['mean']<=1.10)
                    item['passes']=all(checks.values())
                else:item['passes']=None
                if native:
                    times=sorted([r for r in native['branches'] if r['domain']==domain and r['mode']==mode],key=lambda r:r['seed'])
                    assert [r['seed'] for r in times]==list(range(1300,1310))
                    item['native_timings']={count:interval([r['timings'][count]['ratio'] for r in times]) for count in ['1','1024']}
                    metrics=native['native_primary_metrics']
                    own={r['seed']:r['metrics']['acc'] for r in metrics if r['domain']==domain and r['mode']==mode}
                    detached={r['seed']:r['metrics']['acc'] for r in metrics if r['domain']==domain and r['mode']=='detach'}
                    item['native_primary_gain']=interval([own[s]-detached[s] for s in range(1300,1310)])
                    checks.update(native_one_encoder=all(r['encoder_calls']==1 for r in times),
                        native_batch_one_cost=item['native_timings']['1']['mean']<=1.20,
                        native_batch_1024_cost=item['native_timings']['1024']['mean']<=1.10,
                        native_primary_noninferiority=item['native_primary_gain']['ci95'][0]>=-.005)
                    item['passes']=all(checks.values()) if timing else None
                item['checks']=checks
            domain_result[mode]=item
            if domain=='har':
                if mode=='detach':primary_cell=f"${item['primary_accuracy']['mean']:.3f}$"
                else:
                    lo,hi=primary['ci95'];primary_cell=f"${100*primary['mean']:+.2f}\\ [{100*lo:+.2f},{100*hi:+.2f}]$"
                cert=f"${item['certificates']['1024']['count']}/10$" if protected else '---'
                cost=f"${item['native_timings']['1024']['mean']:.3f}\\times$" if item.get('native_timings') else '---'
                table.append(LABELS[mode]+' & '+primary_cell+f" & ${mse['mean']:.3f}$ & ${100*item['disagreement']['mean']:.2f}\\%$ & "+cert+' & '+cost+r' \\')
        result['domains'][domain]=domain_result
    table.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    (ROOT/'paper/alignment_confirmation_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8',newline='\n')
    har=result['domains']['har'];direct=har['protected'];ols=har['direct_ols']
    numbers=dict(FreshMse=f"{direct['auxiliary_mse_ratio']['mean']:.3f}",
        FreshMseLow=f"{direct['auxiliary_mse_ratio']['ci95'][0]:.3f}",FreshMseHigh=f"{direct['auxiliary_mse_ratio']['ci95'][1]:.3f}",
        FreshReduction=f"{100*(1-direct['auxiliary_mse_ratio']['mean']):.1f}",
        FreshGain=f"{100*direct['primary_gain']['mean']:.2f}",
        FreshGainLow=f"{100*direct['primary_gain']['ci95'][0]:.2f}",FreshGainHigh=f"{100*direct['primary_gain']['ci95'][1]:.2f}",
        FreshDirectChanged=f"{100*direct['disagreement']['mean']:.2f}",FreshOlsChanged=f"{100*ols['disagreement']['mean']:.2f}")
    if native:
        numbers.update(FreshDirectLatency=f"{direct['native_timings']['1024']['mean']:.3f}",
            FreshOlsLatency=f"{ols['native_timings']['1024']['mean']:.3f}",
            FreshDirectSingleLatency=f"{direct['native_timings']['1']['mean']:.3f}",
            FreshOlsSingleLatency=f"{ols['native_timings']['1']['mean']:.3f}")
    (ROOT/'paper/alignment_numbers.tex').write_text(''.join('\\newcommand{\\'+key+'}{'+value+'}\n' for key,value in numbers.items()),encoding='utf-8',newline='\n')
    print(json.dumps({d:{m:dict(gain=v['primary_gain']['mean'],mse=v['auxiliary_mse_ratio']['mean'],changed=v['disagreement']['mean'],
        certs={k:w['count'] for k,w in v['certificates'].items()},passes=v.get('passes')) for m,v in item.items()} for d,item in result['domains'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True)
    p.add_argument('--timings',type=Path);p.add_argument('--native-timings',type=Path)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
