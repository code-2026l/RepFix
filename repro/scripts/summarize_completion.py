"""Local-only paired summaries of full baselines and versioned consumers."""
import hashlib,json
from pathlib import Path
from select_joint_alignment import interval

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'repro/results/CPUCONT'
LABELS={'detach':'Detach','detach_ols':'Detach + OLS','joint':'Joint','joint_ols':'Joint + OLS',
        'protected':'Distilled head','protected_ols':'Distilled + OLS','feature_map':'Feature align. ($\\lambda=1$)',
        'feature_map_ols':'Feature align. + OLS','logit_map':'Logit map'}

def main():
    paths=sorted((BASE/'completion_har').glob('h???.json'));records=[json.loads(p.read_text()) for p in paths]
    assert [r['seed'] for r in records]==list(range(1300,1310))
    assert all(r['complete'] and r['phase']=='retrospective_completion' and r['new_bank']==1000 for r in records)
    groups=[{b['mode']:b for b in r['branches']} for r in records]
    out=dict(study='retrospective matched completion',old_confirmation_verdicts_unchanged=True,
        source_record_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        monitor=1024,bank=1000,tail=.05/1000,seed_count=10,recorded_endpoints=100,distinct_endpoints=90,
        selection_uses_old_development_utility=True,branches={})
    for mode in LABELS:
        rows=[g[mode] for g in groups]
        item=dict(primary_gain=interval([r['test']['primary']['acc']-g['detach']['test']['primary']['acc'] for r,g in zip(rows,groups)]),
            primary_difference_to_distilled=interval([r['test']['primary']['acc']-g['protected']['test']['primary']['acc'] for r,g in zip(rows,groups)]),
            auxiliary_mse_ratio=interval([r['test']['auxiliary_mse']/g['detach']['test']['auxiliary_mse'] for r,g in zip(rows,groups)]),contracts={})
        for name in ['classification','coverage80','coverage50']:
            c=[r['test']['contracts'][name] for r in rows]
            d={key:interval([x[key] for x in c]) for key in ['disagreement','coverage','reference_coverage']}
            d['accepted_accuracy']=[x['accepted_accuracy'] for x in c]
            d['certified']=sum(x['certificate']['certified'] for x in c)
            d['upper_bounds']=[x['certificate']['upper_remaining_disagreement'] for x in c]
            if name=='classification':
                d['flows']={k:interval([x['flows'][k]/x['flows']['examples'] for x in c]) for k in ['correct_to_wrong','wrong_to_correct','wrong_to_wrong']}
                d['metrics']={k:interval([x['metrics'][k] for x in c]) for k in ['acc','nll','ece']}
            item['contracts'][name]=d
        out['branches'][mode]=item
    # The original direct_ols and its identically refitted duplicate count once.
    for g in groups:assert g['direct_ols']['reader_sha256']==g['protected_ols']['reader_sha256']
    (BASE/'completion_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8',newline='\n')
    table=[r'\begin{table}[H]',r'\centering',
        r'\caption{Matched HAR completion on the ten saved fresh-confirmation seeds. This retrospective comparison uses development-selected full-feature alignment and training-only OLS; no test fitting or coefficient search. Primary values are paired percentage-point differences to detachment. Each changed/certified cell reports mean disagreement and the number of new $1{,}024$-monitor bounds below $5\%$, with tail $0.05/1{,}000$. Selective consumers use fixed healthy-training confidence thresholds targeting $80\%$ or $50\%$ coverage. These new bounds do not replace the earlier confirmation.}',
        r'\label{tab:completion_controls}',r'\footnotesize',r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{lrrrrr}',r'\toprule',r'Arm & primary & MSE / D & class & selective 80 & selective 50 \\',r'\midrule']
    for mode,label in LABELS.items():
        r=out['branches'][mode];gain=r['primary_gain'];lo,hi=gain['ci95']
        cells=[f"${100*gain['mean']:+.2f}\\ [{100*lo:+.2f},{100*hi:+.2f}]$",f"${r['auxiliary_mse_ratio']['mean']:.3f}$"]
        for c in r['contracts'].values():cells.append(f"${100*c['disagreement']['mean']:.2f}\\%$; ${c['certified']}/10$")
        table.append(label+' & '+' & '.join(cells)+r' \\')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'paper/completion_controls_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8',newline='\n')
    numbers={}
    for mode,prefix in [('joint_ols','CompleteJointOls'),('feature_map','CompleteFeature'),('protected','CompleteDirect')]:
        row=out['branches'][mode]
        for key,value in [('Changed',100*row['contracts']['classification']['disagreement']['mean']),
                          ('Nfr',100*row['contracts']['classification']['flows']['correct_to_wrong']['mean']),
                          ('Gain',100*row['primary_gain']['mean']),('GainLow',100*row['primary_gain']['ci95'][0]),
                          ('GainHigh',100*row['primary_gain']['ci95'][1]),('Mse',row['auxiliary_mse_ratio']['mean'])]:
            numbers[prefix+key]=f'{value:.3f}' if key=='Mse' else f'{value:.2f}'
    (ROOT/'paper/completion_numbers.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in numbers.items()),encoding='utf-8',newline='\n')
    print(json.dumps({m:dict(gain=v['primary_gain'],mse=v['auxiliary_mse_ratio']['mean'],
        difference_to_distilled=v['primary_difference_to_distilled']['ci95']) for m,v in out['branches'].items() if m in ['joint_ols','feature_map','protected']},indent=2))

if __name__=='__main__':main()
