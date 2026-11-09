"""Generate chart tables/macros locally from the complete fixed record family."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'repro/results/CPUCONT'


def main():
    records=[json.loads(path.read_text()) for path in sorted((BASE/'chart_confirmation').glob('c???.json'))]
    assert sorted(row['seed'] for row in records)==list(range(700,710))
    summary=json.loads((BASE/'chart_confirmation_summary.json').read_text())['tasks']['radioml']
    assert summary['selected']=='paired_ridge' and not summary['passes']
    metrics=summary['metrics']
    def average(mode,head,key='acc',repair=None):
        rows=[next(b for b in r['branches'] if b['mode']==mode) for r in records]
        return float(np.mean([(row['repairs'][repair] if repair else row)['test'][head][key] for row in rows]))
    numbers=dict(ChartGain=100*metrics['ordinary_gain']['mean'],ChartGainLow=100*metrics['ordinary_gain']['low'],
        ChartGainHigh=100*metrics['ordinary_gain']['high'],ChartOldGain=100*metrics['old_accuracy_difference']['mean'],
        ChartOldGainLow=100*metrics['old_accuracy_difference']['low'],ChartOldGainHigh=100*metrics['old_accuracy_difference']['high'],
        ChartDisagreement=100*metrics['disagreements']['mean'],ChartLatency=metrics['batch1024_ratio']['mean'],
        ChartHealthy=average('J','reference'),ChartNativeOld=average('J','frozen'),
        ChartRepairedOld=average('J','frozen',repair='paired_ridge'),ChartCurrent=average('J','own'))
    macros=['% Generated locally by write_chart_evidence.py; ten fixed confirmation seeds.']
    for name,value in numbers.items():
        decimals=3 if name in ['ChartLatency','ChartHealthy','ChartNativeOld','ChartRepairedOld','ChartCurrent'] else 2
        macros.append('\\newcommand{\\'+name+'}{'+f'{value:.{decimals}f}'+'}')
    (ROOT/'paper/chart_numbers.tex').write_text('\n'.join(macros)+'\n',encoding='utf-8',newline='\n')
    table=[r'\begin{table}[t]',r'\centering',
        r'\caption{Post-hoc single-encoder chart, ten independent SNR confirmation seeds. Means on the full fixed test corpus. Current logits are algebraically preserved by head rebasing. Old-reader metrics can improve despite extensive prediction changes. The five-point disagreement budget certifies none of the ten repaired joint consumers.}',
        r'\label{tab:chart_confirmation}',r'\footnotesize',r'\begin{tabular}{lrrrr}',r'\toprule',
        r'Consumer & accuracy & NLL & ECE & changed predictions \\',r'\midrule']
    for label,head,repair in [('Healthy old','reference',None),('Native J, old readout','frozen',None),
                             ('Chart J, old readout','frozen','paired_ridge'),('Native J, current head','own',None),
                             ('Chart J, rebased current','own','paired_ridge'),('Supervised refit','supervised_refit',None)]:
        vals=[average('J',head,key,repair) for key in ['acc','nll','ece']]
        if head=='reference':changed='0'
        elif head=='frozen':
            rows=[r['branches'][1] for r in records]
            changed=f"{100*np.mean([(r['repairs'][repair] if repair else r)['test']['disagreement'] for r in rows]):.2f}\\%"
        elif repair:changed='0 (vs. native J)'
        else:changed='---'
        table.append(label+' & '+' & '.join(f'{x:.3f}' for x in vals)+' & '+changed+r' \\')
    table.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (ROOT/'paper/chart_confirmation_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8',newline='\n')
    return numbers


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
