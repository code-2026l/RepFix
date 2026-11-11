"""Utility and decision compatibility are separate in the fixed chart study."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style import COLORS as C, WIDTH, apply_style, panel, save

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'repro/results/CPUCONT'


def main():
    apply_style()
    rows=[json.loads(path.read_text()) for path in sorted((BASE/'chart_confirmation').glob('c???.json'))]
    assert len(rows)==10
    summary=json.loads((BASE/'chart_confirmation_summary.json').read_text())['tasks']['radioml']['metrics']
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.85),gridspec_kw=dict(width_ratios=[1.05,1]))
    fig.subplots_adjust(left=.19,right=.975,top=.80,bottom=.28,wspace=.50)
    ax=axes[0]
    panel(ax,'a','Retained utility')
    contrasts=[('ordinary_gain','Current J − D',C['current']),
               ('old_accuracy_difference','Repaired old − healthy',C['repair'])]
    for i,(key,label,color) in enumerate(contrasts):
        value=summary[key];points=np.asarray(value['values'])*100
        y=1-i;mean=value['mean']*100
        ax.scatter(points,y+np.linspace(-.08,.08,10),s=11,color=color,alpha=.45,linewidths=0,zorder=3)
        ax.errorbar(mean,y,xerr=[[mean-value['low']*100],[value['high']*100-mean]],fmt='D',
                    color=color,markersize=4,capsize=2,zorder=5)
    ax.axvline(0,color=C['reference'],lw=.8,ls='--')
    ax.set(yticks=[1,0],yticklabels=['Current J − D','Repaired old\n− healthy'],ylim=(-.45,1.45),
           xlim=(-.15,2.65),xlabel='Accuracy difference (pp)')
    ax.grid(axis='x',alpha=.7);ax.tick_params(axis='y',length=0)
    ax=axes[1]
    panel(ax,'b','Changed legacy decisions')
    for i,row in enumerate(rows):
        item=row['branches'][1]['repairs']['paired_ridge']
        ax.plot([5,100*item['test']['consumer_audit']['upper_remaining_disagreement']], [i,i],
                color=C['grid'],lw=1,zorder=1)
        ax.scatter(100*item['test']['disagreement'],i,s=13,color=C['frozen'],zorder=4)
        ax.scatter(100*item['test']['consumer_audit']['upper_remaining_disagreement'],i,s=18,
                   marker='|',color=C['ink'],zorder=5)
    ax.axvspan(0,5,color=C['repair'],alpha=.10)
    ax.axvline(5,color=C['repair'],lw=.8,ls='--')
    ax.set(xlim=(0,58),ylim=(-.8,9.8),yticks=[0,4,9],yticklabels=['700','704','709'],
           ylabel='Confirmation seed',xlabel='Disagreement / upper bound (%)')
    fig.text(.975,.94,'0 / 10 certified',ha='right',fontsize=7.3,color=C['frozen'])
    fig.text(.5,.125,'Same encoder shapes  ·  0.990× native CPU latency at batch 1,024',ha='center',fontsize=7.3)
    fig.text(.5,.035,'(a) All seed differences and paired 95% CI.  (b) Dots: observed; ticks: finite-corpus bounds.',
             ha='center',fontsize=7.1)
    result=save(fig,ROOT/'paper/figures/fig_chart_confirmation')
    plt.close(fig)
    return result


if __name__=='__main__':
    main()
