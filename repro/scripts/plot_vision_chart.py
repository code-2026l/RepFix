"""All fixed vision branches and anchor budgets, without seed error bars."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figure_style import COLORS as C, WIDTH, apply_style, panel, save

ROOT=Path(__file__).resolve().parents[2]


def main():
    apply_style()
    record=json.loads((ROOT/'repro/results/CPUCONT/vision_development/f000.json').read_text())
    assert record['complete'] and not record['splits']['test_metrics_computed']
    healthy=record['healthy']['acc']
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.15),sharey=True)
    fig.subplots_adjust(left=.10,right=.98,top=.78,bottom=.29,wspace=.27)
    ax=axes[0]
    panel(ax,'a','Native and refitted heads')
    x=np.arange(len(record['branches']))
    for offset,key,label,color in [(-.24,'own','Current',C['current']),
        (0,'frozen','Frozen',C['frozen']),(.24,'supervised_refit','Supervised refit',C['amber'])]:
        ax.bar(x+offset,[b[key]['acc'] for b in record['branches']],width=.22,color=color,
               alpha=.88,label=label,zorder=3)
    ax.axhline(healthy,color=C['reference'],lw=.85,ls='--',label='Healthy')
    ax.set(xticks=[0,1,2],xticklabels=['D','Rotation 1','Rotation 10'],ylabel='Validation accuracy',ylim=(0,.8),xlim=(-.5,2.5))
    if not record.get('training_complete',True):
        ax.axvspan(1.55,2.45,color=C['grid'],alpha=.55,zorder=0)
        ax.text(2,.30,'Numerical\nfailure',ha='center',va='center',color=C['ink'],fontsize=7.3)
    ax.grid(axis='y',alpha=.7)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,.985),
               columnspacing=1.0,handlelength=1.4,fontsize=7.3)
    ax=axes[1]
    panel(ax,'b','Repair from training anchors')
    colors=[C['reference'],C['repair'],C['purple']]
    budgets=[128,512,1024,2048]
    for branch,color in zip(record['branches'],colors):
        for method,style,marker in [('diagonal_stats','--','o'),('paired_ridge','-','s')]:
            values=[branch['repairs'][str(n)][method]['acc'] for n in budgets]
            ax.plot(budgets,values,color=color,ls=style,marker=marker,ms=3,lw=1.25)
    ax.axhline(healthy,color=C['reference'],lw=.85,ls=(0,(1,2)),alpha=.8)
    ax.set(xscale='log',xticks=budgets,xticklabels=['128','512','1,024','2,048'],xlabel='Unlabeled anchor inputs')
    ax.grid(axis='y',alpha=.7);ax.tick_params(axis='x',labelsize=7.2)
    handles=[Line2D([0],[0],color=color,lw=1.4,label=label) for color,label in
             zip(colors[:len(record['branches'])],['D','Rotation 1','Rotation 10'])]
    handles += [Line2D([0],[0],color=C['ink'],ls=style,marker=marker,ms=3,label=label)
                for style,marker,label in [('--','o','Diagonal'),('-','s','Paired ridge')]]
    fig.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,.095),
               columnspacing=.8,handlelength=1.3,fontsize=7.1)
    fig.text(.5,.035,'One development seed · pretrained ResNet-18 · 64 × 64 inputs · test set unopened',
             ha='center',fontsize=7.1)
    result=save(fig,ROOT/'paper/figures/fig_vision_chart')
    plt.close(fig)
    return result


if __name__=='__main__':
    main()
