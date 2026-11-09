"""Fresh paired utility and finite-corpus compatibility, with all ten seeds."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from figure_style import COLORS as C,WIDTH,apply_style,panel,save

ROOT=Path(__file__).resolve().parents[2]


def main():
    apply_style()
    summary=json.loads((ROOT/'repro/results/CPUCONT/alignment_confirmation_summary.json').read_text())['domains']['har']
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.15))
    fig.subplots_adjust(left=.14,right=.98,top=.82,bottom=.25,wspace=.62)
    specs=[('joint','Joint',C['reference']),('protected','Direct',C['repair']),
           ('logit_map','Logit map',C['current']),('direct_ols','OLS-fp32',C['purple'])]
    ax=axes[0];panel(ax,'a','Primary accuracy retained')
    values=[]
    for i,(mode,label,color) in enumerate(specs):
        row=summary[mode]['primary_gain'];y=3-i
        points=np.array(row['values'])*100;mean=row['mean']*100;lo,hi=np.array(row['ci95'])*100
        values.extend([*points,lo,hi])
        ax.scatter(points,y+np.linspace(-.07,.07,10),s=8,color=color,alpha=.35,linewidths=0)
        ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='D',ms=3.5,color=color,capsize=2)
    ax.axvline(0,color=C['reference'],lw=.7)
    ax.axvline(-.5,color=C['frozen'],ls='--',lw=.7)
    ax.set(yticks=[3,2,1,0],yticklabels=[label for _,label,_ in specs],ylim=(-.45,3.5),
           xlim=(min(-.65,min(values)-.15),max(values)+.15),xlabel='Accuracy difference to D (pp)')
    ax.grid(axis='x');ax.tick_params(axis='y',length=0)
    ax=axes[1];panel(ax,'b','Legacy decisions retained')
    upper=max(max(summary[m]['certificates']['1024']['upper_bounds']) for m,_,_ in specs[1:])*100
    limit=max(7.,upper*1.3)
    for i,(mode,label,color) in enumerate(specs[1:]):
        row=summary[mode];y=2-i
        values=np.array(row['disagreement']['values'])*100
        bounds=np.array(row['certificates']['1024']['upper_bounds'])*100
        offsets=np.linspace(-.07,.07,10)
        for observed,bound,offset in zip(values,bounds,offsets):
            ax.plot([observed,bound],[y+offset]*2,color=color,alpha=.16,lw=.6)
        ax.scatter(values,y+offsets,s=8,color=color,alpha=.55,linewidths=0,zorder=4)
        ax.scatter(bounds,y+offsets,s=14,color=color,marker='|',linewidths=.7,zorder=4)
        ax.text(limit*.98,y,f"{row['certificates']['1024']['count']}/10",ha='right',va='center',fontsize=7.1)
    ax.axvspan(0,5,color=C['repair'],alpha=.055);ax.axvline(5,color=C['repair'],ls='--',lw=.7)
    ax.set(yticks=[2,1,0],yticklabels=[label for _,label,_ in specs[1:]],ylim=(-.45,2.5),
           xlim=(0,limit),xlabel='Disagreement / bound (%)')
    ax.tick_params(axis='y',length=0);ax.grid(axis='x')
    fig.text(.5,.055,'Fresh seeds 1300–1309 · paired checkpoints and minibatches · primary monitor: 1,024 inputs',
             ha='center',fontsize=7.1)
    result=save(fig,ROOT/'paper/figures/fig_alignment_confirmation');plt.close(fig);return result


if __name__=='__main__':main()
