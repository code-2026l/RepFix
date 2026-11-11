"""Cross-view development, reference-level confirmation and native serving."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from figure_style import apply_style,panel,save,COLORS as C
R=Path(__file__).resolve().parents[2]


def main():
    folder=R/'repro/results/CPUCONT'
    dev=json.loads((folder/'contrastive_development_choice_20261111.json').read_text())
    confirm=json.loads((folder/'contrastive_confirmation_20261111.json').read_text())
    native=json.loads((folder/'contrastive_native_summary_20261111.json').read_text())
    apply_style();fig,axs=plt.subplots(2,2,figsize=(5.5,4.6));fig.subplots_adjust(left=.105,right=.98,top=.94,bottom=.115,wspace=.42,hspace=.60)
    ax=axs[0,0]
    for seed,color in zip([6300,6301,6302],[C['current'],C['repair'],C['purple']]):
        rows=sorted([r for r in dev['comparisons'] if r['seed']==seed],key=lambda r:r['alpha'])
        ax.plot([0,1],[r['gain_pp'] for r in rows],color=color,marker='o',ms=3.5,lw=1,label=str(seed))
    pilot=json.loads((folder/'contrastive_first_choice_20261111.json').read_text())['comparisons'][0]
    ax.scatter(-1,pilot['gain_pp'],s=16,facecolor='white',edgecolor=C['current'],linewidth=.8)
    ax.axhline(0,color=C['reference'],lw=.6);ax.axhline(.2,color=C['reference'],ls=':',lw=.7)
    ax.set_xticks([-1,0,1],['0.01','0.1','1']);ax.set_xlim(-1.25,1.25);ax.set_ylim(-.05,1.45)
    ax.set_xlabel('Auxiliary weight');ax.set_ylabel('Gain over matched D (points)');ax.grid(axis='y');ax.legend(fontsize=6.8,ncol=3,loc='upper left',handlelength=.7,columnspacing=.65)
    panel(ax,'a','Development selects weight 0.1')
    for ax,letter,title,groups,stat,key,zero in [
        (axs[0,1],'b','Native primary gain',native['reference_groups'],native['native_gain_pp'],'gain_pp',0),
        (axs[1,0],'c','Held-out auxiliary loss',confirm['reference_groups'],confirm['auxiliary_ratio'],'auxiliary_ratio',.95)]:
        for i,row in enumerate(groups):ax.scatter(row[key],i,s=20,color=C['current'],zorder=3)
        mean=stat['mean'];lo,hi=stat['ci95'];ax.errorbar(mean,4,xerr=[[mean-lo],[hi-mean]],fmt='D',color=C['repair'],ms=4,capsize=2,lw=1.2,zorder=4)
        ax.axvline(zero,color=C['reference'],ls='--',lw=.7);ax.axhline(3.5,color=C['grid'],lw=.6)
        ax.set_yticks(range(5),['6201','6202','6203','6204','Mean + CI']);ax.set_ylim(4.55,-.55);ax.grid(axis='x')
        ax.set_xlabel('Accuracy difference (points)' if key=='gain_pp' else 'Contrastive loss / D');panel(ax,letter,title)
        if key=='auxiliary_ratio':ax.set_xlim(min(.70,lo-.03),max(1,hi+.03))
    ax=axs[1,1]
    for j,(role,label,color,marker) in enumerate([('chosen','Chosen reference',C['repair'],'o'),('full_precision','Full FP32',C['current'],'s')]):
        for i,batch in enumerate(['1','128']):
            stat=native['cost'][batch][role]['ratio'];x=i+(j-.5)*.18;mean=stat['mean'];lo,hi=stat['ci95']
            ax.scatter(x+np.array([-.045,-.015,.015,.045]),stat['reference_values'],s=9,color=color,alpha=.4,zorder=2)
            ax.errorbar(x,mean,yerr=[[mean-lo],[hi-mean]],color=color,fmt=marker,ms=3.7,lw=1,capsize=2,zorder=3,label=label if i==0 else None)
    for x,budget in [(0,1.2),(1,1.1)]:ax.plot([x-.32,x+.32],[budget,budget],color=C['reference'],ls='--',lw=.8)
    ax.axhline(1,color=C['grid'],lw=.65);ax.set_xticks([0,1],['1','128']);ax.set_xlim(-.5,1.5)
    upper = max(native['cost'][batch][role]['ratio']['ci95'][1]
                for batch in ['1','128'] for role in ['chosen','full_precision'])
    ax.set_ylim(.98, upper + .15)
    ax.set_xlabel('Serving batch size');ax.set_ylabel('Latency / FP32 current only');ax.grid(axis='y');ax.legend(fontsize=6.8,loc='upper right',handlelength=.8)
    panel(ax,'d','Fixed programs, four references')
    save(fig,R/'paper/figures/fig_contrastive_benefit');plt.close(fig)


if __name__=='__main__':main()
