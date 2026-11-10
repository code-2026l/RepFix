"""Reference-level query/error frontier; standalone research evidence figure."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from figure_style import apply_style,panel,save,COLORS as C
R=Path(__file__).resolve().parents[2]


def main():
    apply_style();v=json.loads((R/'repro/results/CPUCONT/small_budget_risk_confirmation.json').read_text());assert v['confirm_primary']
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.65));fig.subplots_adjust(left=.10,right=.98,bottom=.27,top=.85,wspace=.30)
    for ax,view,letter,title in zip(axes,['clean','augmented'],['a','b'],['Clean inputs','Augmented inputs']):
        for key,label,color,marker in [('direct','Direct count',C['frozen'],'o'),('covariance','Covariance',C['repair'],'D'),('isotropic','Isotropic',C['current'],'s')]:
            stats=[v['statistics'][view][str(n)]['mae'][key] for n in [32,64,128,256]]
            ax.plot(range(4),[q['mean']*100 for q in stats],marker=marker,color=color,label=label,ms=3.3,lw=1.1,zorder=4)
            for j,q in enumerate(stats):
                ax.scatter(j+np.array([-.08,-.025,.025,.08]),np.array(q['reference_values'])*100,s=9,color=color,alpha=.35,linewidth=0,zorder=3)
        ax.set_xticks(range(4),['32','64','128','256']);ax.set_xlabel('Additional reference queries');ax.set_ylim(0,6);ax.grid(axis='y');panel(ax,letter,title)
    axes[0].set_ylabel('Prediction MAE (points)');axes[1].tick_params(labelleft=False)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.005),ncol=3,handlelength=1.3,columnspacing=1.3)
    save(fig,R/'output/risk_research/fig_risk_queries');plt.close(fig)


if __name__=='__main__':main()
