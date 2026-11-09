"""Separate harmful changes from versioned selective-consumer disagreement."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from figure_style import COLORS as C,WIDTH,apply_style,panel,save

ROOT=Path(__file__).resolve().parents[2]

def main():
    apply_style();s=json.loads((ROOT/'repro/results/CPUCONT/completion_summary.json').read_text())['branches']
    modes=['joint','joint_ols','protected','protected_ols','feature_map','feature_map_ols']
    names=['Joint','Joint + OLS','Distilled head','Distilled + OLS','Feature align.','Feature + OLS']
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.65),gridspec_kw={'width_ratios':[1.05,1.15]})
    fig.subplots_adjust(left=.17,right=.96,top=.85,bottom=.29,wspace=.5)
    ax=axes[0];panel(ax,'a','Which decisions change?')
    y=np.arange(6);left=np.zeros(6)
    for key,label,color in [('correct_to_wrong','Correct → wrong',C['frozen']),
                            ('wrong_to_correct','Wrong → correct',C['repair']),('wrong_to_wrong','Wrong → wrong',C['reference'])]:
        values=np.array([s[m]['contracts']['classification']['flows'][key]['mean'] for m in modes])*100
        ax.barh(y,values,left=left,height=.53,color=color,label=label,edgecolor='white',linewidth=.4);left+=values
    ax.set(yticks=y,yticklabels=names,xlabel='Examples changed (%)',xlim=(0,31));ax.invert_yaxis();ax.tick_params(axis='y',length=0);ax.grid(axis='x')
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,fontsize=7.1,handlelength=1.1,columnspacing=.8)
    ax=axes[1];panel(ax,'b','Which contract survives?')
    contracts=['classification','coverage80','coverage50']
    values=np.array([[100*s[m]['contracts'][c]['disagreement']['mean'] for c in contracts] for m in modes])
    ax.imshow(values,cmap='YlOrBr',norm=LogNorm(vmin=1,vmax=60),aspect='auto')
    for i,m in enumerate(modes):
        for j,c in enumerate(contracts):
            v=s[m]['contracts'][c];color='white' if values[i,j]>17 else C['ink']
            ax.text(j,i,f"{values[i,j]:.1f}%\n{v['certified']}/10",ha='center',va='center',color=color,fontsize=7.2)
    ax.set(xticks=range(3),xticklabels=['Class','Selective\n80%','Selective\n50%'],yticks=[])
    ax.spines[:].set_visible(False);ax.tick_params(axis='x',length=0)
    fig.text(.75,.14,'Mean changes; certified seeds / 10',ha='center',fontsize=7.1)
    fig.text(.75,.10,'New 1,024-input monitor · tail 0.05 / 1,000',ha='center',fontsize=7.1)
    result=save(fig,ROOT/'paper/figures/fig_consumer_contracts');plt.close(fig);return result

if __name__=='__main__':main()
