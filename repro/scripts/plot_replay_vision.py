"""All residual-block capacity-screen results, including the failed gate."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from figure_style import COLORS as C,WIDTH,apply_style,panel,save
ROOT=Path(__file__).resolve().parents[2]

def main():
    apply_style();rows=json.loads((ROOT/'repro/results/CPUCONT/vision_shared_replay_choice.json').read_text())['all_candidates'];ref=rows[0]
    names=['Head\nonly','Dense\nblock']+['%d / %d'%(r['k'],r['rank']) for r in rows]
    accuracy=[ref['head_only'],ref['dense']]+[r['accuracy'] for r in rows];gain=100*(np.asarray(accuracy)-ref['healthy'])
    fig,ax=plt.subplots(figsize=(WIDTH,2.4));fig.subplots_adjust(left=.13,right=.98,bottom=.26,top=.80)
    panel(ax,'a','ResNet-18 / CIFAR-100: larger updates approach the dense control')
    colors=[C['reference'],C['current']]+[C['repair']]*6
    ax.bar(np.arange(8),gain,width=.57,color=colors,alpha=.83,zorder=3)
    ax.axhline(2,color=C['frozen'],lw=1,ls=(0,(3,3)));ax.text(7.42,2.07,'Fixed gain gate',ha='right',color=C['frozen'],fontsize=7.2)
    ax.axhline(0,color=C['reference'],lw=.7);ax.set(xticks=range(8),xticklabels=names,ylim=(-.5,2.75),ylabel='Gain over healthy head (pp)')
    ax.grid(axis='y');ax.tick_params(axis='x',length=0)
    fig.text(.54,.08,'Structured updates: first-layer rows / second-layer rank',ha='center',fontsize=7.2)
    fig.text(.54,.025,'One development seed · old decisions unchanged · official test unused',ha='center',fontsize=7.2)
    for x,v in enumerate(gain):ax.text(x,v+(.08 if v>=0 else -.07),f'{v:.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=7.1)
    result=save(fig,ROOT/'paper/figures/fig_replay_vision');plt.close(fig);return result

if __name__=='__main__':main()
