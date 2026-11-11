"""Matched CPU overhead with separate visual and small-network scales."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from figure_style import COLORS as C,WIDTH,apply_style,panel,save
ROOT=Path(__file__).resolve().parents[2]

def draw(ax,summary,tasks,batch,letter,title,limits,ticks):
    panel(ax,letter,title);bounds=[]
    for y,(key,label) in enumerate(tasks):
        small=key in summary['small_networks']
        cost=summary['small_networks'][key]['cost'][batch] if small else summary['backbones'][key]['cost'][batch]
        rows=[('full_precision',cost['ratio'])] if small else [('full_precision',cost['full_precision']['ratio'])]
        if not small and cost['chosen']['backend']!=cost['full_precision']['backend']:rows.append(('chosen',cost['chosen']['ratio']))
        for kind,q in rows:
            bounds.extend(q['values']);bounds.extend(q['ci95'])
            color=C['current'] if kind=='full_precision' else C['repair'];pos=y+(0 if len(rows)==1 else (-.11 if kind=='full_precision' else .11))
            ax.scatter(q['values'],pos+np.linspace(-.05,.05,len(q['values'])),color=color,s=8,alpha=.26,linewidths=0)
            ax.errorbar(q['mean'],pos,xerr=[[q['mean']-q['ci95'][0]],[q['ci95'][1]-q['mean']]],fmt='D',color=color,
                mfc='white' if kind=='full_precision' else color,markersize=3.7,capsize=2.5,lw=1,zorder=4)
    budget=1.20 if batch=='1' else 1.10
    ax.axvline(1,color=C['reference'],lw=.6);ax.axvline(budget,color=C['frozen'],ls=(0,(3,3)),lw=1)
    limits=(min(limits[0],min(bounds)-.02),max(limits[1],max(bounds)+.02))
    ax.set(xlim=limits,ylim=(-.43,1.43),yticks=[0,1],yticklabels=[v[1] for v in tasks],xticks=ticks);ax.invert_yaxis();ax.grid(axis='x')

def main():
    apply_style();summary=json.loads((ROOT/'repro/results/CPUCONT/served_frontier_summary.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,3.85));fig.subplots_adjust(left=.16,right=.98,bottom=.15,top=.865,wspace=.47,hspace=.72)
    vision=[('resnet18','ResNet-18'),('vit_b32','ViT-B/32')];small=[('har','HAR'),('radioml','SNR')]
    draw(axes[0,0],summary,vision,'1','a','Vision · batch 1',(.98,1.26),[1.0,1.1,1.2])
    draw(axes[0,1],summary,vision,'128','b','Vision · batch 128',(.98,1.30),[1.0,1.1,1.2,1.3])
    draw(axes[1,0],summary,small,'1','c','MLPs · batch 1',(.98,1.65),[1.0,1.2,1.4,1.6])
    draw(axes[1,1],summary,small,'1024','d','MLPs · batch 1024',(.98,2.60),[1.0,1.5,2.0,2.5])
    for ax in axes[1]:ax.set_xlabel('Protected / current latency')
    handles=[plt.Line2D([],[],marker='D',color=C['current'],mfc='white',ls='none',markersize=4,label='FP32 reference'),
        plt.Line2D([],[],marker='D',color=C['repair'],ls='none',markersize=4,label='Selected reference'),
        plt.Line2D([],[],color=C['frozen'],ls=(0,(3,3)),lw=1,label='Budget')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.995),ncol=3,fontsize=7.2,columnspacing=1.4,handletextpad=.5)
    fig.text(.5,.035,'10 endpoint medians · 95% paired intervals · fixed FP32 current function',ha='center',fontsize=7.2,color='#606B76')
    result=save(fig,ROOT/'paper/figures/fig_serving_frontier');plt.close(fig);return result

if __name__=='__main__':main()
