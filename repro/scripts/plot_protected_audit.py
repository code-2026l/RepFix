"""Readout versus encoder adaptation on all saved protected-head seeds."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style import COLORS as C, WIDTH, apply_style, panel, save

ROOT=Path(__file__).resolve().parents[2]


def main():
    apply_style()
    summary=json.loads((ROOT/'repro/results/CPUCONT/protected_head_audit_summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.35))
    fig.subplots_adjust(left=.14,right=.98,top=.84,bottom=.24,wspace=.38)
    keys=[('joint_ols','J + OLS',C['current']),('joint_ridge','J + ridge',C['current']),
          ('protected_online','P + online',C['repair']),('protected_ols','P + OLS',C['purple']),
          ('protected_ridge','P + ridge',C['purple'])]
    for ax,domain,letter,label in zip(axes,['har','radioml'],['a','b'],['HAR reconstruction','RadioML SNR']):
        panel(ax,letter,label)
        data=summary['domains'][domain]['frozen_controls']
        largest=max(max(data[key]['disagreement']['values'])*100 for key,_,_ in keys)
        limit=max(6.8,largest*1.28)
        for i,(key,label,color) in enumerate(keys):
            row=data[key];estimate=row['disagreement'];y=4-i
            points=np.array(estimate['values'])*100;mean=estimate['mean']*100;lo,hi=np.array(estimate['ci95'])*100
            ax.scatter(points,y+np.linspace(-.08,.08,10),s=8,color=color,alpha=.35,linewidths=0)
            ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='D',color=color,ms=3.5,capsize=2,zorder=5)
            ax.text(limit*.98,y,f"{row['certified']}/10",ha='right',va='center',fontsize=7.1,color=C['ink'])
        ax.axvline(5,color=C['reference'],ls='--',lw=.7)
        ax.set(xlim=(0,limit),ylim=(-.5,4.55),yticks=range(4,-1,-1),
               yticklabels=[label for _,label,_ in keys],xlabel='Full-corpus disagreement (%)')
        ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.6)
    fig.text(.5,.045,'Fresh monitor · family of 140 · points / diamonds / bars: seeds / means / 95% CI',
             ha='center',fontsize=7.1)
    result=save(fig,ROOT/'paper/figures/fig_protected_audit');plt.close(fig);return result


if __name__=='__main__':main()
