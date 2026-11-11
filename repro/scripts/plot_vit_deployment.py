"""Official-test gains, teacher retention and complete-image CPU cost."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style import COLORS as C,apply_style,panel,save

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'repro/results/CPUCONT'


def read(name):
    return json.loads((RESULTS/(name+'_20261111.json')).read_text())


def points(ax,summary,y,color,offset=0):
    values=summary['reference_values']
    mean=summary['mean']
    low,high=summary['ci95']
    yy=y+offset
    ax.scatter(values,yy+np.linspace(-.09,.09,len(values)),s=14,color=color,alpha=.5,linewidths=0,zorder=2)
    ax.errorbar(mean,yy,xerr=[[mean-low],[high-mean]],fmt='D',markersize=4,color=color,
                markeredgecolor='white',markeredgewidth=.5,linewidth=1.2,capsize=2.4,zorder=3)


def main():
    audit=read('vit_native_audit')
    cost=read('vit_scripted_timing')
    control=read('vit_supervised_control')
    specificity=read('vit_routing_specificity')['metrics']
    apply_style()
    fig=plt.figure(figsize=(5.5,4.45))
    grid=fig.add_gridspec(2,2,height_ratios=(1,.9))
    fig.subplots_adjust(left=.2,right=.974,bottom=.115,top=.91,wspace=.68,hspace=.68)
    a=fig.add_subplot(grid[0,0])
    for y,(label,summary,color) in enumerate((
        ('Two-view',audit['gains_pp']['two_view'],C['current']),
        ('Double epochs',audit['gains_pp']['double_epoch'],C['purple']),
        ('Direct MLP',control['official_gain_pp'],C['repair']))):
        points(a,summary,y,color)
    a.axvline(0,color=C['reference'],lw=.75)
    a.set(yticks=range(3),yticklabels=('Two-view','Double epochs','Direct MLP'),
          xlim=(-.1,2.8),xticks=(0,1,2),ylim=(2.5,-.5),xlabel='Accuracy gain (pp)')
    a.grid(axis='x')
    panel(a,'a','Matched controls')

    b=fig.add_subplot(grid[0,1])
    points(b,audit['pure_versus_teacher_pp'],0,C['reference'])
    points(b,specificity['random_accuracy_vs_teacher_pp'],1,C['purple'])
    points(b,audit['versus_teacher_pp'],2,C['repair'])
    b.axvline(-.25,color=C['reference'],ls='--',lw=.8)
    b.axvline(0,color=C['grid'],lw=.8)
    b.set(yticks=(0,1,2),yticklabels=('Student','Random','Routed'),ylim=(2.5,-.5),
          xlim=(-1.25,.16),xticks=(-1,-.5,0),xlabel='Accuracy vs. teacher (pp)')
    b.grid(axis='x')
    panel(b,'b','Teacher retention')

    c=fig.add_subplot(grid[1,:])
    for y,batch in enumerate(('1','128')):
        cap=1.2 if batch=='1' else 1.1
        c.fill_betweenx([y-.36,y+.36],.98,cap,color=C['repair'],alpha=.055,zorder=0)
        c.plot([cap,cap],[y-.36,y+.36],color=C['reference'],ls='--',lw=.85,zorder=1)
        for name,offset,color in (('teacher',-.11,C['current']),('rescue_current',.11,C['repair'])):
            points(c,cost['batches'][batch]['ratios'][name],y,color,offset)
    c.set(yticks=(0,1),yticklabels=('Batch 1','Batch 128'),ylim=(1.5,-.5),
          xlabel='Latency ratio',xlim=(.975,max(1.27,max(
              value['ci95'][1] for row in cost['batches'].values()
              for name,value in row['ratios'].items() if name in ('teacher','rescue_current'))+.03)),
          xticks=(1,1.05,1.1,1.15,1.2,1.25))
    c.grid(axis='x')
    c.scatter([],[],s=20,color=C['current'],marker='D',label='vs. original teacher')
    c.scatter([],[],s=20,color=C['repair'],marker='D',label='vs. same routed current')
    c.legend(loc='lower right',bbox_to_anchor=(1,1.16),ncol=2,fontsize=7.2,
             handletextpad=.25,columnspacing=.8)
    panel(c,'c','Complete-image forward cost')
    save(fig,ROOT/'paper/figures/fig_vit_deployment')
    plt.close(fig)


if __name__=='__main__':
    main()
