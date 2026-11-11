"""Shared healthy-path execution and the measured visual deployment frontier."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon,FancyArrowPatch
import numpy as np

from figure_style import COLORS as C,apply_style,panel,save

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'repro/results/CPUCONT'


def read(name):
    return json.loads((RESULTS/(name+'_20261111.json')).read_text())


def block(ax,x,y,width,height,title,subtitle,color):
    offset=.9
    ax.add_patch(Polygon([(x,y+height),(x+offset,y+height+offset),(x+width+offset,y+height+offset),
                          (x+width,y+height)],closed=True,fc=color,ec=color,alpha=.12,lw=.6))
    ax.add_patch(Polygon([(x+width,y),(x+width+offset,y+offset),(x+width+offset,y+height+offset),
                          (x+width,y+height)],closed=True,fc=color,ec=color,alpha=.18,lw=.6))
    ax.add_patch(Rectangle((x,y),width,height,fc='white',ec=color,lw=.8,zorder=2))
    ax.add_patch(Rectangle((x,y),width,height,fc=color,ec='none',alpha=.055,zorder=2))
    ax.text(x+width/2,y+height*.65,title,ha='center',va='center',fontsize=7.2,zorder=3)
    ax.text(x+width/2,y+height*.28,subtitle,ha='center',va='center',fontsize=6.9,color=color,zorder=3)


def wire(ax,points,color,arrow=True,style='-'):
    xy=np.array(points)
    ax.plot(xy[:-1,0],xy[:-1,1],color=color,lw=.85,ls=style,zorder=1)
    if arrow:
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=6.5,
                                    lw=.85,color=color,zorder=1))
    else:
        ax.plot(xy[-2:,0],xy[-2:,1],color=color,lw=.85,zorder=1)


def dots(ax,result,y,color):
    mean=result['mean']
    low,high=result['ci95']
    values=result['reference_values']
    ax.scatter(values,y+np.linspace(-.08,.08,len(values)),s=12,color=color,alpha=.5,linewidths=0,zorder=3)
    ax.errorbar(mean,y,xerr=[[mean-low],[high-mean]],fmt='D',markersize=3.6,color=color,
                markeredgecolor='white',markeredgewidth=.5,capsize=2.2,lw=1.1,zorder=4)


def main():
    audit=read('vit_native_audit')
    costs=read('vit_scripted_timing')
    control=read('vit_supervised_control')
    apply_style()
    fig=plt.figure(figsize=(5.5,3.85))
    diagram=fig.add_axes([.035,.51,.94,.40])
    diagram.set(xlim=(0,124),ylim=(-1,40))
    diagram.axis('off')
    block(diagram,12,15,20,10,'Shared prefix','10 frozen blocks',C['reference'])
    for offset in (1.4,.7,0):
        diagram.add_patch(Rectangle((1.8+offset,16+offset),5,8,fc='white',ec=C['current'],lw=.7))
    diagram.text(4.3,20,'x',ha='center',va='center',fontsize=8)
    wire(diagram,[(7.5,20),(12,20)],C['ink'])
    wire(diagram,[(32,20),(37,20),(37,28),(42,28)],C['reference'])
    wire(diagram,[(37,20),(37,7),(42,7)],C['current'])
    block(diagram,42,24,20,8,'Healthy tail','2 frozen blocks',C['repair'])
    block(diagram,42,3,20,8,'Current tail','selected inputs',C['current'])
    for offset in (1.2,.6,0):
        diagram.add_patch(Rectangle((66+offset,25+offset),7,6,fc='white',ec=C['repair'],lw=.65))
    diagram.text(70,22,r'$h_0,u_0$',ha='center',fontsize=8)
    wire(diagram,[(62,28),(66,28)],C['repair'])
    wire(diagram,[(74,28),(80,28),(80,33),(88,33)],C['repair'])
    block(diagram,88,29,20,8,'Old readers','coarse + fine',C['repair'])
    wire(diagram,[(108,33),(119,33)],C['repair'])
    diagram.text(118,36,r'$z_{\rm old}$',ha='center',fontsize=8,color=C['repair'])
    wire(diagram,[(74,28),(78,28),(78,19.5),(84,19.5)],C['purple'])
    block(diagram,84,16,20,7,'Student','128 hidden units',C['purple'])
    wire(diagram,[(104,19.5),(112,19.5),(112,15)],C['purple'])
    wire(diagram,[(62,7),(112,7),(112,9)],C['current'])
    diagram.add_patch(Rectangle((108,9),8,6,fc='white',ec=C['ink'],lw=.75,zorder=2))
    diagram.text(112,12,'gate',ha='center',va='center',fontsize=7,zorder=3)
    wire(diagram,[(116,12),(122,12)],C['ink'])
    diagram.text(119,16,r'$z_{\rm new}$',ha='center',fontsize=8)
    diagram.text(52,.1,r'$r(x)=1$',ha='center',fontsize=7.3,color=C['current'])
    panel(diagram,'a','One healthy path, selective current tail')

    a=fig.add_axes([.19,.13,.29,.23])
    for y,(summary,color) in enumerate(((audit['gains_pp']['two_view'],C['current']),
                                        (audit['gains_pp']['double_epoch'],C['purple']),
                                        (control['official_gain_pp'],C['repair']))):
        dots(a,summary,y,color)
    a.axvline(0,color=C['reference'],lw=.65)
    a.set(yticks=(0,1,2),yticklabels=('Two-view','Double epochs','Direct MLP'),ylim=(2.5,-.5),
          xlim=(-.1,2.8),xticks=(0,1,2),xlabel='Accuracy gain (pp)')
    a.grid(axis='x')
    panel(a,'b','Official CIFAR-100 test')

    b=fig.add_axes([.68,.13,.29,.23])
    maximum=max(costs['batches'][batch]['ratios']['teacher']['ci95'][1] for batch in ('1','128'))
    for y,batch in enumerate(('1','128')):
        cap=1.2 if batch=='1' else 1.1
        b.fill_betweenx([y-.38,y+.38],.99,cap,color=C['repair'],alpha=.06,zorder=0)
        b.plot([cap,cap],[y-.38,y+.38],color=C['reference'],lw=.75,ls='--')
        dots(b,costs['batches'][batch]['ratios']['teacher'],y,C['repair'])
    b.set(yticks=(0,1),yticklabels=('Batch 1','Batch 128'),ylim=(1.55,-.55),xlim=(.985,max(1.245,maximum+.02)),
          xlabel='Latency vs. teacher',xticks=(1,1.1,1.2))
    b.grid(axis='x')
    panel(b,'c','Full FP32 image forward')
    save(fig,ROOT/'paper/figures/fig_vit_shared_path')
    plt.close(fig)


if __name__=='__main__':
    main()
