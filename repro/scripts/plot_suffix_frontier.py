"""Vector architecture and actual served utility; every dot is a saved seed."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from figure_style import COLORS as C,WIDTH,apply_style,panel,save
from spatial_diagrams import plane,text,line_arrow,small_head,validate_layout
ROOT=Path(__file__).resolve().parents[2]

def mechanism(ax):
    ax.set(xlim=(0,1000),ylim=(0,300));ax.set_axis_off()
    text(ax,12,282,'(a) Share a frozen prefix; learn the complete nonlinear suffix',8.4,bold=True,ha='left')
    for x,y,w,h,color in [(72,88,274,145,C['current']),(400,47,580,209,C['reference'])]:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=10',facecolor=color,alpha=.045,edgecolor=color,lw=.6))
    text(ax,204,213,'Compute once',7.5,C['current'],italic=True)
    text(ax,682,236,'Versioned suffixes',7.5,C['reference'],italic=True)
    text(ax,24,150,'$x$',10);line_arrow(ax,[(39,150),(91,150)])
    plane(ax,90,117,195,54,C['current'],depth=4,slant=18,dx=8,dy=6)
    text(ax,195,143,'Frozen prefix $p_0$',8.3)
    line_arrow(ax,[(320,150),(363,150),(363,198),(431,198)],C['frozen'])
    line_arrow(ax,[(363,150),(363,95),(431,95)],C['repair'])
    plane(ax,432,177,190,40,C['frozen'],depth=3,slant=12,dx=6,dy=5)
    text(ax,533,197,'Healthy suffix $s_0$',8)
    plane(ax,432,74,190,40,C['repair'],depth=3,slant=12,dx=6,dy=5)
    text(ax,533,94,r'Learned suffix $s_\theta$',8)
    line_arrow(ax,[(655,198),(700,198)],C['frozen']);small_head(ax,704,177,242,42,'Old coarse readout\nClass + fixed thresholds',C['frozen'])
    line_arrow(ax,[(655,95),(700,95)],C['repair']);small_head(ax,704,74,242,42,'Current fine-label readout',C['repair'])
    text(ax,690,142,'Two output functions; one shared prefix',7.2,color='#606B76')
    text(ax,193,58,'Reference parameters retained',7.2,C['current'])
    text(ax,502,14,r'$f_0=r_0\circ s_0\circ p_0\qquad f_\theta=r_\theta\circ s_\theta\circ p_0$',8.5)

def gains(ax,row,letter,title):
    panel(ax,letter,title);bounds=[]
    for x,(key,color) in enumerate([('F',C['current']),('H',C['repair'])]):
        q=row['utility']['chosen']['differences'][key];v=100*np.asarray(q['values']);bounds+=v.tolist()+[100*q['ci95'][0],100*q['ci95'][1]]
        ax.scatter(x+np.linspace(-.12,.12,len(v)),v,s=12,color=color,alpha=.55,zorder=3,linewidths=0)
        ax.errorbar(x+.24,100*q['mean'],yerr=[[100*(q['mean']-q['ci95'][0])],[100*(q['ci95'][1]-q['mean'])]],fmt='D',color=C['ink'],capsize=3,markersize=3.6,lw=1.1,zorder=4)
    ax.axhline(0,color=C['reference'],lw=.8,ls=(0,(3,3)));ax.grid(axis='y')
    lo=min(0,min(bounds));hi=max(bounds);ax.set(xlim=(-.27,1.49),ylim=(lo-.08*(hi-lo),hi+.15*(hi-lo)),xticks=[0,1],xticklabels=['vs head only','vs richer probe'],ylabel='Accuracy gain (pp)')

def main():
    apply_style();summary=json.loads((ROOT/'repro/results/CPUCONT/served_frontier_summary.json').read_text())
    fig=plt.figure(figsize=(WIDTH,3.45));ax=fig.add_axes([.015,.49,.97,.50]);mechanism(ax);validate_layout(fig)
    left=fig.add_axes([.12,.155,.335,.25]);right=fig.add_axes([.625,.155,.335,.25])
    gains(left,summary['backbones']['resnet18'],'b','ResNet-18 · CIFAR-100')
    gains(right,summary['backbones']['vit_b32'],'c','ViT-B/32 · CIFAR-100')
    fig.text(.5,.035,'10 fresh continuation seeds · one fixed reference, label mask and split',ha='center',fontsize=7.2,color='#606B76')
    result=save(fig,ROOT/'paper/figures/fig_suffix_frontier');plt.close(fig);return result

if __name__=='__main__':main()
