"""Versioned computation and paired confirmation; no synthetic measurements."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from figure_style import COLORS as C,WIDTH,apply_style,panel,save
from spatial_diagrams import plane,text,line_arrow,small_head,validate_layout
ROOT=Path(__file__).resolve().parents[2]

def main():
    apply_style();s=json.loads((ROOT/'repro/results/CPUCONT/shared_replay_confirmation_summary.json').read_text())['domains']
    fig=plt.figure(figsize=(WIDTH,3.95));ax=fig.add_axes([.015,.48,.97,.49]);ax.set(xlim=(0,1000),ylim=(0,320));ax.set_axis_off()
    text(ax,16,308,'(a) Share the healthy trace; learn the nonlinear correction',8.4,bold=True,ha='left')
    text(ax,33,219,'$x$',9);line_arrow(ax,[(48,219),(88,219)])
    small_head(ax,90,200,130,39,r'$W_1^0,\ b_1^0$',C['current']);line_arrow(ax,[(221,219),(266,219)])
    plane(ax,260,196,95,40,C['current'],depth=3,slant=12,dx=6,dy=5);text(ax,313,216,'$a_1^0$',8.7)
    line_arrow(ax,[(370,219),(410,219)]);small_head(ax,412,200,130,39,r'$W_2^0,\ b_2^0$',C['current'])
    line_arrow(ax,[(543,219),(584,219)]);plane(ax,579,196,100,40,C['current'],depth=3,slant=12,dx=6,dy=5);text(ax,635,216,'$t_2^0$',8.7)
    line_arrow(ax,[(696,219),(804,219)]);small_head(ax,806,200,174,39,'Healthy consumer',C['frozen'])
    text(ax,479,272,'One healthy dense computation',7.8,C['current'],italic=True)
    line_arrow(ax,[(68,219),(68,111),(113,111)],C['repair'])
    small_head(ax,115,91,205,40,r'Sparse rows $S$: $\delta a_1$',C['repair'])
    line_arrow(ax,[(313,196),(313,154),(218,154),(218,132)],C['repair'])
    line_arrow(ax,[(321,111),(365,111)],C['repair'])
    small_head(ax,367,91,295,40,r'$W_2^0[:,S]\,\delta a_1[S]+UVa_1+\delta b_2$',C['repair'])
    line_arrow(ax,[(663,111),(692,111)],C['repair'])
    line_arrow(ax,[(635,196),(666,196),(666,151),(720,151),(720,132)],C['current'])
    small_head(ax,694,91,97,40,r'$\sigma(t_2)$',C['repair'])
    line_arrow(ax,[(792,111),(823,111)],C['repair']);small_head(ax,825,91,155,40,'Current task',C['current'])
    text(ax,217,57,'Nonlinear first-layer learning',7.4,C['repair'],italic=True)
    text(ax,504,57,'Low-rank second-layer update',7.4,C['repair'],italic=True)
    text(ax,516,17,'Exact in real arithmetic; old and current functions use different output paths.',7.2,color='#58616A')
    validate_layout(fig)
    left=fig.add_axes([.13,.15,.31,.245]);right=fig.add_axes([.64,.15,.32,.245])
    panel(left,'b','HAR reconstruction');r=s['har']['auxiliary_mse_ratio'];v=np.asarray(r['values'])
    left.scatter(np.linspace(-.12,.12,10),v,s=12,color=C['repair'],alpha=.60,zorder=3)
    left.errorbar([.29],[r['mean']],yerr=[[r['mean']-r['ci95'][0]],[r['ci95'][1]-r['mean']]],fmt='D',color=C['ink'],capsize=3,markersize=4)
    left.axhline(1,color=C['reference'],ls=(0,(3,3)),lw=.8);left.set(xlim=(-.24,.45),ylim=(.88,1.025),xticks=[0,.29],xticklabels=['10 seeds','Mean'],ylabel='Joint / detached MSE');left.grid(axis='y')
    panel(right,'c','SNR primary accuracy')
    for x,key,color in [(0,'primary_gain',C['repair']),(1,'primary_difference_to_full_joint',C['frozen'])]:
        q=s['radioml'][key];right.scatter(x+np.linspace(-.13,.13,10),100*np.asarray(q['values']),s=12,color=color,alpha=.60,zorder=3)
        right.errorbar([x+.25],[100*q['mean']],yerr=[[100*(q['mean']-q['ci95'][0])],[100*(q['ci95'][1]-q['mean'])]],fmt='D',color=C['ink'],capsize=3,markersize=4)
    right.axhline(0,color=C['reference'],ls=(0,(3,3)),lw=.8);right.set(xlim=(-.3,1.5),xticks=[0,1],xticklabels=['J − D','J − full J'],ylabel='Accuracy difference (pp)');right.grid(axis='y')
    fig.text(.5,.038,'120 / 120 old-contract audits certify · 0 observed changes · CPU cost target not met',ha='center',fontsize=7.2)
    result=save(fig,ROOT/'paper/figures/fig_shared_replay');plt.close(fig)
    return result

if __name__=='__main__':main()
