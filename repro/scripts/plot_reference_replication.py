"""Fixed-width vectors: every auxiliary pilot and nested replication seed."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from figure_style import WIDTH,COLORS as C,apply_style,panel,save
R=Path(__file__).resolve().parents[2];B=R/'repro/results/CPUCONT'


def main():
    apply_style();pilot=json.loads((B/'vision_auxiliary_first_pilot_choice.json').read_text())
    result=json.loads((B/'vision_reference_summary.json').read_text());assert pilot['complete'] and result['complete']
    fig=plt.figure(figsize=(WIDTH,2.65))
    a=fig.add_axes([.10,.25,.34,.61]);b=fig.add_axes([.68,.20,.29,.66])
    panel(a,'a','Real auxiliary: development')
    a.add_patch(Rectangle((.83,.20),.12,.13,facecolor=C['repair'],alpha=.065,edgecolor='none'))
    a.axhline(.20,color=C['reference'],ls=(0,(3,2)),lw=.7)
    a.axvline(.95,color=C['reference'],ls=(0,(3,2)),lw=.7)
    a.axhline(0,color=C['grid'],lw=.7)
    for arch,title,color,marker in [('resnet18','ResNet-18',C['current'],'o'),('vit_b32','ViT-B/32',C['repair'],'s')]:
        rows=pilot['backbones'][arch]['candidates']
        a.scatter([r['dose_matched_mse_ratio'] for r in rows],[r['gain_pp'] for r in rows],
            s=22,color=color,marker=marker,label=title,zorder=4,edgecolor='white',linewidth=.5)
        for r in rows:
            a.annotate(f"{r['alpha']:g}",(r['dose_matched_mse_ratio'],r['gain_pp']),
                xytext=(4,5),textcoords='offset points',fontsize=7.2,color=color)
    a.text(.842,.265,'Joint gate',fontsize=7.2,color=C['repair'])
    a.set(xlim=(.83,1.035),ylim=(-.64,.33),xlabel='Dose-matched MSE ratio',ylabel='Primary gain over D (points)')
    a.set_xticks([.85,.95,1.]);a.set_yticks([-.6,-.4,-.2,0,.2])
    a.legend(loc='upper center',bbox_to_anchor=(.5,-.31),ncol=2,fontsize=7.1,
        handletextpad=.3,columnspacing=.7,borderaxespad=0)
    panel(b,'b','ViT: new fitted references')
    differences=[]
    for i,g in enumerate(result['reference_groups']):
        values=100*(np.asarray(g['accuracies']['D'])-np.asarray(g['accuracies']['H']))
        differences.extend(values.tolist());y=3-i
        b.plot([values.min(),values.max()],[y,y],color=C['repair'],alpha=.3,lw=1.)
        b.scatter(values,y+np.array([-.08,0,.08]),s=13,color=C['repair'],alpha=.5,zorder=4,linewidth=0)
        b.scatter(np.mean(values),y,s=24,marker='D',color=C['repair'],edgecolor='white',linewidth=.5,zorder=5)
    stat=result['reference_level_statistics']['H'];mean=100*stat['mean'];lo,hi=np.array(stat['ci95'])*100
    b.errorbar(mean,-1,xerr=[[mean-lo],[hi-mean]],fmt='D',ms=4.6,color=C['ink'],lw=1.15,capsize=2.8,zorder=5)
    b.axhline(-.5,color=C['grid'],lw=.6);b.axvline(0,color=C['reference'],ls=(0,(3,2)),lw=.7)
    b.set(xlim=(min(-.1,min(differences)-.25,lo-.2),max(max(differences)+.3,hi+.2)),ylim=(-1.6,3.55),xlabel='Gain over H (points)')
    b.set_yticks([3,2,1,0,-1],['6201','6202','6203','6204','Mean\n95% CI'])
    b.tick_params(axis='y',length=0);b.spines['left'].set_visible(False)
    export=save(fig,R/'paper/figures/fig_reference_replication');plt.close(fig);return export


if __name__=='__main__':main()
