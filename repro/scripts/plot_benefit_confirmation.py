"""Local plots of the complete ten-seed frozen-recipe confirmation."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style import COLORS as C,WIDTH,apply_style,panel,save

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'repro/results/CPUCONT'


def main():
    summary_path=BASE/'benefit_confirmation_summary.json'
    summary=json.loads(summary_path.read_text())
    paths=sorted((BASE/'benefit_confirmation').glob('s???.json'))
    assert len(paths)==10
    rows=[json.loads(p.read_text()) for p in paths]
    def b(r,mode):return next(v['test'] for v in r['branches'] if v['mode']==mode)
    apply_style()
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.8),gridspec_kw={'width_ratios':[1,1.08]})
    fig.subplots_adjust(left=.12,right=.975,bottom=.25,top=.84,wspace=.48)
    entries=[('ordinary_auxiliary_gain','J - D','joint','detach',C['current']),
             ('guarded_auxiliary_gain','GJ - GD','guarded_joint','guarded_detach',C['purple']),
             ('guarded_gain_over_ordinary_detach','GJ - D','guarded_joint','detach',C['repair']),
             ('guard_cost_in_accuracy','GJ - J','guarded_joint','joint',C['amber'])]
    all_values=[]
    for i,(name,label,left,right,color) in enumerate(entries):
        values=100*np.array([b(r,left)['own']['acc']-b(r,right)['own']['acc'] for r in rows])
        all_values.extend(values)
        stat=summary['comparisons'][name]
        mean,lo,hi=100*np.array([stat['mean'],*stat['mean_ci95']])
        axes[0].scatter(values,i+np.linspace(-.09,.09,10),s=8,alpha=.33,color=color,linewidths=0,zorder=2)
        axes[0].errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='D',color=color,ms=4,
                         capsize=2.5,lw=1.2,zorder=3)
    axes[0].axvline(0,color='#8A959F',lw=.8,ls='--')
    axes[0].plot([-.5,-.5],[2.69,3.31],color=C['amber'],lw=1,ls=':')
    axes[0].set(yticks=np.arange(4),yticklabels=[e[1] for e in entries],ylim=(3.55,-.6),
                xlabel='Primary-accuracy difference (pp)',xlim=(min(-.8,min(all_values)-.15),max(all_values)+.15))
    axes[0].grid(axis='x')
    panel(axes[0],'a','Real auxiliary benefit retained')
    xvals=[];bounds=[]
    for mode,label,color,marker in [('detach','D',C['reference'],'o'),('joint','J',C['frozen'],'s'),
                    ('guarded_detach','GD',C['purple'],'^'),('guarded_joint','GJ',C['repair'],'D')]:
        x=100*np.array([b(r,mode)['own']['acc']-b(r,'detach')['own']['acc'] for r in rows])
        y=np.array([b(r,mode)['consumer_audit']['upper_remaining_disagreement'] for r in rows])
        xvals.extend(x);bounds.extend(y)
        axes[1].scatter(x,y,color=color,marker=marker,s=12,alpha=.45,linewidths=.3,edgecolors='white',label=label,zorder=3)
        axes[1].scatter([x.mean()],[y.mean()],color=color,marker=marker,s=30,edgecolors='white',linewidths=.5,zorder=4)
    xmax=max(xvals)+.15;xmin=min(-.3,min(xvals)-.15)
    axes[1].fill_between([0,xmax],0,.05,color=C['repair'],alpha=.07,zorder=0)
    axes[1].axhline(.05,color=C['repair'],lw=.8,ls='--')
    axes[1].axvline(0,color='#B1B8BF',lw=.6)
    axes[1].set(xlabel='Primary gain over D (pp)',ylabel='Unlabeled disagreement bound',
                xlim=(xmin,xmax),ylim=(-.035,max(bounds)+.05),yticks=[0,.2,.4,.6])
    axes[1].grid(axis='y')
    axes[1].legend(loc='center left',ncol=2,fontsize=7.3,columnspacing=.6,handletextpad=.25,handlelength=1)
    panel(axes[1],'b','Exact consumer protection')
    fig.text(.29,.065,'Diamonds / bars: means / paired 95% CI',ha='center',fontsize=7.1)
    fig.text(.77,.065,'Small points: all ten seeds',ha='center',fontsize=7.1)
    info=save(fig,ROOT/'paper/figures/fig_benefit_confirmation')
    plt.close(fig)
    manifest=dict(figure=info,source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in paths+[summary_path]},uses_all_ten_confirmation_seeds=True,
                 primary_uncertainty='Paired seed-level two-sided 95% t interval, conditional fixed dataset.',
                 consumer_bound='Separate simultaneous 40-consumer fixed-corpus bank; budget 0.05.')
    (ROOT/'paper/figures/benefit_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
