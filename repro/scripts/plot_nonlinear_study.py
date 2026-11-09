"""Local editable mechanism and complete validation-only selection figures."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style import COLORS as C, WIDTH, apply_style, panel, save
from spatial_diagrams import nonlinear_planes, validate_layout

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'repro/results/CPUCONT'


def main():
    apply_style()
    fig = plt.figure(figsize=(WIDTH,2.85))
    ax = fig.add_axes([.008,.015,.984,.97])
    nonlinear_planes(ax);validate_layout(fig)
    mechanism = save(fig,ROOT/'paper/figures/fig_nonlinear_interface')
    plt.close(fig)
    path = BASE/'nonlinear_interface_summary.json'
    summary = json.loads(path.read_text())
    raw_paths = sorted((BASE/'nonlinear_development').glob('b???.json'))
    records = [json.loads(p.read_text()) for p in raw_paths]
    items = summary['candidates']
    fig,axes = plt.subplots(1,2,figsize=(WIDTH,3.05))
    fig.subplots_adjust(left=.175,right=.975,top=.85,bottom=.20,wspace=.25)
    def acc(record,mode,width,dose):
        return next(row['validation']['own']['acc'] for row in record['branches'] if
                    row['mode']==mode and row['channel_width']==width and row['alpha']==dose)
    for ax,name,right,rwidth,letter,title in [
        (axes[0],'compatible_auxiliary_gain','TD',True,'a','Auxiliary gain: TJ - TD'),
        (axes[1],'compatible_total_gain','D',False,'b','Total gain: TJ - D')]:
        panel(ax,letter,title)
        limits=[0.]
        labels=[]
        for index,item in enumerate(items):
            domain,width,dose=item['domain'],item['width'],item['dose']
            rows=[r for r in records if r['domain']==domain]
            values=np.array([100*(acc(r,'TJ',width,dose)-acc(r,right,width if rwidth else 0,
                                   dose if rwidth else 1.)) for r in rows])
            estimate=item['comparisons'][name];mean=100*estimate['mean'];lo,hi=np.array(estimate['mean_ci95'])*100
            y=5-index;color=C['current'] if domain=='radioml' else C['repair']
            ax.scatter(values,y+np.linspace(-.09,.09,len(values)),s=13,color=color,alpha=.35,linewidths=0)
            ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='D',color=color,markersize=3.5,
                        capsize=2,elinewidth=1.1,zorder=5)
            labels.append(('SNR' if domain=='radioml' else 'HAR')+' / %d / %g'%(width,dose))
            limits.extend([lo,hi,*values])
        span=max(limits)-min(limits)
        ax.set(xlim=(min(limits)-.08*span,max(limits)+.08*span),ylim=(-.45,5.45),
               yticks=range(5,-1,-1),yticklabels=labels if ax is axes[0] else ['']*6,
               xlabel='Validation accuracy difference (pp)')
        ax.axvline(0,color=C['reference'],lw=.75,ls='--')
        ax.axhline(3.5,color='#C5CCD3',lw=.5,ls=(0,(3,3)))
        ax.grid(axis='x',alpha=.65);ax.tick_params(axis='y',length=0)
    fig.text(.04,.925,'Task / width / dose',fontsize=7.2,ha='left',color='#59636C')
    fig.text(.5,.045,'Development only: 3 seeds per task; means / 95% intervals; no test confirmation',
             ha='center',fontsize=7.1)
    validation=save(fig,ROOT/'paper/figures/fig_nonlinear_validation')
    plt.close(fig)
    manifest=dict(mechanism=mechanism,validation=validation,development_only=True,
        no_candidate_passes_complete_gate=True,full_data_latency_measured=False,
        input_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,*raw_paths]})
    (ROOT/'paper/figures/nonlinear_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
