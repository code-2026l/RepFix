"""Standalone evidence figure; never alter the concurrently edited manuscript."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from figure_style import apply_style,panel,save,COLORS as C


def main(a):
    root=Path(__file__).resolve().parents[2];base=root/'repro/results/CPUCONT'
    summary=json.loads((base/'audit_boundary_summary_20261111.json').read_text())
    coupling=json.loads((base/'coupling_control_20261111.json').read_text())
    apply_style();fig,axs=plt.subplots(2,2,figsize=(5.5,4.95))
    fig.subplots_adjust(left=.105,right=.98,top=.94,bottom=.105,hspace=.60,wspace=.38)
    ax=axs[0,0]
    for j,(method,label,col) in enumerate([('uniform','Uniform',C['current']),('stratified','Stratified',C['repair'])]):
        for k,view in enumerate(['clean','augmented']):
            s=summary['statistics'][view]['query_score']['0.01']['censored_query_score'][method];x=k+(j-.5)*.27
            ax.bar(x,s['mean'],width=.24,color=col,alpha=.82,label=label if k==0 else None,zorder=2)
            ax.scatter(x+np.array([-.07,-.025,.025,.07]),s['reference_values'],s=11,facecolor='white',edgecolor=col,linewidth=.7,zorder=3)
    ax.set_xticks([0,1],['Clean','Augmented']);ax.set_ylim(0,175);ax.set_ylabel('Queries to reject 1% budget');ax.grid(axis='y');ax.legend(loc='upper left',fontsize=7,handlelength=1)
    panel(ax,'a','Exact audit does not accelerate')
    ax=axs[0,1];ns=[64,128,256,512,1024]
    for method,label,col,mark in [('direct','Direct',C['frozen'],'o'),('ppi_fixed','PPI',C['repair'],'s'),('power_tuned','Power tuned',C['current'],'^')]:
        stats=[summary['statistics']['clean']['point_mae'][str(n)][method] for n in ns]
        ax.plot(range(5),[s['mean']*100 for s in stats],color=col,marker=mark,ms=3,lw=1.05,label=label,zorder=3)
    ax.set_xticks(range(5),[str(n) for n in ns]);ax.set_xlabel('Reference queries (clean corpus)');ax.set_ylabel('Mean absolute error (points)');ax.set_ylim(0,2.9);ax.grid(axis='y');ax.legend(fontsize=7,loc='upper right',handlelength=1)
    panel(ax,'b','Mean correction gives modest gains')
    ax=axs[1,0]
    for i,e in enumerate(coupling['endpoints']):
        b=e['views']['clean']['coupling_bounds'][0]
        ax.plot([100*b['minimum_error'],100*b['maximum_error']],[i,i],color=C['reference'],lw=2.2,alpha=.65,zorder=1)
        ax.scatter(100*b['paired_error'],i,s=13,color=C['current'],zorder=3,label='Observed pairing' if i==0 else None)
        ax.scatter(100*b['unchanged_gaussian_prediction'],i,s=16,marker='|',color=C['frozen'],zorder=4,label='Marginal forecast' if i==0 else None)
    ax.set_yticks([1,4,7,10],['6201','6202','6203','6204']);ax.invert_yaxis();ax.set_ylim(12,-1);ax.set_xlabel('Class disagreement (%)');ax.set_ylabel('Reference seed (3 endpoints each)');ax.set_xlim(-1,55);ax.grid(axis='x');ax.legend(fontsize=6.8,loc='upper right',handlelength=.6)
    panel(ax,'c','Coupling changes, marginals stay fixed')
    ax=axs[1,1]
    for j,(view,col) in enumerate([('clean',C['current']),('augmented',C['repair'])]):
        stats=summary['statistics'][view]['r2'];x=np.array([0.,1.])+(j-.5)*.15
        values=np.array([stats[k]['reference_values'] for k in ['fit_anchor','oracle']])
        for r in range(4):ax.plot(x,values[:,r],color=col,alpha=.35,lw=.65,zorder=1)
        ax.scatter(np.repeat(x,4),values.flatten(),s=14,color=col,label=view.capitalize(),zorder=3)
    ax.set_xticks([0,1],['Fit-anchor law','Oracle law']);ax.set_ylim(0,.4);ax.set_ylabel(r'Mean error-predictor $r^2$');ax.grid(axis='y');ax.legend(fontsize=7,loc='upper left',handlelength=.8)
    panel(ax,'d','Full residual access is diagnostic only')
    a.out.parent.mkdir(exist_ok=True,parents=True);metadata=save(fig,a.out);plt.close(fig)
    a.out.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
