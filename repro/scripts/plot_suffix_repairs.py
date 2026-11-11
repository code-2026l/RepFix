"""Complete validation-only repair screen; no test-outcome selection."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from figure_style import COLORS as C,WIDTH,apply_style,panel,save
ROOT=Path(__file__).resolve().parents[2]
ARMS=['frozen','mean_only','scalar_only','mean_scalar','diagonal_stats','paired_diagonal','scaled_procrustes','covariance','paired_ridge']
LABELS=['Frozen','Mean','Global scale','Mean + scale','Diagonal statistics','Paired diagonal','Procrustes','Covariance','Paired ridge']

def main():
    apply_style();rows=[json.loads(p.read_text()) for p in sorted((ROOT/'repro/results/CPUCONT/suffix_repair_development').glob('a???.json'))]
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.05));fig.subplots_adjust(left=.245,right=.975,bottom=.18,top=.79,wspace=.36)
    cmap=LinearSegmentedColormap.from_list('contract_drift',['#FAF9F8','#E7BFC7',C['frozen']])
    for i,(ax,r,title) in enumerate(zip(axes,rows,['ResNet-18','ViT-B/32'])):
        matrix=100*np.asarray([[max(b['repairs'][arm]['contracts'][k]['disagreement'] for b in r['branches']) for k in ['classification','coverage80','coverage50']] for arm in ARMS])
        artist=ax.imshow(matrix,cmap=cmap,vmin=0,vmax=30,aspect='auto');panel(ax,'ab'[i],title)
        for y in range(len(ARMS)):
            for x in range(3):ax.text(x,y,f'{matrix[y,x]:.1f}',ha='center',va='center',fontsize=7.3,color='white' if matrix[y,x]>19 else C['ink'])
        ax.set(xticks=[0,1,2],xticklabels=['Class','80%','50%'],yticks=range(len(ARMS)),yticklabels=LABELS if i==0 else ['']*len(ARMS));ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    cbax=fig.add_axes([.36,.905,.45,.027]);fig.colorbar(artist,cax=cbax,orientation='horizontal',ticks=[0,10,20,30]);cbax.xaxis.set_ticks_position('top');cbax.tick_params(labelsize=7.1,length=2)
    fig.text(.5,.86,'Maximum old-decision disagreement (%)',ha='center',fontsize=7.4)
    fig.text(.5,.045,'2048 isolated unlabeled anchors · 3 fixed development seeds · 1% acceptance gate',ha='center',fontsize=7.2,color='#606B76')
    result=save(fig,ROOT/'paper/figures/fig_suffix_repairs');plt.close(fig);return result

if __name__=='__main__':main()
