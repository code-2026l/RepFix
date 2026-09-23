"""Rebuild the three main figures from archived, seed-level evidence.

Run from any directory: python paper/revision/build_figures.py
No training, imputed observations, or fitted performance curves.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
OUT = PAPER / 'figures'
SOURCES = {}
REPORT = {}
COL = dict(har='#2878A6', battery='#008A70', radioml='#CC6633', nyu='#8061A8',
           joint='#C55A3B', stf0='#677482', stfcal='#008A70', stfcalrank='#8061A8')
DOM = dict(har='HAR', battery='Battery', radioml='RadioML', nyu='NYUv2')
plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':7.5, 'axes.labelsize':7.5,
    'axes.titlesize':8, 'xtick.labelsize':7, 'ytick.labelsize':7,
    'legend.fontsize':6.6, 'axes.linewidth':.65, 'lines.linewidth':1.4,
    'text.color':'#223044', 'axes.labelcolor':'#223044',
    'xtick.color':'#445365', 'ytick.color':'#445365',
    'axes.edgecolor':'#A2ADBA', 'axes.axisbelow':True,
    'grid.color':'#E6EAF0', 'grid.linewidth':.5,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'axes.spines.top':False, 'axes.spines.right':False,
    'legend.frameon':False, 'xtick.major.size':3, 'ytick.major.size':3,
})

def source(rel):
    p=ROOT/rel
    blob=p.read_bytes()
    SOURCES[rel]=hashlib.sha256(blob).hexdigest()
    return blob.decode('utf-8')

def read(rel):
    return json.loads(source(rel))

def title(ax, letter, label):
    ax.set_title(f'{letter}  {label}', loc='left', fontweight='bold', pad=9)

def save(fig, name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    # Visible labels must be inside the final canvas. Tick labels outside view
    # limits are not drawn by Matplotlib and are excluded from this check.
    for ax in fig.axes:
        for t in [ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label, *ax.texts]:
            if not t.get_text() or not t.get_visible(): continue
            bb=t.get_window_extent(renderer)
            if bb.width and bb.height:
                assert bb.x0 >= -1 and bb.y0 >= -1, (name,t.get_text(),bb)
                assert bb.x1 <= fig.bbox.width+1 and bb.y1 <= fig.bbox.height+1, (name,t.get_text(),bb)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{ext}', dpi=300, facecolor='white')
    assert b'/Subtype /Type3' not in (OUT/f'{name}.pdf').read_bytes()
    plt.close(fig)

def crossing(rows):
    pts=sorted((float(r['alpha']),float(r['rel_scatter_mean'])) for r in rows)
    for (a,s),(b,t) in zip(pts,pts[1:]):
        if t < .5 <= s and a>0:
            return math.exp(math.log(a)+(.5-s)/(t-s)*math.log(b/a))
    raise ValueError('No bracketed half-height crossing')

def probe_cells():
    sweeps={}
    for domain in ['har','battery']:
        for width in [32,64,128,256]:
            sweeps[domain,width]=read(f'results/BSCALE/xcap_{domain}_m{width}.json')['results']
    for r in read('repro/results/xcap2_radioml.json')['results']:
        sweeps.setdefault(('radioml',int(r['m'])),[]).append(r)
    for width in [32,64,128,256]:
        sweeps['nyu',width]=read(f'results/NYUCAP/xcap_nyu_m{width}.json')['results']
    cells=[]
    for domain in DOM:
        for width in [32,64,128,256]:
            rel=(f'results/WARM/warm_nyu_m{width}_t4.json' if domain=='nyu'
                 else f'results/R3/R3_calib_{domain}_m{width}.json')
            rs=read(rel)['raw']
            vals=[r['cal_rho_u'] for r in rs if np.isfinite(r.get('cal_rho_u',np.nan))]
            assert len(vals)==10
            pred=1/np.mean(vals); meas=crossing(sweeps[domain,width])
            cells.append(dict(domain=domain,width=width,head='fixed variance toxifier',
                              rho=float(np.mean(vals)),predicted=float(pred),half_height=meas,
                              ratio=float(pred/meas),probe_seeds=10))
    REPORT['probe_cells']=cells
    return cells

def fig_gap():
    fig,(a,b)=plt.subplots(1,2,figsize=(5.5,2.60),gridspec_kw={'width_ratios':[1.05,1]})
    fig.subplots_adjust(left=.105,right=.99,bottom=.20,top=.87,wspace=.63)
    real=[]
    specs=[('har_recon','HAR reconstruction','o'),('har_clf','HAR subject-ID','s'),
           ('radioml_recon','RadioML I/Q','^'),('radioml_reg','RadioML SNR','D')]
    for key,label,marker in specs:
        d=read(f'repro/results/A1/A1_{key}_10s.json')
        xs=[r['align'][0]['cos_pa'] for r in d['raw']]
        ys=[r['align'][0]['frac_uc'] for r in d['raw']]
        color=COL['joint'] if key.endswith('_reg') else COL['har']
        a.scatter(xs,ys,s=17,marker=marker,c=color if key.endswith('_reg') else 'white',
                  edgecolor=color,linewidth=.8,label=label,zorder=3)
        real.append(dict(head=key,mean_cos=float(np.mean(xs)),max_abs_cos=float(np.max(np.abs(xs))),
                         mean_radial_mass=float(np.mean(ys)),
                         mean_signed_cos=float(np.mean([r['align'][0]['cos_a_uc'] for r in d['raw']])),
                         mean_negative_pair_fraction=float(np.mean([r['align'][0]['neg_frac'] for r in d['raw']]))))
    a.axvline(0,color='#B6C0CA',lw=.8)
    a.set(xlim=(-.067,.067),ylim=(-.03,1.03),xticks=[-.05,0,.05],yticks=[0,.5,1],
          xlabel='Mean task-gradient cosine',ylabel='Unsigned radial mass')
    a.text(-.063,.70,'SNR: 10/10 collapse',color=COL['joint'],fontsize=7)
    a.legend(loc='center left',bbox_to_anchor=(-.02,.37),handletextpad=.3,labelspacing=.25,fontsize=6.2)
    title(a,'a','Two distinct diagnostic axes')
    methods=['joint','pcgrad','cagrad','mgda','aligned','imtl','nash','famo','uw','gradnorm','imtll']
    names=['Joint','PCGrad','CAGrad','MGDA','Aligned-MTL','IMTL-G','Nash-MTL','FAMO','Uncertainty','GradNorm','IMTL-L']
    mat=[]
    actual={}
    optimizer_audit={}
    for dom in ['har','battery','radioml']:
        d=read(f'repro/results/P02/MOO_{dom}.json')
        assert len(d['records'])==10
        all_modes={r['mode'] for record in d['records'] for r in record['results']}
        grouped={mode:[r for record in d['records'] for r in record['results'] if r['mode']==mode]
                 for mode in all_modes}
        assert all(len(rs)==10 for rs in grouped.values())
        actual[dom]={mode:float(np.mean([r['collapse_rate'] for r in rs])) for mode,rs in grouped.items()}
        optimizer_audit[dom]={mode:dict(n=10,collapse_rate=actual[dom][mode],
                                         metric=float(np.mean([r['metric_mean'] for r in rs])),
                                         alpha=rs[0]['alpha']) for mode,rs in grouped.items()}
    # Match archived mode spellings explicitly; fail if an expected result is absent.
    aliases={'aligned':['aligned','align'],'imtl':['imtl','imtl_g','imtlg'],
             'nash':['nash','nashmtl'],'uw':['uw','uncertainty'],
             'imtll':['imtll','imtl_l']}
    for method in methods:
        vals=[]
        for dom in ['har','battery','radioml']:
            keys=[k for k in aliases.get(method,[method]) if k in actual[dom]]
            if not keys: raise KeyError((method,actual[dom]))
            vals.append(actual[dom][keys[0]])
        mat.append(vals)
    # Detachment controls are in the matched repair files, not P02's combiner rows.
    detach=[]
    for dom in ['har','battery','radioml']:
        d=read(f'results/R3/R3_dose_{dom}_hi.json')
        rs=[r for r in d['raw'] if r['mode']=='stf0']
        assert len(rs)==10 and len({r['seed'] for r in rs})==10
        detach.append(float(np.mean([r['collapse'] for r in rs])))
        optimizer_audit[dom]['detached_control']=dict(n=10,width=rs[0]['m'],alpha=rs[0]['alpha'],
                 collapse_rate=detach[-1],metric=float(np.mean([r['metric'] for r in rs])))
    mat.append(detach); names.append('Detach')
    b.imshow(mat,aspect='auto',cmap=ListedColormap(['#D8EAE5','#E9B29F']),vmin=0,vmax=1)
    b.set_xticks(range(3),['HAR','Battery','RadioML'],fontsize=6.8)
    b.set_yticks(range(len(names)),names,fontsize=6.4)
    b.tick_params(axis='both',length=0)
    for i,row in enumerate(mat):
        for j,v in enumerate(row): b.text(j,i,f'{int(round(v*10))}/10',ha='center',va='center',fontsize=6.1)
    b.set_xticks(np.arange(-.5,3,1),minor=True); b.set_yticks(np.arange(-.5,len(names),1),minor=True)
    b.grid(which='minor',color='white',lw=1.5);b.tick_params(which='minor',length=0)
    for spine in b.spines.values():spine.set_visible(False)
    title(b,'b','High-dose collapse counts')
    REPORT['real_head_alignment']=real
    REPORT['optimizer_collapse_counts']={'methods':names,'domains':['har','battery','radioml'],'rates':mat}
    REPORT['optimizer_audit']=optimizer_audit
    save(fig,'fig_survival_gap')

def fig_boundary(cells):
    fig,axs=plt.subplots(2,2,figsize=(5.5,4.50))
    fig.subplots_adjust(left=.12,right=.97,bottom=.105,top=.91,hspace=.68,wspace=.39)
    a,b,c,d=axs.ravel()
    ratios=np.array([x['ratio'] for x in cells]).reshape(4,4)
    a.imshow(np.log10(ratios),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-2,vcenter=0,vmax=2),aspect='auto')
    a.set_xticks(range(4),[32,64,128,256]);a.set_yticks(range(4),list(DOM.values()))
    a.set_xlabel('Encoder width');a.tick_params(length=0)
    for i in range(4):
        for j in range(4):
            val=ratios[i,j]
            a.text(j,i,f'{val:.1f}' if val>=1 else f'{val:.2f}',ha='center',va='center',
                   fontsize=8,color='white' if abs(math.log10(val))>1.25 else '#223044')
    a.set_xticks(np.arange(-.5,4,1),minor=True);a.set_yticks(np.arange(-.5,4,1),minor=True)
    a.grid(which='minor',color='white',lw=1.5);a.tick_params(which='minor',length=0)
    for spine in a.spines.values():spine.set_visible(False)
    title(a,'a','Dose estimate / half-height')
    a.text(.5,1.03,'Red: overestimates tolerable dose',transform=a.transAxes,
           fontsize=6.5,color='#9B382E',ha='center')
    warm=[]
    for dom in ['har','battery','nyu']:
        rows=[]
        for path in sorted((ROOT/'results/WARM').glob(f'warm_{dom}_m64_t*.json')):
            w=int(re.search(r'_t(\d+)',path.name).group(1))
            raw=read(path.relative_to(ROOT).as_posix())['raw']
            rho=np.mean([r['cal_rho_u'] for r in raw if np.isfinite(r.get('cal_rho_u',np.nan))])
            half=next(x['half_height'] for x in cells if x['domain']==dom and x['width']==64)
            rows.append((w,1/rho/half))
        rows.sort();warm.extend(dict(domain=dom,warmup=w,ratio=float(y)) for w,y in rows)
        b.plot(*np.array(rows).T,marker='o',ms=3.2,color=COL[dom],label=DOM[dom])
    b.axhspan(1/3,3,color='#E7EFEA',zorder=-1)
    b.axhline(1,color='#85968D',lw=.7,ls='--')
    b.axvline(4,color='#A9B4C0',lw=.7,ls=':')
    b.set(xscale='log',yscale='log',xticks=[1,4,16,32],xticklabels=['1','4','16','32'],
          xlabel='Detached warm-up epochs',ylabel='Dose estimate / half-height',ylim=(1e-3,2e3))
    b.legend(loc='lower left',fontsize=6.2,handlelength=1.2,labelspacing=.25)
    title(b,'b','Calibration-time sensitivity')
    drift=[]
    for dom in ['har','battery']:
        raw=read(f'results/RHO/xcap_{dom}_zero_m64.json')['raw']
        traces=[r['rho_trace'] for r in raw if float(r['alpha'])==0 and r['rho_trace']]
        assert len(traces)==5
        values=np.array([[t['rho_u'] for t in r] for r in traces])
        steps=np.array([t['step'] for t in traces[0]],float)
        y=values/values[:,:1]
        c.plot(steps/steps[-1],np.median(y,axis=0),color=COL[dom],label=DOM[dom])
        c.fill_between(steps/steps[-1],y.min(0),y.max(0),color=COL[dom],alpha=.15,lw=0)
        drift.append(dict(domain=dom,initial_mean=float(values[:,0].mean()),final_mean=float(values[:,-1].mean()),
                          final_scatter_mean=float(np.mean([r['rel_scatter'] for r in raw if r['alpha']==0]))))
    c.set(yscale='log',xlabel='Fraction of training completed',ylabel=r'Probe drift $\hat\rho_t/\hat\rho_0$',
          xlim=(0,1),ylim=(.3,3e7),yticks=[1,1e3,1e6])
    c.text(.04,.69,r'Zero auxiliary dose: $\alpha=0$',transform=c.transAxes,fontsize=6.8)
    c.legend(loc='center right',bbox_to_anchor=(1,.40),fontsize=6.5)
    title(c,'c','Drift at zero auxiliary dose')
    gate=[]
    for w in [1,2,4,8,16,32]:
        raw=read(f'results/WARMU/warmu_har_m64_a0p03_t{w}.json')['raw']
        rs=[r for r in raw if r['mode']=='stfcal']
        gate.append(dict(warmup=w,n=len(rs),gate=float(np.mean([r['cal_gate'] for r in rs])),
                         collapse=float(np.mean([r['collapse'] for r in rs])),scatter=float(np.mean([r['rel_scatter'] for r in rs]))))
    x=[r['warmup'] for r in gate]
    d.plot(x,[r['gate'] for r in gate],'o-',ms=4,color=COL['stfcal'],label='Gate open')
    d.plot(x,[r['collapse'] for r in gate],'s--',ms=3.8,mfc='white',color=COL['joint'],label='Final collapse')
    d.set(xscale='log',xticks=x,xticklabels=[str(w) for w in x],ylim=(-.08,1.1),yticks=[0,.5,1],
          xlabel='Detached warm-up epochs',ylabel='Fraction of seeds')
    d.text(.35,.82,r'HAR, $\alpha=0.03$',transform=d.transAxes,fontsize=6.8)
    d.legend(loc='center right',fontsize=6.4,handlelength=1.4,labelspacing=.4)
    title(d,'d','Calibration and gate failure')
    REPORT.update(warmup_ratios=warm,zero_dose_drift=drift,warmup_intervention=gate)
    save(fig,'fig_probe_boundary')

def dose_records():
    src=ROOT/'repro/results/_screena'
    pat=re.compile(r'mode=(\w+) seed=(\d+) metric=([\d.eE+-]+) aux=[\d.eE+-]+ rel_scatter=([\d.eE+-]+) collapse=(True|False)')
    records={};duplicates=0
    for p in sorted(src.glob('*.txt')):
        if p.name.startswith('_reg_'):
            m=64; tag=re.search(r'_a([\dp]+)_s',p.name)
            if not tag:continue
            alpha=float(tag.group(1).replace('p','.'))
        elif p.name.startswith('_dense_rmlreg_'):
            m=int(re.search(r'_m(\d+)',p.name).group(1));alpha=float(re.search(r'_a(\d+)',p.name).group(1))
        else:continue
        if m!=64:continue
        for line in source(p.relative_to(ROOT).as_posix()).splitlines():
            match=pat.search(line)
            if not match:continue
            mode,seed,metric,scatter,collapse=match.groups()
            if mode not in ['joint','stf0','stfcal']:continue
            key=(alpha,mode,int(seed));v=dict(alpha=alpha,mode=mode,seed=int(seed),metric=float(metric),
                                           scatter=float(scatter),collapse=(collapse=='True'))
            if key in records:
                assert records[key]==v, (p,key,'Conflicting duplicate')
                duplicates+=1
            records[key]=v
    for alpha in {k[0] for k in records}:
        for mode in ['joint','stf0','stfcal']:
            assert sum(k[0]==alpha and k[1]==mode for k in records)==5,(alpha,mode)
    REPORT['dose_duplicate_records_removed']=duplicates
    REPORT['dose_records']=list(records.values())
    return records

def fig_intervention():
    records=dose_records()
    # Stack the dose responses on a shared x axis; reserve the full right
    # column for seed-level survival/utility and collision-free labels.
    fig=plt.figure(figsize=(5.5,3.25))
    gs=fig.add_gridspec(2,2,width_ratios=[1.08,1],left=.115,right=.985,
                        bottom=.145,top=.81,wspace=.43,hspace=.40)
    a=fig.add_subplot(gs[0,0])
    b=fig.add_subplot(gs[1,0],sharex=a)
    c=fig.add_subplot(gs[:,1])
    alphas=sorted({key[0] for key in records})
    labels=dict(joint='Joint',stf0='Detach',stfcal='Calibrated STF')
    for mode in labels:
        rs=[[records[alpha,mode,s] for s in range(5)] for alpha in alphas]
        acc=np.array([[r['metric'] for r in row] for row in rs]); scat=np.array([[r['scatter'] for r in row] for row in rs])
        for ax,ys in [(a,acc),(b,scat)]:
            ax.plot(alphas,ys.mean(1),marker='o',ms=2.4,lw=1.25,
                    color=COL[mode],label=labels[mode],zorder=3)
            ax.fill_between(alphas,ys.mean(1)-ys.std(1),ys.mean(1)+ys.std(1),color=COL[mode],alpha=.13,lw=0,zorder=1)
        for i,alpha in enumerate(alphas):
            if mode=='joint' and any(r['collapse'] for r in rs[i]):
                a.plot(alpha,acc.mean(1)[i],marker='x',color=COL[mode],ms=4,mew=.9,zorder=4)
    for ax in [a,b]:
        ax.set(xscale='log',xlim=(.075,1350),xticks=[.1,1,10,100,1000],
               xticklabels=['0.1','1','10','100','1000'])
        ax.grid(axis='y',lw=.45,color='#E6EAF0')
        ax.tick_params(which='minor',bottom=False)
    a.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
    a.set(ylabel='Accuracy',ylim=(.19,.32),yticks=[.20,.25,.30])
    a.text(.025,.065,'20% labels / 5 seeds',transform=a.transAxes,
           fontsize=6.5,color='#677482',
           bbox=dict(facecolor='white',edgecolor='none',pad=.8,alpha=.94))
    b.axhline(.3,color=COL['joint'],ls=':',lw=.9,zorder=2)
    b.set(xlabel=r'Auxiliary dose $\alpha$',ylabel='Relative scatter',ylim=(0,1.4),yticks=[0,.3,1])
    title(a,'a','Task utility');title(b,'b','Representation survival')
    d=read('results/R5aux/R5aux_radioml_reg_10s.json')
    mode_labels=dict(joint='Joint',stf0='Detach',stfcal='Radial',stfcalrank='Rank-3')
    # Same real auxiliary, separate fully labeled protocol: show both outcomes.
    real_results=[]
    for mode in mode_labels:
        rs=[r for r in d['raw'] if r['mode']==mode]
        x=np.array([r['rel_scatter'] for r in rs]); y=np.array([r['metric'] for r in rs])
        c.scatter(x,y,s=9,color=COL[mode],alpha=.24,linewidths=0,zorder=2)
        c.errorbar(x.mean(),y.mean(),xerr=x.std(),yerr=y.std(),fmt='o',ms=4,
                   color=COL[mode],capsize=2,lw=.85,zorder=4)
        positions={'joint':(.035,.184),'stf0':(.59,.319),
                   'stfcal':(.44,.174),'stfcalrank':(.63,.232)}
        c.annotate(mode_labels[mode],xy=(x.mean(),y.mean()),xytext=positions[mode],
                   fontsize=7,color=COL[mode],ha='left',va='center',
                   arrowprops=dict(arrowstyle='-',color=COL[mode],lw=.65,
                                   shrinkA=3,shrinkB=5,
                                   connectionstyle='arc3,rad=0'),zorder=5)
        real_results.append(dict(mode=mode,n=len(rs),accuracy=float(y.mean()),scatter=float(x.mean()),
                                 collapse_count=sum(r['collapse'] for r in rs)))
    c.axvline(.3,color=COL['joint'],ls=':',lw=.9,zorder=1)
    c.set(xlabel='Relative scatter',ylabel='Accuracy',xlim=(0,1.16),ylim=(.16,.35),
          xticks=[0,.3,.6,1],yticks=[.20,.25,.30])
    c.grid(axis='y',lw=.45,color='#E6EAF0')
    c.text(.04,.955,r'Full labels / $\alpha=3000$ / 10 seeds',transform=c.transAxes,
           fontsize=6.25,color='#677482',va='top',
           bbox=dict(facecolor='white',edgecolor='none',pad=1,alpha=.97))
    title(c,'c','Survival versus utility')
    fig.legend(*a.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,.995),
               ncol=3,fontsize=7,handlelength=1.8,columnspacing=1.7)
    REPORT['real_snr_full_label']=real_results
    save(fig,'fig_diagnostic_intervention')

if __name__=='__main__':
    cells=probe_cells()
    fig_gap();fig_boundary(cells);fig_intervention()
    REPORT['source_sha256']=SOURCES
    (PAPER/'revision/evidence_audit.json').write_text(
        json.dumps(REPORT,indent=2),encoding='utf-8',newline='\n')
    print('Built 3 vector figures (9 panels), PNG previews, editable SVGs, and evidence audit.')
    print('Deduplicated dose records:',REPORT['dose_duplicate_records_removed'])
    for dom in DOM:
        ratios=[c['ratio'] for c in cells if c['domain']==dom]
        print(dom,'ratio range',min(ratios),max(ratios))
