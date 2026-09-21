"""Battery protocol v2: cell-disjoint validation and train-only preprocessing.

Historical data and drivers are never overwritten. Checkpoint selection sees
validation only. Test metrics are computed after the selected weights restore.
"""
import argparse
import json
import random
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
import battery_bmtl_v3 as base
from battery_representation_audit import representation_stats


def split_cells(cycles, seed):
    ids = sorted({str(c.get('_cid', 'cell')) for c in cycles})
    if len(ids) < 3:
        raise ValueError('at least three independent cells are needed')
    random.Random(seed).shuffle(ids)
    nt = max(1, len(ids)-int(.85*len(ids)))
    nv = max(1, int(.15*len(ids)))
    return dict(train=ids[:-(nt+nv)], validation=ids[-(nt+nv):-nt], test=ids[-nt:])


def prepare(cycles, seed, max_ts):
    cycles = [c for c in cycles if float(c.get('SOH', 0)) > 0]
    split = split_cells(cycles, seed)
    tr = [c for c in cycles if str(c.get('_cid','cell')) in split['train']]
    limits = {}
    for feat in base.FEATS:
        arrays = [np.asarray(c.get(feat) if c.get(feat) is not None else [], dtype=np.float32) for c in tr]
        arrays = [x for x in arrays if x.size]
        limits[feat] = [float(min(x.min() for x in arrays)),float(max(x.max() for x in arrays))] if arrays else [0.,1.]
    rul_scale = max(1., max(float(c['RUL']) for c in tr))
    thresholds = dict(rate=float(np.quantile([c['C_rate'] for c in tr],.75)),
                      temp_lo=float(np.quantile([c['temp'] for c in tr],.1)),
                      temp_hi=float(np.quantile([c['temp'] for c in tr],.9)))
    datasets = {}
    for name, cells in split.items():
        rows = [c for c in cycles if str(c.get('_cid','cell')) in cells]
        x = np.zeros((len(rows),max_ts,len(base.FEATS)),np.float32)
        for i,c in enumerate(rows):
            for j,feat in enumerate(base.FEATS):
                z=np.asarray(c.get(feat) if c.get(feat) is not None else [],np.float32)[-max_ts:]
                lo,hi=limits[feat]
                if z.size:
                    x[i,-len(z):,j]=np.clip((z-lo)/max(hi-lo,1e-6),0,1)
        cond=[1 if c['C_rate']>thresholds['rate'] else
              (2 if c['temp']<thresholds['temp_lo'] or c['temp']>thresholds['temp_hi'] else 0) for c in rows]
        datasets[name]=base.CycDataset(x,[c['SOH']/100 for c in rows],
            [c['RUL']/rul_scale for c in rows],np.ones(len(rows)),cond,[c.get('cens',0) for c in rows])
    meta=dict(protocol='cell_validation_v2', cells=split, feature_limits=limits,
              target_scale=[100.,rul_scale],condition_thresholds=thresholds,
              sample_counts={k:len(v) for k,v in datasets.items()},
              selection='minimum validation SOH RMSE', preprocessing_fit='training cells only',
              auxiliary_weighting='unweighted', test_use='after checkpoint selection only')
    return datasets,meta


@torch.no_grad()
def evaluate(model, loader, device, scale):
    se=ae=0.; n=0; cens_ae=obs_ae=0.; nc=no=0
    for x,s,r,_,_,cens in loader:
        x,s,r=x.to(device),s.to(device),r.to(device)
        z=model.embed(x)
        ps=model.head_soh(z).squeeze(-1);pr=model.head_rul(z).squeeze(-1)
        err=(pr-r).abs()
        se+=float((ps-s).square().sum());ae+=float(err.sum());n+=len(x)
        cm=cens.to(device)>0
        cens_ae+=float(torch.relu(r-pr)[cm].sum());nc+=int(cm.sum())
        obs_ae+=float(err[~cm].sum());no+=int((~cm).sum())
    return dict(soh_rmse=(se/n)**.5*scale[0],rul_mae_legacy=ae/n*scale[1],
                rul_observed_mae=obs_ae/no*scale[1] if no else None,
                rul_censored_shortfall=cens_ae/nc*scale[1] if nc else None,
                n=n,n_observed=no,n_censored=nc)


def self_test():
    rows=[dict(_cid=str(i//3),SOH=95.,RUL=i+1,C_rate=1.,temp=20.,cens=0,
               **{f:[float(i)]*4 for f in base.FEATS}) for i in range(24)]
    ds,meta=prepare(rows,0,4)
    a,b,c=map(set,meta['cells'].values())
    assert not (a&b or a&c or b&c) and a|b|c==set(str(i) for i in range(8))
    changed=[dict(x) for x in rows]
    for x in changed:
        if x['_cid'] in b|c:
            x['RUL']=1e8
            for f in base.FEATS:x[f]=[1e8]*4
    ds2,m2=prepare(changed,0,4)
    assert meta['feature_limits']==m2['feature_limits'] and meta['target_scale']==m2['target_scale']
    torch.testing.assert_close(ds['train'].X,ds2['train'].X)
    print('PASS: group disjointness and held-out perturbations cannot change fitted preprocessing')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['MATR','NASA'],default='MATR')
    ap.add_argument('--data-json')
    ap.add_argument('--out')
    ap.add_argument('--seeds',default='0')
    ap.add_argument('--seed-cell',type=int,default=0)
    ap.add_argument('--epochs',type=int,default=150)
    ap.add_argument('--d',type=int,default=128)
    ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--max-ts',type=int,default=128)
    ap.add_argument('--rank-max',type=int,default=128)
    ap.add_argument('--configs',default='single_soh,joint,stf0,stfcal,stfcalrank')
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args();torch.set_num_threads(1)
    if a.self_test:self_test();return
    if not a.out:ap.error('--out required')
    source=Path(a.data_json or base._path('battery_parsed',a.dataset.lower()+'_cycles.json'))
    datasets,meta=prepare(json.loads(source.read_text()),a.seed_cell,a.max_ts)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(metadata=meta,dataset=a.dataset,epochs=a.epochs,d=a.d,rank_max=a.rank_max,
                seeds=a.seeds,rows=[],complete=False)
    out.write_text(json.dumps(result,indent=2))
    print(meta,flush=True)
    if a.prepare_only:return
    device='cuda' if torch.cuda.is_available() else 'cpu'
    loaders={k:base._make_loader(v,a.batch,k=='train',0,False) for k,v in datasets.items()}
    modes={'single_soh':(False,'off'),'joint':(False,'off'),'stf0':(True,'off'),'stfcal':(False,'cal'),'stfcalrank':(False,'calrank')}
    for seed in map(int,a.seeds.split(',')):
        for mode in a.configs.split(','):
            stf,cal=modes[mode]
            base.set_seed(seed)
            model=base.BMTL(len(base.FEATS),d=a.d,n_ts=a.max_ts).to(device)
            opt=torch.optim.Adam(model.parameters(),lr=2e-3)
            val=base.run(model,loaders['train'],loaders['validation'],opt,a.epochs,
                device,1.,stf,0,1.,meta['target_scale'],stf_mode=cal,cal_warm_epochs=3,
                cal_rank_max=a.rank_max,cal_basis='pca',single='soh' if mode=='single_soh' else False)
            model.eval()
            test=evaluate(model,loaders['test'],device,meta['target_scale'])
            diag=representation_stats(model,loaders['test'],device)
            ckpt=out.with_name(out.stem+f'_{mode}_seed{seed}.pt')
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},ckpt)
            row=dict(seed=seed,mode=mode,validation=val,test=test,representation=diag,checkpoint=ckpt.name)
            result['rows'].append(row);out.write_text(json.dumps(result,indent=2))
            print({k:v for k,v in row.items() if k!='representation'},flush=True)
    result['complete']=True;out.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
