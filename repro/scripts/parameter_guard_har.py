"""Matched SGD experiment; subject validation; test evaluated after selection.

Uses released UCI-HAR features, with any additional standardization fitted on
training subjects only. The fixed guard batch is drawn only from training data.
Floor, dose and learning rate are experimental choices, not derived constants.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from real_aux_mtl import RealAuxModel, cdm
from parameter_variance_guard import project_update, checked_step
from stf_rms_experiment import variance_guard


def load():
    root=Path(cdm._path('uci_har','extracted','UCI HAR Dataset'))
    x=np.loadtxt(root/'train/X_train.txt').astype('float32')
    y=np.loadtxt(root/'train/y_train.txt').astype('int64')-1
    subject=np.loadtxt(root/'train/subject_train.txt').astype('int64')
    ids=np.unique(subject);np.random.default_rng(20260921).shuffle(ids)
    valids=ids[:4];mask=np.isin(subject,valids)
    mean=x[~mask].mean(0);sd=x[~mask].std(0).clip(1e-6)
    xt=np.loadtxt(root/'test/X_test.txt').astype('float32')
    yt=np.loadtxt(root/'test/y_test.txt').astype('int64')-1
    datasets={k:(torch.from_numpy((xx-mean)/sd),torch.from_numpy(yy))
              for k,xx,yy in [('train',x[~mask],y[~mask]),('validation',x[mask],y[mask]),('test',xt,yt)]}
    return datasets,dict(validation_subjects=valids.tolist(),training_subjects=ids[4:].tolist(),
                        preprocessing='released features; additional mean/std fit on training subjects',
                        selection='minimum validation primary cross entropy; test only after selection')


def flat_grad(loss,params,retain=False):
    grads=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,grads)])


def variance(model,x):
    h=model(x)
    return (h-h.mean(0)).square().sum()/(2*len(x))


# Fixed scalar attenuations, to bracket the calibrated one from both sides.
# `scale_01` is the historical matched baseline (a hard-coded 0.1).
FIXED_SCALES={'scale_0001':1e-4,'scale_001':0.01,'scale_01':0.1,'scale_03':0.3,'scale_06':0.6}


def probe_rho(model,guard):
    """Order parameters on the detached healthy anchor, in the same convention as
    the battery probe: gradients are taken with respect to the embedding only, so
    no parameter gradients are created and the probe cannot perturb the run.

    `u`  is rho_u = mean_b ||Pi_c (dL_a/dh)|| / ||dL_p/dh||, the paper's rank-1
         order parameter: the component along the per-sample centred direction
         u_c = h - E_b[h].
    `all` is the unprojected ratio ||dL_a/dh|| / ||dL_p/dh||

    The two differ by the projection onto a single one of the m representation
    coordinates, which for an isotropic auxiliary gradient is a factor ~sqrt(m).
    The contraction the theory describes acts on the representation as a whole, so
    the unprojected ratio is the one that sets a dose; rho_u is reported beside it
    because it is what the battery gate reads."""
    x,y=guard
    with torch.enable_grad():
        h=model(x).detach().requires_grad_(True)
        lp=F.cross_entropy(model.primary(h),y)
        la=F.mse_loss(model.aux(h),x)
        gp=torch.autograd.grad(lp,h,retain_graph=True,allow_unused=True)[0]
        ga=torch.autograd.grad(la,h,retain_graph=False,allow_unused=True)[0]
    if gp is None or ga is None:return None
    with torch.no_grad():
        hc=h.detach()-h.detach().mean(0,keepdim=True)
        u=hc/hc.norm(dim=1,keepdim=True).clamp_min(1e-8)
        proj=((ga*u).sum(-1,keepdim=True))*u
        gp_n=gp.norm(dim=1)+1e-8
        return dict(u=float((proj.norm(dim=1)/gp_n).mean()),
                    all=float((ga.norm(dim=1)/gp_n).mean()))


@torch.no_grad()
def evaluate(model,data):
    x,y=data;ce=mse=correct=0.;hs=[]
    for start in range(0,len(x),256):
        xb=x[start:start+256];yb=y[start:start+256];h=model(xb);p=model.primary(h)
        ce+=float(F.cross_entropy(p,yb,reduction='sum'))
        mse+=float(F.mse_loss(model.aux(h),xb,reduction='sum'))/x.shape[1]
        correct+=int((p.argmax(1)==yb).sum());hs.append(h.double())
    h=torch.cat(hs)
    if not torch.isfinite(h).all() or not np.isfinite(ce+mse):
        raise RuntimeError('nonfinite evaluation: numerical training failure')
    hc=h-h.mean(0);ev=torch.linalg.eigvalsh(hc.T@hc/(len(h)-1)).clamp_min(0)
    return dict(cross_entropy=ce/len(x),accuracy=correct/len(x),reconstruction_mse=mse/len(x),
                covariance_trace=float(ev.sum()),participation_rank=float(ev.sum().square()/ev.square().sum().clamp_min(1e-30)))


def run(data,mode,seed,a):
    torch.manual_seed(seed)
    model=RealAuxModel(data['train'][0].shape[1],6,a.m,'recon',data['train'][0].shape[1])
    params=list(model.parameters());x,y=data['train'];batch_rng=torch.Generator().manual_seed(seed+1000)
    guard_idx=torch.randperm(len(x),generator=torch.Generator().manual_seed(seed+2000))[:a.batch]
    fixed=x[guard_idx];guard=(fixed,y[guard_idx]);floor=None;best=float('inf');beststate=None;bestepoch=None
    stats=dict(steps=0,corrections=0,backtracks=0,rejections=0)
    start=time.monotonic();anchor=None;rho=None;q=1.0
    for epoch in range(a.epochs):
        if epoch==a.warm:
            anchor=float(variance(model,fixed).detach());floor=a.floor_fraction*anchor
            if mode=='cal_scale':
                # One probe on the healthy anchor fixes the multiplier; the dose
                # controller is calibrated, not tuned.  q = (1-eps)/(alpha*rho)
                # makes the effective dose alpha*q = (1-eps)/rho, i.e. a fixed
                # margin 1-eps below the critical shell.
                rho=probe_rho(model,guard)
                q=min(1.,(1.-a.eps)/max(a.alpha*rho['all'],1e-8))
        order=torch.randperm(len(x),generator=batch_rng)
        for idx in order.split(a.batch):
            xb=x[idx];yb=y[idx];h=model(xb);lp=F.cross_entropy(model.primary(h),yb)
            warm=epoch<a.warm
            ha=h.detach() if warm or mode=='detach' else variance_guard(h) if mode=='feature_guard' else h
            la=F.mse_loss(model.aux(ha),xb)
            if warm:scale=1.
            elif mode=='cal_scale':scale=q
            elif mode in FIXED_SCALES:scale=FIXED_SCALES[mode]
            else:scale=1.
            if mode=='pcgrad' and not warm:
                gp=flat_grad(lp,params,True);ga=flat_grad(a.alpha*la,params)
                dot=gp@ga
                u=gp+ga
                if float(dot)<0:
                    u=gp-dot/(ga@ga).clamp_min(1e-30)*ga+ga-dot/(gp@gp).clamp_min(1e-30)*gp
            else:
                loss=lp+a.alpha*scale*la
                if mode=='variance_penalty' and not warm:
                    loss=loss+a.penalty_weight*F.relu(1.-torch.sqrt(h.var(0,unbiased=False)+1e-4)).mean()
                u=flat_grad(loss,params)
            if mode=='parameter_guard' and not warm:
                val=variance(model,fixed);v=flat_grad(val,params)
                projected=project_update(u,v,float(val.detach())-floor,a.rate)
                stats['corrections']+=int(not torch.equal(u,projected))
                result=checked_step(params,projected,a.lr,lambda:variance(model,fixed),floor)
                stats['backtracks']+=result['backtracks'];stats['rejections']+=int(not result['accepted'])
            else:
                with torch.no_grad():
                    offset=0
                    for p in params:
                        p.add_(u[offset:offset+p.numel()].view_as(p),alpha=-a.lr);offset+=p.numel()
            stats['steps']+=1
        if epoch>=a.warm:
            val=evaluate(model,data['validation'])
            if not np.isfinite(val['cross_entropy']):raise RuntimeError('nonfinite validation loss')
            if val['cross_entropy']<best:
                best=val['cross_entropy'];beststate=copy.deepcopy(model.state_dict());bestepoch=epoch
    model.load_state_dict(beststate)
    if a.save_checkpoints:
        checkpoint=Path(a.out).with_name(Path(a.out).stem+f'_{mode}_seed{seed}.pt')
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),checkpoint)
    return dict(mode=mode,seed=seed,alpha=a.alpha,selected_epoch=bestepoch,validation=evaluate(model,data['validation']),
                test=evaluate(model,data['test']),anchor_variance=anchor,guard_floor=floor,
                probe_rho=rho,dose_multiplier=q,
                selected_guard_variance=float(variance(model,fixed).detach()),training=stats,seconds=time.monotonic()-start)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--warm',type=int,default=3)
    ap.add_argument('--m',type=int,default=64);ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--lr',type=float,default=.03);ap.add_argument('--alpha',type=float,default=12.)
    ap.add_argument('--floor-fraction',type=float,default=.5);ap.add_argument('--rate',type=float,default=1.)
    ap.add_argument('--penalty-weight',type=float,default=1.);ap.add_argument('--seeds',default='0,1,2')
    ap.add_argument('--eps',type=float,default=.1)
    ap.add_argument('--save-checkpoints',action='store_true')
    ap.add_argument('--modes',default='joint,detach,scale_01,pcgrad,variance_penalty,feature_guard,parameter_guard')
    a=ap.parse_args()
    if a.epochs<=a.warm or not 0<a.floor_fraction<1:ap.error('epochs must exceed warm; floor fraction in (0,1)')
    if not 0<a.eps<1:ap.error('eps must be in (0,1)')
    valid={'joint','detach','pcgrad','variance_penalty','feature_guard','parameter_guard','cal_scale'}|set(FIXED_SCALES)
    if not set(a.modes.split(','))<=valid:ap.error('unknown mode')
    torch.set_num_threads(1);data,meta=load();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(experimental=True,optimizer='plain SGD, no momentum or decay',config=vars(a),metadata=meta,rows=[],complete=False)
    for seed in map(int,a.seeds.split(',')):
        for mode in a.modes.split(','):
            try:
                row=run(data,mode,seed,a)
                row['status']='completed'
            except RuntimeError as exc:
                row=dict(mode=mode,seed=seed,alpha=a.alpha,status='failed',error=str(exc))
            result['rows'].append(row);out.write_text(json.dumps(result,indent=2));print(row,flush=True)
    result['complete']=True;out.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
