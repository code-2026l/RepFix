"""Frozen real-decoder basis audit on training/validation data, never test data.

For a ReLU decoder, the input Hessian of squared reconstruction loss equals its
Gauss-Newton matrix away from activation boundaries. Normalized gradient energy
uses residual covariance as well; these matrices generally differ.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from parameter_guard_har import load,flat_grad,variance
from real_aux_mtl import RealAuxModel


def audit(model,x,y):
    h=model(x);lp=F.cross_entropy(model.primary(h),y);la=F.mse_loss(model.aux(h),x)
    gp,=torch.autograd.grad(lp,h,retain_graph=True)
    ga,=torch.autograd.grad(la,h,retain_graph=True)
    params=list(model.parameters());pg=flat_grad(lp,params,True);ag=flat_grad(la,params,True)
    vv=flat_grad((h-h.mean(0)).square().sum()/(2*len(x)),params)
    with torch.no_grad():
        hd=h.double()-h.double().mean(0);ch=hd.T@hd/len(h)
        norm=gp.double().norm(dim=1,keepdim=True)+1e-8
        weighted=ga.double()/norm;cg=weighted.T@weighted/len(h)
        w1=model.aux[0].weight.double();w2=model.aux[2].weight.double()
        mask=(model.aux[0](h)>0).double()
        # E[D_i W2^T W2 D_i] = (W2^T W2) * E[mask_i mask_i^T].
        a=2/x.shape[1]*w1.T@((w2.T@w2)*(mask.T@mask/len(x)))@w1
        curves={};spectra={}
        for name,matrix in [('pca',ch),('hessian',a),('gradient_energy',cg)]:
            ev,u=torch.linalg.eigh(matrix);u=u.flip(1);spectra[name]=ev.flip(0).tolist()
            curves[name]=[]
            for r in range(h.shape[1]+1):
                q=u[:,:r];res=weighted-(weighted@q)@q.T
                curves[name].append(dict(rank=r,mean_ratio=float(res.norm(dim=1).mean()),rms_ratio=float(res.square().sum(1).mean().sqrt())))
        # Validate the RMS basis objective at every rank, including full deletion.
        for r in range(h.shape[1]+1):
            opt=curves['gradient_energy'][r]['rms_ratio']
            assert all(opt<=curves[k][r]['rms_ratio']+1e-6 for k in curves)
            assert curves['gradient_energy'][r]['mean_ratio']<=opt+1e-8
        comm=lambda a,b:float((a@b-b@a).norm()/(a.norm()*b.norm()).clamp_min(1e-30))
        residual=model.aux(h)-x
        return dict(n=len(x),curves=curves,spectra=spectra,
            representation_gradient_commutator=comm(ch,cg),hessian_gradient_commutator=comm(a,cg),
            signed_auxiliary_contraction=float(vv@ag),signed_primary_restoration=float(-vv@pg),
            parameter_gradient_cosine=float((pg@ag)/(pg.norm()*ag.norm()).clamp_min(1e-30)),
            reconstruction_mse=float(residual.square().mean()),
            scope='empirical frozen-batch residual-energy optimality; not mean-ratio optimality or training safety')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--m',type=int,default=64);a=ap.parse_args();torch.set_num_threads(1)
    data,meta=load();model=RealAuxModel(561,6,a.m,'recon',561)
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True));model.eval()
    results=dict(metadata=meta,checkpoint=Path(a.checkpoint).name,rows={})
    for split in ('train','validation'):
        x,y=data[split];idx=torch.randperm(len(x),generator=torch.Generator().manual_seed(739))[:256]
        results['rows'][split]=audit(model,x[idx],y[idx])
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(results,indent=2))
    print('PASS: real-decoder RMS basis optimality and Jensen bound on train/validation probes')


if __name__=='__main__':main()
