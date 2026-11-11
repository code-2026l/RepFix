"""Unlabeled development-only readers and conditional-reference gates; CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from consumer_routing import decisions,decision_margin,oscillation,oracle_referral

TARGETS=[0.,.05,.10,.20,.30,.40,.50,.60,.80,1.]
GATES=['confidence','boundary','normalized_boundary','learned_disagreement']

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save_json(path,value):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(path)

def ridge(x,y,coefficient):
    x=x.double();y=y.double();mean=x.mean(0);sd=x.std(0).clamp_min(1e-6);a=(x-mean)/sd
    ym=y.mean(0);w=torch.linalg.solve(a.T@a/len(a)+coefficient*torch.eye(a.shape[1]),a.T@(y-ym)/len(a))
    weight=(w.T/sd).float();bias=(ym-mean@(w/sd[:,None])).float()
    return dict(weight=weight,bias=bias),mean.float(),sd.float()

class CheapReader(nn.Module):
    def __init__(self,affine,mean,sd,residual):
        super().__init__();d=len(mean);classes=affine['weight'].shape[0]
        self.linear=nn.Linear(d,classes);self.linear.load_state_dict(affine);self.linear.requires_grad_(False)
        self.register_buffer('mean',mean);self.register_buffer('sd',sd)
        self.hidden=nn.Linear(d,64) if residual else None
        self.output=nn.Linear(64,classes) if residual else None
        if residual:nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        y=self.linear(x)
        if self.hidden is not None:y=y+self.output(F.gelu(self.hidden((x-self.mean)/self.sd)))
        return y

def fit_disagreement(x,errors):
    mean=x.mean(0);sd=x.std(0).clamp_min(1e-6);xx=(x-mean)/sd
    model=nn.Linear(x.shape[1],1).double();nn.init.zeros_(model.weight)
    p=errors.double().mean().clamp(.001,.999)
    with torch.no_grad():model.bias.fill_(float(p.logit()))
    opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=100,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=F.binary_cross_entropy_with_logits(model(xx.double()).squeeze(1),errors.double())+.01*model.weight.square().sum();loss.backward();return loss
    opt.step(closure)
    return dict(weight=(model.weight.detach()/sd.double()).float(),bias=(model.bias.detach()-(model.weight.detach()*mean.double()/sd.double()).sum()).float())

@torch.no_grad()
def gate_scores(features,cheap,thresholds,residual_model,disagreement_model):
    margin=decision_margin(cheap,thresholds)
    residual=F.linear(features,residual_model['weight'],residual_model['bias']).squeeze(1).double().exp().clamp_min(1e-6)
    error=F.linear(features,disagreement_model['weight'],disagreement_model['bias']).squeeze(1).double().sigmoid()
    return dict(confidence=cheap.double().softmax(1).max(1).values,boundary=margin,
                normalized_boundary=margin/residual,learned_disagreement=1-error)

def main(a):
    started=time.time();ref=torch.load(a.reference,map_location='cpu',weights_only=True)
    state=torch.load(a.anchors,map_location='cpu',weights_only=True);thresholds=list(ref['thresholds'].values());old=ref['heads']['coarse']
    ax=state['anchor_current'].float();target=F.linear(state['anchor_target'].float(),old['weight'],old['bias'])
    vx=state['validation_features'].float();reference=state['reference_logits'].float()
    assert len(ax)==2048 and len(vx)==5000
    order=torch.randperm(2048,generator=torch.Generator().manual_seed(6100));fitting,gatefit,calibration=order[:1024],order[1024:1536],order[1536:]
    assert len(set(fitting.tolist())&set(gatefit.tolist()))==0
    affine,mean,sd=ridge(ax[fitting],target[fitting],1e-3)
    out=dict(complete=False,device='cpu',phase='development',arch=ref['arch'],seed=a.seed,plan_sha256=a.plan_sha256,
        reference_sha256=digest(a.reference),anchor_artifact_sha256=digest(a.anchors),new_labels_for_fitting=False,
        primary_model_not_modified=True,splits=dict(reader=1024,gate=512,threshold=512,validation=5000,official_test_loaded=False),
        source_sha256={n:digest(Path(__file__).with_name(n)) for n in ['conditional_reference_cpu.py','consumer_routing.py']},branches=[],readers={})
    saved=dict(partition=order,thresholds=thresholds,reference=reference,coarse_labels=state['coarse_labels'],readers={})
    for kind in ['affine','residual64']:
        torch.manual_seed(6100);model=CheapReader(affine,mean,sd,kind=='residual64');trace=[]
        if kind=='residual64':
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
            generator=torch.Generator().manual_seed(6100);scale=target[fitting].var(0,unbiased=False).mean().clamp_min(1e-6)
            for epoch in range(100):
                losses=[]
                for idx in fitting[torch.randperm(len(fitting),generator=generator)].split(128):
                    loss=(model(ax[idx])-target[idx]).square().mean()/scale
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reader loss')
                    opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
                trace.append(float(np.mean(losses)))
        model.eval().requires_grad_(False)
        with torch.no_grad():
            fitcheap=model(ax);cheap=model(vx)
            train_errors=(decisions(fitcheap[gatefit],thresholds)!=decisions(target[gatefit],thresholds)).any(1)
            residual=oscillation(target[gatefit]-fitcheap[gatefit]).clamp_min(1e-6).log()[:,None]
            residual_model,_,_=ridge(ax[gatefit],residual,1.)
        disagreement_model=fit_disagreement(ax[gatefit],train_errors)
        with torch.no_grad():
            cal_scores=gate_scores(ax[calibration],fitcheap[calibration],thresholds,residual_model,disagreement_model)
            scores=gate_scores(vx,cheap,thresholds,residual_model,disagreement_model)
            errors=(decisions(cheap,thresholds)!=decisions(reference,thresholds)).numpy()
        safe=decision_margin(cheap,thresholds)>oscillation(reference-cheap)
        assert not errors[safe.numpy()].any()
        out['readers'][kind]=dict(trace=trace,unrouted_disagreements=errors.mean(0).tolist(),
            exact_residual_margin_fraction=float(safe.double().mean()),oracle={str(e):oracle_referral(errors,e) for e in [0.,.01,.05]})
        cutoffs={}
        for gate in GATES:
            cutoffs[gate]={}
            for fraction in TARGETS:
                cutoff=None if fraction in [0.,1.] else float(torch.quantile(cal_scores[gate],fraction))
                referred=torch.full((len(vx),),fraction==1.,dtype=torch.bool) if cutoff is None else scores[gate]<=cutoff
                remaining=errors & ~referred.numpy()[:,None];rates=remaining.mean(0).tolist()
                row=dict(reader=kind,gate=gate,target_referral=fraction,cutoff=cutoff,referral=float(referred.double().mean()),
                    disagreements=rates,maximum_disagreement=max(rates),passes_consumer_gate=max(rates)<=.01+1e-12)
                cutoffs[gate][str(fraction)]=cutoff;out['branches'].append(row)
        saved['readers'][kind]=dict(state=model.state_dict(),residual_model=residual_model,disagreement_model=disagreement_model,
            cheap=cheap,scores=scores,cutoffs=cutoffs,feature_dimension=vx.shape[1])
        print(json.dumps(dict(reader=kind,arch=ref['arch'],seed=a.seed,unrouted=errors.mean(0).tolist(),
            best_referral=min(v['referral'] for v in out['branches'] if v['reader']==kind and v['passes_consumer_gate']))),flush=True)
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True);torch.save(saved,folder/'routing.pt')
    out.update(complete=True,elapsed_s=time.time()-started,state_file='routing.pt',state_sha256=digest(folder/'routing.pt'))
    save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--anchors',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
