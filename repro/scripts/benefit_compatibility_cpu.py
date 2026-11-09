"""Predeclared CPU calculation: real auxiliary benefit with output preservation.

Development uses independent validation only. Confirmation scores the frozen
recipe on the test corpus and a separate 40-consumer finite-corpus audit bank.
All source authoring, statistical analysis and manuscript work remain local.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F

import paired_readout_continuation as paired
from consumer_risk import certificate
from readout_nullspace import guarded_features,masked_primary_loss,readout_projector
from readout_transfer_audit import fit_head,stratified_split


def load(args):
    d=paired.cdm.load_domain(args.domain)
    x=torch.from_numpy(d['X_tr']).float()
    y=torch.from_numpy(d['y_tr']).long()
    fit,val=stratified_split(y,.2,args.seed)
    if args.limit_fit:
        order=torch.randperm(len(fit),generator=torch.Generator().manual_seed(args.seed+301))
        fit=fit[order[:args.limit_fit]]
    frac=.1 if args.domain=='har' else .2
    mask=torch.zeros(len(fit),dtype=torch.bool)
    order=torch.randperm(len(fit),generator=torch.Generator().manual_seed(args.seed+501))
    mask[order[:max(1,round(len(fit)*frac))]]=True
    if len(torch.unique(y[fit][mask]))!=d['n_classes']:
        raise ValueError('Labeled fitting subset omits a primary class')
    if args.domain=='har':
        a=x.clone();at=torch.from_numpy(d['X_te']).float();kind='recon';adim=d['d_in']
    else:
        raw=torch.from_numpy(np.asarray(d['aux_tr'])).float()
        mean,sd=raw[fit].mean(),raw[fit].std(unbiased=False).clamp_min(1e-8)
        a=(raw-mean)/sd
        at=(torch.from_numpy(np.asarray(d['aux_te'])).float()-mean)/sd
        kind='snr';adim=1
    model=paired.ram.RealAuxModel(d['d_in'],d['n_classes'],64,
                                   'recon' if kind=='recon' else 'reg',adim)
    limit=args.limit_eval or len(val)
    val=val[:limit]
    data=dict(x=x[fit],y=y[fit],a=a[fit],labeled=mask,
              validation=(x[val],y[val],a[val]))
    if args.phase=='confirmation':
        xt=torch.from_numpy(d['X_te']).float();yt=torch.from_numpy(d['y_te']).long()
        lim=args.limit_eval or len(yt)
        data['test']=(xt[:lim],yt[:lim],at[:lim])
    splits=dict(fitting_examples=len(fit),labeled_fitting_examples=int(mask.sum()),
        primary_label_fraction=frac,validation_examples=len(val),
        fitting_indices_sha256=paired.state_hash(fit),validation_indices_sha256=paired.state_hash(val),
        labeled_mask_sha256=paired.state_hash(mask),validation_used_for_encoder=False,
        validation_used_for_healthy_head=False,test_metrics_computed=args.phase=='confirmation')
    return model,data,kind,splits


def current_logits(model,h):
    return F.linear(h.double(),model.primary.weight.double(),model.primary.bias.double())


def step(model,opt,x,y,a,mask,kind,alpha,detached,reference=None,projector=None):
    h=model(x)
    if projector is not None:
        h=guarded_features(h,reference,projector)
    lp=masked_primary_loss(current_logits(model,h),y,mask)
    ha=h.detach() if detached else h
    la=paired.auxiliary_loss(model,ha.float(),a,kind)
    loss=lp+alpha*la
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite operating loss')
    opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return float(lp.detach()),float(la.detach())


@torch.no_grad()
def feature_eval(model,x,base,projector):
    raw=paired.embed(model,x)
    return guarded_features(raw,base,projector) if projector is not None else raw


@torch.no_grad()
def diagnose(model,x,base,head,mu,sd,projector):
    h=feature_eval(model,x,base,projector)
    logits=head((h-mu)/sd)
    expected=head((base-mu)/sd)
    changed=int((logits.argmax(1)!=expected.argmax(1)).sum())
    return dict(anchor_examples=len(x),prediction_changes=changed,disagreement=changed/len(x),
                max_absolute_logit_change=float((logits-expected).abs().max()),
                threshold=.05,triggered=changed/len(x)>.05)


@torch.no_grad()
def score(model,part,base,head,mu,sd,projector,kind,monitor=None):
    x,y,a=part
    h=feature_eval(model,x,base,projector)
    z=head((h-mu)/sd);z0=head((base-mu)/sd)
    hcenter=h-h.mean(0);bcenter=base-base.mean(0)
    delta=float((z-z0).abs().max())
    changes=int((z.argmax(1)!=z0.argmax(1)).sum())
    if projector is not None and (delta>=1e-8 or changes):
        raise ArithmeticError('Guarded consumer output is not preserved numerically')
    result=dict(frozen=paired.metrics(z,y),own=paired.metrics(current_logits(model,h),y),
        reference=paired.metrics(z0,y),max_absolute_healthy_logit_error=delta,
        prediction_changes=changes,disagreement=changes/len(y),
        relative_trace=float(hcenter.square().sum()/bcenter.square().sum().clamp_min(1e-30)),
        auxiliary_mse=float(paired.auxiliary_loss(model,h.float(),a,kind)))
    if monitor is not None:
        bound=certificate(z.argmax(1).numpy(),z0.argmax(1).numpy(),monitor,.05/40)
        remaining=np.setdiff1d(np.arange(len(y)),monitor)
        old_correct=z0.argmax(1).numpy()==y.numpy()
        new_correct=z.argmax(1).numpy()==y.numpy()
        bound.update(budget=.05,certified=bound['upper_remaining_disagreement']<=.05,
                     remainder_loss_increase=float(old_correct[remaining].mean()-new_correct[remaining].mean()),
                     remainder_disagreement=float(np.mean(z.argmax(1).numpy()[remaining]!=z0.argmax(1).numpy()[remaining])))
        result['consumer_audit']=bound
    return result


def run(args):
    started=time.time()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    model,data,kind,splits=load(args)
    x,y,a,mask=[data[k] for k in ('x','y','a','labeled')]
    warm=args.warm_epochs or (60 if args.domain=='har' else 25)
    cont=args.cont_epochs or (30 if args.domain=='har' else 25)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    rng=torch.Generator().manual_seed(args.seed+10000)
    for epoch in range(warm):
        order=torch.randperm(len(x),generator=rng)
        for j in range(0,len(order),256):
            idx=order[j:j+256]
            step(model,opt,x[idx],y[idx],a[idx],mask[idx],kind,1.,True)
        if epoch==0 or (epoch+1)%10==0:
            print(json.dumps(dict(stage='warm',epoch=epoch+1,seed=args.seed,elapsed_s=time.time()-started)),flush=True)
    model.eval()
    reference=copy.deepcopy(model.state_dict());moments=copy.deepcopy(opt.state_dict())
    base=paired.embed(model,x)
    head,mu,sd=fit_head(base[mask],y[mask])
    projector,projection_info=readout_projector(head.weight.detach(),sd)
    bases={name:paired.embed(model,part[0]) for name,part in data.items() if name in ('validation','test')}
    model_sha,moment_sha=paired.state_hash(reference),paired.state_hash(moments)
    source_names=['benefit_compatibility_cpu.py','readout_nullspace.py','paired_readout_continuation.py','consumer_risk.py']
    out=dict(protocol='predeclared real auxiliary benefit and exact retained-affine-output control',
        plan_sha256=args.plan_sha256,selection_sha256=args.selection_sha256,
        phase=args.phase,seed=args.seed,domain=args.domain,auxiliary=kind,
        device='cpu',cuda_visibility=os.environ['CUDA_VISIBLE_DEVICES'],torch_threads=torch.get_num_threads(),
        torch_version=torch.__version__,python_version=platform.python_version(),splits=splits,
        configuration=dict(warm_epochs=warm,continuation_epochs=cont,doses=args.doses,width=64,batch=256,lr=.001),
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in source_names},
        reference_model_sha256=model_sha,reference_optimizer_sha256=moment_sha,
        projector=projection_info,reference_encoder_required_at_inference=True,
        validation=score(model,data['validation'],bases['validation'],head,mu,sd,None,kind),
        branches=[],complete=False)
    states=args.out.parent/(args.out.stem+'_states');states.mkdir(exist_ok=True,parents=True)
    torch.save(dict(model=reference,optimizer=moments,head=head.state_dict(),mean=mu,sd=sd,
                    projector=projector,splits=splits),states/'reference.pt')
    doses=[float(z) for z in args.doses.split(',')]
    detached_dose=doses[0] if len(doses)==1 else 1.
    branches=[('detach',detached_dose),('guarded_detach',detached_dose)]
    branches += [(mode,dose) for dose in doses for mode in ('joint','guarded_joint')]
    anchor_count=min(512,len(x));anchor_x=x[:anchor_count];anchor_ref=base[:anchor_count]
    monitor=None
    if args.phase=='confirmation':
        n=len(data['test'][1]);nmon=min(1024,n-1)
        monitor=np.random.default_rng(1850001+args.seed).choice(n,nmon,replace=False)
        out['consumer_bank']=dict(candidates=40,delta=.05,budget=.05,monitor_inputs=nmon,
             monitor_membership_sha256=hashlib.sha256(monitor.astype(np.int64).tobytes()).hexdigest(),
             separate_from_earlier_bank=True,used_for_training=False)
    for mode,alpha in branches:
        model.load_state_dict(reference)
        opt=torch.optim.Adam(model.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(moments))
        assert paired.state_hash(model.state_dict())==model_sha
        assert paired.state_hash(opt.state_dict())==moment_sha
        guarded=mode.startswith('guarded_');detached=mode.endswith('detach')
        p=projector if guarded else None
        row=dict(mode=mode,alpha=alpha,initial_model_sha256=model_sha,initial_optimizer_sha256=moment_sha,
                 guard_fixed_before_training=guarded,diagnostic_trace=[dict(epoch=0,**diagnose(model,anchor_x,anchor_ref,head,mu,sd,p))])
        rng=torch.Generator().manual_seed(args.seed+20000);order_hash=hashlib.sha256()
        begin=time.time()
        for epoch in range(cont):
            order=torch.randperm(len(x),generator=rng);order_hash.update(order.numpy().tobytes())
            for j in range(0,len(order),256):
                idx=order[j:j+256]
                step(model,opt,x[idx],y[idx],a[idx],mask[idx],kind,alpha,detached,
                     reference=base[idx] if guarded else None,projector=p)
            if epoch==0 or (epoch+1)%5==0 or epoch+1==cont:
                trace=dict(epoch=epoch+1,**diagnose(model,anchor_x,anchor_ref,head,mu,sd,p))
                row['diagnostic_trace'].append(trace)
                print(json.dumps(dict(stage='continue',mode=mode,alpha=alpha,epoch=epoch+1,
                    seed=args.seed,anchor_disagreement=trace['disagreement'],elapsed_s=time.time()-begin)),flush=True)
        row.update(validation=score(model,data['validation'],bases['validation'],head,mu,sd,p,kind),
                   minibatch_order_sha256=order_hash.hexdigest(),
                   final_model_sha256=paired.state_hash(model.state_dict()),
                   final_optimizer_sha256=paired.state_hash(opt.state_dict()),elapsed_s=time.time()-begin)
        if args.phase=='confirmation':
            row['test']=score(model,data['test'],bases['test'],head,mu,sd,p,kind,monitor)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict()),states/('%s_%g.pt'%(mode,alpha)))
        out['branches'].append(row)
        paired.save_json(args.out.with_suffix('.partial.json'),out)
    assert len({b['minibatch_order_sha256'] for b in out['branches']})==1
    if args.phase=='confirmation':
        old_model=copy.deepcopy(model);old_model.load_state_dict(reference);old_model.eval()
        sample=data['validation'][0][:min(1024,len(data['validation'][0]))]
        timings={}
        with torch.no_grad():
            for name,operation in [('ordinary',lambda:current_logits(model,model(sample))),
                  ('guarded_two_encoders',lambda:current_logits(model,guarded_features(
                      model(sample),old_model(sample),projector)))]:
                for _ in range(3):operation()
                times=[]
                for _ in range(9):
                    clock=time.perf_counter();operation();times.append(time.perf_counter()-clock)
                timings[name]=dict(batch_size=len(sample),median_seconds=float(np.median(times)),repeats=9)
        out['inference_latency']=timings
    out.update(complete=True,elapsed_s=time.time()-started)
    paired.save_json(args.out,out)
    print(json.dumps(dict(stage='complete',output=args.out.name,elapsed_s=out['elapsed_s'])),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--domain',choices=['har','radioml'],required=True)
    parser.add_argument('--phase',choices=['development','confirmation'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--doses',required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--selection-sha256')
    parser.add_argument('--warm-epochs',type=int,default=0)
    parser.add_argument('--cont-epochs',type=int,default=0)
    parser.add_argument('--limit-fit',type=int,default=0)
    parser.add_argument('--limit-eval',type=int,default=0)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if len(args.plan_sha256)!=64 or any(c not in '0123456789abcdef' for c in args.plan_sha256):
        parser.error('The fixed local plan SHA-256 is required')
    if args.phase=='confirmation' and not args.limit_fit and not args.selection_sha256:
        parser.error('Confirmation requires the frozen development-selection SHA-256')
    run(args)


if __name__=='__main__':
    main()
