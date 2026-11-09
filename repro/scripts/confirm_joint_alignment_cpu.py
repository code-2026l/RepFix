"""Fresh-seed confirmation of the frozen alignment-development choices."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

import joint_alignment_cpu as ja
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from readout_transfer_audit import fit_head


def main(args):
    start=time.time();choice=json.loads(args.choice_file.read_text())
    assert hashlib.sha256(args.choice_file.read_bytes()).hexdigest()==args.selection_sha256
    assert choice['plan_sha256']==args.plan_sha256 and choice['choices_frozen_before_confirmation']
    selected=choice['domains'][args.domain];assert any(v['confirm'] for v in selected.values())
    assert args.seed in range(1300,1310)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    cfg=SimpleNamespace(domain=args.domain,seed=args.seed,phase='confirmation',limit_fit=0,limit_eval=0)
    model,data,kind,splits=bc.load(cfg);data['domain']=args.domain
    warm=60 if args.domain=='har' else 25;epochs=30 if args.domain=='har' else 25
    dose=100. if args.domain=='har' else 1.
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    rng=torch.Generator().manual_seed(args.seed+10000)
    for epoch in range(warm):
        order=torch.randperm(len(data['x']),generator=rng)
        for j in range(0,len(order),256):
            idx=order[j:j+256]
            bc.step(model,opt,data['x'][idx],data['y'][idx],data['a'][idx],data['labeled'][idx],kind,1.,True)
    model.eval();reference=copy.deepcopy(model.state_dict());moments=copy.deepcopy(opt.state_dict())
    healthy=paired.embed(model,data['x']);head,mu,sd=fit_head(healthy[data['labeled']],data['y'][data['labeled']])
    A=head.weight.detach().double()/sd.reshape(-1);B=head.bias.detach().double()-A@mu.reshape(-1)
    bases={part:paired.embed(model,data[part][0]) for part in ['validation','test']}
    directory=args.out.parent/(args.out.stem+'_states');directory.mkdir(exist_ok=True,parents=True)
    torch.save(dict(model=reference,optimizer=moments,A=A,B=B,splits=splits),directory/'reference.pt')
    out=dict(complete=False,device='cpu',cuda_visibility='',phase='confirmation',domain=args.domain,seed=args.seed,
        plan_sha256=args.plan_sha256,selection_sha256=args.selection_sha256,splits=splits,
        reference_model_sha256=paired.state_hash(reference),reference_optimizer_sha256=paired.state_hash(moments),
        configuration=dict(warm_epochs=warm,continuation_epochs=epochs,dose=dose,primary_monitor=1024,
            secondary_monitor=2048,primary_budget=.05,consumer_bank=160,threads=4),branches=[],
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ja.SOURCES},
        confirmation_driver_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    arms=[('detach',0.),('joint',0.),('protected',.1 if args.domain=='har' else 10.)]
    arms += [(m,selected[m]['selected_lambda']) for m in ['feature_map','logit_map'] if selected[m]['confirm']]
    protected_state=None
    for mode,lam in arms:
        row,state=ja.train_branch(model,data,kind,reference,moments,A,B,healthy,mode,lam,args.seed,epochs,dose)
        row.update(ja.evaluate(model,data,bases,A,B,kind,state['reader'],'confirmation',args.seed,mode not in ['detach','joint']))
        row['state_file']=mode+'_%g.pt'%lam
        if state['mapper'] is not None:
            with torch.no_grad():
                h=paired.embed(model,data['test'][0]).float()
                unfused=torch.nn.functional.linear(torch.nn.functional.linear(h,state['mapper']['weight']),A.float(),B.float())
                fused=ja.reader_logits(h,state['reader'])
                row['folding_precision']=dict(maximum_logit_error=float((fused-unfused).abs().max()),
                    prediction_changes=int((fused.argmax(1)!=unfused.argmax(1)).sum()))
        torch.save(state,directory/row['state_file']);out['branches'].append(row)
        if mode=='protected':protected_state=state
        paired.save_json(args.out.with_suffix('.partial.json'),out)
    if selected['direct_ols']['confirm']:
        model.load_state_dict(protected_state['model']);model.eval()
        reader,fit,difference=ja.add_ols(model,data,healthy,A,B)
        row=dict(mode='direct_ols',lam=.1 if args.domain=='har' else 10.,
            reader_sha256=paired.state_hash(reader),final_model_sha256=paired.state_hash(model.state_dict()),
            precision_comparison=difference,state_file='direct_ols.pt',
            nonlinear_layers_changed=ja.state_metadata(model,reference),
            minibatch_order_sha256=next(r['minibatch_order_sha256'] for r in out['branches'] if r['mode']=='protected'))
        row.update(ja.evaluate(model,data,bases,A,B,kind,reader,'confirmation',args.seed))
        torch.save(dict(model=model.state_dict(),reader=reader,ols64=fit),directory/'direct_ols.pt')
        out['branches'].append(row)
    assert len({b['minibatch_order_sha256'] for b in out['branches']})==1
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(args.out,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--domain',choices=['har','radioml'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--selection-sha256',required=True);p.add_argument('--choice-file',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);main(a)
