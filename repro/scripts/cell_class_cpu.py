"""Separately specified classification-cell ablation; thresholds still scored."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from consumer_contracts import confidence_thresholds
from cell_joint_cpu import train_branch,LAMBDAS,SOURCES
from cell_recovery_cpu import score_reader


def run(a):
    start=time.time();original=json.loads(a.record.read_text());domain,seed=original['domain'],original['seed']
    assert original['complete'] and original['phase']=='development'
    torch.manual_seed(seed);np.random.seed(seed)
    model,data,kind,splits=bc.load(SimpleNamespace(domain=domain,seed=seed,phase='development',limit_fit=0,limit_eval=0))
    assert splits==original['splits'] and 'test' not in data
    folder=a.record.parent/(a.record.stem+'_states');ref=torch.load(folder/'reference.pt',map_location='cpu',weights_only=True)
    assert paired.state_hash(ref['model'])==original['reference_model_sha256']
    A=ref['head']['weight'].double()/ref['sd'].reshape(-1);B=ref['head']['bias'].double()-A@ref['mean'].reshape(-1)
    model.load_state_dict(ref['model']);model.eval();healthy=paired.embed(model,data['x'],batch=1024)
    target=healthy@A.T+B;oldval=paired.embed(model,data['validation'][0],batch=1024)@A.T+B
    thresholds=confidence_thresholds(target);epochs=30 if domain=='har' else 25;dose=100. if domain=='har' else 1.
    out=dict(complete=False,device='cpu',cuda_visibility='',phase='development',route='classification_secondary',domain=domain,seed=seed,
        plan_sha256=a.plan_sha256,metric_addendum_plan_sha256=a.addendum_sha256,resource_addendum_plan_sha256=a.resource_sha256,
        secondary_plan_sha256=a.secondary_sha256,source_record_sha256=hashlib.sha256(a.record.read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['cell_class_cpu.py',*SOURCES]},
        splits=splits,thresholds=thresholds,configuration=dict(epochs=epochs,dose=dose,batch=256,lr=.001,gradient_clip=5.,
            protected_consumers=['classification'],scored_consumers=['classification','coverage80','coverage50']),branches=[])
    destination=a.out.parent/(a.out.stem+'_states');destination.mkdir(exist_ok=True,parents=True)
    for arch in ['affine','residual32']:
        for objective in ['cell_kl','cell_logodds']:
            for lam in LAMBDAS:
                row,saved=train_branch(model,data,kind,ref['model'],ref['optimizer'],A,B,healthy,target,{},
                    'protected',arch,objective,lam,seed,epochs,dose)
                assert row['initial_optimizer_sha256']==paired.state_hash(ref['optimizer'])
                row['objective']='class_'+objective
                h=paired.embed(model,data['validation'][0],batch=1024)
                row.update(score_reader(model,h,data['validation'],oldval,thresholds,saved['reader']))
                row['auxiliary_mse']=float(paired.auxiliary_loss(model,h.float(),data['validation'][2],kind).detach())
                name=arch+'_'+objective+'_%g.pt'%lam;row['state_file']=name;torch.save(saved,destination/name)
                out['branches'].append(row);paired.save_json(a.out.with_suffix('.partial.json'),out)
                print(json.dumps(dict(seed=seed,domain=domain,architecture=arch,objective=row['objective'],lam=lam,
                    own=row['own']['acc'],changes={c:r['disagreement'] for c,r in row['contracts'].items()},elapsed_s=time.time()-start)),flush=True)
    assert len({r['minibatch_order_sha256'] for r in out['branches']})==1
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--addendum-sha256',required=True);p.add_argument('--resource-sha256',required=True)
    p.add_argument('--secondary-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);run(a)
