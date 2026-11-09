"""Independent replay and fixed retrospective reader controls; numerical CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='4'

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from consumer_risk import certificate
from legacy_head_fit import fit_legacy_heads


def load_record(path):
    record=json.loads(path.read_text())
    assert record['complete'] and record['phase']=='confirmation'
    for name,expected in record['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==expected
    cfg=SimpleNamespace(domain=record['domain'],seed=record['seed'],phase='confirmation',limit_fit=0,limit_eval=0)
    torch.manual_seed(record['seed']);np.random.seed(record['seed'])
    model,data,kind,splits=bc.load(cfg)
    assert splits==record['splits']
    states=path.parent/(path.stem+'_states')
    healthy=torch.load(states/'reference.pt',map_location='cpu',weights_only=True)
    assert paired.state_hash(healthy['model'])==record['reference_model_sha256']
    assert paired.state_hash(healthy['optimizer'])==record['reference_optimizer_sha256']
    model.load_state_dict(healthy['model']);model.eval()
    weight=healthy['head']['weight']/healthy['sd'].reshape(-1)
    bias=healthy['head']['bias']-weight@healthy['mean'].reshape(-1)
    return record,model,data,kind,states,weight,bias


def score(logits,reference,labels,monitor,alpha):
    bound=certificate(logits.argmax(1).numpy(),reference.argmax(1).numpy(),monitor,alpha)
    bound['certified']=bound['upper_remaining_disagreement']<=.05
    return dict(metrics=paired.metrics(logits,labels),disagreement=float((logits.argmax(1)!=reference.argmax(1)).double().mean()),
                maximum_logit_error=float((logits-reference).abs().max()),consumer_audit=bound)


@torch.no_grad()
def replay(args):
    record,model,data,kind,states,A,B=load_record(args.record)
    x,y=data['x'],data['y']
    parts=['validation','test']
    bases={part:paired.embed(model,data[part][0]) for part in parts}
    targets=(paired.embed(model,x)@A.T+B).float().double()
    n=len(data['test'][1])
    original_monitor=np.random.default_rng(1850001+record['seed']).choice(n,min(1024,n-1),replace=False)
    new_monitor=np.random.default_rng(6290001+record['seed']+(100000 if record['domain']=='har' else 0)).choice(n,min(1024,n-1),replace=False)
    result=dict(complete=False,device='cpu',cuda_visibility='',domain=record['domain'],seed=record['seed'],
        original_record_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        audit_plan_sha256=args.plan_sha256,
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fitter_source_sha256=hashlib.sha256(Path(__file__).with_name('legacy_head_fit.py').read_bytes()).hexdigest(),
        original_plan_sha256=record['plan_sha256'],retrospective_frozen_feature_control=True,
        new_family_size=140,new_monitor_sha256=hashlib.sha256(new_monitor.astype(np.int64).tobytes()).hexdigest(),
        no_encoder_updates=True,labels_used_for_fitting=False,branches=[])
    fitted_dir=args.out.parent/(args.out.stem+'_readers');fitted_dir.mkdir(exist_ok=True,parents=True)
    largest=0.
    for branch in record['branches']:
        mode,lam=branch['mode'],branch['lam']
        saved=torch.load(states/('%s_%g.pt'%(mode,lam)),map_location='cpu',weights_only=True)
        assert paired.state_hash(saved['model'])==branch['final_model_sha256']
        assert branch['initial_model_sha256']==record['reference_model_sha256']
        assert branch['initial_optimizer_sha256']==record['reference_optimizer_sha256']
        model.load_state_dict(saved['model']);model.eval()
        if mode=='protected':assert paired.state_hash(saved['protected'])==branch['protected_head_sha256']
        original_hash=paired.state_hash(model.state_dict())
        features={part:paired.embed(model,data[part][0]) for part in parts}
        donor=dict(mode=mode,lam=lam,replayed=True,original_model_sha256=original_hash,fitted_readers={})
        for part in parts:
            h=features[part];labels=data[part][1]
            native=bc.current_logits(model,h)
            fixed=h@A.T+B;reference=bases[part]@A.T+B
            if saved['protected'] is None:
                legacy=fixed
            else:
                legacy=F.linear(h.float(),saved['protected']['weight'],saved['protected']['bias']).double()
            for name,logits in [('own',native),('fixed_weights',fixed),('frozen',legacy),('reference',reference)]:
                actual=paired.metrics(logits,labels)
                for key,value in actual.items():
                    difference=abs(value-branch[part][name][key]);largest=max(largest,difference)
                    assert difference<1e-7,(mode,part,name,key,difference)
            assert int((legacy.argmax(1)!=reference.argmax(1)).sum())==branch[part]['prediction_changes']
            assert abs(float((legacy-reference).abs().max())-branch[part]['max_absolute_healthy_logit_error'])<1e-6
            mse=float(paired.auxiliary_loss(model,h.float(),data[part][2],kind))
            assert abs(mse-branch[part]['auxiliary_mse'])<1e-7
            if part=='test':
                bound=certificate(legacy.argmax(1).numpy(),reference.argmax(1).numpy(),original_monitor,.05/40)
                for key,value in bound.items():assert abs(value-branch[part]['consumer_audit'][key])<1e-12
                if mode=='protected':donor['online_head_fresh_audit']=score(legacy,reference,labels,new_monitor,.05/140)
        training_features=paired.embed(model,x)
        heads=fit_legacy_heads(training_features,targets)
        for method,head in heads.items():
            item={key:value for key,value in head.items() if key not in ['weight','bias']}
            for part in parts:
                logits=F.linear(features[part],head['weight'],head['bias'])
                if part=='test':item[part]=score(logits,bases[part]@A.T+B,data[part][1],new_monitor,.05/140)
                else:item[part]=dict(metrics=paired.metrics(logits,data[part][1]),
                    disagreement=float((logits.argmax(1)!=(bases[part]@A.T+B).argmax(1)).double().mean()))
            item['head_sha256']=paired.state_hash(head)
            torch.save(head,fitted_dir/(mode+'_'+method+'.pt'))
            donor['fitted_readers'][method]=item
        assert paired.state_hash(model.state_dict())==original_hash
        result['branches'].append(donor)
    result.update(complete=True,maximum_original_metric_error=largest,original_consumers_replayed=3)
    paired.save_json(args.out,result)


@torch.no_grad()
def benchmark(args):
    paths=sorted(args.records_dir.glob('ph*_conf_*.json'))
    assert len(paths)==20
    result=dict(complete=False,device='cpu',cuda_visibility='',audit_plan_sha256=args.plan_sha256,
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        benchmark='current output versus both outputs from exactly one encoder call',branches=[])
    for path in paths:
        record,model,data,kind,states,A,B=load_record(path)
        row=next(row for row in record['branches'] if row['mode']=='protected')
        state=torch.load(states/('protected_%g.pt'%row['lam']),map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);model.eval()
        p=state['protected'];calls=[]
        def current(sample):return bc.current_logits(model,model(sample))
        def both(sample):
            features=model(sample)
            return bc.current_logits(model,features),F.linear(features.float(),p['weight'],p['bias'])
        def legacy(sample):return F.linear(model(sample).float(),p['weight'],p['bias'])
        hook=model.register_forward_hook(lambda *_:calls.append(1))
        both(data['validation'][0][:1]);hook.remove()
        assert len(calls)==1
        timings={}
        for batch in [1,1024]:
            sample=data['validation'][0][:batch]
            operations=dict(current=lambda:current(sample),both=lambda:both(sample),legacy=lambda:legacy(sample))
            for op in operations.values():
                for _ in range(5):op()
            times={key:[] for key in operations}
            for repeat in range(51):
                for key in (list(operations) if repeat%2 else list(reversed(operations))):
                    start=time.perf_counter();operations[key]();times[key].append(time.perf_counter()-start)
            medians={key:float(np.median(values)) for key,values in times.items()}
            timings[str(batch)]=dict(medians_seconds=medians,samples_seconds=times,
                both_vs_current_ratio=medians['both']/medians['current'])
        result['branches'].append(dict(seed=record['seed'],domain=record['domain'],timings=timings,
            encoder_calls_for_both=1,protected_head_parameters=sum(t.numel() for t in p.values())))
    result.update(complete=True,torch_threads=torch.get_num_threads())
    paired.save_json(args.out,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['replay','benchmark'],required=True)
    parser.add_argument('--record',type=Path)
    parser.add_argument('--records-dir',type=Path)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    (replay if args.mode=='replay' else benchmark)(args)
