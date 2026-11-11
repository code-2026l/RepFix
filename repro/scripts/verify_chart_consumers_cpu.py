"""Independent tensor-based replay of native and emitted chart consumers."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import benefit_compatibility_cpu as loader
import paired_readout_continuation as paired
from compiled_readout import TrainableInterface
from consumer_risk import certificate


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    record=json.loads(args.record.read_text())
    assert record['complete'] and record['phase']=='confirmation'
    cfg=SimpleNamespace(domain=record['domain'],seed=record['seed'],phase='confirmation',limit_fit=0,limit_eval=0)
    torch.manual_seed(record['seed']);np.random.seed(record['seed'])
    base,data,kind,splits=loader.load(cfg)
    assert splits==record['splits']
    reference=TrainableInterface(base)
    states=args.record.parent/(args.record.stem+'_states')
    healthy=torch.load(states/'reference.pt',map_location='cpu',weights_only=True)
    assert paired.state_hash(healthy['model'])==record['reference_model_sha256']
    assert paired.state_hash(healthy['optimizer'])==record['reference_optimizer_sha256']
    assert paired.state_hash(healthy['anchors'])==record['anchor_membership_sha256']
    reference.load_state_dict(healthy['model']);reference.eval()
    x,y,aux=data['test']
    h0=paired.embed(reference,x)
    weight,bias=healthy['head']['weight'],healthy['head']['bias']
    mean,sd=healthy['mean'],healthy['sd']
    z0=F.linear((h0-mean)/sd,weight,bias)
    monitor=np.random.default_rng(3870001+record['seed']).choice(len(y),min(1024,len(y)-1),replace=False)
    result=dict(complete=False,device='cpu',cuda_visibility='',seed=record['seed'],domain=record['domain'],
        record_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),branches=[])
    for branch in record['branches']:
        mode=branch['mode']
        state=torch.load(states/(mode+'.pt'),map_location='cpu',weights_only=True)
        assert paired.state_hash(state['model'])==branch['final_model_sha256']
        assert paired.state_hash(state['optimizer'])==branch['final_optimizer_sha256']
        model=copy.deepcopy(reference)
        model.load_state_dict(state['model']);model.eval()
        native=paired.embed(model,x)
        native_current=F.linear(native,state['model']['primary.weight'].double(),state['model']['primary.bias'].double())
        for metric,value in paired.metrics(native_current,y).items():
            assert abs(value-branch['test']['own'][metric])<1e-10
        for candidate,item in branch['repairs'].items():
            saved=torch.load(states/(mode+'_'+candidate+'.pt'),map_location='cpu',weights_only=True)
            chart,deployed=saved['chart'],saved['deployed']
            assert paired.state_hash(chart)==item['chart_sha256']
            # Direct affine evaluation deliberately bypasses fold_chart() and
            # the train-time effective-affine implementation.
            h=torch.cat([F.linear(model.trunk(x[j:j+2048]).double(),deployed['final.weight'],
                                  deployed['final.bias']) for j in range(0,len(x),2048)])
            feature_error=float((h-(native@chart['matrix']+chart['offset'])).abs().max())
            assert feature_error<1e-9
            current=F.linear(h,deployed['primary.weight'],deployed['primary.bias'])
            old=F.linear((h-mean)/sd,weight,bias)
            maximum=0.
            for name,logits in [('own',current),('frozen',old),('reference',z0)]:
                for metric,value in paired.metrics(logits,y).items():
                    difference=abs(value-item['test'][name][metric])
                    assert difference<1e-10,(mode,candidate,name,metric,difference)
                    maximum=max(maximum,difference)
            changes=int((current.argmax(1)!=native_current.argmax(1)).sum())
            logit_error=float((current-native_current).abs().max())
            assert changes==item['test']['current_prediction_changes']==0
            assert logit_error<1e-8
            assert int((old.argmax(1)!=z0.argmax(1)).sum())==item['test']['prediction_changes']
            bounds=certificate(old.argmax(1).numpy(),z0.argmax(1).numpy(),monitor,.05/40)
            for key,value in bounds.items():
                assert abs(value-item['test']['consumer_audit'][key])<1e-12
            result['branches'].append(dict(mode=mode,candidate=candidate,all_metrics_replayed=True,
                bounds_replayed=True,maximum_metric_error=maximum,current_prediction_changes=changes,
                current_logit_error=logit_error,fold_feature_error=feature_error))
    result.update(complete=True)
    paired.save_json(args.out,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
