"""Independent saved-state consumer replay; numerical CPU execution only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import benefit_compatibility_cpu as experiment
import paired_readout_continuation as paired
from consumer_risk import certificate


@torch.no_grad()
def run(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    r=json.loads(args.record.read_text())
    assert r['complete'] and r['phase']=='confirmation'
    torch.manual_seed(r['seed']);np.random.seed(r['seed'])
    config=SimpleNamespace(domain=r['domain'],seed=r['seed'],phase='confirmation',limit_fit=0,limit_eval=0)
    model,data,kind,splits=experiment.load(config)
    assert splits==r['splits']
    states=args.record.parent/(args.record.stem+'_states')
    ref=torch.load(states/'reference.pt',map_location='cpu',weights_only=True)
    assert paired.state_hash(ref['model'])==r['reference_model_sha256']
    model.load_state_dict(ref['model']);model.eval()
    x,y,a=data['test'];h0=paired.embed(model,x)
    w,b,mu,sd=ref['head']['weight'],ref['head']['bias'],ref['mean'],ref['sd']
    z0=F.linear((h0-mu)/sd,w,b)
    n=len(y);monitor=np.random.default_rng(1850001+r['seed']).choice(n,1024,replace=False)
    assert hashlib.sha256(monitor.astype(np.int64).tobytes()).hexdigest()==r['consumer_bank']['monitor_membership_sha256']
    out=dict(complete=False,device='cpu',cuda_visibility='',torch_threads=1,seed=r['seed'],
        record_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),branches=[])
    for row in r['branches']:
        state=torch.load(states/('%s_%g.pt'%(row['mode'],row['alpha'])),map_location='cpu',weights_only=True)
        assert paired.state_hash(state['model'])==row['final_model_sha256']
        model.load_state_dict(state['model']);model.eval()
        h=paired.embed(model,x)
        if row['mode'].startswith('guarded'):
            # Reconstruct directly, independently of the experimental wrapper.
            h=h0+(h-h0)@ref['projector']
        z=F.linear((h-mu)/sd,w,b)
        own=F.linear(h,model.primary.weight.double(),model.primary.bias.double())
        actual=dict(frozen=paired.metrics(z,y),own=paired.metrics(own,y),
            auxiliary_mse=float(paired.auxiliary_loss(model,h.float(),a,kind)),
            max_absolute_healthy_logit_error=float((z-z0).abs().max()),
            prediction_changes=int((z.argmax(1)!=z0.argmax(1)).sum()))
        maximum=0.
        for head in ['frozen','own']:
            for metric in ['acc','nll','ece']:
                error=abs(actual[head][metric]-row['test'][head][metric]);maximum=max(maximum,error)
                assert error<1e-7,(row['mode'],head,metric,error)
        assert abs(actual['auxiliary_mse']-row['test']['auxiliary_mse'])<1e-7
        assert actual['prediction_changes']==row['test']['prediction_changes']
        bound=certificate(z.argmax(1).numpy(),z0.argmax(1).numpy(),monitor,.05/40)
        for key,value in bound.items():
            assert abs(value-row['test']['consumer_audit'][key])<1e-12
        out['branches'].append(dict(mode=row['mode'],alpha=row['alpha'],all_scores_replayed=True,
            maximum_metric_error=maximum,prediction_changes=actual['prediction_changes'],
            max_absolute_healthy_logit_error=actual['max_absolute_healthy_logit_error'],bounds_replayed=True))
    out['complete']=True
    paired.save_json(args.out,out)
    print(json.dumps(dict(seed=r['seed'],all_four_saved_consumers_replayed=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);run(parser.parse_args())
