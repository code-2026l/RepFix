"""Independent saved-reader metrics/bounds replay, without fitting or training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from consumer_risk import certificate


@torch.no_grad()
def run(args):
    record=json.loads(args.record.read_text());assert record['complete'] and record['phase']=='confirmation'
    for name,sha in record['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==sha
    assert hashlib.sha256(Path(__file__).with_name('confirm_joint_alignment_cpu.py').read_bytes()).hexdigest()==record['confirmation_driver_source_sha256']
    cfg=SimpleNamespace(domain=record['domain'],seed=record['seed'],phase='confirmation',limit_fit=0,limit_eval=0)
    model,data,kind,splits=bc.load(cfg);assert splits==record['splits']
    directory=args.record.parent/(args.record.stem+'_states')
    ref=torch.load(directory/'reference.pt',map_location='cpu',weights_only=True)
    assert paired.state_hash(ref['model'])==record['reference_model_sha256']
    assert paired.state_hash(ref['optimizer'])==record['reference_optimizer_sha256']
    model.load_state_dict(ref['model']);model.eval();A,B=ref['A'],ref['B']
    teacher={part:paired.embed(model,data[part][0])@A.T+B for part in ['validation','test']}
    train_teacher=(paired.embed(model,data['x'])@A.T+B).float().double()
    maximum=stationarity=0.;out=dict(complete=False,device='cpu',cuda_visibility='',
        confirmation_record_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed=record['seed'],domain=record['domain'],branches=[],refitting_performed=False)
    for row in record['branches']:
        saved=torch.load(directory/row['state_file'],map_location='cpu',weights_only=True)
        assert paired.state_hash(saved['model'])==row['final_model_sha256']
        assert paired.state_hash(saved['reader'])==row['reader_sha256']
        model.load_state_dict(saved['model']);model.eval();reader=saved['reader']
        if saved.get('mapper') is not None:
            assert torch.equal(reader['weight'],A.float()@saved['mapper']['weight'])
            assert torch.equal(reader['bias'],B.float())
        for part in ['validation','test']:
            x,y,a=data[part];features=paired.embed(model,x)
            logits=F.linear(features.to(reader['weight'].dtype),reader['weight'],reader['bias']).double()
            current=F.linear(features,model.primary.weight.double(),model.primary.bias.double())
            fixed=F.linear(features,A,B)
            for name,z in [('own',current),('legacy',logits),('reference',teacher[part]),('fixed',fixed)]:
                for key,value in paired.metrics(z,y).items():
                    diff=abs(value-row[part][name][key]);maximum=max(maximum,diff);assert diff<1e-9
            changed=logits.argmax(1)!=teacher[part].argmax(1)
            assert int(changed.sum())==row[part]['prediction_changes']
            assert abs(float(paired.auxiliary_loss(model,features.float(),a,kind))-row[part]['auxiliary_mse'])<1e-8
            if part=='test' and 'certificates' in row[part]:
                indices=np.random.default_rng(7390001+record['seed']+(100000 if record['domain']=='har' else 0)).permutation(len(y))
                for count in [1024,2048]:
                    monitor=indices[:min(count,len(y)-1)]
                    actual=certificate(logits.argmax(1).numpy(),teacher[part].argmax(1).numpy(),monitor,.05/160)
                    expected=row[part]['certificates'][str(count)]
                    for key,value in actual.items():assert value==expected[key]
                    assert expected['certified']==(actual['upper_remaining_disagreement']<=.05)
                    assert expected['monitor_sha256']==hashlib.sha256(monitor.astype(np.int64).tobytes()).hexdigest()
        if row['mode']=='direct_ols':
            fitted=saved['ols64']
            assert torch.equal(reader['weight'],fitted['weight'].float()) and torch.equal(reader['bias'],fitted['bias'].float())
            h=paired.embed(model,data['x']);mu=h.mean(0,keepdim=True);sd=h.std(0,keepdim=True).clamp_min(1e-6)
            residual=F.linear(h,fitted['weight'],fitted['bias'])-train_teacher
            error=max(float((((h-mu)/sd).T@residual/len(h)).abs().max()),float(residual.mean(0).abs().max()))
            assert error<1e-6;stationarity=max(stationarity,error)
        out['branches'].append(dict(mode=row['mode'],metrics_and_bounds_replayed=True))
    out.update(complete=True,maximum_metric_error=maximum,maximum_ols_stationarity_residual=stationarity)
    paired.save_json(args.out,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);run(a)
