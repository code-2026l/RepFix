"""Independent endpoint/contract replay for the fixed retrospective HAR audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from consumer_risk import certificate

@torch.no_grad()
def main(a):
    row=json.loads(a.record.read_text());original=json.loads(a.original_record.read_text())
    assert row['source_record_sha256']==hashlib.sha256(a.original_record.read_bytes()).hexdigest()
    assert row['complete'] and row['new_bank']==1000
    for n,sha in row['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==sha
    folder=a.record.parent/(a.record.stem+'_states');old=a.original_record.parent/(a.original_record.stem+'_states')
    ref=torch.load(old/'reference.pt',map_location='cpu',weights_only=True)
    model,data,kind,_=bc.load(SimpleNamespace(domain='har',seed=row['seed'],phase='confirmation',limit_fit=0,limit_eval=0))
    model.load_state_dict(ref['model']);model.eval();A,B=ref['A'],ref['B']
    healthy={p:paired.embed(model,data[p][0])@A.T+B for p in ['validation','test']}
    teach=(paired.embed(model,data['x'])@A.T+B).float().double()
    monitor=np.random.default_rng(8490001+row['seed']).choice(len(data['test'][1]),1024,replace=False)
    assert hashlib.sha256(monitor.astype(np.int64).tobytes()).hexdigest()==row['monitor_sha256']
    states={r['mode']:torch.load(old/r['state_file'],map_location='cpu',weights_only=True) for r in original['branches']}
    states['feature_map']=torch.load(folder/'feature_map.pt',map_location='cpu',weights_only=True)
    out=dict(complete=False,device='cpu',cuda_visibility='',seed=row['seed'],branches=[],
        completion_record_sha256=hashlib.sha256(a.record.read_bytes()).hexdigest(),
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    maximum=stationarity=0.
    for b in row['branches']:
        saved=states[b['donor']];model.load_state_dict(saved['model']);model.eval()
        assert b['model_sha256']==paired.state_hash(saved['model'])
        if b['mode'].endswith('_ols'):
            fitted=saved if b['mode']==b['donor'] else torch.load(folder/(b['mode']+'.pt'),map_location='cpu',weights_only=True)
            reader=fitted['reader']
            assert fitted.get('model_sha256',paired.state_hash(saved['model']))==b['model_sha256']
            assert all(torch.equal(reader[k],fitted['ols64'][k].float()) for k in ['weight','bias'])
            h=paired.embed(model,data['x']);residual=F.linear(h,fitted['ols64']['weight'],fitted['ols64']['bias'])-teach
            x=(h-h.mean(0))/h.std(0).clamp_min(1e-6)
            error=max(float((x.T@residual/len(x)).abs().max()),float(residual.mean(0).abs().max()))
            assert error<1e-6;stationarity=max(stationarity,error)
        else:reader=saved['reader']
        assert b['reader_sha256']==paired.state_hash(reader)
        for part in ['validation','test']:
            h=paired.embed(model,data[part][0]);y=data[part][1]
            z=F.linear(h.to(reader['weight'].dtype),reader['weight'],reader['bias']).double()
            for key,value in paired.metrics(bc.current_logits(model,h),y).items():
                error=abs(value-b[part]['primary'][key]);maximum=max(maximum,error);assert error<1e-9
            c,p=z.softmax(1).max(1);c0,p0=healthy[part].softmax(1).max(1)
            for contract,threshold in [('classification',None),*row['thresholds'].items()]:
                actual=p.clone();reference=p0.clone()
                if threshold is not None:actual[c<threshold]=-1;reference[c0<threshold]=-1
                expected=b[part]['contracts'][contract]
                assert float((actual!=reference).double().mean())==expected['disagreement']
                assert float((actual>=0).double().mean())==expected['coverage']
                if contract=='classification':
                    flows=expected['flows'];right=reference==y;now=actual==y
                    assert int((right&~now).sum())==flows['correct_to_wrong']
                    assert int((~right&now).sum())==flows['wrong_to_correct']
                    assert int(((actual!=reference)&~right&~now).sum())==flows['wrong_to_wrong']
                    for key,value in paired.metrics(z,y).items():
                        error=abs(value-expected['metrics'][key]);maximum=max(maximum,error);assert error<1e-9
                if part=='test':
                    bound=certificate(actual.numpy(),reference.numpy(),monitor,.05/1000)
                    assert all(value==expected['certificate'][key] for key,value in bound.items())
                    assert expected['certificate']['certified']==(bound['upper_remaining_disagreement']<=.05)
        out['branches'].append(dict(mode=b['mode'],metrics_and_contracts_replayed=True))
    out.update(complete=True,maximum_metric_error=maximum,maximum_ols_stationarity_residual=stationarity)
    paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--original-record',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);main(a)
