"""Independent coarse-consumer metrics, checkpoint and fitted-reader replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models import resnet18
from torch import nn
import paired_readout_continuation as paired
from vision_chart_cpu import preprocess,bn_state

@torch.no_grad()
def main(a):
    r=json.loads(a.record.read_text());assert r['complete'] and not r['smoke']
    ref=torch.load(a.cache,map_location='cpu',weights_only=True)
    saved=torch.load(a.record.parent/(a.record.stem+'_states')/'endpoint.pt',map_location='cpu',weights_only=True)
    assert paired.state_hash(saved['encoder'])==r['final_encoder_sha256']
    for name,sha in r['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==sha
    train=Path(os.environ['REPFIX_DATA_X'])/'v037/cifar-100-python/train'
    assert hashlib.sha256(train.read_bytes()).hexdigest()==ref['data_sha256']
    with train.open('rb') as stream:raw=pickle.load(stream,encoding='bytes')
    images=torch.from_numpy(raw[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1).copy())
    fine=torch.tensor(raw[b'fine_labels'])[ref['validation']];coarse=torch.tensor(raw[b'coarse_labels'])[ref['validation']]
    model=resnet18(weights=None);model.fc=nn.Identity();model.to(memory_format=torch.channels_last)
    model.load_state_dict(saved['encoder']);model.eval()
    maximum_feature_error=float((model(preprocess(images[ref['validation'][:128]]))-saved['validation_features'][:128]).abs().max())
    assert maximum_feature_error<1e-4
    def linear(h,reader):return F.linear(h.float(),reader['weight'].float(),reader['bias'].float())
    h=saved['validation_features'];old=linear(ref['healthy_validation'],ref['heads']['coarse']).double()
    current=linear(h,saved['primary']);maximum=0.;stationarity=0.
    for key,value in paired.metrics(current,fine).items():
        error=abs(value-r['fine'][key]);assert error<1e-9;maximum=max(maximum,error)
    if 'ols64' in saved:
        fit=saved['fitting_features'].double();target=linear(ref['healthy_fit'],ref['heads']['coarse']).double()
        coef=saved['ols64'];residual=F.linear(fit,coef['weight'],coef['bias'])-target
        x=(fit-fit.mean(0))/fit.std(0).clamp_min(1e-6)
        stationarity=max(float((x.T@residual/len(x)).abs().max()),float(residual.mean(0).abs().max()));assert stationarity<1e-6
    out=dict(complete=False,device='cpu',cuda_visibility='',seed=r['seed'],mode=r['mode'],lam=r['lam'],branches=[],
        source_record_sha256=hashlib.sha256(a.record.read_bytes()).hexdigest(),
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for b in r['branches']:
        reader=saved['readers'][b['mode']];assert paired.state_hash(reader)==b['reader_sha256']
        z=linear(h,reader).double();confidence,pred=z.softmax(1).max(1);confidence0,pred0=old.softmax(1).max(1)
        for name,threshold in [('classification',None),*ref['thresholds'].items()]:
            p,q=pred.clone(),pred0.clone()
            if threshold is not None:p[confidence<threshold]=-1;q[confidence0<threshold]=-1
            expected=b['contracts'][name];accepted=p>=0;accepted0=q>=0
            assert float((p!=q).double().mean())==expected['disagreement']
            assert float(accepted.double().mean())==expected['coverage']
            assert float(accepted0.double().mean())==expected['reference_coverage']
            accuracy=float((p[accepted]==coarse[accepted]).double().mean()) if accepted.any() else None
            assert accuracy==expected['accepted_accuracy']
            if name=='classification':
                for key,value in paired.metrics(z,coarse).items():
                    error=abs(value-expected['metrics'][key]);assert error<1e-9;maximum=max(maximum,error)
                assert int(((q==coarse)&(p!=coarse)).sum())==expected['flows']['correct_to_wrong']
        out['branches'].append(dict(mode=b['mode'],metrics_and_selective_decisions_replayed=True))
    out.update(complete=True,maximum_metric_error=maximum,maximum_feature_error=maximum_feature_error,maximum_ols_stationarity_residual=stationarity)
    paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1);main(a)
