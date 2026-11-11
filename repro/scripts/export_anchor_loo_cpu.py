"""Reconstruct only cached fitting-anchor design; no new reference queries."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,time
from pathlib import Path
import torch
import vision_auxiliary_cpu as base
import nonlinear_suffix_cpu as factory
from ridge_loo import fixed_design_loo


@torch.no_grad()
def main(a):
    started=time.time();record=json.loads(a.record.read_text());risk=json.loads(a.risk_record.read_text())
    folder=a.record.parent/(a.record.stem+'_states');ref=torch.load(folder/'reference.pt',map_location='cpu',weights_only=True)
    state=torch.load(folder/'D.pt',map_location='cpu',weights_only=True);cache=torch.load(a.population,map_location='cpu',weights_only=True,mmap=True)
    risk_path=a.risk_record.parent/(a.risk_record.stem+'_states')/'risk.pt';assert base.digest(risk_path)==risk['state_sha256']
    raw=torch.load(risk_path,map_location='cpu',weights_only=True);assert risk['training_record_sha256']==base.digest(a.record)
    _,model=factory.build_for(cache);model.load_state_dict(state['model']);model.eval().requires_grad_(False)
    fit=torch.randperm(2048,generator=torch.Generator().manual_seed(6500))[:1024]
    features=base.embeds(model,cache['x'][0,ref['anchors'][fit]]).double()
    design=(features-features.mean(0))/features.std(0).clamp_min(1e-6)
    target=raw['anchor_target'][fit];fitted,loo,leverage=fixed_design_loo(design,target,1024*.001)
    error=float((fitted-raw['anchor_cheap'][fit]).abs().max());assert error<.0002,error
    outfolder=a.out.parent/(a.out.stem+'_states');outfolder.mkdir(exist_ok=True,parents=True);path=outfolder/'loo.pt'
    torch.save(dict(fitted=fitted,loo=loo,leverage=leverage,target=target,fit_indices=fit),path)
    base.save(a.out,dict(complete=True,device='cpu',phase='cached_anchor_loo_export',reference_seed=raw['reference_seed'],seed=raw['seed'],
        raw_state_sha256=risk['state_sha256'],source_sha256={p.name:base.digest(p) for p in [Path(__file__),Path(__file__).with_name('ridge_loo.py')]},
        plan_sha256=a.plan_sha256,state_sha256=base.digest(path),fit_logit_max_error=error,additional_reference_queries=0,initial_paired_anchors=1024,
        fixed_normalizer=True,penalty=.001*1024,official_test_loaded=False,elapsed_s=time.time()-started,branches=[]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--risk-record',type=Path,required=True)
    p.add_argument('--population',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
