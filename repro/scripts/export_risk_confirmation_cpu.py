"""Reconstruct fixed saved models and export unlabeled risk-study logits; CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
import nonlinear_suffix_cpu as factory
from conditional_reference_cpu import ridge


@torch.no_grad()
def main(a):
    start=time.time();row=json.loads(a.record.read_text());assert row['complete'] and row['phase']=='confirmation'
    folder=a.record.parent/(a.record.stem+'_states');reference=torch.load(folder/'reference.pt',map_location='cpu',weights_only=True)
    cache=torch.load(a.population,map_location='cpu',weights_only=True,mmap=True);state=torch.load(folder/'D.pt',map_location='cpu',weights_only=True)
    _,model=factory.build_for(cache);model.load_state_dict(state['model']);model.eval().requires_grad_(False)
    embed=lambda x:torch.cat([model(b) for b in x.split(256)])
    anchors=reference['anchors'];val=reference['validation'];x=embed(cache['x'][0,anchors]);old=reference['heads']['coarse']
    target=F.linear(cache['h'][0,anchors],old['weight'],old['bias']);target-=target.mean(1,keepdim=True)
    order=torch.randperm(2048,generator=torch.Generator().manual_seed(6500));reader,_,_=ridge(x[order[:1024]],target[order[:1024]],.001)
    anchor_cheap=F.linear(x,reader['weight'],reader['bias']);views={}
    for view,name in [(0,'clean'),(1,'augmented')]:
        h=embed(cache['x'][view,val]);cheap=F.linear(h,reader['weight'],reader['bias']);healthy=F.linear(cache['h'][view,val],old['weight'],old['bias'])
        views[name]=dict(cheap=cheap,reference=healthy)
    payload=dict(reference_seed=row['reference_seed'],seed=row['seed'],anchor_cheap=anchor_cheap,anchor_target=target,views=views,thresholds=reference['thresholds'])
    outfolder=a.out.parent/(a.out.stem+'_states');outfolder.mkdir(exist_ok=True,parents=True);path=outfolder/'risk.pt';torch.save(payload,path)
    result=dict(complete=True,device='cpu',phase='risk_confirmation_export',reference_seed=row['reference_seed'],seed=row['seed'],
        plan_sha256=a.plan_sha256,source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        training_record_sha256=hashlib.sha256(a.record.read_bytes()).hexdigest(),state_file='risk.pt',state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        official_test_loaded=False,labels_loaded_for_fitting=False,branches=[],elapsed_s=time.time()-start)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--population',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
