"""Label-free development screen of protected-only BF16 matrix products."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
import torch
import nonlinear_suffix_cpu as study
import late_vision_cpu as late
import paired_readout_continuation as paired
from paired_suffix import pack_suffix
from consumer_contracts import evaluate_contracts

@torch.no_grad()
def main(a):
    start=time.time();ref=torch.load(a.reference,map_location='cpu',weights_only=True)
    part=torch.load(a.reference.with_name('cache.pt'),map_location='cpu',weights_only=True)['validation']
    _,healthy=study.build_for(ref);healthy.load_state_dict(ref['healthy']);healthy.requires_grad_(False)
    old=late.linear(part['h'],ref['heads']['coarse']);results={}
    for low in [False,True]:
        model=pack_suffix(healthy,healthy,ref['heads']['coarse'],ref['heads']['coarse'],False,low)
        z=torch.cat([model(part['x'][j:j+128]) for j in range(0,len(old),128)])
        contracts=evaluate_contracts(z.double(),old.double(),part['coarse'],ref['thresholds'])
        results['bf16_products' if low else 'fp32_products']=dict(contracts=contracts,maximum_logit_error=float((z-old).abs().max()),
            passes=all(v['disagreement']<=.01 for v in contracts.values()))
    paired.save_json(a.out,dict(complete=True,device='cpu',phase='reference_precision_development',arch=ref['arch'],plan_sha256=a.plan_sha256,
        reference_state_sha256=paired.state_hash(ref),source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['reference_precision_cpu.py','paired_suffix.py','mixed_reference.py',*study.NAMES]},
        new_labels_used_for_selection=False,results=results,branches=[],elapsed_s=time.time()-start))
    print(json.dumps(dict(arch=ref['arch'],results=results,seconds=time.time()-start)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
