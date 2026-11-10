"""Read-only verification of the protected-head teacher and initial function."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
import paired_readout_continuation as paired
import late_vision_cpu as late
import nonlinear_suffix_cpu as study
from consumer_contracts import evaluate_contracts

@torch.no_grad()
def main(a):
    start=time.time();ref=torch.load(a.reference,map_location='cpu',weights_only=True)
    cache=torch.load(a.reference.with_name('cache.pt'),map_location='cpu',weights_only=True,mmap=True)
    _,healthy=study.build_for(ref);healthy.load_state_dict(ref['healthy']);healthy.requires_grad_(False)
    old=ref['heads']['coarse'];out=dict(complete=False,device='cpu',arch=ref['arch'],plan_sha256=a.plan_sha256,
        reference_state_sha256=paired.state_hash(ref),source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in [Path(__file__).name,*study.NAMES]},branches=[])
    for name in ['train','validation']:
        part=cache[name];x=part['x'].flatten(0,1) if name=='train' else part['x'];h=part['h'].flatten(0,1) if name=='train' else part['h']
        actual=study.embed(healthy,x);z=late.linear(actual,old);target=late.linear(h,old)
        error=float((z-target).abs().max());kl=float(F.kl_div(z.log_softmax(1),target.softmax(1),reduction='batchmean'))
        row=dict(split=name,examples=len(x),maximum_feature_error=float((actual-h).abs().max()),maximum_teacher_logit_error=error,initial_kl=kl,
            class_disagreement=float((z.argmax(1)!=target.argmax(1)).float().mean()))
        if name=='validation':row['contracts']=evaluate_contracts(z.double(),target.double(),part['coarse'],ref['thresholds'])
        assert error<1e-3 and abs(kl)<1e-5 and row['class_disagreement']<=.001,row
        out['branches'].append(row)
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out);print(json.dumps(out['branches']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
