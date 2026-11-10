"""Fixed validation-only whole-AMP utility control on selected saved states."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
import torch
import nonlinear_suffix_cpu as study
import late_vision_cpu as late
import paired_readout_continuation as paired
from benchmark_suffix_cpu import load_suffix
from whole_amp import WholeAMPReadout

@torch.no_grad()
def main(a):
    start=time.time();ref=torch.load(a.reference,map_location='cpu',weights_only=True);prefix,healthy=study.build_for(ref);healthy.load_state_dict(ref['healthy']);healthy.requires_grad_(False)
    images,y,_,_=late.images(True);idx=ref['validation_indices'];images=images[idx];y=y[idx]
    out=dict(complete=False,device='cpu',phase='whole_amp_utility_development',arch=ref['arch'],plan_sha256=a.plan_sha256,
        reference_state_sha256=paired.state_hash(ref),source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['whole_amp_utility_cpu.py','whole_amp.py','benchmark_suffix_cpu.py','mixed_reference.py','paired_suffix.py',*study.NAMES]},branches=[])
    fast=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
    try:
        for path in a.records:
            r=json.loads(path.read_text());assert r['phase']=='development' and not r['splits']['test_metrics_computed'];rows={b['mode']:b for b in r['branches']}
            current=load_suffix(ref,healthy,rows['U'],path.parent/(path.stem+'_states'))(False)
            model=WholeAMPReadout(prefix,current.current,dict(weight=current.weight,bias=current.bias)).eval().requires_grad_(False)
            z=torch.cat([model(late.preprocess(images[j:j+128],ref['metadata']['image_size'])) for j in range(0,len(images),128)])
            metrics=paired.metrics(z,y);acc=metrics['acc'];passes=(acc>=r['healthy']['acc']+.02-1e-12 and acc>=rows['F']['validation']['own']['acc']+.02-1e-12 and all(acc>=rows[m]['validation']['own']['acc']-.005-1e-12 for m in ['N','H']))
            out['branches'].append(dict(seed=r['seed'],own=metrics,passes=passes,source_record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_state_sha256=rows['U']['state_sha256']))
    finally:torch.backends.mha.set_fastpath_enabled(fast)
    out.update(complete=True,all_utility_gates=all(b['passes'] for b in out['branches']),elapsed_s=time.time()-start);paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--records',type=Path,nargs='+',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
