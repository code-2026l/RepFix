"""Fixed-shape CPU compilation experiment; report setup and serving separately."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',TORCHINDUCTOR_COMPILE_THREADS='4',MAX_JOBS='4',TORCHINDUCTOR_FREEZING='1')
import argparse,json,time,hashlib
from pathlib import Path
import torch
import numpy as np
from shared_reference_replay import serve_mlp,merged_serving,serve_merged_mlp,packed_serving,serve_packed_mlp
from paired_readout_continuation import save_json

@torch.no_grad()
def main(a):
    out=dict(complete=False,device='cpu',threads=4,backend='inductor-max-autotune',branches=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    choice=json.loads(a.choice.read_text())
    for domain,stem in [('har','d000'),('radioml','d003')]:
        c=choice['domains'][domain]['selected'];p=torch.load(a.records/(stem+'_states')/('joint_k%d_r%d.pt'%(c['k'],c['rank'])),map_location='cpu',weights_only=True)['serving']
        m=merged_serving(p);packed=packed_serving(p)
        ops={'current':lambda x:serve_merged_mlp(x,m),'replay':lambda x:serve_mlp(x,p),'packed_replay':lambda x:serve_packed_mlp(x,packed)}
        for batch in [1,1024]:
            x=torch.randn(batch,p['first_weight'].shape[1],generator=torch.Generator().manual_seed(81000+batch));compiled={};setup={};errors={}
            for name,op in ops.items():
                start=time.perf_counter();compiled[name]=torch.compile(op,fullgraph=True,dynamic=False,mode='max-autotune');actual=compiled[name](x)
                setup[name]=time.perf_counter()-start;expected=op(x)
                errors[name]=max(float((u-v).abs().max()) for u,v in zip(actual,expected)) if isinstance(actual,tuple) else float((actual-expected).abs().max())
            samples={name:[] for name in ops}
            for op in compiled.values():
                for _ in range(20):op(x)
            for rep in range(101):
                for name in (list(ops) if rep%2 else list(reversed(ops))):
                    start=time.perf_counter();compiled[name](x);samples[name].append(time.perf_counter()-start)
            medians={name:float(np.median(v)) for name,v in samples.items()}
            row=dict(domain=domain,batch=batch,setup_seconds=setup,errors=errors,samples_seconds=samples,medians_seconds=medians,
                ratios={name:v/medians['current'] for name,v in medians.items()});out['branches'].append(row)
            save_json(a.out.with_suffix('.partial.json'),out);print(json.dumps(dict(domain=domain,batch=batch,ratios=row['ratios'],setup_seconds=setup)),flush=True)
    out['complete']=True;save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--choice',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);main(a)
