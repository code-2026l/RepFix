"""Isolated native-float32 serving of two parameter versions; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time,warnings
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from shared_reference_replay import serve_mlp,merged_serving,serve_merged_mlp,packed_serving,serve_packed_mlp,serve_prefix_mlp
from paired_readout_continuation import save_json,state_hash


@torch.no_grad()
def main(a):
    choice=json.loads(a.choice.read_text());out=dict(complete=False,device='cpu',threads=4,dtype='float32',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),choice_sha256=hashlib.sha256(a.choice.read_bytes()).hexdigest(),branches=[])
    for path in sorted(a.records.glob('d???.json')):
        record=json.loads(path.read_text());sel=choice['domains'][record['domain']]['selected'];k,r=sel['k'],sel['rank']
        folder=path.parent/(path.stem+'_states')
        for mode in ['detach','joint']:
            p=torch.load(folder/('%s_k%d_r%d.pt'%(mode,k,r)),map_location='cpu',weights_only=True)['serving']
            merged=merged_serving(p);packed=packed_serving(p)
            def ordinary(x):return serve_merged_mlp(x,merged)
            def both(x):return serve_mlp(x,p)
            def optimized(x):return serve_packed_mlp(x,packed)
            def prefix(x):return serve_prefix_mlp(x,p,merged)
            def independent(x):
                old=F.relu(F.linear(x,p['first_weight'],p['first_bias']));old=F.relu(F.linear(old,p['second_weight'],p['second_bias']))
                return ordinary(x),F.linear(old,p['old_weight'],p['old_bias'])
            operations=dict(current=ordinary,replay=both,packed_replay=optimized,prefix_shared=prefix,two_independent=independent)
            sample=torch.randn(1024,p['first_weight'].shape[1],generator=torch.Generator().manual_seed(record['seed']+90000))
            errors={name:float((op(sample)[0]-both(sample)[0]).abs().max()) for name,op in operations.items() if name!='current'}
            errors['current']=float((ordinary(sample)-both(sample)[0]).abs().max())
            row=dict(domain=record['domain'],seed=record['seed'],mode=mode,k=k,rank=r,serving_sha256=state_hash(p),
                packed_parameters=sum(v.numel() for v in packed.values() if isinstance(v,torch.Tensor)),
                current_parameters=sum(v.numel() for v in merged.values()),random_input_current_maximum_errors=errors,backends={})
            for backend in ['eager','torchscript']:
                start=time.perf_counter()
                if backend=='eager':ops=operations
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',torch.jit.TracerWarning)
                        ops={name:torch.jit.trace(op,sample[:1],check_trace=True) for name,op in operations.items()}
                timing={};setup=time.perf_counter()-start
                for batch in [1,1024]:
                    x=sample[:batch]
                    for op in ops.values():
                        for _ in range(12):op(x)
                    durations={name:[] for name in ops}
                    for rep in range(71):
                        for name in (list(ops) if rep%2 else list(reversed(ops))):
                            begin=time.perf_counter();ops[name](x);durations[name].append(time.perf_counter()-begin)
                    medians={name:float(np.median(v)) for name,v in durations.items()}
                    timing[str(batch)]=dict(medians_seconds=medians,samples_seconds=durations,
                        ratios_to_current={name:value/medians['current'] for name,value in medians.items()})
                row['backends'][backend]=dict(setup_seconds=setup,timings=timing)
            out['branches'].append(row);save_json(a.out.with_suffix('.partial.json'),out)
            print(json.dumps(dict(domain=record['domain'],seed=record['seed'],mode=mode,
                ratios={be:{b:v['ratios_to_current']['packed_replay'] for b,v in x['timings'].items()} for be,x in row['backends'].items()})),flush=True)
    out['complete']=True;save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--choice',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);main(a)
