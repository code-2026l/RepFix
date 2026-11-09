"""Native serving of fixed stronger controls; isolated after all training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired

@torch.no_grad()
def main(a):
    out=dict(complete=False,device='cpu',cuda_visibility='',threads=4,primary_runtime_dtype='float32',legacy_runtime_dtype='float32',
        benchmark_driver_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),branches=[])
    for path in sorted(a.records.glob('h???.json')):
        r=json.loads(path.read_text());model,data,_,_=bc.load(SimpleNamespace(domain='har',seed=r['seed'],phase='confirmation',limit_fit=0,limit_eval=0))
        folder=path.parent/(path.stem+'_states');original=a.original_records/('c%03d.json'%(r['seed']-1300))
        orig=json.loads(original.read_text());old=original.parent/(original.stem+'_states')
        for mode in ['detach_ols','joint_ols','feature_map','feature_map_ols']:
            b=next(b for b in r['branches'] if b['mode']==mode)
            state=torch.load(folder/'feature_map.pt' if b['donor']=='feature_map' else old/next(x['state_file'] for x in orig['branches'] if x['mode']==b['donor']),map_location='cpu',weights_only=True)
            model.load_state_dict(state['model']);model.eval()
            reader=torch.load(folder/(mode+'.pt'),map_location='cpu',weights_only=True)['reader'] if mode.endswith('_ols') else state['reader']
            def current(sample):return model.primary(model(sample))
            def both(sample):
                h=model(sample);return model.primary(h),F.linear(h,reader['weight'],reader['bias'])
            calls=[];hook=model.register_forward_hook(lambda *_:calls.append(1));both(data['validation'][0][:1]);hook.remove();assert len(calls)==1
            timing={}
            for size in [1,1024]:
                sample=data['validation'][0][:size];ops={'current':current,'both':both}
                for op in ops.values():
                    for _ in range(5):op(sample)
                durations={k:[] for k in ops}
                for repeat in range(51):
                    for key in (list(ops) if repeat%2 else list(reversed(ops))):
                        start=time.perf_counter();ops[key](sample);durations[key].append(time.perf_counter()-start)
                medians={k:float(np.median(v)) for k,v in durations.items()}
                timing[str(size)]=dict(medians_seconds=medians,samples_seconds=durations,ratio=medians['both']/medians['current'])
            out['branches'].append(dict(seed=r['seed'],mode=mode,encoder_calls=1,reader_sha256=paired.state_hash(reader),
                reader_parameters=sum(t.numel() for t in reader.values()),timings=timing))
    out['complete']=True;paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--original-records',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);main(a)
