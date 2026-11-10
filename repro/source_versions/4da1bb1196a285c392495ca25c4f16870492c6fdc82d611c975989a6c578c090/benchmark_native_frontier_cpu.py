"""Matched native/torchscript cost and actual served-consumer audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time,warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
import dense_replay as dense
import shared_reference_replay as sparse
from consumer_contracts import evaluate_contracts
from native_replay_cpu import compile_kernel,pack,NativeReplay

@torch.no_grad()
def main(a):
    start=time.time();selected=None if a.choice is None else json.loads(a.choice.read_text())
    out=dict(complete=False,device='cpu',phase=a.phase,threads=4,plan_sha256=a.plan_sha256,
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['benchmark_native_frontier_cpu.py','native_replay_cpu.py','native_replay.cpp','dense_replay.py','shared_reference_replay.py','consumer_contracts.py']},
        choice_sha256=None if a.choice is None else hashlib.sha256(a.choice.read_bytes()).hexdigest(),compiler={},branches=[])
    kernels={}
    for folder,domain in [(a.sparse_records,'har'),(a.dense_records,'radioml')]:
        records=[]
        for path in sorted(folder.glob('*.json')):
            if path.name.endswith('.partial.json') or path.name in ['queue.json','global_guard.json']:continue
            record=json.loads(path.read_text())
            if record.get('domain')==domain and record.get('phase')==a.phase and record.get('complete'):records.append((path,record))
        assert len(records)==(3 if a.phase=='development' else 10),(domain,len(records))
        supported=domain=='har';s,r=(4,4) if supported else (16,12)
        d,c=(561,6) if supported else (256,11);binary,info=compile_kernel(a.out.parent/'kernels',d,c,s,r,supported)
        out['compiler'][domain]=info
        module=sparse if supported else dense
        for path,record in records:
            seed=record['seed'];torch.manual_seed(seed)
            healthy,data,_,splits=bc.load(SimpleNamespace(domain=domain,seed=seed,phase=a.phase,limit_fit=0,limit_eval=0));assert splits==record['splits']
            states=path.parent/(path.stem+'_states');reference=torch.load(states/'reference.pt',map_location='cpu',weights_only=True)
            healthy.load_state_dict(reference['reference']['model']);part=data['test' if a.phase=='confirmation' else 'validation'];x,y,_=part
            h2=torch.cat([F.relu(healthy.encoder[2](F.relu(healthy.encoder[0](x[j:j+1024])))) for j in range(0,len(x),1024)])
            old=F.linear(healthy.encoder[4](h2).double(),reference['A'],reference['B'])
            for mode in ['detach','joint']:
                branch=next(b for b in record['branches'] if b['mode']==mode and ((b.get('k')==s and b.get('rank')==r) if supported else (b['rank1']==s and b['rank2']==r)))
                raw=torch.load(states/branch['state_file'],map_location='cpu',weights_only=True);p=raw['serving']
                assert paired.state_hash(p)==branch['serving_sha256'] and paired.state_hash(raw['model'])==branch['final_state_sha256']
                native=NativeReplay(binary,pack(p,supported),c)
                merged=module.merged_serving(p);packed=module.packed_serving(p)
                def current(t):return module.serve_merged_mlp(t,merged)
                def replay(t):return module.serve_packed_mlp(t,packed)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',torch.jit.TracerWarning)
                    ts_current=torch.jit.trace(current,x[:1],check_trace=False);ts_replay=torch.jit.trace(replay,x[:1],check_trace=False)
                row=dict(domain=domain,seed=seed,mode=mode,source_record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),serving_sha256=paired.state_hash(p),timings={})
                for batch,repeats in [(1,101),(1024,31)]:
                    sample=x[:batch];array=np.ascontiguousarray(sample.numpy());one=np.empty((batch,c),np.float32);both=np.empty((batch,2*c),np.float32)
                    ops={'native_current':native.timed_call(array,'current',one),'native_replay':native.timed_call(array,'replay',both),
                        'native_independent':native.timed_call(array,'independent',both),'native_empty':native.timed_call(array,'empty',one),
                        'torchscript_current':lambda:ts_current(sample),'torchscript_replay':lambda:ts_replay(sample)}
                    for op in ops.values():
                        for _ in range(12):op()
                    values={name:[] for name in ops}
                    for rep in range(repeats):
                        for name in (list(ops) if rep%2 else list(reversed(ops))):
                            begin=time.perf_counter();ops[name]();values[name].append(time.perf_counter()-begin)
                    internal={}
                    if batch==1:
                        for name in ['current','replay','independent','empty']:
                            op=native.repeat_call(array,name,both,1000);duration=[]
                            for _ in range(21):
                                begin=time.perf_counter();op();duration.append((time.perf_counter()-begin)/1000)
                            internal[name]=duration
                    row['timings'][str(batch)]=dict(samples_seconds=values,medians_seconds={name:float(np.median(v)) for name,v in values.items()},kernel_samples_seconds=internal)
                # Audit the selected post-development serving path on the entire corpus.
                actual=[];comparison=[]
                backend='native_replay' if selected is None else selected['domains'][domain]['batches']['1']['protected']
                for j in range(0,len(x),1024):
                    block=x[j:j+1024]
                    if backend.startswith('native'):
                        z=torch.from_numpy(native.call(block.numpy(),'independent' if backend=='native_independent' else 'replay').copy());actual.append((z[:,:c],z[:,c:]))
                    else:actual.append(ts_replay(block))
                    comparison.append(ts_current(block))
                own,legacy=[torch.cat([v[i] for v in actual]) for i in [0,1]];target=torch.cat(comparison)
                monitor=None if a.phase=='development' else np.random.default_rng(12700001+seed+(100000 if supported else 0)).choice(len(x),1024,replace=False)
                row['audit']=dict(backend=backend,own=paired.metrics(own,y),contracts=evaluate_contracts(legacy.double(),old,y,reference['thresholds'],monitor,.05/2000),
                    maximum_legacy_logit_error=float((legacy.double()-old).abs().max()),maximum_current_merge_logit_error=float((own-target).abs().max()),
                    current_merge_prediction_changes=int((own.argmax(1)!=target.argmax(1)).sum()))
                out['branches'].append(row);paired.save_json(a.out.with_suffix('.partial.json'),out)
                print(json.dumps(dict(domain=domain,seed=seed,mode=mode,seconds=time.time()-start)),flush=True)
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sparse-records',type=Path,required=True);p.add_argument('--dense-records',type=Path,required=True)
    p.add_argument('--phase',choices=['development','confirmation'],required=True);p.add_argument('--choice',type=Path);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);main(a)
