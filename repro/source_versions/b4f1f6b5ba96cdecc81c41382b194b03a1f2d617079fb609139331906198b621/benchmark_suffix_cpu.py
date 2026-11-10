"""Isolated matched TorchScript serving and full-prefix endpoint replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,time,warnings
from pathlib import Path
import numpy as np
import torch
import paired_readout_continuation as paired
import late_vision_cpu as late
import nonlinear_suffix_cpu as study
from consumer_contracts import evaluate_contracts
from suffix_serving import SuffixReadout,WholeNetwork
from paired_suffix import pack_suffix

def trace(module,sample):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',torch.jit.TracerWarning)
        return torch.jit.freeze(torch.jit.trace(module.eval(),sample,check_trace=False))

def load_suffix(ref,healthy,row,folder):
    state=torch.load(folder/row['state_file'],map_location='cpu',weights_only=True)
    assert paired.state_hash(state)==row['state_sha256']
    mode=row['mode'];current=None;hidden=mean=sd=None
    if mode not in ['F','N','H']:
        current,_,_=study.make_model(ref,healthy,mode,3800)
        current.load_state_dict(state['update'],strict=mode!='L' and mode!='R64');current=current.dense().requires_grad_(False)
    if mode=='N':
        head=study.init_head(ref);head=study.ResidualHead(head);head.load_state_dict(state['head'])
        hidden=head.hidden;mean=head.base.mean;sd=head.base.sd
    def make(both):
        return SuffixReadout(healthy,current,state['reader'],ref['heads']['coarse'],mode,both,hidden,mean,sd).eval().requires_grad_(False)
    return make

@torch.no_grad()
def main(a):
    start=time.time();record=json.loads(a.record.read_text());phase=record['phase'];seed=record['seed'];arch=record['arch']
    ref=torch.load(a.reference,map_location='cpu',weights_only=True);assert paired.state_hash(ref)==record['reference_state_sha256']
    prefix,healthy=study.build_for(ref);healthy.load_state_dict(ref['healthy']);healthy.requires_grad_(False)
    choice=None if a.choice is None else json.loads(a.choice.read_text())
    if phase=='confirmation':
        assert choice is not None
        cache=torch.load(a.test_cache,map_location='cpu',weights_only=True);part=cache['test'];assert cache['reference_state_sha256']==paired.state_hash(ref)
        images,y,coarse,data_sha=late.images(False);assert data_sha==cache['data_sha256']
    else:
        part=torch.load(a.reference.with_name('cache.pt'),map_location='cpu',weights_only=True)['validation']
        images,y,coarse,data_sha=late.images(True);indices=ref['validation_indices'];images=images[indices];y=y[indices];coarse=coarse[indices]
    assert torch.equal(y,part['y']) and torch.equal(coarse,part['coarse'])
    sample=late.preprocess(images[:128],ref['metadata']['image_size']);ts_prefix=trace(prefix,sample[:1])
    physical=[]
    for j in range(0,len(images),128):physical.append(ts_prefix(late.preprocess(images[j:j+128],ref['metadata']['image_size'])))
    physical=torch.cat(physical);old=late.linear(part['h'],ref['heads']['coarse'])
    monitor=None if phase=='development' else np.random.default_rng(13700001+seed).choice(len(images),1024,replace=False)
    names=['benchmark_suffix_cpu.py','suffix_serving.py','paired_suffix.py',*study.NAMES]
    out=dict(complete=False,device='cpu',phase=phase,arch=arch,seed=seed,threads=12,plan_sha256=a.plan_sha256,
        training_record_sha256=hashlib.sha256(a.record.read_bytes()).hexdigest(),reference_state_sha256=paired.state_hash(ref),
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names},
        choice_sha256=None if a.choice is None else hashlib.sha256(a.choice.read_bytes()).hexdigest(),
        prefix_sha256=paired.state_hash(prefix.state_dict()),healthy=record['healthy'],splits=record['splits'],
        forward_latency_excludes=['decoding','preprocessing','compilation','warmup'],audits={},serving={},branches=[])
    folder=a.record.parent/(a.record.stem+'_states')
    for row in record['branches']:
        mode=row['mode'];make=load_suffix(ref,healthy,row,folder);s0=make(False);s1=make(True)
        ts0=trace(s0,physical[:1]);ts1=trace(s1,physical[:1])
        actual=[ts1(physical[j:j+128]) for j in range(0,len(images),128)]
        z,legacy=[torch.cat([v[i] for v in actual]) for i in [0,1]]
        original=row['test' if phase=='confirmation' else 'validation']['own']
        audit=dict(own=paired.metrics(z,y),contracts=evaluate_contracts(legacy.double(),old.double(),coarse,ref['thresholds'],monitor,.05/2000),
            state_sha256=row['state_sha256'],maximum_legacy_logit_error=float((legacy-old).abs().max()),
            cached_prefix_accuracy_difference=paired.metrics(z,y)['acc']-original['acc'])
        out['audits'][mode]=audit
        if mode not in ['U','H']:continue
        mono0=trace(WholeNetwork(prefix,s0),sample[:1]);mono1=trace(WholeNetwork(prefix,s1),sample[:1])
        packed0=packed1=None
        if mode=='U':
            reader=dict(weight=s0.weight,bias=s0.bias)
            packed0=trace(WholeNetwork(prefix,pack_suffix(healthy,s0.current,reader,ref['heads']['coarse'],False)),sample[:1])
            packed1=trace(WholeNetwork(prefix,pack_suffix(healthy,s0.current,reader,ref['heads']['coarse'],True)),sample[:1])
        old_only=trace(WholeNetwork(prefix,SuffixReadout(healthy,None,ref['heads']['coarse'],ref['heads']['coarse'],'F',False)),sample[:1])
        timing={}
        for batch in [1,128]:
            x=sample[:batch]
            operations=dict(mono_current=lambda:mono0(x),mono_protected=lambda:mono1(x),
                modular_current=lambda:ts0(ts_prefix(x)),modular_protected=lambda:ts1(ts_prefix(x)),
                independent=lambda:(mono0(x),old_only(x)))
            operations['modular_independent']=lambda:(ts0(ts_prefix(x)),old_only(x))
            if packed0 is not None:operations.update(packed_current=lambda:packed0(x),packed_protected=lambda:packed1(x),packed_independent=lambda:(packed0(x),old_only(x)))
            for op in operations.values():
                for _ in range(12):op()
            times={n:[] for n in operations}
            for rep in range(31):
                for name in (list(operations) if rep%2 else list(reversed(operations))):
                    begin=time.perf_counter();operations[name]();times[name].append(time.perf_counter()-begin)
            timing[str(batch)]=dict(samples_seconds=times,medians_seconds={n:float(np.median(v)) for n,v in times.items()})
        chosen={}
        if choice is not None:
            # Batch membership can change CPU arithmetic. Audit each chosen ABI
            # at its registered chunk size, rather than only its graph name.
            for batch in [1,128]:
                backend=choice['backbones'][arch]['modes'][mode]['batches'][str(batch)]['protected']
                result=[]
                for j in range(0,len(images),batch):
                    inputs=late.preprocess(images[j:j+batch],ref['metadata']['image_size'])
                    if backend=='mono_protected':result.append(mono1(inputs))
                    elif backend=='packed_protected':result.append(packed1(inputs))
                    else:result.append(ts1(ts_prefix(inputs)))
                own,legacy=[torch.cat([v[i] for v in result]) for i in [0,1]]
                chosen[str(batch)]=dict(backend=backend,own=paired.metrics(own,y),contracts=evaluate_contracts(legacy.double(),old.double(),coarse,ref['thresholds'],monitor,.05/2000),
                    maximum_legacy_logit_error=float((legacy-old).abs().max()),
                    cached_prefix_accuracy_difference=paired.metrics(own,y)['acc']-original['acc'])
        out['serving'][mode]=dict(timings=timing,chosen=chosen)
        paired.save_json(a.out.with_suffix('.partial.json'),out)
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)
    print(json.dumps(dict(arch=arch,seed=seed,elapsed_s=time.time()-start)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--test-cache',type=Path);p.add_argument('--choice',type=Path);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
