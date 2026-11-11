"""Isolated matched TorchScript serving and full-prefix endpoint replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,io,json,time,warnings
from pathlib import Path
import numpy as np
import torch
import paired_readout_continuation as paired
import late_vision_cpu as late
import nonlinear_suffix_cpu as study
from consumer_contracts import evaluate_contracts
from suffix_serving import SuffixReadout,WholeNetwork
from paired_suffix import pack_suffix
from mixed_reference import MixedReference,AutocastReference
from whole_amp import WholeAMPReadout,WholeAMPTwin

def trace(module,sample):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',torch.jit.TracerWarning)
        return torch.jit.freeze(torch.jit.trace(module.eval(),sample,check_trace=False))

def export_bytes(module):
    stream=io.BytesIO();torch.jit.save(module,stream);return stream.tell()

def cache_save(path,value):
    path.parent.mkdir(exist_ok=True,parents=True);temporary=path.with_suffix('.%d.tmp'%os.getpid())
    torch.save(value,temporary);temporary.replace(path)

def trace_amp(module,sample):
    fast=torch.backends.mha.get_fastpath_enabled();prior=torch._C._jit_set_autocast_mode(False);torch.backends.mha.set_fastpath_enabled(False)
    try:return trace(module,sample)
    finally:torch.backends.mha.set_fastpath_enabled(fast);torch._C._jit_set_autocast_mode(prior)

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
    names=['benchmark_suffix_cpu.py','suffix_serving.py','paired_suffix.py','mixed_reference.py','whole_amp.py',*study.NAMES]
    out=dict(complete=False,device='cpu',phase=phase,arch=arch,seed=seed,threads=12,plan_sha256=a.plan_sha256,
        training_record_sha256=hashlib.sha256(a.record.read_bytes()).hexdigest(),reference_state_sha256=paired.state_hash(ref),
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names},
        choice_sha256=None if a.choice is None else hashlib.sha256(a.choice.read_bytes()).hexdigest(),
        prefix_sha256=paired.state_hash(prefix.state_dict()),healthy=record['healthy'],splits=record['splits'],
        forward_latency_excludes=['decoding','preprocessing','compilation','warmup'],audits={},serving={},branches=[],
        timing_only=a.timing_only,audit_only=a.audit_only,reference_precision=a.reference_precision,only_u=a.only_u)
    header=dict(reference=out['reference_state_sha256'],source=out['source_sha256'],data=data_sha,phase=phase,prefix=out['prefix_sha256'],choice=out['choice_sha256'])
    prefix_path=None if a.shared_cache is None else a.shared_cache/(arch+'_'+phase+'_prefix.pt')
    if a.timing_only:physical=ts_prefix(sample[:1])
    elif prefix_path is not None and prefix_path.exists():
        saved=torch.load(prefix_path,map_location='cpu',weights_only=True,mmap=True);assert saved['header']==header;physical=saved['physical']
    else:
        physical=torch.cat([ts_prefix(late.preprocess(images[j:j+128],ref['metadata']['image_size'])) for j in range(0,len(images),128)])
        if prefix_path is not None:cache_save(prefix_path,dict(header=header,physical=physical))
    old=late.linear(part['h'],ref['heads']['coarse'])
    monitor=None if phase=='development' else np.random.default_rng((14700001 if a.reference_precision else 13700001)+seed).choice(len(images),1024,replace=False)
    folder=a.record.parent/(a.record.stem+'_states');raw_controls={}
    for row in record['branches']:
        mode=row['mode']
        if a.timing_only and (mode not in ['U','H'] or (a.only_u and mode!='U')):continue
        make=load_suffix(ref,healthy,row,folder)
        cache_path=None if mode!='H' or a.shared_cache is None else a.shared_cache/(arch+'_'+phase+('_H_timing.pt' if a.timing_only else '_H_audit.pt'))
        h_header=dict(header,state=row['state_sha256']);saved=None
        if cache_path is not None and cache_path.exists():
            saved=torch.load(cache_path,map_location='cpu',weights_only=True);assert saved['header']==h_header
            if a.timing_only:
                out['serving'][mode]=dict(timings=saved['timings'],chosen={},same_function_timing_reused=True,unique_state_sha256=row['state_sha256']);continue
        s0=make(False);s1=make(True)
        ts0=trace(s0,physical[:1]);ts1=trace(s1,physical[:1])
        original=row['test' if phase=='confirmation' else 'validation']['own']
        canonical=None
        if not a.timing_only:
            if saved is None:
                actual=[ts1(physical[j:j+128]) for j in range(0,len(images),128)]
                z,legacy=[torch.cat([v[i] for v in actual]) for i in [0,1]]
            else:z,legacy=saved['canonical']
            canonical=(z,legacy)
            raw_controls[mode]=canonical
            out['audits'][mode]=dict(own=paired.metrics(z,y),contracts=evaluate_contracts(legacy.double(),old.double(),coarse,ref['thresholds'],monitor,.05/2000),
                state_sha256=row['state_sha256'],maximum_legacy_logit_error=float((legacy-old).abs().max()),
                cached_prefix_accuracy_difference=paired.metrics(z,y)['acc']-original['acc'])
        if mode not in ['U','H']:continue
        if a.only_u and mode!='U':
            if cache_path is not None and saved is None:cache_save(cache_path,dict(header=h_header,timings={},canonical=canonical,chosen={}))
            continue
        mono0=trace(WholeNetwork(prefix,s0),sample[:1]);mono1=trace(WholeNetwork(prefix,s1),sample[:1])
        packed0=packed1=None
        if mode=='U':
            reader=dict(weight=s0.weight,bias=s0.bias)
            packed0=trace(WholeNetwork(prefix,pack_suffix(healthy,s0.current,reader,ref['heads']['coarse'],False)),sample[:1])
            packed1=trace(WholeNetwork(prefix,pack_suffix(healthy,s0.current,reader,ref['heads']['coarse'],True)),sample[:1])
        precision32=precision16=precision16mono=amp0=amp1=reference_amp=reference_amp_mono=None
        if a.reference_precision and mode=='U':
            reader=dict(weight=s0.weight,bias=s0.bias)
            precision32=trace(MixedReference(healthy,s0.current,reader,ref['heads']['coarse'],False),physical[:1])
            precision16=trace(MixedReference(healthy,s0.current,reader,ref['heads']['coarse'],True),physical[:1])
            precision16mono=trace(WholeNetwork(prefix,MixedReference(healthy,s0.current,reader,ref['heads']['coarse'],True)),sample[:1])
            if a.reference_autocast:
                reference_amp=trace_amp(AutocastReference(healthy,s0.current,reader,ref['heads']['coarse']),physical[:1])
                reference_amp_mono=trace_amp(WholeNetwork(prefix,AutocastReference(healthy,s0.current,reader,ref['heads']['coarse'])),sample[:1])
            if a.whole_amp_control:
                amp0=trace_amp(WholeAMPReadout(prefix,s0.current,reader),sample[:1]);amp1=trace_amp(WholeAMPTwin(prefix,healthy,s0.current,reader,ref['heads']['coarse']),sample[:1])
        old_only=trace(WholeNetwork(prefix,SuffixReadout(healthy,None,ref['heads']['coarse'],ref['heads']['coarse'],'F',False)),sample[:1])
        timing={}
        for batch in ([] if a.audit_only else [1,128]):
            x=sample[:batch]
            operations=dict(mono_current=lambda:mono0(x),mono_protected=lambda:mono1(x),
                modular_current=lambda:ts0(ts_prefix(x)),modular_protected=lambda:ts1(ts_prefix(x)),
                independent=lambda:(mono0(x),old_only(x)))
            operations['modular_independent']=lambda:(ts0(ts_prefix(x)),old_only(x))
            if packed0 is not None:operations.update(packed_current=lambda:packed0(x),packed_protected=lambda:packed1(x),packed_independent=lambda:(packed0(x),old_only(x)))
            if a.reference_precision:
                fixed=choice['backbones'][arch]['modes'][mode]['batches'][str(batch)];current_name=fixed['current'];current_op=operations[current_name]
                extra=dict(bf16_reference_mono=lambda:precision16mono(x),bf16_reference_modular=lambda:precision16(ts_prefix(x)),fp32_reference_modular=lambda:precision32(ts_prefix(x)))
                if reference_amp is not None:extra.update(autocast_reference_mono=lambda:reference_amp_mono(x),autocast_reference_modular=lambda:reference_amp(ts_prefix(x)))
                if phase=='development':
                    operations={current_name:current_op,**extra,'independent_fp32':lambda:(current_op(),old_only(x))}
                    if a.reference_autocast:operations.update(mono_protected=lambda:mono1(x),modular_protected=lambda:ts1(ts_prefix(x)))
                    if amp0 is not None:operations.update(whole_amp_current=lambda:amp0(x),whole_amp_protected=lambda:amp1(x))
                else:
                    protected_ops=dict(extra,mono_protected=lambda:mono1(x),modular_protected=lambda:ts1(ts_prefix(x)))
                    full=fixed.get('full_precision_protected','fp32_reference_modular')
                    operations={current_name:current_op,fixed['protected']:protected_ops[fixed['protected']],full:protected_ops[full]}
                    if amp0 is not None:operations.update(whole_amp_current=lambda:amp0(x),whole_amp_protected=lambda:amp1(x))
            elif choice is not None:
                fixed=choice['backbones'][arch]['modes'][mode]['batches'][str(batch)]
                operations={n:operations[n] for n in dict.fromkeys([fixed['current'],fixed['protected'],fixed['independent']])}
            for op in operations.values():
                for _ in range(12):op()
            times={n:[] for n in operations}
            for rep in range(31):
                for name in (list(operations) if rep%2 else list(reversed(operations))):
                    begin=time.perf_counter();operations[name]();times[name].append(time.perf_counter()-begin)
            timing[str(batch)]=dict(samples_seconds=times,medians_seconds={n:float(np.median(v)) for n,v in times.items()})
        chosen={};raw_chosen={};full_precision={};raw_full={}
        if choice is not None and not a.timing_only:
            # Batch membership can change CPU arithmetic. Audit each chosen ABI
            # at its registered chunk size, rather than only its graph name.
            def serve(backend,batch):
                result=[]
                for j in range(0,len(images),batch):
                    inputs=late.preprocess(images[j:j+batch],ref['metadata']['image_size'])
                    if backend=='mono_protected':result.append(mono1(inputs))
                    elif backend=='packed_protected':result.append(packed1(inputs))
                    elif backend=='bf16_reference_mono':result.append(precision16mono(inputs))
                    elif backend=='bf16_reference_modular':result.append(precision16(ts_prefix(inputs)))
                    elif backend=='fp32_reference_modular':result.append(precision32(ts_prefix(inputs)))
                    elif backend=='autocast_reference_mono':result.append(reference_amp_mono(inputs))
                    elif backend=='autocast_reference_modular':result.append(reference_amp(ts_prefix(inputs)))
                    else:
                        assert backend=='modular_protected',backend
                        result.append(ts1(ts_prefix(inputs)))
                return tuple(torch.cat([v[i] for v in result]) for i in [0,1])
            def report(backend,own,legacy):
                return dict(backend=backend,own=paired.metrics(own,y),contracts=evaluate_contracts(legacy.double(),old.double(),coarse,ref['thresholds'],monitor,.05/2000),
                    maximum_legacy_logit_error=float((legacy-old).abs().max()),cached_prefix_accuracy_difference=paired.metrics(own,y)['acc']-original['acc'])
            for batch in [1,128]:
                fixed=choice['backbones'][arch]['modes'][mode]['batches'][str(batch)];backend=fixed['protected']
                if saved is not None:own,legacy=saved['chosen'][str(batch)]
                else:own,legacy=serve(backend,batch)
                raw_chosen[str(batch)]=(own,legacy)
                chosen[str(batch)]=report(backend,own,legacy)
                if a.reference_precision and mode=='U':
                    full=fixed['full_precision_protected']
                    full_own,full_legacy=(own,legacy) if full==backend else serve(full,batch)
                    raw_full[str(batch)]=(full_own,full_legacy)
                    full_precision[str(batch)]=report(full,full_own,full_legacy)
        out['serving'][mode]=dict(timings=timing,chosen=chosen,full_precision=full_precision)
        if a.timing_only and mode=='U' and a.reference_precision:
            # Storage is a descriptive cost, collected after latency sampling.
            prefix_bytes=export_bytes(ts_prefix)
            sizes=dict(mono_current=export_bytes(mono0),modular_current=prefix_bytes+export_bytes(ts0),
                mono_protected=export_bytes(mono1),modular_protected=prefix_bytes+export_bytes(ts1),
                fp32_reference_modular=prefix_bytes+export_bytes(precision32),bf16_reference_mono=export_bytes(precision16mono),
                bf16_reference_modular=prefix_bytes+export_bytes(precision16))
            if reference_amp is not None:sizes.update(autocast_reference_mono=export_bytes(reference_amp_mono),autocast_reference_modular=prefix_bytes+export_bytes(reference_amp))
            out['serving'][mode]['serialized_program_bytes']=sizes
        raw_amp={}
        if a.whole_amp_control and not a.timing_only:
            amp_results={}
            for batch in [1,128]:
                actual=[amp1(late.preprocess(images[j:j+batch],ref['metadata']['image_size'])) for j in range(0,len(images),batch)]
                own,legacy=[torch.cat([v[i] for v in actual]) for i in [0,1]];raw_amp[str(batch)]=(own,legacy)
                amp_results[str(batch)]=dict(own=paired.metrics(own,y),contracts=evaluate_contracts(legacy.double(),old.double(),coarse,ref['thresholds'],monitor,.05/2000),maximum_legacy_logit_error=float((legacy-old).abs().max()))
            out['serving'][mode]['whole_amp_chosen']=amp_results
        if a.audit_only and mode=='U':
            tensor_path=a.out.parent/(a.out.stem+'_states')/'served_logits.pt'
            cache_save(tensor_path,dict(header=header,source_state_sha256=row['state_sha256'],canonical=canonical,chosen=raw_chosen,
                whole_amp=raw_amp,full_precision=raw_full,controls=raw_controls,reference=old,primary_labels=y,coarse_labels=coarse,monitor=None if monitor is None else torch.from_numpy(monitor)))
            out['serving'][mode]['served_logits_sha256']=hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        if cache_path is not None and saved is None:cache_save(cache_path,dict(header=h_header,timings=timing,canonical=canonical,chosen=raw_chosen))
        if saved is not None:out['serving'][mode]['same_function_output_reused']=True
        paired.save_json(a.out.with_suffix('.partial.json'),out)
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)
    print(json.dumps(dict(arch=arch,seed=seed,elapsed_s=time.time()-start)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--test-cache',type=Path);p.add_argument('--choice',type=Path);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--timing-only',action='store_true');p.add_argument('--audit-only',action='store_true');p.add_argument('--shared-cache',type=Path)
    p.add_argument('--reference-precision',action='store_true');p.add_argument('--only-u',action='store_true');p.add_argument('--whole-amp-control',action='store_true')
    p.add_argument('--reference-autocast',action='store_true')
    a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
