"""Frozen-ABI native image replay and isolated forward timing, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,json,time,warnings
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import vision_auxiliary_cpu as base
import nonlinear_suffix_cpu as factory
import late_vision_cpu as late
import paired_readout_continuation as paired
from suffix_serving import SuffixReadout,WholeNetwork
from mixed_reference import MixedReference,AutocastReference
from consumer_contracts import evaluate_contracts


def trace(module,x,amp=False):
    fast=torch.backends.mha.get_fastpath_enabled();prior=torch._C._jit_set_autocast_mode(False)
    if amp:torch.backends.mha.set_fastpath_enabled(False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',torch.jit.TracerWarning)
            return torch.jit.freeze(torch.jit.trace(module.eval(),x,check_trace=False))
    finally:
        torch.backends.mha.set_fastpath_enabled(fast);torch._C._jit_set_autocast_mode(prior)


@torch.no_grad()
def main(a):
    started=time.time();record=json.loads(a.record.read_text());assert record['complete']
    cache=torch.load(a.population,map_location='cpu',weights_only=True,mmap=True)
    path=a.record.parent/(a.record.stem+'_states')/'endpoint.pt';assert base.digest(path)==record['state_sha256']
    state=torch.load(path,map_location='cpu',weights_only=True);ref=state['reference'];val=ref['validation']
    images,y,coarse,data_sha=late.images(True);assert data_sha==cache['data_sha256'];images=images[val];y=y[val];coarse=coarse[val]
    size=cache['metadata']['image_size'];sample=late.preprocess(images[:128],size)
    prefix,healthy=factory.build_for(cache);healthy.load_state_dict(cache['healthy']);healthy.eval().requires_grad_(False)
    current=copy.deepcopy(healthy);current.load_state_dict(state['model']);current.requires_grad_(False)
    legacy=ref['heads']['coarse'];reader=state['reader'];choice=json.loads(a.choice.read_text())['backbones']['vit_b32']['modes']['U']['batches']
    setup=time.time();pfx=trace(prefix,sample[:1]);physical=pfx(sample)
    single=SuffixReadout(healthy,current,reader,legacy,'U',False).eval();both=SuffixReadout(healthy,current,reader,legacy,'U',True).eval()
    tail=trace(single,physical[:1]);mono_current=trace(WholeNetwork(prefix,single),sample[:1])
    mono_bf16=trace(WholeNetwork(prefix,MixedReference(healthy,current,reader,legacy,True)),sample[:1])
    fp32_mod=trace(MixedReference(healthy,current,reader,legacy,False),physical[:1])
    amp_mod=trace(AutocastReference(healthy,current,reader,legacy),physical[:1],True)
    mono_full=trace(WholeNetwork(prefix,both),sample[:1])
    programs=dict(mono_current=mono_current,modular_current=lambda x:tail(pfx(x)),
        bf16_reference_mono=mono_bf16,fp32_reference_modular=lambda x:fp32_mod(pfx(x)),
        autocast_reference_modular=lambda x:amp_mod(pfx(x)),mono_protected=mono_full)
    setup_s=time.time()-setup;audits={};timings={};arrays={}
    if a.mode=='audit':
        physical=torch.cat([prefix(late.preprocess(images[j:j+128],size)) for j in range(0,len(images),128)])
        raw_current=late.linear(base.embeds(current,physical),reader)
        old=late.linear(torch.cat([healthy(b) for b in physical.split(128)]),legacy)
        assert record['baseline']
        control=json.loads(a.baseline_record.read_text());info=next(b for b in control['branches'] if b['mode']=='D')
        bp=a.baseline_record.parent/(a.baseline_record.stem+'_states')/'D.pt';assert base.digest(bp)==info['state_sha256']==record['baseline']['state_sha256']
        detached=torch.load(bp,map_location='cpu',weights_only=True);dmodel=copy.deepcopy(healthy);dmodel.load_state_dict(detached['model'])
        dz=late.linear(base.embeds(dmodel,physical),detached['reader'])
        monitor=torch.randperm(len(val),generator=torch.Generator().manual_seed(7300))[:1024].numpy()
        arrays=dict(labels=y,coarse=coarse,healthy_logits=old,detached_logits=dz,current_logits=raw_current,thresholds=ref['thresholds'],served={})
        for batch in ['1','128']:
            audits[batch]={}
            for role,key in [('chosen','protected'),('full_precision','full_precision_protected')]:
                name=choice[batch][key];values=[programs[name](late.preprocess(images[j:j+128],size)) for j in range(0,len(images),128)]
                z,legacy_z=[torch.cat([v[k] for v in values]) for k in [0,1]]
                contracts=evaluate_contracts(legacy_z,old,coarse,ref['thresholds'],monitor,.05/(12*2*2*3))
                audits[batch][role]=dict(backend=name,own=paired.metrics(z,y),contracts=contracts,
                    current_logit_max_error=float((z-raw_current).abs().max()),current_class_changes=int((z.argmax(1)!=raw_current.argmax(1)).sum()),
                    old_logit_max_error=float((legacy_z-old).abs().max()))
                arrays['served'][batch+'_'+role]=dict(current_predictions=z.argmax(1),legacy_logits=legacy_z)
        individual={}
        for role,key in [('chosen','protected'),('full_precision','full_precision_protected')]:
            values=[programs[choice['1'][key]](sample[i:i+1]) for i in range(128)];z,old_z=[torch.cat([v[k] for v in values]) for k in [0,1]]
            individual[role]=dict(examples=128,current_class_changes=int((z.argmax(1)!=raw_current[:128].argmax(1)).sum()),
                current_logit_max_error=float((z-raw_current[:128]).abs().max()),contracts=evaluate_contracts(old_z,old[:128],coarse[:128],ref['thresholds']))
        arrays['monitor']=torch.from_numpy(monitor)
        native=dict(current=paired.metrics(raw_current,y),detached=paired.metrics(dz,y),individual_batch1=individual,
            difference_from_cached_primary=paired.metrics(raw_current,y)['acc']-record['validation']['acc'])
    else:
        native=None
        for batch in ['1','128']:
            xx=sample[:int(batch)];names=[choice[batch]['current'],choice[batch]['protected'],choice[batch]['full_precision_protected']]
            assert len(set(names))==3
            for name in names:
                for _ in range(12):programs[name](xx)
            durations={name:[] for name in names}
            for rep in range(31):
                for name in (names if rep%2==0 else names[::-1]):
                    start=time.perf_counter();programs[name](xx);durations[name].append(time.perf_counter()-start)
            timings[batch]=dict(backends=choice[batch],samples_seconds=durations,medians_seconds={k:float(np.median(v)) for k,v in durations.items()})
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    state_sha=None
    if arrays:torch.save(arrays,folder/'replay.pt');state_sha=base.digest(folder/'replay.pt')
    result=dict(complete=True,device='cpu',phase='contrastive_native_'+a.mode,reference_seed=record['reference_seed'],seed=record['seed'],
        source_sha256={Path(__file__).name:base.digest(Path(__file__))},record_sha256=base.digest(a.record),checkpoint_sha256=record['state_sha256'],
        choice_sha256=base.digest(a.choice),plan_sha256=a.plan_sha256,official_test_loaded=False,threads=12,
        native=native,audits=audits,timings=timings,state_sha256=state_sha,trace_setup_s=setup_s,
        host_load_average=list(os.getloadavg()),elapsed_s=time.time()-started,branches=[])
    base.save(a.out,result);print(json.dumps(dict(complete=True,mode=a.mode,reference=record['reference_seed'],seed=record['seed'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--population',type=Path,required=True)
    p.add_argument('--choice',type=Path,required=True);p.add_argument('--baseline-record',type=Path);p.add_argument('--mode',choices=['audit','timing'],required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
