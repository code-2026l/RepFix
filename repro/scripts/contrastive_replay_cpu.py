"""Numerical saved-weight replay and matched CE-only baseline, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,time
from pathlib import Path
import torch
import contrastive_benefit_cpu as study
import vision_auxiliary_cpu as base
import nonlinear_suffix_cpu as factory
import late_vision_cpu as late
from consumer_routing import decisions


@torch.no_grad()
def main(a):
    start=time.time();cache=torch.load(a.population,map_location='cpu',weights_only=True,mmap=True)
    records=[(p,json.loads(p.read_text())) for p in sorted(a.records.glob('p???.json'))]
    assert len(records)==4 and all(r['complete'] for _,r in records)
    baseline=torch.load(a.baseline,map_location='cpu',weights_only=True)
    _,healthy=factory.build_for(cache);healthy.load_state_dict(cache['healthy']);healthy.eval().requires_grad_(False)
    first=torch.load(records[0][0].parent/(records[0][0].stem+'_states')/'endpoint.pt',map_location='cpu',weights_only=True)
    ref=first['reference'];val=ref['validation'];x=cache['x'][0,val]
    old_h=torch.cat([healthy(b) for b in x.split(128)])
    old=late.linear(old_h,ref['heads']['coarse']);cached=late.linear(cache['h'][0,val],ref['heads']['coarse'])
    changed=(decisions(old,list(ref['thresholds'].values()))!=decisions(cached,list(ref['thresholds'].values()))).sum(0)
    assert not changed.any()
    rows=[];arrays=dict(old_logits=old,cached_old_logits=cached,thresholds=ref['thresholds'],branches={})
    for path,record in records:
        weights=path.parent/(path.stem+'_states')/'endpoint.pt';assert base.digest(weights)==record['state_sha256']
        state=torch.load(weights,map_location='cpu',weights_only=True)
        _,model=factory.build_for(cache);model.load_state_dict(state['model']);model.eval().requires_grad_(False)
        h=base.embeds(model,x);z=late.linear(h,state['reader']);assert torch.equal(z,state['own_logits'])
        accuracy=float((z.argmax(1)==cache['y'][val]).double().mean());assert accuracy==record['validation']['acc']
        head=base.initial_head(state['reference']);head.load_state_dict(state['head']);aux=study.evaluate_aux(model,cache['x'][:,val],head)
        assert abs(aux-record['contrastive_loss'])<1e-12
        assert torch.equal(state['old_logits'],cached)
        entry=dict(id=path.stem,alpha=record['alpha'],accuracy=accuracy,contrastive_loss=aux,state_sha256=base.digest(weights),own_logit_max_error=float((z-state['own_logits']).abs().max()))
        if record['alpha']==0:
            same_model=all(torch.equal(value,baseline['model'][key]) for key,value in state['model'].items())
            entry['prior_D_model_exact']=same_model
            entry['prior_D_logit_max_error']=float((z-baseline['own_logits']).abs().max())
            assert same_model and torch.equal(z,baseline['own_logits'])
        arrays['branches'][path.stem]=dict(predictions=z.argmax(1),accuracy=accuracy,contrastive_loss=aux);rows.append(entry)
        print(json.dumps(dict(replayed=path.stem,accuracy=accuracy)),flush=True)
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True);torch.save(arrays,folder/'replay.pt')
    result=dict(complete=True,device='cpu',phase='contrastive_independent_replay',branches=rows,
        source_sha256={p.name:base.digest(p) for p in [Path(__file__),Path(study.__file__)]},
        old_logit_max_error=float((old-cached).abs().max()),old_changed_by_consumer=changed.tolist(),
        validation_images=len(val),official_test_loaded=False,replay_sha256=base.digest(folder/'replay.pt'),elapsed_s=time.time()-start)
    base.save(a.out,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--population',type=Path,required=True);p.add_argument('--records',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
