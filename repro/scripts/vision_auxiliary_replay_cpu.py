"""Independently execute saved suffixes and the retained healthy consumer; CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import nonlinear_suffix_cpu as factory
from consumer_contracts import evaluate_contracts


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def main(a):
    start=time.time();record=json.loads(a.record.read_text())
    assert record['complete'] and not record['splits']['official_test_loaded']
    cache=torch.load(a.population,map_location='cpu',weights_only=True,mmap=True)
    assert sha(a.population)==record['population_sha256']
    state_folder=a.record.parent/(a.record.stem+'_states')
    reference=torch.load(state_folder/'reference.pt',map_location='cpu',weights_only=True)
    val=reference['validation'];fit=reference['fitting'];labels=cache['y'][val]
    x=cache['x'][0,val];target=cache['targets'][0,val]
    scale=cache['targets'][:,fit].flatten(0,1).var(0,unbiased=False).mean().clamp_min(1e-6)
    _,healthy=factory.build_for(cache);healthy.load_state_dict(cache['healthy']);healthy.eval().requires_grad_(False)
    # Preparation used batches of 128. This is a new execution from weights,
    # rather than the training driver's cached reference-versus-itself check.
    old_h=torch.cat([healthy(part) for part in x.split(128)])
    old=F.linear(old_h,reference['heads']['coarse']['weight'],reference['heads']['coarse']['bias'])
    cached_old=F.linear(cache['h'][0,val],reference['heads']['coarse']['weight'],reference['heads']['coarse']['bias'])
    contracts=evaluate_contracts(old,cached_old,cache['coarse'][val],reference['thresholds'])
    rows=[];replay=dict(labels=labels,coarse=cache['coarse'][val],old_logits=old,
        cached_old_logits=cached_old,thresholds=reference['thresholds'],branches={})
    for branch in record['branches']:
        mode=branch['mode'];path=state_folder/(mode+'.pt');state=torch.load(path,map_location='cpu',weights_only=True)
        def inspect(value):
            if isinstance(value,torch.Tensor):
                assert not value.is_floating_point() or torch.isfinite(value).all()
                return 1
            if isinstance(value,dict):return sum(inspect(v) for v in value.values())
            if isinstance(value,(list,tuple)):return sum(inspect(v) for v in value)
            return 0
        checked_tensors=inspect(state)
        if mode in ['F','H']:features=cache['h'][0,val]
        elif mode=='N':
            h=state['head'];features=cache['h'][0,val]
            hidden=F.gelu(F.linear((features-h['base.mean'])/h['base.sd'],h['hidden.weight'],h['hidden.bias']))
            features=torch.cat([features,hidden],1)
        else:
            _,model=factory.build_for(cache);model.load_state_dict(state['model']);model.eval().requires_grad_(False)
            features=torch.cat([model(part) for part in x.split(256)])
        if mode=='H':features=torch.cat([features,cache['hidden'][0,val]],1)
        z=F.linear(features,state['reader']['weight'],state['reader']['bias'])
        acc=float((z.argmax(1)==labels).double().mean())
        assert abs(acc-branch['validation']['own']['acc'])<1e-12
        entry=dict(mode=mode,accuracy=acc,correct=int((z.argmax(1)==labels).sum()),
            state_sha256=sha(path),finite_saved_tensors=checked_tensors,own_logit_max_error=None,mse_by_dose={})
        saved=dict(own_logits=z)
        if mode not in ['F','N','H']:
            assert entry['state_sha256']==branch['state_sha256']
            entry['own_logit_max_error']=float((z-state['own_logits']).abs().max())
            assert torch.equal(z,state['own_logits'])
            assert torch.equal(cached_old,state['old_logits'])
            saved['squared_errors']={}
            doses=list(branch['detached_mse_by_dose']) if mode=='D' else [str(branch['alpha'])]
            for dose,weights in zip(doses,state['decoders']):
                decoder=nn.Sequential(nn.Linear(features.shape[1],256),nn.ReLU(),nn.Linear(256,192))
                decoder.load_state_dict(weights);decoder.eval()
                pred=torch.cat([decoder(part) for part in features.split(256)])
                errors=(pred-target).square().mean(1)/scale
                mse=float((pred-target).square().mean()/scale)
                expected=branch['detached_mse_by_dose'][dose] if mode=='D' else branch['validation']['reconstruction_mse']
                assert abs(mse-expected)<1e-7
                entry['mse_by_dose'][dose]=mse;saved['squared_errors'][dose]=errors
        rows.append(entry);replay['branches'][mode]=saved
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    torch.save(replay,folder/'replay.pt')
    result=dict(complete=True,device='cpu',phase='independent_saved_state_replay',
        arch=record['arch'],reference_seed=record['reference_seed'],seed=record['seed'],
        plan_sha256=record['plan_sha256'],record_sha256=sha(a.record),population_sha256=record['population_sha256'],
        source_sha256={p.name:sha(p) for p in [Path(__file__),*[Path(__file__).with_name(n) for n in factory.NAMES]]},
        old_logit_max_error=float((old-cached_old).abs().max()),contracts=contracts,
        all_old_decisions_unchanged=all(v['disagreement']==0 for v in contracts.values()),branches=rows,
        state_file='replay.pt',state_sha256=sha(folder/'replay.pt'),elapsed_s=time.time()-start)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(arch=record['arch'],branches=len(rows),elapsed_s=time.time()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--population',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
