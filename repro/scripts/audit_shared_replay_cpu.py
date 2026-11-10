"""Reload raw endpoints and independently reconstruct both consumer functions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from consumer_contracts import evaluate_contracts
from shared_reference_replay import SharedReferenceMLP,serve_packed_mlp,packed_serving

@torch.no_grad()
def main(a):
    out=dict(complete=False,device='cpu',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),branches=[])
    for path in sorted(a.records.glob('c???.json')):
        row=json.loads(path.read_text());domain,seed=row['domain'],row['seed'];folder=path.parent/(path.stem+'_states')
        ref=torch.load(folder/'reference.pt',map_location='cpu',weights_only=True);old=ref['reference'];cfg=row['configuration']
        model,data,kind,splits=bc.load(SimpleNamespace(domain=domain,seed=seed,phase='confirmation',limit_fit=0,limit_eval=0))
        assert splits==row['splits'];model.load_state_dict(old['model']);assert paired.state_hash(old['model'])==row['reference_model_sha256']
        x,y,aux=data['test'];old2=torch.cat([F.relu(model.encoder[2](F.relu(model.encoder[0](x[j:j+1024])))) for j in range(0,len(x),1024)])
        reference=F.linear(model.encoder[4](old2).double(),ref['A'],ref['B']);monitor=np.random.default_rng(10700001+seed+(100000 if domain=='har' else 0)).choice(len(y),1024,replace=False)
        for branch in row['branches']:
            raw=torch.load(folder/branch['state_file'],map_location='cpu',weights_only=True);assert paired.state_hash(raw['model'])==branch['final_state_sha256']
            updated=SharedReferenceMLP(model,ref['support'][:cfg['k']],ref['right_vectors'][:cfg['rank']],ref['A'],ref['B']);updated.load_state_dict(raw['model'])
            assert paired.state_hash(updated.serving())==branch['serving_sha256']==paired.state_hash(raw['serving'])
            p=packed_serving(raw['serving']);values=[serve_packed_mlp(x[j:j+1024],p) for j in range(0,len(x),1024)]
            current,legacy=[torch.cat([v[k] for v in values]) for k in [0,1]]
            direct=updated.merged();direct_h=paired.embed(direct,x).float()
            # Reconstruct merged matrices without the replay serving helper.
            w1=raw['serving']['first_weight'].clone();b1=raw['serving']['first_bias'].clone();s=ref['support'][:cfg['k']]
            w1[s]+=raw['model']['delta1'];b1[s]+=raw['model']['delta1_bias']
            w2=raw['serving']['second_weight']+raw['model']['u']@raw['model']['v'];b2=raw['serving']['second_bias']+raw['model']['delta2_bias']
            dense=torch.cat([F.linear(F.relu(F.linear(F.relu(F.linear(x[j:j+1024],w1,b1)),w2,b2)),raw['serving']['current_weight'],raw['serving']['current_bias']) for j in range(0,len(x),1024)])
            metrics=paired.metrics(current,y)
            assert max(abs(metrics[k]-branch['test']['own'][k]) for k in metrics)<1e-10
            mse=float(paired.auxiliary_loss(updated,paired.embed(updated,x).float(),aux,kind))
            assert abs(mse-branch['test']['auxiliary_mse'])<1e-7
            contracts=evaluate_contracts(legacy.double(),reference,y,ref['thresholds'],monitor,.05/600)
            for name,value in contracts.items():
                expected=branch['test']['contracts'][name]
                assert value['disagreement']==expected['disagreement']
                assert value['certificate']==expected['certificate']
            exact=updated.legacy_logits(old2)
            assert torch.equal(exact,reference)
            out['branches'].append(dict(domain=domain,seed=seed,mode=branch['mode'],current_metrics_match=True,auxiliary_mse_matches=True,
                contracts_match=True,full_reference_bitwise_equal=True,maximum_packed_to_dense_error=float((current-dense).abs().max()),
                packed_to_dense_decision_changes=int((current.argmax(1)!=dense.argmax(1)).sum()),state_sha256=paired.state_hash(raw['model'])))
        print(json.dumps(dict(seed=seed,domain=domain,verified=2)),flush=True)
    assert len(out['branches'])==40;out['complete']=True;paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);main(a)
