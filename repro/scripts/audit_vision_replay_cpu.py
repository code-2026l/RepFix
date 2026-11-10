"""Reload both completed residual-block grids and re-evaluate every endpoint."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
import paired_readout_continuation as paired
from shared_reference_vision import SharedReferenceBlock,NormalizedHead,dense_block
from vision_consumer_cpu import load_train
from vision_shared_replay_cpu import healthy_features,cached
from consumer_contracts import evaluate_contracts

@torch.no_grad()
def main(a):
    out=dict(complete=False,device='cpu',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),branches=[])
    for path in a.records:
        record=json.loads(path.read_text());folder=path.parent/(path.stem+'_states');ref=torch.load(folder/'reference.pt',map_location='cpu',weights_only=True)
        images,fine,coarse,sha=load_train();assert sha==record['data_sha256']
        e=resnet18(weights=None).to(memory_format=torch.channels_last).eval()
        prefix=nn.Sequential(e.conv1,e.bn1,e.relu,e.maxpool,e.layer1,e.layer2,e.layer3,e.layer4[0]);prefix.load_state_dict(ref['prefix'])
        first=nn.Conv2d(512,512,3,padding=1).to(memory_format=torch.channels_last);second=copy.deepcopy(first)
        first.load_state_dict(ref['first']);second.load_state_dict(ref['second'])
        normalized=ref['normalized'];head=NormalizedHead(normalized['linear.weight'],normalized['linear.bias'],normalized['mean'],normalized['sd']);head.load_state_dict(normalized)
        val=ref['validation'];x,h=healthy_features(prefix,first,second,images,val);cache=cached(first,second,x)
        reference=F.linear(h,ref['heads']['coarse']['weight'],ref['heads']['coarse']['bias'])
        for branch in record['branches']:
            raw=torch.load(folder/branch['state_file'],map_location='cpu',weights_only=True);assert paired.state_hash(raw)==branch['state_sha256']
            mode=branch['mode'];current=[];legacy=[];max_merge=0.;changed=0
            hh=copy.deepcopy(head)
            if mode=='F':hh.load_state_dict(raw['head'])
            elif mode=='U':
                f,s=copy.deepcopy(first),copy.deepcopy(second);f.load_state_dict(raw['first']);s.load_state_dict(raw['second']);hh.load_state_dict(raw['head'])
            else:
                m=SharedReferenceBlock(first,second,ref['support'][:branch['k']],ref['right_vectors'][:branch['rank']],head)
                state=m.state_dict();state.update(raw['updates']);m.load_state_dict(state);f,s=m.merged()
            for j in range(0,len(x),256):
                xx=x[j:j+256];old=F.relu(cache[1][j:j+256]+xx).mean((-2,-1))
                if mode=='F':z=hh(h[j:j+256])
                elif mode=='U':z=hh(dense_block(xx,f,s).mean((-2,-1)))
                else:
                    now,_=m.forward_states(xx,tuple(t[j:j+256] for t in cache));z=m.head(now.mean((-2,-1)))
                    dense=m.head(dense_block(xx,f,s).mean((-2,-1)));max_merge=max(max_merge,float((z-dense).abs().max()));changed+=int((z.argmax(1)!=dense.argmax(1)).sum())
                current.append(z);legacy.append(F.linear(old,ref['heads']['coarse']['weight'],ref['heads']['coarse']['bias']))
            z=torch.cat(current);old=torch.cat(legacy);metrics=paired.metrics(z,fine[val]);error=max(abs(metrics[k]-branch['validation']['own'][k]) for k in metrics)
            assert error<1e-10,(mode,error)
            contracts=evaluate_contracts(old,reference,coarse[val],ref['thresholds'])
            assert all(v['disagreement']==branch['validation']['contracts'][name]['disagreement'] for name,v in contracts.items())
            out['branches'].append(dict(seed=record['seed'],source_record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mode=mode,k=branch['k'],rank=branch['rank'],
                maximum_metric_error=error,contracts_match=True,maximum_legacy_logit_error=float((old-reference).abs().max()),maximum_current_merge_logit_error=max_merge,current_merge_changes=changed))
        print(json.dumps(dict(record=hashlib.sha256(path.read_bytes()).hexdigest(),verified=5)),flush=True)
    assert len(out['branches'])==10;out['complete']=True;paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1);main(a)
