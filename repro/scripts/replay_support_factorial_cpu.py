"""Development-only separation of first-map support and second-map rank."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
import numpy as np
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
import shared_replay_cpu as sparse
import dense_replay_cpu as dense
from consumer_contracts import confidence_thresholds

def run(a):
    start=time.time();old=json.loads(a.record.read_text());seed=old['seed'];assert old['phase']=='development' and old['domain']=='radioml'
    torch.manual_seed(seed);np.random.seed(seed)
    healthy,data,kind,splits=bc.load(SimpleNamespace(domain='radioml',seed=seed,phase='development',limit_fit=0,limit_eval=0))
    assert splits==old['splits'] and 'test' not in data
    ref=torch.load(a.record.parent/(a.record.stem+'_states')/'reference.pt',map_location='cpu',weights_only=True)
    healthy.load_state_dict(ref['model']);healthy.eval();A=ref['head']['weight'].double()/ref['sd'].reshape(-1);B=ref['head']['bias'].double()-A@ref['mean'].reshape(-1)
    caches=sparse.cache(healthy,data['x']);cv=sparse.cache(healthy,data['validation'][0])
    oldfit=F.linear(healthy.encoder[4](F.relu(caches[1])).double(),A,B).detach()
    oldval=F.linear(healthy.encoder[4](F.relu(cv[1])).double(),A,B).detach();thresholds=confidence_thresholds(oldfit)
    support,v,selection=sparse.calibration(healthy,data,kind,1.,seed);q,vd,cal=dense.calibration(healthy,data,kind,1.,seed)
    assert torch.equal(v,vd) and selection['indices_sha256']==cal['indices_sha256']
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    torch.save(dict(reference=ref,A=A,B=B,support=support,q=q,v=v,thresholds=thresholds),folder/'reference.pt')
    names=['replay_support_factorial_cpu.py','shared_replay_cpu.py','shared_reference_replay.py','dense_replay_cpu.py','dense_replay.py','consumer_contracts.py']
    out=dict(complete=False,device='cpu',phase='development',domain='radioml',seed=seed,plan_sha256=a.plan_sha256,splits=splits,
        source_record_sha256=hashlib.sha256(a.record.read_bytes()).hexdigest(),source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names},branches=[])
    for family in ['sparse','dense']:
        for rank2 in [8,12]:
            for mode in ['detach','joint']:
                if family=='sparse':
                    model,opt,row=sparse.train(healthy,ref['optimizer'],data,kind,caches,A,B,support,v,16,rank2,mode,seed,25,1.)
                    row['validation']=sparse.score(model,data['validation'],kind,cv,oldval,thresholds);d1=model.delta1.detach();d2=(model.u@model.v).detach()
                else:
                    model,opt,row=dense.train(healthy,ref['optimizer'],data,kind,caches,A,B,q,v,16,rank2,mode,seed,25,1.)
                    row['validation']=dense.score(model,data['validation'],kind,cv,oldval,thresholds);d1=(model.p@model.q).detach();d2=(model.u@model.v).detach()
                row.update(family=family,first_rank=int(torch.linalg.matrix_rank(d1)),second_rank=int(torch.linalg.matrix_rank(d2)),
                    update_parameters=sum(p.numel() for p in model.update_parameters()),state_file='%s_%d_%s.pt'%(family,rank2,mode))
                torch.save(dict(model=model.state_dict(),serving=model.serving()),folder/row['state_file'])
                out['branches'].append(row);paired.save_json(a.out.with_suffix('.partial.json'),out)
                print(json.dumps(dict(seed=seed,family=family,rank2=rank2,mode=mode,acc=row['validation']['own']['acc'],elapsed_s=time.time()-start)),flush=True)
    assert len({b['minibatch_order_sha256'] for b in out['branches']})==1
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);run(a)
