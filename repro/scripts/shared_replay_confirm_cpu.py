"""Fresh-seed paired confirmation of a frozen shared-reference configuration."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from readout_transfer_audit import fit_head
from consumer_contracts import confidence_thresholds,evaluate_contracts
from shared_reference_replay import packed_serving,serve_packed_mlp
from shared_replay_cpu import cache,calibration,train,score


def ordinary(healthy,moments,data,kind,seed,epochs,dose,mode):
    model=copy.deepcopy(healthy);opt=torch.optim.Adam(model.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(moments))
    order_sha=hashlib.sha256();rng=torch.Generator().manual_seed(seed+20000)
    for epoch in range(epochs):
        order=torch.randperm(len(data['x']),generator=rng);order_sha.update(order.numpy().tobytes())
        for j in range(0,len(order),256):
            idx=order[j:j+256];bc.step(model,opt,data['x'][idx],data['y'][idx],data['a'][idx],data['labeled'][idx],kind,dose,mode=='detach')
    return model,opt,order_sha.hexdigest()


@torch.no_grad()
def full_score(model,part,kind):
    x,y,a=part;h=paired.embed(model,x).float()
    # Native serving folds the final encoder affine into the task head.
    h2=torch.cat([F.relu(model.encoder[2](F.relu(model.encoder[0](x[j:j+1024])))) for j in range(0,len(x),1024)])
    weight=(model.primary.weight.double()@model.encoder[4].weight.double()).float()
    bias=(model.primary.bias.double()+model.primary.weight.double()@model.encoder[4].bias.double()).float()
    return dict(own=paired.metrics(F.linear(h2,weight,bias),y),auxiliary_mse=float(paired.auxiliary_loss(model,h,a,kind)))


@torch.no_grad()
def audited_score(model,part,kind,cached,old,thresholds,monitor):
    row=score(model,part,kind,cached,old,thresholds);x,y,_=part
    p=packed_serving(model.serving());z=[serve_packed_mlp(x[j:j+1024],p) for j in range(0,len(x),1024)]
    own,legacy=[torch.cat([v[i] for v in z]) for i in [0,1]]
    row['unpacked_own']=row['own'];row['own']=paired.metrics(own,y)
    row['packed_serving']=dict(own=paired.metrics(own,y),contracts=evaluate_contracts(legacy.double(),old,y,thresholds,monitor,alpha=.05/600),
        maximum_legacy_logit_error=float((legacy.double()-old).abs().max()))
    row['contracts']=row['packed_serving']['contracts']
    row['full_reference_contracts']=evaluate_contracts(model.legacy_logits(F.relu(cached[1])),old,y,thresholds,monitor,alpha=.05/600)
    current,old2=model.forward_states(x,cached)
    row['changed_second_relu_gates']=int(((current>0)!=(old2>0)).sum())
    row['changed_first_relu_gates']=int(((cached[0][:,model.support]+F.linear(x,model.delta1,model.delta1_bias)>0)!=(cached[0][:,model.support]>0)).sum())
    return row


def run(a):
    start=time.time();choice=json.loads(a.choice.read_text());selected=choice['domains'][a.domain]['selected'];assert selected['utility_compatible']
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    healthy,data,kind,splits=bc.load(SimpleNamespace(domain=a.domain,seed=a.seed,phase='confirmation',limit_fit=0,limit_eval=0))
    warm=60 if a.domain=='har' else 25;epochs=30 if a.domain=='har' else 25;dose=100. if a.domain=='har' else 1.
    opt=torch.optim.Adam(healthy.parameters(),lr=.001);rng=torch.Generator().manual_seed(a.seed+10000);warm_hash=hashlib.sha256()
    for epoch in range(warm):
        order=torch.randperm(len(data['x']),generator=rng);warm_hash.update(order.numpy().tobytes())
        for j in range(0,len(order),256):
            idx=order[j:j+256];bc.step(healthy,opt,data['x'][idx],data['y'][idx],data['a'][idx],data['labeled'][idx],kind,1.,True)
        if epoch==0 or (epoch+1)%10==0:print(json.dumps(dict(stage='warm',domain=a.domain,seed=a.seed,epoch=epoch+1,elapsed_s=time.time()-start)),flush=True)
    healthy.eval();moments=copy.deepcopy(opt.state_dict());head,mean,sd=fit_head(paired.embed(healthy,data['x'])[data['labeled']],data['y'][data['labeled']])
    A=head.weight.detach().double()/sd.reshape(-1);B=head.bias.detach().double()-A@mean.reshape(-1)
    reference=dict(model=copy.deepcopy(healthy.state_dict()),optimizer=moments,head=head.state_dict(),mean=mean,sd=sd,splits=splits)
    fitting_cache=cache(healthy,data['x']);caches={name:cache(healthy,data[name][0]) for name in ['validation','test']}
    oldfit=F.linear(healthy.encoder[4](F.relu(fitting_cache[1])).double(),A,B).detach();thresholds=confidence_thresholds(oldfit)
    old={name:F.linear(healthy.encoder[4](F.relu(caches[name][1])).double(),A,B).detach() for name in caches}
    support,right,selection=calibration(healthy,data,kind,dose,a.seed);n=len(data['test'][1])
    monitor=np.random.default_rng(10700001+a.seed+(100000 if a.domain=='har' else 0)).choice(n,1024,replace=False)
    states=a.out.parent/(a.out.stem+'_states');states.mkdir(parents=True,exist_ok=True)
    torch.save(dict(reference=reference,A=A,B=B,thresholds=thresholds,support=support,right_vectors=right),states/'reference.pt')
    names=['shared_replay_confirm_cpu.py','shared_replay_cpu.py','shared_reference_replay.py','consumer_contracts.py',
        'benefit_compatibility_cpu.py','readout_transfer_audit.py','paired_readout_continuation.py']
    out=dict(complete=False,device='cpu',phase='confirmation',domain=a.domain,seed=a.seed,plan_sha256=a.plan_sha256,
        selection_sha256=hashlib.sha256(a.choice.read_bytes()).hexdigest(),source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names},
        splits=splits,reference_model_sha256=paired.state_hash(reference['model']),reference_optimizer_sha256=paired.state_hash(moments),
        warm_order_sha256=warm_hash.hexdigest(),support_calibration=selection,thresholds=thresholds,configuration=dict(k=selected['k'],rank=selected['rank'],
            warm_epochs=warm,epochs=epochs,dose=dose,batch=256,lr=.001,gradient_clip=None,threads=4),
        consumer_bank=dict(alpha=.05/600,candidates=600,monitor_inputs=1024,monitor_sha256=paired.state_hash(torch.from_numpy(monitor)),
            monitor_used_for_training=False,budget=.05),full_controls={},branches=[])
    for mode in ['detach','joint']:
        model,optimizer,order=ordinary(healthy,moments,data,kind,a.seed,epochs,dose,mode);model.eval()
        row=dict(minibatch_order_sha256=order,model_sha256=paired.state_hash(model.state_dict()),
            validation=full_score(model,data['validation'],kind),test=full_score(model,data['test'],kind),state_file='full_'+mode+'.pt')
        out['full_controls'][mode]=row;torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),states/row['state_file'])
        paired.save_json(a.out.with_suffix('.partial.json'),out)
    for mode in ['detach','joint']:
        model,optimizer,row=train(healthy,moments,data,kind,fitting_cache,A,B,support,right,selected['k'],selected['rank'],mode,a.seed,epochs,dose)
        model.eval();row['validation']=audited_score(model,data['validation'],kind,caches['validation'],old['validation'],thresholds,None)
        row['test']=audited_score(model,data['test'],kind,caches['test'],old['test'],thresholds,monitor);row['state_file']=mode+'.pt'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),serving=model.serving()),states/row['state_file'])
        out['branches'].append(row);paired.save_json(a.out.with_suffix('.partial.json'),out)
        print(json.dumps(dict(stage=mode,domain=a.domain,seed=a.seed,accuracy=row['test']['own']['acc'],mse=row['test']['auxiliary_mse'],elapsed_s=time.time()-start)),flush=True)
    assert len({row['minibatch_order_sha256'] for row in [*out['branches'],*out['full_controls'].values()]})==1
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--domain',choices=['har','radioml'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--choice',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);run(a)
