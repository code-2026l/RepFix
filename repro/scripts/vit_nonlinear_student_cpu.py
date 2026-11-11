"""Nonlinear fine-logit students sharing a frozen healthy ViT."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from vit_exact_student_cpu import TEACHERS, base, factory

WIDTHS = (128, 512)


class NonlinearVisualStudent(nn.Module):
    def __init__(self, healthy, state, legacy, both=True):
        super().__init__()
        self.healthy = healthy
        self.readout = LogitStudent(len(state['mean']), state['hidden.weight'].shape[0],
                                    state['base_weight'], state['base_bias'], state['mean'], state['sd'])
        self.readout.load_state_dict(state)
        self.register_buffer('old_weight', legacy['weight'])
        self.register_buffer('old_bias', legacy['bias'])
        self.both = both

    def forward(self, tokens):
        h, hidden = self.healthy.features(tokens)
        z = self.readout(torch.cat((h, hidden),1))
        if self.both:
            return z, F.linear(h,self.old_weight,self.old_bias)
        return z


class LogitStudent(nn.Module):
    def __init__(self, dimension, width, weight, bias, mean, sd):
        super().__init__()
        for name,value in (('base_weight',weight),('base_bias',bias),('mean',mean),('sd',sd)):
            self.register_buffer(name,value)
        self.hidden=nn.Linear(dimension,width)
        self.output=nn.Linear(width,100)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,x):
        phi=(x-self.mean)/self.sd
        return F.linear(phi,self.base_weight,self.base_bias)+self.output(F.gelu(self.hidden(phi)))


def fit(phi,target,weight,bias,mean,sd,width,seed):
    torch.manual_seed(seed+50000)
    model=LogitStudent(phi.shape[1],width,weight,bias,mean,sd)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    centered=target-target.mean(1,keepdim=True)
    variance=centered.square().mean().clamp_min(1e-6)
    probs=centered.softmax(1)
    generator=torch.Generator().manual_seed(seed+70000)
    for epoch in range(30):
        total=0.
        for ids in torch.randperm(len(phi),generator=generator).split(256):
            z=model(phi[ids])
            z=z-z.mean(1,keepdim=True)
            loss=F.mse_loss(z,centered[ids])/variance+F.kl_div(z.log_softmax(1),probs[ids],reduction='batchmean')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total+=loss.item()
        if epoch in (0,9,19,29):
            print(json.dumps(dict(width=width,epoch=epoch+1,loss=total)),flush=True)
    return model.eval().requires_grad_(False)


def run(args):
    start=time.perf_counter()
    cache=torch.load(args.population,map_location='cpu',weights_only=True,mmap=True)
    states,records={},{}
    for name,path in zip(TEACHERS,(args.joint,args.two_view,args.double_epoch)):
        record=json.loads(path.read_text())
        assert record['complete'] and record['reference_seed']==args.reference_seed and record['seed']==args.seed
        endpoint=path.parent/(path.stem+'_states')/'endpoint.pt'
        assert base.digest(endpoint)==record['state_sha256']
        states[name]=torch.load(endpoint,map_location='cpu',weights_only=True)
        records[name]=record
    ref=states['joint']['reference']
    _,healthy=factory.build_for(cache)
    healthy.load_state_dict(cache['healthy'])
    healthy.eval().requires_grad_(False)
    fitting,validation=ref['fitting'],ref['validation']
    phi_fit=torch.cat((cache['h'][:,fitting],cache['hidden'][:,fitting]),-1).flatten(0,1).contiguous()
    phi_val=torch.cat((cache['h'][0,validation],cache['hidden'][0,validation]),-1)
    x=phi_fit.double()
    mean,sd=x.mean(0),x.std(0,unbiased=False).clamp_min(.001)
    design=(x-mean)/sd
    gram=design.T@design/len(design)
    cholesky=torch.linalg.cholesky(gram+.001*torch.eye(gram.shape[0],dtype=gram.dtype))
    widths=WIDTHS
    if args.stage=='confirmation':
        choice=json.loads(args.choice.read_text())
        assert choice['plan_sha256']==args.plan_sha256
        widths=(choice['width'],)
    predictions,models,teacher_predictions,rows={},{},{},[]
    with torch.no_grad():
        old=base.late.linear(cache['h'][0,validation],ref['heads']['coarse'])
    for name,state in states.items():
        with torch.no_grad():
            model=copy.deepcopy(healthy)
            model.load_state_dict(state['model'])
            model.eval().requires_grad_(False)
            h=torch.cat([base.embeds(model,view) for view in cache['x'][:,fitting]])
            target=base.late.linear(h,state['reader'])
            val_logits=base.late.linear(base.embeds(model,cache['x'][0,validation]),state['reader'])
            torch.testing.assert_close(val_logits,state['own_logits'],atol=2e-4,rtol=2e-4)
            teacher_predictions[name]=val_logits.argmax(1)
            reference_logits=base.late.linear(cache['h'][:,fitting].flatten(0,1),state['reader'])
            residual=(target-reference_logits).double()
            target_mean=residual.mean(0)
            coeff=torch.cholesky_solve(design.T@(residual-target_mean)/len(design),cholesky)
            weight=coeff.T.float()
            weight[:,:768]+=state['reader']['weight']*sd[:768].float()
            bias=target_mean.float()+state['reader']['bias']+state['reader']['weight']@mean[:768].float()
            print(json.dumps(dict(prepared_teacher=name,elapsed_s=time.perf_counter()-start)),flush=True)
        for width in widths:
            student=fit(phi_fit,target,weight,bias,mean.float(),sd.float(),width,args.seed)
            with torch.no_grad():
                z=torch.cat([student(batch) for batch in phi_val.split(256)])
            key=f'{name}/{width}'
            predictions[key]=z.argmax(1)
            models[key]=student.state_dict()
            rows.append(dict(key=key,teacher=name,width=width,
                             accuracy=float((z.argmax(1)==cache['y'][validation]).double().mean()),
                             teacher_accuracy=float((val_logits.argmax(1)==cache['y'][validation]).double().mean())))
        del model,h,target
    folder=args.out.parent/(args.out.stem+'_states')
    folder.mkdir(exist_ok=True,parents=True)
    replay,program=folder/'replay.pt',folder/'program.pt'
    torch.save(dict(labels=cache['y'][validation],predictions=predictions,teachers=teacher_predictions,
                    old=old,protected=base.late.linear(cache['h'][0,validation],ref['heads']['coarse']),
                    thresholds=list(ref['thresholds'].values())),replay)
    torch.save(dict(models=models,reference=ref,records={name:records[name]['state_sha256'] for name in TEACHERS},
                    healthy_sha256=base.paired.state_hash(cache['healthy'])),program)
    base.save(args.out,dict(complete=True,device='cpu',stage=args.stage,reference_seed=args.reference_seed,
        seed=args.seed,plan_sha256=args.plan_sha256,official_test_loaded=False,
        choice_sha256=base.digest(args.choice) if args.choice else None,
        outcomes=rows,replay_sha256=base.digest(replay),program_sha256=base.digest(program),
        source_sha256={Path(__file__).name:base.digest(Path(__file__))},
        elapsed_s=time.perf_counter()-start,branches=[]))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('pilot','confirmation'),required=True)
    parser.add_argument('--population',type=Path,required=True)
    parser.add_argument('--joint',type=Path,required=True)
    parser.add_argument('--two-view',type=Path,required=True)
    parser.add_argument('--double-epoch',type=Path,required=True)
    parser.add_argument('--reference-seed',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--choice',type=Path)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    run(parser.parse_args())
