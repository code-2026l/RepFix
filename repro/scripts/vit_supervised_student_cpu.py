"""Same-feature, same-label supervised nonlinear control for visual students."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from vit_nonlinear_student_cpu import LogitStudent,base


def run(args):
    cache=torch.load(args.population,map_location='cpu',mmap=True,weights_only=True)
    row=json.loads(args.student_record.read_text())
    source=args.student_record.parent/(args.student_record.stem+'_states/program.pt')
    assert base.digest(source)==row['program_sha256']
    state=torch.load(source,map_location='cpu',weights_only=True)
    ref=state['reference']
    fit,val=ref['fitting'],ref['validation']
    params=state['models']['joint/128']
    mean,sd=params['mean'],params['sd']
    x=torch.cat((cache['h'][:,fit],cache['hidden'][:,fit]),-1).flatten(0,1)
    labels=cache['y'][fit].repeat(2)
    mask=ref['label_mask'].repeat(2)
    reader=ref['heads']['fine']
    weight=torch.zeros_like(params['base_weight'])
    weight[:,:768]=reader['weight']*sd[:768]
    bias=reader['bias']+reader['weight']@mean[:768]
    torch.manual_seed(row['seed']+50000)
    model=LogitStudent(len(mean),128,weight,bias,mean,sd)
    del model._buffers['base_weight']
    del model._buffers['base_bias']
    model.register_parameter('base_weight',nn.Parameter(weight))
    model.register_parameter('base_bias',nn.Parameter(bias))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator().manual_seed(row['seed']+70000)
    for epoch in range(30):
        for ids in torch.randperm(len(x),generator=generator).split(256):
            z=model(x[ids])
            chosen=mask[ids]
            loss=F.cross_entropy(z[chosen],labels[ids][chosen]) if chosen.any() else z.sum()*0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval().requires_grad_(False)
    phi=torch.cat((cache['h'][0,val],cache['hidden'][0,val]),1)
    with torch.no_grad():
        logits=torch.cat([model(b) for b in phi.split(256)])
    folder=args.out.parent/(args.out.stem+'_states')
    folder.mkdir(exist_ok=True,parents=True)
    replay,program=folder/'replay.pt',folder/'program.pt'
    torch.save(dict(labels=cache['y'][val],predictions=logits.argmax(1)),replay)
    torch.save(model.state_dict(),program)
    base.save(args.out,dict(complete=True,device='cpu',stage='supervised',
        reference_seed=row['reference_seed'],seed=row['seed'],plan_sha256=args.plan_sha256,
        official_test_loaded=False,student_program_sha256=row['program_sha256'],
        replay_sha256=base.digest(replay),program_sha256=base.digest(program),
        source_sha256={Path(__file__).name:base.digest(Path(__file__))},branches=[]))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--population',type=Path,required=True)
    parser.add_argument('--student-record',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    run(parser.parse_args())
