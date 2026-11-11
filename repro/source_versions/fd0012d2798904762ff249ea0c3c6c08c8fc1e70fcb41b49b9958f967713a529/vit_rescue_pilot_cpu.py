"""Select the fixed fitting-margin percentile for conditional current-tail rescue."""
import argparse
import json
from pathlib import Path

import torch

from vit_exact_student_cpu import TEACHERS, base
from vit_nonlinear_student_cpu import LogitStudent

PERCENTILES=(.1,.25,.5)


@torch.no_grad()
def run(args):
    cache=torch.load(args.population,map_location='cpu',mmap=True,weights_only=True)
    record=json.loads(args.student_record.read_text())
    assert record['reference_seed']==6200 and record['seed']==6300 and record['stage']=='pilot'
    source=args.student_record.parent/(args.student_record.stem+'_states/program.pt')
    assert base.digest(source)==record['program_sha256']
    state=torch.load(source,map_location='cpu',weights_only=True)
    ref=state['reference']
    fitting,validation=ref['fitting'],ref['validation']
    xfit=torch.cat((cache['h'][0,fitting],cache['hidden'][0,fitting]),1)
    xval=torch.cat((cache['h'][0,validation],cache['hidden'][0,validation]),1)
    labels=cache['y'][validation]
    results,predictions,fit_margins,val_margins,teachers={},{},{},{},{}
    for name,path in zip(TEACHERS,(args.joint,args.two_view,args.double_epoch)):
        original=json.loads(path.read_text())
        endpoint=path.parent/(path.stem+'_states/endpoint.pt')
        assert base.digest(endpoint)==state['records'][name]==original['state_sha256']
        teacher=torch.load(endpoint,map_location='cpu',weights_only=True)['own_logits']
        params=state['models'][name+'/128']
        model=LogitStudent(len(params['mean']),128,params['base_weight'],params['base_bias'],params['mean'],params['sd'])
        model.load_state_dict(params)
        model.eval().requires_grad_(False)
        train_logits=torch.cat([model(x) for x in xfit.split(256)])
        val_logits=torch.cat([model(x) for x in xval.split(256)])
        train_top=train_logits.topk(2,dim=1).values
        val_top=val_logits.topk(2,dim=1).values
        train_margin=train_top[:,0]-train_top[:,1]
        val_margin=val_top[:,0]-val_top[:,1]
        fit_margins[name],val_margins[name]=train_margin,val_margin
        teachers[name]=teacher.argmax(1)
        for percentile in PERCENTILES:
            cutoff=float(torch.quantile(train_margin,percentile))
            selected=val_margin<=cutoff
            served=val_logits.clone()
            served[selected]=teacher[selected]
            key=f'{name}/{percentile}'
            predictions[key]=served.argmax(1)
            results[key]=dict(accuracy=float((served.argmax(1)==labels).double().mean()),
                              teacher_accuracy=float((teacher.argmax(1)==labels).double().mean()),
                              cutoff=cutoff,fraction=float(selected.double().mean()))
    folder=args.out.parent/(args.out.stem+'_states')
    folder.mkdir(exist_ok=True,parents=True)
    replay=folder/'replay.pt'
    torch.save(dict(labels=labels,predictions=predictions,teachers=teachers,
                    fit_margins=fit_margins,val_margins=val_margins),replay)
    base.save(args.out,dict(complete=True,device='cpu',stage='pilot',reference_seed=6200,seed=6300,
        plan_sha256=args.plan_sha256,official_test_loaded=False,outcomes=results,
        replay_sha256=base.digest(replay),student_program_sha256=record['program_sha256'],
        source_sha256={Path(__file__).name:base.digest(Path(__file__))},branches=[]))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--population',type=Path,required=True)
    parser.add_argument('--student-record',type=Path,required=True)
    parser.add_argument('--joint',type=Path,required=True)
    parser.add_argument('--two-view',type=Path,required=True)
    parser.add_argument('--double-epoch',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    torch.set_num_threads(12)
    torch.set_num_interop_threads(1)
    run(parser.parse_args())
