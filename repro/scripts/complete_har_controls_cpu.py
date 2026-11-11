"""Fixed retrospective strong controls and deployment-contract audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import joint_alignment_cpu as ja
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from consumer_contracts import confidence_thresholds,evaluate_contracts
from legacy_head_fit import fit_legacy_heads


def run(a):
    start=time.time();record=json.loads(a.record.read_text());assert record['complete'] and record['domain']=='har'
    torch.manual_seed(record['seed']);np.random.seed(record['seed'])
    cfg=SimpleNamespace(domain='har',seed=record['seed'],phase='confirmation',limit_fit=0,limit_eval=0)
    model,data,kind,splits=bc.load(cfg);assert splits==record['splits'];data['domain']='har'
    folder=a.record.parent/(a.record.stem+'_states');ref=torch.load(folder/'reference.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ref['model']);model.eval();A,B=ref['A'],ref['B']
    healthy=paired.embed(model,data['x']);target=(healthy@A.T+B).float().double()
    reference={p:paired.embed(model,data[p][0])@A.T+B for p in ['validation','test']}
    thresholds=confidence_thresholds(healthy@A.T+B)
    monitor=np.random.default_rng(8490001+record['seed']).choice(len(data['test'][1]),1024,replace=False)
    destination=a.out.parent/(a.out.stem+'_states');destination.mkdir(exist_ok=True,parents=True)
    row,state=ja.train_branch(model,data,kind,ref['model'],ref['optimizer'],A,B,healthy,'feature_map',1.,record['seed'],30,100.)
    assert row['minibatch_order_sha256']==record['branches'][0]['minibatch_order_sha256']
    torch.save(state,destination/'feature_map.pt')
    donors={r['mode']:torch.load(folder/r['state_file'],map_location='cpu',weights_only=True) for r in record['branches']}
    donors['feature_map']=state
    out=dict(complete=False,device='cpu',cuda_visibility='',phase='retrospective_completion',seed=record['seed'],domain='har',
        plan_sha256=a.plan_sha256,source_record_sha256=hashlib.sha256(a.record.read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in
            ['complete_har_controls_cpu.py','consumer_contracts.py','joint_alignment_cpu.py','alignment_objectives.py','legacy_head_fit.py']},
        thresholds=thresholds,new_bank=1000,monitor_sha256=hashlib.sha256(monitor.astype(np.int64).tobytes()).hexdigest(),
        feature_training=row,branches=[])
    for mode,saved in donors.items():
        model.load_state_dict(saved['model']);model.eval()
        readers={mode:saved['reader']}
        if mode in ['detach','joint','protected','feature_map']:
            fit=fit_legacy_heads(paired.embed(model,data['x']),target)['ols']
            reader={k:fit[k].float() for k in ['weight','bias']};readers[mode+'_ols']=reader
            torch.save(dict(reader=reader,ols64=fit,model_sha256=paired.state_hash(saved['model'])),destination/(mode+'_ols.pt'))
        features={p:paired.embed(model,data[p][0]) for p in ['validation','test']}
        for name,reader in readers.items():
            item=dict(mode=name,donor=mode,model_sha256=paired.state_hash(saved['model']),reader_sha256=paired.state_hash(reader))
            for part in ['validation','test']:
                h=features[part];y=data[part][1];z=ja.reader_logits(h,reader).double()
                item[part]=dict(primary=paired.metrics(bc.current_logits(model,h),y),
                    auxiliary_mse=float(paired.auxiliary_loss(model,h.float(),data[part][2],kind)),
                    contracts=evaluate_contracts(z,reference[part],y,thresholds,monitor if part=='test' else None))
                if name.endswith('_ols'):
                    z64=ja.reader_logits(h,fit).double()
                    item[part]['float32_comparison']=dict(maximum_logit_error=float((z-z64).abs().max()),
                        changed_predictions=int((z.argmax(1)!=z64.argmax(1)).sum()))
            out['branches'].append(item)
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);run(a)
