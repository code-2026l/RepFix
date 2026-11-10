"""Local output replay: recalculate actual serving metrics and certificates.

This checks recovered output arrays, not a second execution of the pretrained
backbone. The numerical driver separately reconstructs and hashes its weights.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
import paired_readout_continuation as paired
from consumer_contracts import evaluate_contracts

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def compare(actual,recorded):
    if isinstance(recorded,dict):
        assert actual.keys()==recorded.keys(),(actual.keys(),recorded.keys())
        for key in recorded:compare(actual[key],recorded[key])
    elif isinstance(recorded,(list,tuple)):
        assert len(actual)==len(recorded)
        for x,y in zip(actual,recorded):compare(x,y)
    elif isinstance(recorded,float):assert math.isclose(actual,recorded,rel_tol=2e-6,abs_tol=2e-8),(actual,recorded)
    else:assert actual==recorded,(actual,recorded)

def main(a):
    start=time.time();execution=json.loads(a.transport_audit.read_text());assert execution['complete'] or a.allow_partial
    locations_path=Path(__file__).resolve().parents[2]/'repro/results/CPUCONT/native_serving_record_locations.json'
    locations=json.loads(locations_path.read_text());assert locations['complete'] or a.allow_partial
    out=dict(complete=False,analysis='local',method='recovered served-output replay; not a second backbone execution',
        transport_audit_sha256=digest(a.transport_audit),source_sha256={Path(__file__).name:digest(Path(__file__))},endpoints=[])
    references={}
    for arch,kind,index in [('resnet18','next_visionfullprepare',0),('vit_b32','next_visiontwoprepare',2)]:
        references[arch]=torch.load(a.state_root/kind/('p%03d_states/reference.pt'%index),map_location='cpu',weights_only=True)
    for identifier,location in sorted(locations['records'].items()):
        path=a.state_root/location['queue']/(identifier+'.json')
        if not path.exists() and a.allow_partial:continue
        assert digest(path)==location['record_sha256']
        record=json.loads(path.read_text());assert record['complete'] and record['audit_only'];arch=record['arch'];seed=record['seed'];ref=references[arch]
        assert paired.state_hash(ref)==record['reference_state_sha256']
        artifact=path.parent/(path.stem+'_states')/'served_logits.pt'
        assert digest(artifact)==record['serving']['U']['served_logits_sha256']
        state=torch.load(artifact,map_location='cpu',weights_only=True)
        assert state['header']['reference']==record['reference_state_sha256']
        assert state['source_state_sha256']==record['audits']['U']['state_sha256']
        monitor=np.random.default_rng(14700001+seed).choice(len(state['primary_labels']),1024,replace=False)
        assert np.array_equal(monitor,state['monitor'].numpy())
        endpoints=[]
        for mode,(own,legacy) in state['controls'].items():
            compare(paired.metrics(own,state['primary_labels']),record['audits'][mode]['own'])
            contracts=evaluate_contracts(legacy.double(),state['reference'].double(),state['coarse_labels'],ref['thresholds'],monitor,.05/2000)
            compare(contracts,record['audits'][mode]['contracts'])
            endpoints.append(dict(kind='control',mode=mode,consumers=len(contracts)))
        for kind in ['chosen','full_precision']:
            for batch,(own,legacy) in state[kind].items():
                row=record['serving']['U'][kind][batch]
                compare(paired.metrics(own,state['primary_labels']),row['own'])
                contracts=evaluate_contracts(legacy.double(),state['reference'].double(),state['coarse_labels'],ref['thresholds'],monitor,.05/2000)
                compare(contracts,row['contracts'])
                endpoints.append(dict(kind=kind,batch=int(batch),consumers=len(contracts),
                    maximum_disagreement=max(v['disagreement'] for v in contracts.values()),
                    certified=sum(v['certificate']['certified'] for v in contracts.values())))
        out['endpoints'].append(dict(arch=arch,seed=seed,queue=location['queue'],state_sha256=digest(artifact),checks=endpoints))
        print(json.dumps(dict(arch=arch,seed=seed,outputs_replayed=len(endpoints))),flush=True)
    complete=len(out['endpoints'])==20 and execution['complete'] and locations['complete']
    if not a.allow_partial:assert complete and {r['seed'] for r in out['endpoints']}==set(range(3810,3820))
    out.update(complete=complete,elapsed_s=time.time()-start);a.out.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state-root',type=Path,required=True);p.add_argument('--transport-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);main(a)
