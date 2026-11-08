"""Verify retrieved numerical checkpoints and pairing before local delivery."""
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from paired_readout_continuation import state_hash

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'repro/results/CPUCONT'
expected_source=hashlib.sha256((ROOT/'repro/scripts/paired_readout_continuation.py').read_bytes()).hexdigest()
files=[]
checkpoints=0
branches=0
max_rss=0
job_wall=0
for folder,prefix,count in [('har','h',5),('replication','v',5),('projected','p',5),('real','r',5)]:
    paths=sorted((BASE/folder).glob(prefix+'???.json'))
    if len(paths)!=count:
        raise AssertionError((folder,len(paths),count))
    queue=json.loads((BASE/folder/'queue.json').read_text())
    if not queue['finished'] or queue['running'] or queue['pending'] or queue['failed'] or len(queue['completed'])!=count:
        raise AssertionError('Numerical campaign did not finish cleanly')
    max_rss=max(max_rss,queue['peak_aggregate_rss_bytes'])
    job_wall+=sum(j['wall_s'] for j in queue['completed'])
    for path in paths:
        data=json.loads(path.read_text())
        if not data['complete'] or data['source_sha256']!=expected_source:
            raise AssertionError('Incomplete run or executed source mismatch')
        if data['device']!='cpu' or data['cuda_visibility']!='' or data['torch_threads']!=1:
            raise AssertionError('CPU-only requirement not met')
        if data['splits']['calibration_used_for_encoder'] or data['splits']['calibration_used_for_head']:
            raise AssertionError('Calibration leakage flag')
        states=path.parent/(path.stem+'_states')
        reference=torch.load(states/'reference.pt',map_location='cpu',weights_only=True)
        if state_hash(reference['model'])!=data['reference_model_sha256']:
            raise AssertionError('Reference model hash mismatch')
        if state_hash(reference['optimizer'])!=data['reference_optimizer_sha256']:
            raise AssertionError('Reference Adam state hash mismatch')
        checkpoints+=1
        primary_hash=None
        orders=set()
        for branch in data['branches']:
            if branch['initial_model_sha256']!=data['reference_model_sha256'] or branch['initial_optimizer_sha256']!=data['reference_optimizer_sha256']:
                raise AssertionError('Unpaired initial state')
            checkpoint=states/('%s_%g.pt'%(branch['mode'],branch['alpha']))
            loaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if any(v.device.type!='cpu' for v in loaded['model'].values() if isinstance(v,torch.Tensor)):
                raise AssertionError('A saved parameter is not on CPU')
            shared={k:v for k,v in loaded['model'].items() if k.startswith(('encoder.','primary.'))}
            digest=state_hash(shared)
            if digest!=branch['final_shared_sha256']:
                raise AssertionError('Downloaded final checkpoint hash mismatch')
            if branch['mode']=='primary':
                primary_hash=digest
            if branch['mode']=='detach' and digest!=primary_hash:
                raise AssertionError('Detached endpoint differs from primary-only control')
            orders.add(branch['minibatch_order_sha256'])
            checkpoints+=1
            branches+=1
        if len(orders)!=1:
            raise AssertionError('Future minibatch ordering differs')
        files.append(path)
out=dict(verified_runs=len(files),verified_branch_endpoints=branches,
         verified_model_optimizer_checkpoints=checkpoints,
         all_detached_controls_bitwise_equal=True,all_initial_states_and_orders_match=True,
         all_saved_parameters_cpu=True,calibration_examples_excluded_from_fitting=True,
         all_queues_finished=True,maximum_concurrent_workers=4,
         monitored_peak_aggregate_rss_bytes=max_rss,sum_job_wall_seconds=job_wall,
         executed_driver_sha256=expected_source,
         input_record_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(BASE/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='input_record_sha256'},indent=2))
