"""Isolated two-output timing, excluding partial records and including OLS heads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import audit_protected_head_cpu as audit
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired


class CompletedDirectory:
    def __init__(self,path):self.path=path
    def glob(self,pattern):
        return [p for p in self.path.glob(pattern) if not p.name.endswith('.partial.json')]


@torch.no_grad()
def run(args):
    # Preserve the frozen replay source. Its original glob also matched progress
    # snapshots; only this wrapper filters those non-endpoint input records.
    args.records_dir=CompletedDirectory(args.records_dir)
    audit.benchmark(args)
    result=json.loads(args.out.read_text())
    files=sorted(args.records_dir.glob('ph*_conf_*.json'))
    for path in files:
        record,model,data,kind,states,A,B=audit.load_record(path)
        i=(0 if record['domain']=='har' else 10)+record['seed']-300
        fitted=args.fitted_dir/('r%03d_readers'%i)/'protected_ols.pt'
        head=torch.load(fitted,map_location='cpu',weights_only=True)
        protected=next(row for row in record['branches'] if row['mode']=='protected')
        state=torch.load(states/('protected_%g.pt'%protected['lam']),map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);model.eval()
        def both(sample):
            features=model(sample)
            return bc.current_logits(model,features),F.linear(features.double(),head['weight'],head['bias'])
        timings={}
        for batch in [1,1024]:
            sample=data['validation'][0][:batch]
            operations=dict(current=lambda:bc.current_logits(model,model(sample)),both=lambda:both(sample))
            for op in operations.values():
                for _ in range(5):op()
            samples={key:[] for key in operations}
            for repeat in range(51):
                for name in (list(operations) if repeat%2 else list(reversed(operations))):
                    start=time.perf_counter();operations[name]();samples[name].append(time.perf_counter()-start)
            medians={name:float(np.median(values)) for name,values in samples.items()}
            timings[str(batch)]=dict(medians_seconds=medians,samples_seconds=samples,
                                     both_vs_current_ratio=medians['both']/medians['current'])
        row=next(row for row in result['branches'] if row['domain']==record['domain'] and row['seed']==record['seed'])
        row['posthoc_ols_timings']=timings
    result['timing_wrapper_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['progress_snapshots_excluded']=True
    paired.save_json(args.out,result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--records-dir',type=Path,required=True)
    p.add_argument('--fitted-dir',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    run(args)
