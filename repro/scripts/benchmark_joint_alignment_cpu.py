"""Isolated actual current-plus-legacy serving cost on fresh confirmation states."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired
from alignment_objectives import reader_logits


@torch.no_grad()
def main(args):
    paths=sorted(args.records.glob('c???.json'));assert paths
    out=dict(complete=False,device='cpu',cuda_visibility='',plan_sha256=args.plan_sha256,
        benchmark_driver_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        benchmark='current and legacy outputs, one encoder call, alternating medians',threads=4,branches=[])
    for path in paths:
        record=json.loads(path.read_text());assert record['complete'] and record['phase']=='confirmation'
        cfg=SimpleNamespace(domain=record['domain'],seed=record['seed'],phase='confirmation',limit_fit=0,limit_eval=0)
        model,data,_,_=bc.load(cfg)
        directory=path.parent/(path.stem+'_states')
        for row in record['branches']:
            if row['mode'] in ['detach','joint']:continue
            saved=torch.load(directory/row['state_file'],map_location='cpu',weights_only=True)
            model.load_state_dict(saved['model']);model.eval();reader=saved['reader']
            def both(sample):
                features=model(sample)
                return bc.current_logits(model,features),reader_logits(features,reader)
            calls=[];hook=model.register_forward_hook(lambda *_:calls.append(1));both(data['validation'][0][:1]);hook.remove()
            assert len(calls)==1
            times={}
            for batch in [1,1024]:
                sample=data['validation'][0][:batch]
                operations=dict(current=lambda:bc.current_logits(model,model(sample)),both=lambda:both(sample))
                for op in operations.values():
                    for _ in range(5):op()
                durations={key:[] for key in operations}
                for repeat in range(51):
                    for key in (list(operations) if repeat%2 else list(reversed(operations))):
                        start=time.perf_counter();operations[key]();durations[key].append(time.perf_counter()-start)
                medians={k:float(np.median(v)) for k,v in durations.items()}
                times[str(batch)]=dict(medians_seconds=medians,samples_seconds=durations,ratio=medians['both']/medians['current'])
            out['branches'].append(dict(domain=record['domain'],seed=record['seed'],mode=row['mode'],
                reader_sha256=paired.state_hash(reader),encoder_calls=1,
                encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
                reader_parameters=sum(t.numel() for t in reader.values()),
                reader_bytes=sum(t.numel()*t.element_size() for t in reader.values()),timings=times))
    out.update(complete=True);paired.save_json(args.out,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);main(a)
