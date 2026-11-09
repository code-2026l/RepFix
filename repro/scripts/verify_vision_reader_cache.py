"""Local replay of saved visual reader inputs; this does not re-encode images."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import paired_readout_continuation as paired
from consumer_risk import certificate, first_certified


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    record=json.loads(args.record.read_text())
    assert record['complete'] and not record['training_complete']
    assert hashlib.sha256(args.cache.read_bytes()).hexdigest()==record['reader_cache_sha256']
    with np.load(args.cache,allow_pickle=False) as cache:
        arrays={key:torch.from_numpy(cache[key]).double() for key in cache.files}
    labels=arrays['validation_labels'].long()
    weight,bias,mean,sd=[arrays['healthy_'+key] for key in ['weight','bias','mean','sd']]
    reference=F.linear((arrays['healthy_validation']-mean)/sd,weight,bias)
    monitor=np.random.default_rng(5900900).choice(len(labels),1024,replace=False)
    result=dict(complete=True,device='cpu',cuda_visibility='',seed=record['seed'],
        record_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        reader_cache_sha256=record['reader_cache_sha256'],encoder_reexecuted=False,
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        posthoc_consumer_audit=True,consumer_family_size=18,delta=.05,monitor_inputs=1024,
        monitor_indices_sha256=hashlib.sha256(monitor.astype(np.int64).tobytes()).hexdigest(),branches=[])
    def compare(actual,expected):
        difference=max(abs(actual[k]-expected[k]) for k in actual)
        assert difference<1e-7,difference
        return difference
    maximum=compare(paired.metrics(reference,labels),record['healthy'])
    for row in record['branches']:
        mode=row['mode'];features=arrays[mode+'_validation']
        frozen=F.linear((features-mean)/sd,weight,bias)
        own=F.linear(features,arrays[mode+'_weight'],arrays[mode+'_bias'])
        refit=F.linear((features-arrays[mode+'_refit_mean'])/arrays[mode+'_refit_sd'],
                      arrays[mode+'_refit_weight'],arrays[mode+'_refit_bias'])
        for key,logits in [('own',own),('frozen',frozen),('supervised_refit',refit)]:
            maximum=max(maximum,compare(paired.metrics(logits,labels),row[key]))
        candidates={}
        def audit(name,logits):
            bounds=certificate(logits.argmax(1).numpy(),reference.argmax(1).numpy(),monitor,.05/18)
            remainder=np.setdiff1d(np.arange(len(labels)),monitor)
            bounds.update(observed_remainder_disagreement=float((logits.argmax(1)[remainder]!=reference.argmax(1)[remainder]).double().mean()),
                observed_remainder_accuracy_difference=float((logits.argmax(1)[remainder]==labels[remainder]).double().mean()
                    -(reference.argmax(1)[remainder]==labels[remainder]).double().mean()))
            candidates[name]=bounds
        audit('native',frozen)
        for budget,maps in row['repairs'].items():
            for method,expected in maps.items():
                prefix=mode+'_'+budget+'_'+method
                z=F.linear(features@arrays[prefix+'_matrix']+arrays[prefix+'_offset'],weight,bias)
                maximum=max(maximum,compare(paired.metrics(z,labels),expected))
                audit(budget+'_'+method,z)
        for method,expected in row['prior_shift_stress'].items():
            prefix=mode+'_prior_shift_'+method
            z=F.linear(features@arrays[prefix+'_matrix']+arrays[prefix+'_offset'],weight,bias)
            maximum=max(maximum,compare(paired.metrics(z,labels),expected))
        order=list(candidates)
        selections={str(budget):first_certified(candidates,budget,order) for budget in [.05,.10]}
        result['branches'].append(dict(mode=mode,native_and_refitted_metrics_replayed=True,
            all_repair_budgets_replayed=True,prior_shift_stress_replayed=True,consumer_candidates=candidates,
            selections=selections))
    result.update(maximum_metric_error=maximum,replayed_healthy_consumers=1,replayed_native_or_refitted_consumers=6,
                  replayed_repair_consumers=20)
    paired.save_json(args.out,result)
    print(json.dumps(dict(maximum_metric_error=maximum,selections=[row['selections'] for row in result['branches']]),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
