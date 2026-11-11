"""Finalize the interrupted-by-divergence probe without repeating any training.

Numerical controls and a compact reader cache are recovered from saved tensors.
The original incomplete record and failed branch remain explicitly incomplete.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch
from torch import nn

import paired_readout_continuation as paired
from vision_chart_cpu import repair_metrics


def run(args):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    partial=json.loads(args.record.read_text())
    assert not partial['complete'] and [b['mode'] for b in partial['branches']]==['D','J1']
    assert 'FloatingPointError: Non-finite vision continuation loss' in args.failure_log.read_text()
    states=args.record.parent/(args.record.name.split('.')[0]+'_states')
    reference=torch.load(states/'reference.pt',map_location='cpu',weights_only=True)
    assert paired.state_hash(reference['encoder'])==partial['reference_encoder_sha256']
    source=Path(os.environ['REPFIX_DATA_X'])/'v037/cifar-100-python/train'
    with source.open('rb') as stream:training=pickle.load(stream,encoding='bytes')
    labels=torch.tensor(training[b'fine_labels'],dtype=torch.long)
    val_labels=labels[reference['validation']]
    head=nn.Linear(512,100).double();head.load_state_dict(reference['head']);head.eval()
    mean,sd=reference['mean'],reference['sd']
    healthy_val=reference['healthy_validation_features'];healthy_anchor=reference['healthy_anchor_features']
    healthy_metrics=paired.metrics(head((healthy_val-mean)/sd),val_labels)
    for key in healthy_metrics:
        assert abs(healthy_metrics[key]-partial['healthy'][key])<1e-10
    arrays=dict(healthy_validation=healthy_val.float().numpy(),healthy_anchor=healthy_anchor.float().numpy(),
        validation_labels=val_labels.numpy(),anchor_labels=labels[reference['anchors']].numpy(),
        healthy_weight=head.weight.detach().numpy(),healthy_bias=head.bias.detach().numpy(),
        healthy_mean=mean.numpy(),healthy_sd=sd.numpy())
    for row in partial['branches']:
        mode=row['mode']
        saved=torch.load(states/(mode+'.pt'),map_location='cpu',weights_only=True)
        assert paired.state_hash(saved['encoder'])==row['final_encoder_sha256']
        for key,source_key in [('validation','validation_features'),('anchor','anchor_features')]:
            features=saved[source_key]
            assert torch.equal(features.float().double(),features)
            arrays[mode+'_'+key]=features.float().numpy()
        arrays[mode+'_weight']=saved['primary']['weight'].numpy()
        arrays[mode+'_bias']=saved['primary']['bias'].numpy()
        arrays[mode+'_refit_weight']=saved['refit']['weight'].numpy()
        arrays[mode+'_refit_bias']=saved['refit']['bias'].numpy()
        arrays[mode+'_refit_mean']=saved['refit_mean'].numpy()
        arrays[mode+'_refit_sd']=saved['refit_sd'].numpy()
        for budget,maps in saved['maps'].items():
            for method,affine in maps.items():
                arrays[mode+'_'+budget+'_'+method+'_matrix']=affine['matrix'].numpy()
                arrays[mode+'_'+budget+'_'+method+'_offset']=affine['offset'].numpy()
    scale=.25
    control,_=repair_metrics(healthy_anchor,healthy_anchor*scale,healthy_val*scale,val_labels,head,mean,sd)
    scale_control=dict(scale=scale,artificial=True,
        frozen=paired.metrics(head((healthy_val*scale-mean)/sd),val_labels),repairs=control)
    center=healthy_anchor.mean(0,keepdim=True)
    cov=(healthy_anchor-center).T@(healthy_anchor-center)/(len(healthy_anchor)-1)
    values,vectors=torch.linalg.eigh(cov)
    values=values.clamp_min(1e-8*values.max())
    white=(vectors*values.rsqrt())@vectors.T;color=(vectors*values.sqrt())@vectors.T
    rotation=torch.linalg.qr(torch.randn(512,512,generator=torch.Generator().manual_seed(partial['seed']+41000),dtype=torch.float64)).Q
    matrix=white@rotation@color
    shifted_anchor=(healthy_anchor-center)@matrix+center
    shifted_val=(healthy_val-center)@matrix+center
    controls,_=repair_metrics(healthy_anchor,shifted_anchor,shifted_val,val_labels,head,mean,sd)
    new_cov=(shifted_anchor-center).T@(shifted_anchor-center)/(len(shifted_anchor)-1)
    rotation_control=dict(artificial=True,relative_anchor_covariance_error=float((new_cov-cov).norm()/cov.norm()),
        frozen=paired.metrics(head((shifted_val-mean)/sd),val_labels),repairs=controls,
        known_inverse=paired.metrics(head((((shifted_val-center)@torch.linalg.inv(matrix)+center)-mean)/sd),val_labels))
    cache=args.out.with_suffix('.npz')
    np.savez_compressed(cache,**arrays)
    result=dict(partial)
    result.update(protocol='finalization of fixed visual probe with a preserved divergent branch',
        complete=True,training_complete=False,completed_endpoint_modes=['D','J1'],
        numerical_failure=dict(mode='J10',kind='non_finite_continuation_loss',endpoint_scored=False,
            successful_epochs=0,retuned=False,retrained=False),
        original_partial_sha256=hashlib.sha256(args.record.read_bytes()).hexdigest(),
        finalizer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        imposed_scale_control=scale_control,covariance_rotation_control=rotation_control,
        reader_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        reader_cache_bytes=cache.stat().st_size,
        saved_state_file_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(states.glob('*.pt'))})
    paired.save_json(args.out,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--failure-log',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
