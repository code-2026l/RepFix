"""Fixed existing affine-repair hierarchy; isolated anchors, no labels fit maps."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
import torch
import paired_readout_continuation as paired
import late_vision_cpu as late
import nonlinear_suffix_cpu as study
from late_replay import build_backbone
from readout_alignment import ARMS,fit_repairs
from consumer_contracts import evaluate_contracts

@torch.no_grad()
def main(a):
    start=time.time();ref=torch.load(a.reference,map_location='cpu',weights_only=True)
    original=torch.load(a.original_reference,map_location='cpu',weights_only=True)
    assert paired.state_hash(original)==ref['source_reference_sha256']
    images,_,_,data_sha=late.images(True);assert data_sha==original['data_sha256']
    anchors=original['anchors'];assert len(anchors)==2048
    assert not set(anchors.tolist()) & set(original['fitting'].tolist())
    assert not set(anchors.tolist()) & set(original['validation_indices'].tolist())
    oldprefix,oldtail,norm,_=build_backbone(ref['arch'],True)
    oldtail.load_state_dict(original['tail']);norm.load_state_dict(original['norm'])
    target=late.collect(oldprefix,oldtail,norm,images,anchors,ref['metadata']['image_size'])['h']
    del oldprefix,oldtail,norm
    prefix,healthy=study.build_for(ref);healthy.load_state_dict(ref['healthy']);healthy.requires_grad_(False)
    anchor_x=study.collect(prefix,images[anchors],ref['metadata']['image_size']);del prefix,images
    part=torch.load(a.reference.with_name('cache.pt'),map_location='cpu',weights_only=True)['validation']
    reference=late.linear(part['h'],ref['heads']['coarse'])
    out=dict(complete=False,device='cpu',phase='development',arch=ref['arch'],plan_sha256=a.plan_sha256,
        reference_state_sha256=paired.state_hash(ref),anchor_sha256=paired.state_hash(anchors),
        splits=ref['splits'],source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in [Path(__file__).name,'readout_alignment.py',*study.NAMES]},branches=[])
    folder=a.out.parent/(a.out.stem+'_states');folder.mkdir(exist_ok=True,parents=True)
    for record_path in a.records:
        record=json.loads(record_path.read_text());assert record['phase']=='development' and record['complete']
        assert record['reference_state_sha256']==out['reference_state_sha256']
        row=next(b for b in record['branches'] if b['mode']=='U')
        state=torch.load(record_path.parent/(record_path.stem+'_states')/row['state_file'],map_location='cpu',weights_only=True)
        assert paired.state_hash(state)==row['state_sha256']
        current,_,_=study.make_model(ref,healthy,'U',record['seed']);current.load_state_dict(state['update']);current=current.dense().requires_grad_(False)
        anchors_new=study.embed(current,anchor_x);validation_new=study.embed(current,part['x'])
        maps=fit_repairs(target,anchors_new);mu=target.double().mean(0,keepdim=True);sd=target.double().std(0,keepdim=True).clamp_min(1e-6)
        repairs={};fitted={}
        for name in ARMS:
            affine=maps[name];matrix=affine['matrix']*sd;offset=affine['offset']*sd+mu
            weight=ref['heads']['coarse']['weight'].double()@matrix.T
            bias=ref['heads']['coarse']['bias'].double()+offset.squeeze(0)@ref['heads']['coarse']['weight'].double().T
            folded=dict(weight=weight.float(),bias=bias.float())
            if name=='frozen':folded=ref['heads']['coarse']
            output=late.linear(validation_new,folded)
            repairs[name]=dict(paired_anchors=affine['paired_anchors'],contracts=evaluate_contracts(output.double(),reference.double(),part['coarse'],ref['thresholds']))
            fitted[name]=folded
        branch=dict(seed=record['seed'],mode='U',source_state_sha256=row['state_sha256'],repairs=repairs,
            training_record_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest())
        state_path=folder/('seed_%d.pt'%record['seed']);torch.save(dict(readers=fitted,anchor_target=target,anchor_current=anchors_new,
            validation_features=validation_new,reference_logits=reference,coarse_labels=part['coarse']),state_path)
        branch['state_sha256']=hashlib.sha256(state_path.read_bytes()).hexdigest();out['branches'].append(branch)
        paired.save_json(a.out.with_suffix('.partial.json'),out)
        print(json.dumps(dict(seed=record['seed'],maximum_disagreement={n:max(v['disagreement'] for v in r['contracts'].values()) for n,r in repairs.items()})),flush=True)
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--original-reference',type=Path,required=True)
    p.add_argument('--records',type=Path,nargs='+',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(12);torch.set_num_interop_threads(1);main(a)
