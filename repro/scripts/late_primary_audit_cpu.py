"""Training-label-matched primary refits on all saved late-map pilot states."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,copy,hashlib,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
import paired_readout_continuation as paired
import late_vision_cpu as protocol
from late_replay import build_backbone
from readout_transfer_audit import fit_head
from shared_reference_vision import NormalizedHead

@torch.no_grad()
def features(update,head,mode,part):
    values=[]
    for j in range(0,len(part['h']),256):
        h=part['h'][j:j+256]
        if mode=='N':h=torch.cat([h,F.gelu(head.hidden((h-head.base.mean)/head.base.sd))],1)
        elif update is not None:h=update(part['x'][j:j+256],part['residual'][j:j+256])
        values.append(h)
    return torch.cat(values)

def run(a):
    start=time.time();record=json.loads(a.record.read_text());ref=torch.load(a.reference,map_location='cpu',weights_only=True)
    cached=torch.load(a.reference.with_name('cache.pt'),map_location='cpu',weights_only=True);_,tail,norm,_=build_backbone(ref['arch'],True)
    tail.load_state_dict(ref['tail']);norm.load_state_dict(ref['norm']);folder=a.record.parent/(a.record.stem+'_states')
    out=dict(complete=False,device='cpu',phase='retrospective_development_reader_audit',arch=ref['arch'],plan_sha256=a.plan_sha256,
        source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['late_primary_audit_cpu.py','late_vision_cpu.py','late_replay.py','readout_transfer_audit.py']},
        source_record_sha256=hashlib.sha256(a.record.read_bytes()).hexdigest(),splits=ref['splits'],branches=[])
    destination=a.out.parent/(a.out.stem+'_states');destination.mkdir(exist_ok=True,parents=True)
    for mode in ['F','U','N','r16','r64','r128']:
        raw=torch.load(folder/(mode+'.pt'),map_location='cpu',weights_only=True);update,head,_=protocol.construct(ref,tail,norm,mode,3800);head.load_state_dict(raw['head'])
        if mode=='U':update.tail.load_state_dict(raw['tail'])
        elif mode.startswith('r'):
            state=update.state_dict();state.update(raw['update']);update.load_state_dict(state)
        ht=features(update,head,mode,cached['train']);hv=features(update,head,mode,cached['validation'])
        fit,mean,sd=fit_head(ht.double(),cached['train']['y']);reader=NormalizedHead(fit.weight.detach(),fit.bias.detach(),mean,sd).affine()
        z=protocol.linear(hv,reader);row=dict(mode=mode,own=paired.metrics(z,cached['validation']['y']),reader_sha256=paired.state_hash(reader),state_file=mode+'.pt')
        torch.save(dict(reader=reader),destination/row['state_file']);out['branches'].append(row)
        paired.save_json(a.out.with_suffix('.partial.json'),out);print(json.dumps(dict(arch=ref['arch'],mode=mode,accuracy=row['own']['acc'],elapsed_s=time.time()-start)),flush=True)
    out.update(complete=True,elapsed_s=time.time()-start);paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(12);torch.set_num_interop_threads(1);run(a)
