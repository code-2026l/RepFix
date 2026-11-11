"""Locally check recovered CPU tensors against their transport fingerprints."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
import torch

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def tensors(value):
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,dict):
        for item in value.values():yield from tensors(item)
    elif isinstance(value,(list,tuple)):
        for item in value:yield from tensors(item)

def main(a):
    start=time.time();audit=json.loads(a.transport_audit.read_text());assert audit['complete']
    counts=dict(states=0,tensors=0,floating_tensors=0,floating_elements=0);files={}
    for kind,q in audit['queues'].items():
        for name,entry in q['files'].items():
            if not name.endswith(('.pt','.pth')):continue
            path=a.state_root/kind/name;assert path.is_file() and digest(path)==entry['sha256'],(kind,name)
            value=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            for tensor in tensors(value):
                counts['tensors']+=1
                if tensor.is_floating_point() or tensor.is_complex():
                    flat=tensor.reshape(-1);counts['floating_tensors']+=1;counts['floating_elements']+=tensor.numel()
                    for j in range(0,len(flat),1024**2):assert torch.isfinite(flat[j:j+1024**2]).all(),(kind,name)
            counts['states']+=1;files[kind+'/'+name]=entry['sha256'];del value
        print(json.dumps(dict(queue=kind,states=counts['states'])),flush=True)
    assert counts['states']==audit['verified_state_files']
    result=dict(complete=True,analysis='local',gpu_compute_used=False,all_floating_tensors_finite=True,**counts,
        transport_audit_sha256=digest(a.transport_audit),files_sha256=files,
        source_sha256={Path(__file__).name:digest(Path(__file__))},elapsed_s=time.time()-start)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(counts))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state-root',type=Path,required=True);p.add_argument('--transport-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);main(a)
