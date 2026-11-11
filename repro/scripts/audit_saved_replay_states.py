"""Check transferred state digests and every saved floating tensor locally."""
import argparse,hashlib,json
from pathlib import Path
import torch

def floating_tensors(value):
    if isinstance(value,torch.Tensor):
        if value.is_floating_point() or value.is_complex():yield value
    elif isinstance(value,dict):
        for item in value.values():yield from floating_tensors(item)
    elif isinstance(value,(tuple,list)):
        for item in value:yield from floating_tensors(item)

def main(a):
    audit=json.loads(a.audit.read_text());states=tensors=elements=0
    for family,row in audit['queues'].items():
        category,_,name=family.partition('_');base=a.state_root/(category+'_campaign')/name
        for path,sha in row['files_sha256'].items():
            if not path.endswith('.pt'):continue
            source=base/path;assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
            saved=torch.load(source,map_location='cpu',weights_only=True)
            for tensor in floating_tensors(saved):
                assert torch.isfinite(tensor).all(),str(source)
                tensors+=1;elements+=tensor.numel()
            states+=1
    assert states==audit['new_state_files_sha256_verified']
    result=dict(complete=True,device='cpu',all_floating_tensors_finite=True,states=states,floating_tensors=tensors,floating_elements=elements,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),transport_audit_sha256=hashlib.sha256(a.audit.read_bytes()).hexdigest())
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state-root',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);main(a)
