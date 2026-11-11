"""Check every locally returned follow-up tensor file, without remote access."""
import argparse,hashlib,json
from pathlib import Path
import torch
from check_frontier_states import tensors,digest


def main(a):
    execution=json.loads(a.execution.read_text());assert execution['complete']
    files={};total=0;floating=0
    for name,expected in execution['locally_recovered_files'].items():
        if not name.endswith('.pt'):continue
        kind,relative=name.split('/',1);path=a.local_root/('conditional_'+kind)/relative
        assert digest(path)==expected['sha256']
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        for tensor in tensors(state):
            total+=1
            if tensor.is_floating_point() or tensor.is_complex():
                floating+=1;assert torch.isfinite(tensor).all(),name
        files[name]=expected['sha256']
    result=dict(complete=True,analysis='local',all_returned_floating_tensors_finite=True,
        states=len(files),tensors=total,floating_tensors=floating,files_sha256=files,
        transport_audit_sha256=digest(a.execution),source_sha256={Path(__file__).name:digest(Path(__file__))})
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(states=len(files),tensors=total,floating_tensors=floating)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execution',type=Path,required=True);p.add_argument('--local-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);main(a)
