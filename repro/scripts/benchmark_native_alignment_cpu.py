"""Fixed native-float32 serving audit; no training, fitting or kernel search."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
import benchmark_joint_alignment_cpu as benchmark
import benefit_compatibility_cpu as bc
import paired_readout_continuation as paired


@torch.no_grad()
def main(args):
    # Only this isolated numerical process changes the diagnostic closure.
    bc.current_logits=lambda model,h:F.linear(h.float(),model.primary.weight,model.primary.bias)
    benchmark.main(args)
    out=json.loads(args.out.read_text())
    out.update(primary_runtime_dtype='float32',legacy_runtime_dtype='float32',
        native_audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_primary_metrics=[])
    for path in sorted(args.records.glob('c???.json')):
        record=json.loads(path.read_text())
        cfg=SimpleNamespace(domain=record['domain'],seed=record['seed'],phase='confirmation',limit_fit=0,limit_eval=0)
        model,data,_,_=bc.load(cfg);directory=path.parent/(path.stem+'_states')
        for row in record['branches']:
            state=torch.load(directory/row['state_file'],map_location='cpu',weights_only=True)
            model.load_state_dict(state['model']);model.eval()
            h=paired.embed(model,data['test'][0]);y=data['test'][1]
            z32=F.linear(h.float(),model.primary.weight,model.primary.bias).double()
            z64=F.linear(h,model.primary.weight.double(),model.primary.bias.double())
            out['native_primary_metrics'].append(dict(domain=record['domain'],seed=record['seed'],mode=row['mode'],
                metrics=paired.metrics(z32,y),maximum_logit_error=float((z32-z64).abs().max()),
                prediction_changes=int((z32.argmax(1)!=z64.argmax(1)).sum())))
    paired.save_json(args.out,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);main(a)
