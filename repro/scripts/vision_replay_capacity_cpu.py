"""Prospective larger-capacity screen; original failed screen is retained."""
import argparse,hashlib
from pathlib import Path
import torch
import vision_shared_replay_cpu as experiment

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--capacity-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.configs=[(64,32),(128,64),(256,128)];a.update_lr=.0001;a.smoke=False
    # The wrapper's source/plan are added to the numerical record after execution.
    torch.set_num_threads(8);torch.set_num_interop_threads(1);experiment.run(a)
    import json
    row=json.loads(a.out.read_text());row['capacity_plan_sha256']=a.capacity_sha256
    row['source_sha256']['vision_replay_capacity_cpu.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    experiment.paired.save_json(a.out,row)
