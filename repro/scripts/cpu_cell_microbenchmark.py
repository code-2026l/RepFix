"""Synthetic CPU scheduling check; no scientific inputs or outcome selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
import paired_readout_continuation as paired
from consumer_cells import cell_spec,cell_logodds_loss
from recovery_readers import RecoveryReader

def main(a):
    out=dict(complete=False,device='cpu',cuda_visibility='',synthetic_only=True,branches=[])
    for domain,d,c in [('har',561,6),('radioml',256,11)]:
        for threads in [1,2,4]:
            torch.set_num_threads(threads);torch.manual_seed(42)
            model=paired.ram.RealAuxModel(d,c,64,'recon' if domain=='har' else 'reg',d if domain=='har' else 1)
            x=torch.randn(256,d);y=torch.randint(c,(256,));aux=x if domain=='har' else torch.randn(256)
            initial=dict(weight=torch.randn(c,64,dtype=torch.float64)*.1,bias=torch.zeros(c,dtype=torch.float64))
            reader=RecoveryReader(64,c,'residual32',torch.zeros(64),torch.ones(64),initial)
            with torch.no_grad():spec=cell_spec(reader(model(x)).double(),[.5,.8])
            parameters=list(model.parameters())+list(reader.parameters());opt=torch.optim.Adam(parameters,lr=.001)
            start=time.perf_counter()
            for step in range(120):
                h=model(x);z=reader(h);loss=F.cross_entropy(model.primary(h),y)+cell_logodds_loss(z,spec)
                loss=loss+(100. if domain=='har' else 1.)*paired.auxiliary_loss(model,h,aux,'recon' if domain=='har' else 'snr')
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5);opt.step()
            out['branches'].append(dict(domain=domain,threads=threads,seconds_per_step=(time.perf_counter()-start)/120))
    out['complete']=True;paired.save_json(a.out,out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_interop_threads(1);main(a)
