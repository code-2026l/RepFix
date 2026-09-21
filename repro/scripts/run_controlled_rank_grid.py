"""Bounded CPU grid: exact-rank eigenvalue/trace-normalized controls."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import argparse


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--primary-loss',choices=['legacy','logistic'],default='legacy')
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[2]
    folder='CONTROLLED_RANK' if args.primary_loss=='legacy' else 'CONTROLLED_RANK_LOGISTIC'
    out=root/'results'/folder;out.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for m in (32,64,128):
        for r in (4,16):
            for norm in ('eigenvalue','trace'):
                name=f'm{m}_r{r}_{norm}'
                cmd=[sys.executable,'-u','repro/scripts/controlled_rank_scan.py','--m',str(m),'--r',str(r),
                     '--normalization',norm,'--seeds','0,1,2','--epochs','800',
                     '--primary-loss',args.primary_loss,
                     '--alphas','0,0.0003,0.003,0.03,0.3,3,30','--out',f'results/{folder}/{name}.json']
                jobs.append(dict(name=name,command=cmd))
    # Persist portable commands; the executable is supplied by the current environment.
    (out/'manifest.json').write_text(json.dumps([dict(name=j['name'],command=['python',*j['command'][1:]]) for j in jobs],indent=2))
    def run(job):
        target=out/(job['name']+'.json')
        if target.exists() and json.loads(target.read_text()).get('complete'):
            return dict(name=job['name'],state='already_complete')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (out/(job['name']+'.log')).open('w') as log:
            try:
                p=subprocess.run(job['command'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
                return dict(name=job['name'],state='completed' if p.returncode==0 else 'failed',returncode=p.returncode)
            except subprocess.TimeoutExpired:
                return dict(name=job['name'],state='timeout')
    status=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run,jobs):
            status.append(result);(out/'status.json').write_text(json.dumps(status,indent=2));print(result,flush=True)


if __name__=='__main__':main()
