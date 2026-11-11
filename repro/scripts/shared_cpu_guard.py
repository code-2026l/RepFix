"""Bound combined explicitly owned numerical queues; never signal other jobs."""
import argparse,hashlib,json,os,re,time
from pathlib import Path

def main(a):
    config=json.loads(a.manifest.read_text());roots=[Path(p).resolve() for p in config['roots']]
    assert all(p.parent==Path.home().resolve() and re.fullmatch('[0-9a-f]{32}',p.name) for p in roots)
    result=dict(complete=False,device='cpu',branches=[],peak_workers=0,peak_aggregate_rss_bytes=0,
        max_workers=config['max_workers'],max_rss_bytes=config['max_rss_bytes'],stop_triggered=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    start=time.time()
    while True:
        workers=memory=0;finished=True
        for root in roots:
            try:q=json.loads((root/'r/queue.json').read_text())
            except (OSError,json.JSONDecodeError):finished=False;continue
            finished &= q['finished']
            for r in q.get('running',{}).values():
                workers+=1
                try:memory+=int(Path('/proc/%d/statm'%r['pid']).read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                except (OSError,IndexError):pass
        result['peak_workers']=max(workers,result['peak_workers'])
        result['peak_aggregate_rss_bytes']=max(memory,result['peak_aggregate_rss_bytes'])
        if workers>config['max_workers'] or memory>config['max_rss_bytes'] or time.time()-start>7500:
            for root in roots:(root/'STOP').touch()
            result['stop_triggered']=True
        if finished:break
        time.sleep(2)
    result.update(complete=True,elapsed_s=time.time()-start)
    a.out.write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
