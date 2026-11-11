"""Let owned numerical children finish under a host-memory reserve guard.

Pause only their registered queue supervisor, preserve its original ledger, and
resume it with STOP after those children finish. Numerical programs are unchanged.
"""
import argparse,hashlib,json,os,signal,time
from pathlib import Path

def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def available():
    return next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))

def status(pid):
    try:
        fields=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()
        rss=int(Path('/proc/%d/statm'%pid).read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        return fields[0],int(fields[1]),rss
    except (FileNotFoundError,ProcessLookupError):return None,None,0

def main(a):
    root=a.campaign_root.resolve();queue=root/'r/queue.json';assert len(root.name)==32 and root.name.isalnum()
    original=json.loads(queue.read_text());parent=original['pid']
    command=Path('/proc/%d/cmdline'%parent).read_bytes().replace(b'\0',b' ').decode()
    assert str(root) in command and 'bounded_cpu_runner.py' in command and not original['finished']
    atomic(root/'r/resource_handover_before.json',original)
    out=dict(complete=False,device='cpu',operation='resource supervisor handover; numerical programs unchanged',
        resource_plan_sha256=a.plan_sha256,source_sha256={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        reserve_bytes=a.reserve_gib*2**30,parent_pid=parent,peak_child_rss_bytes=0,minimum_available_bytes=available(),adopted_completed=[],branches=[])
    children={};paused=False;start=time.monotonic()
    try:
        os.kill(parent,signal.SIGSTOP);paused=True
        for _ in range(100):
            if status(parent)[0] in ['T','t']:break
            time.sleep(.02)
        else:raise RuntimeError('Queue supervisor did not pause')
        ids={v['pid']:k for k,v in original['running'].items()}
        for pid in map(int,Path('/proc/%d/task/%d/children'%(parent,parent)).read_text().split()):
            state,ppid,rss=status(pid);assert ppid==parent and os.getpgid(pid)==pid
            args=Path('/proc/%d/cmdline'%pid).read_bytes().split(b'\0')
            if state!='Z':
                values=[x.decode() for x in args if x];assert str(root) in ' '.join(values) and 'benchmark_suffix_cpu.py' in ' '.join(values)
                path=Path(values[values.index('--out')+1]).resolve();assert path.parent==root/'r';identifier=path.stem
            else:identifier=ids[pid]
            children[pid]=dict(id=identifier,peak_rss_bytes=rss)
        while True:
            live=[];free=available();out['minimum_available_bytes']=min(out['minimum_available_bytes'],free)
            for pid,row in children.items():
                state,ppid,rss=status(pid);row['peak_rss_bytes']=max(row['peak_rss_bytes'],rss)
                if state not in [None,'Z']:live.append(pid)
            out['peak_child_rss_bytes']=max(out['peak_child_rss_bytes'],sum(status(pid)[2] for pid in live))
            if not live:break
            if free<a.reserve_gib*2**30 or time.monotonic()-start>1800:raise RuntimeError('Host reserve or handover deadline reached')
            time.sleep(2)
        for pid,row in children.items():
            path=root/'r'/(row['id']+'.json');data=json.loads(path.read_text())
            assert data['complete'] and data['device']=='cpu'
            row['record_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();out['adopted_completed'].append(row)
        out.update(complete=True,elapsed_s=time.monotonic()-start)
    finally:
        (root/'STOP').write_text('Supervisor handover: retain completed numerical outputs; requeue pending jobs with host-reserve protection.\n')
        if paused:
            try:os.kill(parent,signal.SIGCONT)
            except ProcessLookupError:pass
        atomic(a.out,out)
    print(json.dumps(dict(complete=out['complete'],retained=[x['id'] for x in out['adopted_completed']],minimum_available_bytes=out['minimum_available_bytes'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--reserve-gib',type=int,default=256)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
