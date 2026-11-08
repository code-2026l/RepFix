"""Run a predeclared numerical queue with bounded CPU, memory, and time."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time


def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def mem_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return 0


def rss(pid):
    try:
        return int(Path('/proc/%d/statm' % pid).read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, IndexError, ValueError):
        return 0


def child_limits(cpu_seconds):
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 10))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))


def stop_own_job(process):
    if process.poll() is None and os.getpgid(process.pid) == process.pid:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent.resolve()
    out = root / 'r'
    out.mkdir(exist_ok=True)
    jobs = manifest['jobs']
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Unique job ids are required')
    state = dict(pid=os.getpid(), started=time.time(), pending=[j['id'] for j in jobs],
                 running={}, completed=[], failed=[], finished=False,
                 peak_aggregate_rss_bytes=0, max_workers=manifest['max_workers'])
    running = {}
    pending = list(jobs)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', PYTHONUNBUFFERED='1',
               REPFIX_DATA_X=manifest['data_root'])
    try:
        while pending or running:
            if (root / 'STOP').exists() or time.time() - state['started'] > manifest['campaign_wall_s']:
                raise RuntimeError('Campaign stop condition reached')
            for key, item in list(running.items()):
                process = item['process']
                memory = rss(process.pid)
                item['peak_rss'] = max(memory, item['peak_rss'])
                elapsed = time.time() - item['start']
                reason = None
                if memory > manifest['per_job_rss_bytes']:
                    reason = 'per_job_memory_limit'
                elif elapsed > manifest['per_job_wall_s']:
                    reason = 'per_job_wall_limit'
                if reason:
                    stop_own_job(process)
                code = process.poll()
                if code is not None:
                    item['log'].close()
                    record = dict(id=key, exit_code=code, wall_s=elapsed,
                                  peak_rss_bytes=item['peak_rss'], stop_reason=reason)
                    destination = out / (key + '.json')
                    valid = False
                    if code == 0 and destination.is_file():
                        data = json.loads(destination.read_text())
                        valid = data.get('complete') is True and data.get('device') == 'cpu'
                        valid = valid and all(b.get('matches_primary_only_exactly', True) for b in data['branches'])
                    state['completed' if valid else 'failed'].append(record)
                    del running[key]
            total_rss = sum(rss(v['process'].pid) for v in running.values())
            state['peak_aggregate_rss_bytes'] = max(total_rss, state['peak_aggregate_rss_bytes'])
            if total_rss > manifest['aggregate_rss_bytes'] or mem_available() < 8 * 2**30:
                raise RuntimeError('Aggregate or host memory stop condition reached')
            if len(state['failed']) >= 2:
                raise RuntimeError('Two failed jobs: stop launching and inspect locally')
            enough_space = shutil.disk_usage(root).free > 5 * 2**30
            enough_memory = mem_available() > 24 * 2**30
            while pending and len(running) < manifest['max_workers'] and enough_memory and enough_space:
                job = pending.pop(0)
                identifier = job['id']
                if not identifier.isalnum():
                    raise ValueError('Opaque alphanumeric job ids are required')
                log = (out / (identifier + '.log')).open('wb')
                argv = [manifest['python'], '-u', str(root / 'repro/scripts/paired_readout_continuation.py'),
                        *job['argv'], '--out', str(out / (identifier + '.json'))]
                process = subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                           preexec_fn=lambda: child_limits(manifest['per_job_cpu_s']))
                running[identifier] = dict(process=process, log=log, start=time.time(), peak_rss=0)
            state['pending'] = [j['id'] for j in pending]
            state['running'] = {key: dict(pid=v['process'].pid, wall_s=time.time() - v['start'],
                                          rss_bytes=rss(v['process'].pid), peak_rss_bytes=v['peak_rss'])
                                for key, v in running.items()}
            state['updated'] = time.time()
            atomic_json(out / 'queue.json', state)
            time.sleep(2)
        state['finished'] = True
    except BaseException as error:
        state['error'] = str(error)
        for item in running.values():
            stop_own_job(item['process'])
            item['log'].close()
        state['running'] = {}
        state['finished'] = True
        raise
    finally:
        state['updated'] = time.time()
        state['running'] = {}
        atomic_json(out / 'queue.json', state)


if __name__ == '__main__':
    main()
