#!/usr/bin/env python3
"""Parallel job scheduler for the RepFix battery/diagnostics experiment suite.

Runs a queue of independent training jobs with N concurrent slots on the single
A100 (80 GB is ample for many small jobs) and 24 vCPUs.  Each job is a
(script, argv, out_json, log_path) tuple.  The runner:

  * keeps at most --slots jobs alive;
  * retries a failed job up to --retries times;
  * skips a job whose out_json already exists and is non-empty (resume);
  * writes a live status file (jobs_state.json) and a human log (runner.log);
  * enforces a per-job timeout (kill + retry).

Job list is a JSON file:
  [{"name": "...", "script": "/abs/path.py", "args": ["--flag", "..."],
    "out": "/abs/out.json", "log": "/abs/out.log"}, ...]
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import time

PY = 'miniconda3/envs/ml/bin/python'
ENV = dict(os.environ, REPFIX_DATA_X='data_x',
           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
           TOKENIZERS_PARALLELISM='false')


def load_jobs(p):
    with open(p) as f:
        d = json.load(f)
    # accept both a bare array and {"note": ..., "jobs": [...]}
    if isinstance(d, dict):
        d = d.get("jobs", [])
    if not isinstance(d, list):
        raise ValueError("job file must be an array or {'jobs': [...]}: %s" % p)
    return d


def done(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def log_line(logpath, msg):
    line = '[%s] %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(logpath, 'a') as f:
        f.write(line + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True, help='job list JSON')
    ap.add_argument('--slots', type=int, default=8)
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--timeout', type=int, default=6 * 3600)
    ap.add_argument('--log', default='results/runner.log')
    a = ap.parse_args()

    jobs = load_jobs(a.jobs)
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    log_line(a.log, 'runner start: %d jobs, %d slots' % (len(jobs), a.slots))

    state = {j['name']: dict(status='pending', tries=0, t0=None, t1=None)
             for j in jobs}
    state_path = a.log.replace('.log', '_state.json')

    def save_state():
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=1)

    # pending queue
    queue = [j for j in jobs if not done(j['out'])]
    skipped = len(jobs) - len(queue)
    for j in jobs:
        if done(j['out']):
            state[j['name']]['status'] = 'skipped(done)'
    log_line(a.log, 'resume: %d already done, %d to run' % (skipped, len(queue)))

    running = []  # (proc, job, t0)

    while queue or running:
        # launch
        while queue and len(running) < a.slots:
            j = queue.pop(0)
            st = state[j['name']]
            st['tries'] += 1
            st['status'] = 'running'
            st['t0'] = time.time()
            os.makedirs(os.path.dirname(j['log']), exist_ok=True)
            cmd = [PY, '-u', j['script']] + list(j['args'])
            lg = open(j['log'], 'ab')
            p = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT,
                                 env=ENV, preexec_fn=os.setsid)
            running.append((p, j, time.time(), lg))
            log_line(a.log, 'LAUNCH %-42s pid=%d slot=%d/%d queue=%d'
                     % (j['name'], p.pid, len(running), a.slots, len(queue)))
            save_state()

        time.sleep(5)

        # reap
        still = []
        for (p, j, t0, lg) in running:
            rc = p.poll()
            if rc is None:
                if time.time() - t0 > a.timeout:
                    log_line(a.log, 'TIMEOUT %s (>%ds), killing' % (j['name'], a.timeout))
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        pass
                    rc = -9
                else:
                    still.append((p, j, t0, lg))
                    continue
            lg.close()
            st = state[j['name']]
            st['t1'] = time.time()
            dt = st['t1'] - st['t0']
            ok = (rc == 0) and done(j['out'])
            if ok:
                st['status'] = 'done'
                log_line(a.log, 'DONE   %-42s rc=%s %.1fmin' % (j['name'], rc, dt / 60))
            else:
                if st['tries'] <= a.retries:
                    st['status'] = 'retry'
                    queue.append(j)
                    log_line(a.log, 'FAIL   %-42s rc=%s try=%d -> requeue'
                             % (j['name'], rc, st['tries']))
                else:
                    st['status'] = 'failed'
                    log_line(a.log, 'FAIL   %-42s rc=%s giving up' % (j['name'], rc))
            save_state()
        running = still

    n_ok = sum(1 for v in state.values() if v['status'] in ('done', 'skipped(done)'))
    log_line(a.log, 'runner FINISHED: %d/%d ok' % (n_ok, len(jobs)))
    save_state()


if __name__ == '__main__':
    main()
