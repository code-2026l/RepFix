#!/usr/bin/env python
"""Scan all result JSONs for HAR/RadioML/Battery runs that contain STF modes,
so the MTL-baseline table can be filled with same-protocol numbers."""

import os
R = os.environ.get("REPFIX_RESULTS",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "results"))

import json, glob, os

PATTERNS = [os.path.join(R, '**/*.json')]
files = []
for p in PATTERNS:
    files += glob.glob(p, recursive=True)

def modes_of(d):
    out = []
    for r in (d.get('results') or []):
        if isinstance(r, dict) and 'mode' in r and 'metric_mean' in r:
            out.append((r['mode'], r.get('metric_mean'), r.get('collapse_rate'),
                        r.get('rel_scatter_mean'), r.get('alpha')))
    return out

hits = 0
for f in sorted(files):
    if os.path.basename(f).startswith(('_smoke', 'E1v6_runner_state', 'E2c_', 'E2c_lite', 'E2c_rml', 'E2c_runner', 'E2c_bcap', 'E1v5_runner')):
        pass
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    dom = d.get('domain')
    if dom not in ('har', 'radioml', 'battery'):
        continue
    ms = modes_of(d)
    stf = [m for m in ms if m[0] in ('stf0', 'stfhard', 'stfsoft')]
    if not stf:
        continue
    hits += 1
    print(f"--- {f.replace(os.path.dirname(R) + '/', '')} domain={dom} kind={d.get('kind')}")
    for m in ms:
        print(f"      {m[0]:<8} metric={m[1]} collapse={m[2]} rel={m[3]} alpha={m[4]}")
print('total files with STF modes:', hits)
