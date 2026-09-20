#!/usr/bin/env python
"""Dump real-aux (E2d) results: per-mode primary metric, aux metric, collapse, rho."""

import os
R = os.environ.get("REPFIX_RESULTS",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "results"))

import json, glob, os

for f in sorted(glob.glob(os.path.join(R, 'E2d/E2d_*.json'))):
    d = json.load(open(f))
    name = os.path.basename(f)
    recs = d if isinstance(d, list) else d.get('results', d.get('records', []))
    print(f'=== {name} ===')
    if isinstance(recs, dict):
        print('  keys:', list(recs.keys())); continue
    for r in recs:
        mode = r.get('mode', '?')
        mm = r.get('metric_mean'); ms = r.get('metric_std')
        am = r.get('aux_metric_mean'); asd = r.get('aux_metric_std')
        cr = r.get('collapse_rate'); rl = r.get('rel_scatter_mean'); rho = r.get('rho_mean')
        rr = r.get('recovery_rate', None)
        def fmt(v, p=4):
            return ('%.' + str(p) + 'f') % v if isinstance(v, (int, float)) else str(v)
        line = f"  {mode:<12} metric={fmt(mm)}±{fmt(ms)} aux={fmt(am)}±{fmt(asd)} collapse={fmt(cr)} rel={fmt(rl)} rho={fmt(rho)}"
        if rr is not None:
            line += f" recovery={fmt(rr)}"
        print(line)
    # also show any top-level fields worth reporting
    if isinstance(d, dict):
        extra = {k: v for k, v in d.items() if k not in ('results', 'records', 'args', 'config') and not isinstance(v, (list, dict))}
        if extra:
            print('  top:', extra)
