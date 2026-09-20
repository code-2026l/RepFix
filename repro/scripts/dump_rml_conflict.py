#!/usr/bin/env python
"""Dump E2c_radioml_conflict.json (contrast-type): per-method collapse & metric."""

import os
R = os.environ.get("REPFIX_RESULTS",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "results"))

import json

d = json.load(open(os.path.join(R, 'E2c/E2c_radioml_conflict.json')))
recs = d if isinstance(d, list) else d.get('results', d.get('records', []))
if isinstance(recs, dict):
    print('keys:', list(recs.keys()))
else:
    for r in recs:
        print({k: r.get(k) for k in ('method', 'alpha', 'mode', 'collapse_rate', 'metric_mean', 'metric_std', 'rho_mean', 'rel_scatter_mean') if k in r})
if isinstance(d, dict):
    extra = {k: v for k, v in d.items() if k not in ('results', 'records', 'args', 'config') and not isinstance(v, (list, dict))}
    if extra:
        print('top:', extra)
