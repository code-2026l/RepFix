#!/usr/bin/env python
"""Aggregate fin_fold{0..5}_{pcgrad,famo,cagrad}.json -> per-method mean+-std."""

import os
R = os.environ.get("REPFIX_RESULTS",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "results"))

import json, glob, os
import numpy as np
from collections import defaultdict

acc = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(os.path.join(R, 'E2d/fin_fold*.json'))):
    d = json.load(open(f))
    m = d.get('method', os.path.basename(f).split('_')[2] if '_' in f else '?')
    for k in ('oos_ic', 'val_ic', 'sigma_h', 'collapse'):
        if k in d:
            acc[m][k].append(float(d[k]))

for m in sorted(acc):
    print(f'--- {m} (n={len(acc[m]["oos_ic"])})')
    for k in ('oos_ic', 'val_ic', 'sigma_h', 'collapse'):
        v = np.array(acc[m][k])
        print(f'  {k:<8} {v.mean():.4f} +- {v.std(ddof=1) if len(v) > 1 else 0:.4f}  folds={[round(x, 4) for x in v]}')
