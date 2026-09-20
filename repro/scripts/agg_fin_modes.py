#!/usr/bin/env python
"""Aggregate fin_fold{0..5}_{stfoff,detach1,stfhard,stfsoft}.json per mode."""

import os
R = os.environ.get("REPFIX_RESULTS",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "results"))

import json, glob, os
import numpy as np
from collections import defaultdict

acc = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(os.path.join(R, 'fin_fold*.json'))):
    if 'E2d' in f:
        continue
    d = json.load(open(f))
    b = os.path.basename(f).replace('.json', '')
    mode = b.split('_', 1)[1]
    r = d[0] if isinstance(d, list) else d
    for k in ('oos_ic', 'val_ic', 'sigma_h', 'collapse', 'aux_val'):
        if k in r and r[k] == r[k]:  # skip NaN
            acc[mode][k].append(float(r[k]))

for m in sorted(acc):
    print(f'--- {m} (n={len(acc[m]["oos_ic"])})')
    for k in ('oos_ic', 'sigma_h', 'collapse', 'aux_val'):
        if k in acc[m]:
            v = np.array(acc[m][k])
            print(f'  {k:<9} {v.mean():+.4f} +- {v.std(ddof=1) if len(v) > 1 else 0:.4f}  folds={[round(x, 3) for x in v]}')
