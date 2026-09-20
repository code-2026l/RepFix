"""Find the alpha regime + epochs where battery STF beats joint on log-SOH,
and report single-task health ceiling."""
import numpy as np
import cross_domain_mtl as C

for epochs, alpha in [(60, 3000.0), (60, 10000.0), (60, 30000.0), (120, 10000.0), (120, 30000.0)]:
    recs, agg = C.collect_domain('battery', ['single', 'joint', 'stf0', 'stfsoft'],
                                 [0, 1, 2, 3], alpha, 128, epochs, 1e-3, 256, 0.02, 8.0)
    d = {}
    for a in agg:
        d[a['mode']] = (a['metric_mean'], a['collapse_rate'], a['rel_scatter_mean'])
    s = d.get('single', (None,))[0]
    j = d.get('joint', (None,))[0]
    stf0 = d.get('stf0', (None,))[0]
    stfs = d.get('stfsoft', (None,))[0]
    print(f"ep={epochs:3d} a={alpha:6.0f} single={s:.4f} joint={j:.4f} "
          f"stf0={stf0:.4f} stfsoft={stfs:.4f} | coll J/soft={d.get('joint',(0,0))[1]:.1f}/{d.get('stfsoft',(0,0,0))[1]:.1f}")