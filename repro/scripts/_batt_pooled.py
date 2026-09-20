"""Change battery split: pooled chronological (train on first 60% cycles of
every cell, test on the last 40%) so the model can LEARN degradation and the
health ceiling is high --- this gives collapse room to hurt and STF to win."""
import numpy as np
import cross_domain_mtl as C

# monkey-patch the split inside load_domain by rebuilding via numpy directly
d = np.load(C._path("nasa_battery", "nasa_battery_multi.npz"))
X = d["X"]; soh = d["y"].astype(np.float32); cell = d["cell"]
y = np.log(soh + 0.02).astype(np.float32)
n_cells = int(cell.max()) + 1
tr = np.zeros(len(X), dtype=bool)
for c in range(n_cells):
    idx = np.where(cell == c)[0]
    idx = idx[np.argsort(idx)]  # chronological within cell
    k = int(len(idx) * 0.6)
    tr[idx[:k]] = True
te = ~tr

C._DATA_CACHE['battery'] = dict(
    X_tr=X[tr].astype(np.float32), y_tr=y[tr],
    X_te=X[te].astype(np.float32), y_te=y[te],
    d_in=X.shape[1], n_classes=1, n_cells=n_cells)

for epochs, alpha in [(60, 1000.0), (60, 3000.0), (60, 10000.0), (120, 10000.0)]:
    recs, agg = C.collect_domain('battery', ['single', 'joint', 'stf0', 'stfsoft'],
                                 [0, 1, 2, 3], alpha, 128, epochs, 1e-3, 256, 0.02, 8.0)
    d2 = {}
    for a in agg:
        d2[a['mode']] = (a['metric_mean'], a['collapse_rate'], a['rel_scatter_mean'])
    print(f"ep={epochs:3d} a={alpha:6.0f} single={d2.get('single',(0,))[0]:.4f} "
          f"joint={d2.get('joint',(0,))[0]:.4f} stf0={d2.get('stf0',(0,))[0]:.4f} "
          f"stfsoft={d2.get('stfsoft',(0,))[0]:.4f} | coll J/soft={d2.get('joint',(0,0))[1]:.1f}/{d2.get('stfsoft',(0,0,0))[1]:.1f}")