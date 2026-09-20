"""Analyze xcap2_battery per-seed data for saddle-node phase-coexistence evidence.
The saddle-node collapse transition implies that near alpha_c, seeds split into
TWO clusters: a healthy cluster (rel_scatter ~ 1) and a collapsed cluster
(rel_scatter ~ 0).  This bi-modality -- NOT a smooth interpolation -- is the
trademark fingerprint we want to put in the paper."""
import json, os
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def load(name):
    fp = os.path.join(R, name)
    return json.load(open(fp)) if os.path.exists(fp) else None

d = load("xcap2_battery.json")
print("kind", d.get("kind"), "domain", d.get("domain"), "n_raw",
      len(d.get("raw", [])), "n_agg", len(d.get("results", [])))

raw = d["raw"]
ms = sorted(set(r["m"] for r in raw))
print("widths:", ms)

for m in ms:
    rows = [r for r in raw if r["m"] == m and r.get("rel_scatter") is not None]
    bya = {}
    for r in rows:
        bya.setdefault(r["alpha"], []).append(r["rel_scatter"])
    print(f"\n=== m={m} ===")
    for a in sorted(bya):
        v = np.array(bya[a])
        lo, hi = v.min(), v.max()
        # bi-modality: max gap between the two extreme clusters
        gap = hi - lo
        # separation index: how far apart the min cluster and max cluster are,
        # relative to their scale
        bipart = (hi - lo) / max(float(v.mean()), 1e-9)
        n_coll = int((v < 0.3).sum())
        print(f"  alpha={a:9.0f} n={len(v):2d} scatter[min={lo:.3f} mean={v.mean():.3f} max={hi:.3f}] "
              f"gap={gap:.3f} bi={bipart:.2f} n_coll_threadal0.3={n_coll}")