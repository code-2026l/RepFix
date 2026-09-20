import pickle, sys, os
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p = os.environ.get("REPFIX_STOCK_CACHE",
                   os.path.join(_ROOT, "data", "stock_data_cache.pkl"))
with open(p, "rb") as f:
    d = pickle.load(f)
print("type:", type(d))
if isinstance(d, dict):
    ks = list(d.keys())
    print("n_keys:", len(ks))
    print("keys_sample:", ks[:12])
    shapes = {}
    for k in ks[:6]:
        v = d[k]
        s = getattr(v, "shape", None)
        tt = getattr(v, "dtype", type(v))
        if isinstance(v, dict):
            print(" subdict", k, list(v.keys())[:8])
            continue
        print(" key", k, "type", type(v).__name__, "shape", s, "dtype", tt)
elif isinstance(d, (list, tuple)):
    print("len:", len(d))
    print("first:", d[0])