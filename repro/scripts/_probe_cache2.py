import pickle, os
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p = os.environ.get("REPFIX_STOCK_CACHE",
                   os.path.join(_ROOT, "data", "stock_data_cache.pkl"))
with open(p, "rb") as f:
    d = pickle.load(f)
k = "301536.SZ"
v = d[k]
for kk, arr in v.items():
    if kk == "date_idx":
        print(f"date_idx type={type(arr).__name__!s}")
        print("   has get/keys:", {kk2 for kk2 in ("get",) if callable(getattr(arr,"get",None))})
        if isinstance(arr, dict):
            ks = list(arr.keys())
            print("   dict n:", len(ks), "sample:", ks[:5])
            print("   one value:", arr[ks[0]] if ks else None)
        continue
    a = np.asarray(arr)
    print(f"{kk:12s} shape={a.shape} dtype={a.dtype} sample={a[:4]}")