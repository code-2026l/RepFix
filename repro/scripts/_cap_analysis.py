import json, numpy as np
from collections import defaultdict
d = json.load(open("../results/th_critical.json"))
by = defaultdict(list)
for r in d["records"]:
    by[r["m"]].append(r)

def alpha_at(m, target=0.0015):
    rows = sorted(by[m], key=lambda r: r["alpha"])
    for i in range(1, len(rows)):
        a0, s0 = rows[i-1]["alpha"], rows[i-1]["sigma_h"]
        a1, s1 = rows[i]["alpha"], rows[i]["sigma_h"]
        if s1 < target <= s0:
            t = (np.log(target) - np.log(s0)) / (np.log(s1) - np.log(s0))
            return float(np.exp(np.log(a0) + t * (np.log(a1) - np.log(a0))))
    return None

ms = sorted(by)
av = []
for m in ms:
    a = alpha_at(m)
    print(f"m={m}: alpha(sigma=0.0015) = {a}")
    if a:
        av.append((m, a))
s1 = {m: round(next(r["sigma_h"] for r in by[m] if abs(r["alpha"]-1.0) < 1e-9), 4) for m in ms}
print("sigma_h at alpha=1.0:", s1)
if len(av) >= 3:
    lx, ly = np.log([m for m,_ in av]), np.log([a for _,a in av])
    g = np.polyfit(lx, ly, 1)
    print("gamma(alpha_c ~ m^g):", round(g[0], 3))