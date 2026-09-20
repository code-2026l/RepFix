"""Battery v2: full repair stats (20 seeds) + capacity alpha_c from fine sweep."""
import json, os
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def load(name):
    fp = os.path.join(R, name)
    return json.load(open(fp)) if os.path.exists(fp) else None

rep = load("xrep2_battery.json")
if rep:
    print("=== xrep2_battery (20 seeds) ===")
    for r in rep["results"]:
        print(f"  {r['mode']:8s} RMSE={r['metric_mean']:.4f}+-{r['metric_std']:.4f} "
              f"coll={r['collapse_rate']:.2f} rec={r['recovery_rate']:.2f} "
              f"rel_s={r.get('rel_scatter_mean'):.4f} rho={r.get('rho_mean'):.3e}")
    joint = [r for r in rep["results"] if r["mode"] == "joint"][0]["metric_mean"]
    for r in rep["results"]:
        if r["mode"] != "joint":
            print(f"  {r['mode']}: vs joint {(r['metric_mean']/joint)*100-100:+.1f}%")

# capacity law: alpha_c at rel_scatter=0.3 threshold per width
d = load("xcap2_battery.json")
if d:
    print("\n=== battery alpha_c (rel_scatter=0.3) per width ===")
    raw = d["raw"]
    for m in sorted(set(r["m"] for r in raw)):
        rows = sorted([r for r in raw if r["m"] == m and r.get("rel_scatter") is not None],
                      key=lambda r: r["alpha"])
        # all below 0.3 already at lowest alpha -> alpha_c < 100 (under-resolved)
        v0 = np.mean([r["rel_scatter"] for r in rows if r["alpha"] == rows[0]["alpha"]])
        vN = np.mean([r["rel_scatter"] for r in rows if r["alpha"] == rows[-1]["alpha"]])
        ac = "ERR"
        for i in range(1, len(rows)):
            a1, a2 = rows[i-1]["alpha"], rows[i]["alpha"]
            s1 = np.mean([r["rel_scatter"] for r in rows if r["alpha"]==a1])
            s2 = np.mean([r["rel_scatter"] for r in rows if r["alpha"]==a2])
            if s2 < 0.3 <= s1:
                t = (0.3 - s1)/((s2 - s1) or 1e-9)
                ac = f"{float(np.exp(np.log(a1)+t*(np.log(a2)-np.log(a1)))):.0f}"
                break
        print(f"  m={m:3d} rel_scatter[a0={v0:.3f} aN={vN:.3f}] alpha_c={ac}")