import json, os, numpy as np

# Capacity sweeps are read from the repository's own results tree.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F = os.environ.get("REPFIX_RESULTS", os.path.join(_ROOT, "repro", "results"))

SWEEP = {"har": ("xcap2_har_fine.json", "xcap_har_fine.json"),
         "radioml": ("xcap2_radioml.json", "xcap_radioml.json"),
         "battery": ("xcap2_battery.json", "xcap_battery_multi.json")}


def load_first(names):
    for n in names:
        p = os.path.join(F, n)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


def across(rows, m):
    rs = sorted([r for r in rows if r["m"] == m and r.get("rel_scatter_mean") is not None],
                key=lambda r: r["alpha"])
    if len(rs) < 2:
        return None
    health = rs[0]["rel_scatter_mean"]
    if health < 0.6:
        return None
    half = health / 2.0
    if rs[-1]["rel_scatter_mean"] >= half:
        return None
    for i in range(1, len(rs)):
        if rs[i]["rel_scatter_mean"] < half:
            a0, a1 = rs[i - 1], rs[i]
            l0, l1 = np.log(a0["alpha"]), np.log(a1["alpha"])
            y0 = np.log(a0["rel_scatter_mean"])
            y1 = np.log(a1["rel_scatter_mean"])
            t = (np.log(half) - y0) / (y1 - y0)
            return float(np.exp(l0 + t * (l1 - l0)))
    return None


for dm in SWEEP:
    d = load_first(SWEEP[dm])
    if not d:
        print(dm, "NO DATA")
        continue
    rows = d["results"]
    ms = sorted(set(r["m"] for r in rows))
    pts = [(m, across(rows, m)) for m in ms]
    pts = [(m, a) for m, a in pts if a is not None and a > 0]
    if len(pts) < 3:
        print(dm, "TOO FEW", pts)
        continue
    ms_ = [p[0] for p in pts]
    ac = [p[1] for p in pts]
    g = np.polyfit(np.log(ms_), np.log(ac), 1)
    print(f"{dm:8s} gamma={g[0]:.4f}  m-range=({min(ms_)},{max(ms_)}) "
          f"ac-range=({min(ac):.4g},{max(ac):.4g})  endpoints: "
          f"first=({ms_[0]},{ac[0]:.4g}) last=({ms_[-1]},{ac[-1]:.4g})")
