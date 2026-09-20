"""Estimate the capacity-law exponent gamma: alpha_c(m) ~ C * m^gamma.

For each domain and width m, alpha_c is the toxicity control where relative
representational scatter crosses the collapse level (rel_scatter < thresh).
We interpolate log-log across m and fit gamma by ordinary least squares.
Print a LaTeX-ready table.
"""

import os
R = os.environ.get("REPFIX_RESULTS",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "results"))

import json, os
import numpy as np

THRESH = 0.30  # collapse in fraction-of-single-task scatter

def load(name):
    fp = os.path.join(R, name)
    if not os.path.exists(fp):
        return None
    return json.load(open(fp))

def alpha_c_m(rows, thresh):
    """rows: list of {alpha, rel_scatter_mean, m}.  Interpolate alpha_c where
    rel_scatter first drops below thresh (strictly between neighbors)."""
    rs = sorted(rows, key=lambda r: r["alpha"])
    prev = None
    for r in rs:
        s = r.get("rel_scatter_mean")
        if s is None:
            continue
        if prev is not None:
            a0, s0 = prev
            if s < thresh <= s0:
                t = (thresh - s0) / ((s - s0) if (s - s0) != 0 else 1e-9)
                return float(np.exp(np.log(a0) + t * (np.log(r["alpha"]) - np.log(a0))))
        prev = (r["alpha"], s)
    # never crossed (all healthy) or already collapsed everywhere
    rs0 = [r for r in rs if r.get("rel_scatter_mean") is not None]
    if rs0 and rs0[0]["rel_scatter_mean"] < thresh:
        return float(rs0[0]["alpha"])  # collapsed at smallest alpha tested
    return float("nan")

FILEMAP = {
    "har": "xcap_har_fine.json",
    "radioml": "xcap_radioml.json",
    "battery": "xcap_battery_multi.json",
    "battery_old": "xcap_battery.json",
}

def gamma_fit(points):
    pts = [(m, a) for m, a in points if np.isfinite(a) and a > 0 and m > 0]
    if len(pts) < 3:
        return None, pts
    x = np.log(np.array([p[0] for p in pts], dtype=float))
    y = np.log(np.array([p[1] for p in pts], dtype=float))
    # y = gamma*x + log(C)  -> least squares
    A = np.vstack([x, np.ones_like(x)]).T
    gamma, logC = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(gamma), pts

for domain, fname in FILEMAP.items():
    d = load(fname)
    if not d:
        print(f"[{domain}] no data ({fname}); skip")
        continue
    rows = d["results"]
    by_m = {}
    for r in rows:
        m = r.get("m", d.get("m"))
        by_m.setdefault(m, []).append(r)
    points = []
    for m in sorted(by_m):
        ac = alpha_c_m(by_m[m], THRESH)
        points.append((m, ac))
        print(f"  {domain} m={m:5d} alpha_c={ac:.4g}")
    gamma, pts = gamma_fit(points)
    if gamma is not None:
        print(f"  ** {domain}: gamma = {gamma:.3f}  (n_widths={len(pts)})")
    else:
        print(f"  ** {domain}: cannot fit gamma (need >=3 widths)")

# theory target
print("\nTheory prediction (typical AE capacity law): gamma ~ 1.5")