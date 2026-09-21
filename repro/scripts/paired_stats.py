#!/usr/bin/env python
"""Recompute the paired significance statement printed in Section 5.2.

The paper reports, over per-seed values (n = 10 per domain):

  * the joint-vs-STFcal relative-scatter reduction is significant at p < 0.01
    in every scientific domain (Wilcoxon signed-rank; Cohen d_z > 20);
  * battery log-SOH RMSE improves by 0.050 (95% CI [0.033, 0.067], d_z = 1.8);
  * all five primary comparisons survive Holm correction.

Every number there is derived from the per-seed records that the unified
protocol writes into the "raw" list of its output files, so this script reads
those records, pairs them by seed and reproduces the statement end to end.  No
scipy: the exact null distribution of the signed-rank statistic is enumerated
over the 2^n sign patterns, which is exact and cheap for n <= 20.

The four scientific domains are the n = 10 cells.  CelebA is a five-seed
toxicity check and is reported apart from the Holm family, as in the paper.

Usage:
    python repro/scripts/paired_stats.py
    python repro/scripts/paired_stats.py --root /path/to/checkout
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

# domain, path relative to the checkout root, supercritical dose
DOSE_RUNS = [
    ("har",     os.path.join("results", "R3", "R3_dose_har_hi.json"),      100.0),
    ("radioml", os.path.join("results", "R3", "R3_dose_radioml_hi.json"), 3000.0),
    ("battery", os.path.join("results", "R3", "R3_dose_battery_hi.json"),  100.0),
    ("nyu",     os.path.join("results", "R7", "R7_dose_nyu_a1.json"),        1.0),
]
CELEBA_RUN = os.path.join("results", "R6", "R6_dose_celeba_hi.json")

# t_{0.975} for the sample sizes used, for the t-interval sensitivity check
T95 = {5: 2.776, 10: 2.262}


def load_raw(root, rel):
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as fh:
        d = json.load(fh)
    return d.get("raw") if isinstance(d, dict) else None


def by_seed(raw, alpha, mode, field):
    out = {}
    for x in raw or []:
        if not isinstance(x, dict):
            continue
        if x.get("alpha") is None or abs(float(x["alpha"]) - alpha) > 1e-9:
            continue
        if str(x.get("mode")) != mode:
            continue
        v = x.get(field)
        if v is None or v != v:          # NaN
            continue
        out[x.get("seed")] = float(v)
    return out


def wilcoxon_exact(diffs):
    """Two-sided exact Wilcoxon signed-rank p, zeros dropped."""
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return 1.0, 0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    dist = Counter()
    for mask in range(1 << n):
        dist[round(sum(ranks[i] for i in range(n) if mask >> i & 1), 6)] += 1
    w = round(w_plus, 6)
    total = float(1 << n)
    lower = sum(c for x, c in dist.items() if x <= w + 1e-6) / total
    upper = sum(c for x, c in dist.items() if x >= w - 1e-6) / total
    return min(1.0, 2.0 * min(lower, upper)), n


def cohen_dz(diffs):
    n = len(diffs)
    if n < 2:
        return float("nan")
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    return m / sd if sd > 0 else float("inf")


def normal_ci(diffs, z=1.96):
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    half = z * sd / math.sqrt(n)
    return m, m - half, m + half, sd


def holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, pvals[i] * (m - rank)))
        adj[i] = prev
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    ap.add_argument("--field", default="rel_scatter")
    ap.add_argument("--metric", default="metric")
    a = ap.parse_args()

    print("=" * 84)
    print("A. joint vs stfcal, relative scatter, paired by seed")
    print("=" * 84)
    names, pvals, dzs, ns = [], [], [], []
    for dom, rel, alpha in DOSE_RUNS:
        raw = load_raw(a.root, rel)
        if raw is None:
            print("  %-8s MISSING %s" % (dom, rel))
            continue
        x = by_seed(raw, alpha, "joint", a.field)
        y = by_seed(raw, alpha, "stfcal", a.field)
        seeds = sorted(set(x) & set(y), key=lambda v: (v is None, v))
        if len(seeds) < 2:
            print("  %-8s n=%d (insufficient pairs)" % (dom, len(seeds)))
            continue
        diffs = [x[s] - y[s] for s in seeds]
        p, n = wilcoxon_exact(diffs)
        d = cohen_dz(diffs)
        names.append("%s scatter" % dom)
        pvals.append(p)
        dzs.append(d)
        ns.append(n)
        print("  %-8s alpha=%-7s n=%2d  joint=%.4f  stfcal=%.4f  median drop=%+.4f"
              "  p=%.5f  d_z=%+.1f"
              % (dom, alpha, n, sum(x[s] for s in seeds) / n,
                 sum(y[s] for s in seeds) / n,
                 sorted(diffs)[n // 2], p, d))

    print()
    print("=" * 84)
    print("B. battery primary metric (log-SOH RMSE), joint vs stfcal")
    print("=" * 84)
    rel = DOSE_RUNS[2][1]
    raw = load_raw(a.root, rel)
    if raw:
        x = by_seed(raw, 100.0, "joint", a.metric)
        y = by_seed(raw, 100.0, "stfcal", a.metric)
        seeds = sorted(set(x) & set(y), key=lambda v: (v is None, v))
        diffs = [x[s] - y[s] for s in seeds]
        m, lo, hi, sd = normal_ci(diffs)
        p, n = wilcoxon_exact(diffs)
        d = cohen_dz(diffs)
        print("  n=%d  joint=%.4f  stfcal=%.4f" % (n, sum(x[s] for s in seeds) / n,
                                                   sum(y[s] for s in seeds) / n))
        print("  mean improvement %+.4f  95%% CI [%+.4f, %+.4f]  (normal approx;"
              " t interval [%+.4f, %+.4f])  d_z=%+.2f  p=%.5f"
              % (m, lo, hi,
                 m - T95[n] * sd / math.sqrt(n), m + T95[n] * sd / math.sqrt(n), d, p))
        names.append("battery SOH")
        pvals.append(p)
        dzs.append(d)
        ns.append(n)

    print()
    print("=" * 84)
    print("C. Holm correction over the %d primary comparisons" % len(pvals))
    print("=" * 84)
    adj = holm(pvals)
    for nm, p, d, n, hp in zip(names, pvals, dzs, ns, adj):
        print("  %-18s n=%2d  raw p=%.5f  Holm p=%.4f  d_z=%+.2f  %s"
              % (nm, n, p, hp, d, "reject H0" if hp < 0.05 else "retain H0"))
    scatter = [(nm, p, d) for nm, p, d in zip(names, pvals, dzs)
               if nm.endswith("scatter")]
    print()
    print("  all five survive Holm at 0.05     : %s" % all(v < 0.05 for v in adj))
    print("  every scatter comparison p < 0.01 : %s"
          % all(p < 0.01 for _, p, _ in scatter))
    print("  every scatter |d_z| > 20          : %s"
          % all(abs(d) > 20 for _, _, d in scatter))

    print()
    print("=" * 84)
    print("D. CelebA (five seeds; a toxicity check, not in the Holm family)")
    print("=" * 84)
    raw = load_raw(a.root, CELEBA_RUN)
    if raw:
        x = by_seed(raw, 100.0, "joint", a.field)
        y = by_seed(raw, 100.0, "stfcal", a.field)
        seeds = sorted(set(x) & set(y), key=lambda v: (v is None, v))
        diffs = [x[s] - y[s] for s in seeds]
        p, n = wilcoxon_exact(diffs)
        print("  n=%d  joint=%.4f  stfcal=%.4f  p=%.5f  d_z=%+.1f"
              % (n, sum(x[s] for s in seeds) / n, sum(y[s] for s in seeds) / n,
                 p, cohen_dz(diffs)))
        print("  (the smallest attainable two-sided p at n=5 is %.4f)"
              % (2.0 / (1 << n)))
    else:
        print("  MISSING %s" % CELEBA_RUN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
