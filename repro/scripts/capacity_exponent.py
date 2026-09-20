#!/usr/bin/env python
"""Capacity exponent gamma from alpha sweeps: alpha_c(m) ~ m^gamma.

Rule
----
The paper normalises the representational scatter by the *same seed's*
single-task scatter, so a healthy cell sits at 1.0 and alpha_c is where the
mean relative scatter first crosses **0.5** (log-alpha interpolation between
neighbouring grid points).  This is the rule behind every real-domain exponent
in the paper; the synthetic testbed is the exception (its healthy level is not
1.0 by construction, so its anchor is read off the curve instead).

A width whose smallest alpha still reads below 0.5 never reaches the healthy
branch, and its "crossing" would be an artefact of where the grid starts rather
than a measurement.  Such widths are printed and dropped, not fitted.

Input
-----
Any number of paths, each a sweep JSON or a directory of them.  A sweep file is
read from its ``results`` list, whose rows carry ``domain``, ``m``, ``alpha``
and ``rel_scatter_mean``; per-(domain, width) files (one width each) and
multi-width sweeps (many widths in one file) are both handled, since grouping
is done on the row fields rather than on the file name.

Usage
-----
    python repro/scripts/capacity_exponent.py results/BSCALE
    python repro/scripts/capacity_exponent.py repro/results/xcap2_radioml.json

For each domain this prints the per-width crossing, the max/min movement across
the width range, and the OLS fit of log alpha_c on log m at the four widths of
the main text and at every width present, with R^2 and a 95% interval on the
slope (two-sided t, residual degrees of freedom).

Two other shipped scripts carry the same rule for their own inputs:
``repro/scripts/plot_unified_v4.py`` implements it as ``_across`` for the
figure panels (same absolute level, same guard), and
``repro/scripts/analyze_radioml_cap.py`` applies it to the pooled RadioML
passes, where the guard is a stricter 0.6 margin.  This script is the one that
reproduces the printed exponents from the shipped sweeps.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

# 95% two-sided t quantiles, by residual degrees of freedom.
T95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469,
       7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788,
       13: 2.1604, 14: 2.1448, 15: 2.1314, 16: 2.1199, 17: 2.1098, 18: 2.1009}

LEVEL = 0.5
MAIN_WIDTHS = (32, 64, 128, 256)


def cross(pts, level=LEVEL):
    """First log-alpha crossing of `level`, interpolated between neighbours."""
    for i in range(1, len(pts)):
        a0, s0 = pts[i - 1]
        a1, s1 = pts[i]
        if s0 >= level > s1:
            if a0 <= 0 or a1 <= 0 or s0 == s1:
                return a1
            w = (s0 - level) / (s0 - s1)
            return math.exp(math.log(a0) + w * (math.log(a1) - math.log(a0)))
    return None


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sl = sxy / sxx
    sst = sum((y - my) ** 2 for y in ys)
    ssr = sum((y - (my + sl * (x - mx))) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ssr / sst if sst > 0 else float("nan")
    se = math.sqrt(ssr / (n - 2) / sxx) if n > 2 and ssr > 0 else float("nan")
    return sl, r2, se, n


def ci(g, se, n):
    d = n - 2
    if g is None or se is None or se != se or d not in T95:
        return ""
    h = T95[d] * se
    return "[%+.3f, %+.3f]" % (g - h, g + h)


def files(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "xcap*.json"))))
        else:
            out.extend(sorted(glob.glob(p)))
    return out


def load(paths):
    """(domain, width) -> sorted [(alpha, rel_scatter_mean), ...]"""
    out = {}
    for p in files(paths):
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        rows = j.get("results") or []
        for r in rows:
            dom, m = r.get("domain"), r.get("m")
            rel = r.get("rel_scatter_mean")
            if dom is None or m is None or rel is None:
                continue
            out.setdefault((dom, int(m)), {})[float(r["alpha"])] = float(rel)
    return {k: sorted(v.items()) for k, v in out.items()}


def main(argv):
    paths = argv[1:] or [os.path.join("results", "BSCALE")]
    data = load(paths)
    if not data:
        print("no sweep rows found in: %s" % ", ".join(paths))
        return 1
    print("sources: %s" % ", ".join(files(paths)))

    domains = sorted({d for d, _ in data})
    for dom in domains:
        ws = sorted(w for d, w in data if d == dom)
        print("\n=== %s  (%d widths) ===" % (dom, len(ws)))
        print("  %-6s %-11s %-11s %s" % ("m", "min rel_s", "alpha_c", "note"))
        pairs = []
        for w in ws:
            pts = data[(dom, w)]
            lo = pts[0][1]
            ac = cross(pts)
            note = ""
            if lo < LEVEL:
                note = "GRID DOES NOT REACH HEALTHY BRANCH -- dropped"
            elif ac is None:
                note = "no crossing -- dropped"
            else:
                pairs.append((w, ac))
            print("  %-6d %-11.3f %-11s %s"
                  % (w, lo, ("%.5g" % ac) if ac else "-", note))

        if pairs:
            acs = [a for _, a in pairs]
            print("    %-28s %.4g .. %.4g  (max/min = %.2fx over %dx in m)"
                  % ("crossing range", min(acs), max(acs),
                     max(acs) / min(acs), pairs[-1][0] // pairs[0][0]))

        subsets = [("all widths", pairs)]
        sub4 = [p for p in pairs if p[0] in MAIN_WIDTHS]
        if len(sub4) != len(pairs):
            subsets.append(("main-text 4 widths", sub4))
        for tag, sub in subsets:
            if len(sub) < 3:
                continue
            g, r2, se, n = ols([math.log(m) for m, _ in sub],
                               [math.log(a) for _, a in sub])
            print("    %-28s gamma = %+.4f  R2 = %.4f  95%% CI %s  (n=%d)"
                  % (tag, g, r2, ci(g, se, n), n))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
