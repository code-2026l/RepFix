#!/usr/bin/env python
"""What the calibration read depends on: the warm-up length.

The one-probe calibration reads rho-hat ONCE, on a detached healthy anchor,
after a fixed number of primary-only epochs (--cal-warm-epochs, four in every
shipped run).  This script measures what that read is worth.

Part 1 -- the read is a trajectory quantity (``results/WARM``)
--------------------------------------------------------------
``warm_<domain>_m<width>_t<tau>.json`` is the shipped calibration recipe
(stfcal, alpha = 1e-9, 40 epochs, ten seeds, widths 32..512) with ONLY the
warm-up length varied over tau in {1,2,4,8,16,32}.  The tau = 4 column is the
shipped configuration and reproduces ``results/R3`` bit for bit, which is the
control on the sweep.

Part 2 -- the warm-up sets the gate's floor (``results/WARMU``)
---------------------------------------------------------------
``warmu_har_m64_a<alpha>_t<tau>.json`` runs the shipped HAR repair recipe
(joint,stf0,stfcal at m = 64, 60 epochs, ten seeds) with the same sweep of tau.
The calibrated gate is alpha * rho-hat >= 1, so a warm-up that reads a smaller
rho-hat raises the dose at which the gate opens.  At alpha = 0.03, 3.3x the
measured transition, tau = 1 and 2 leave it shut in every seed and the run
collapses exactly as unfiltered joint does; at alpha = 100 the gate opens for
any plausible read and the repair is tau-independent.

Usage
-----
    python repro/scripts/warmup_sensitivity.py
    python repro/scripts/warmup_sensitivity.py --warm 'results/WARM/warm_har_*'
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys

T95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469,
       7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788}

WARM = "results/WARM/warm_*.json"
WARMU = "results/WARMU/warmu_*.json"
SWEEP = re.compile(r"warm_([a-z0-9]+)_m(\d+)_t(\d+)\.json$")
REPAIR = re.compile(r"warmu_har_m(\d+)_a([0-9p]+)_t(\d+)\.json$")


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
    if se is None or se != se or d not in T95:
        return ""
    h = T95[d] * se
    return "[%+.3f,%+.3f]" % (g - h, g + h)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def read_probe(j):
    """mean cal_rho_u over the seeds that carry one"""
    return [float(r["cal_rho_u"]) for r in (j.get("raw") or [])
            if r.get("cal_rho_u") is not None and r["cal_rho_u"] == r["cal_rho_u"]
            and float(r["cal_rho_u"]) > 0]


def part1(pattern):
    cells = {}
    for p in sorted(glob.glob(pattern)):
        m = SWEEP.search(os.path.basename(p))
        if not m:
            continue
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        v = read_probe(j)
        if v:
            cells[(m.group(1), int(m.group(3)), int(m.group(2)))] = (1.0 / mean(v),
                                                                     mean(v))
    if not cells:
        print("part 1: no rows under %s" % pattern)
        return
    doms = sorted({d for d, _, _ in cells})
    taus = sorted({t for _, t, _ in cells})

    print("=" * 90)
    print("PART 1  the probe read is a trajectory quantity")
    print("=" * 90)
    for dom in doms:
        ws = sorted({w for d, _, w in cells if d == dom})
        print("\n%s -- rho-hat (mean over seeds) by warm-up length" % dom)
        print("  %-5s %s" % ("tau", "".join("  m%-9d" % w for w in ws)))
        for t in taus:
            row = "".join("  %9.3g" % cells[(dom, t, w)][1] if (dom, t, w) in cells
                          else "  %9s" % "-" for w in ws)
            print("  %-5d %s" % (t, row))

    print("\n%s" % ("-" * 90))
    print("probe exponent gamma_probe(tau) = slope of log(1/rho-hat) on log(m)")
    print("%s" % ("-" * 90))
    for dom in doms:
        ws = sorted({w for d, _, w in cells if d == dom})
        print("\n%s" % dom)
        print("  %-5s %-10s %-7s %-21s %s"
              % ("tau", "gamma", "R2", "95% CI", "alpha_c = 1/rho-hat per width"))
        for t in taus:
            pts = [(w, cells[(dom, t, w)][0]) for w in ws if (dom, t, w) in cells]
            if len(pts) < 3:
                continue
            g, r2, se, n = ols([math.log(w) for w, _ in pts],
                               [math.log(a) for _, a in pts])
            print("  %-5d %-10s %-7.3f %-21s %s"
                  % (t, "%+.3f" % g, r2, ci(g, se, n),
                     " ".join("%.3g" % a for _, a in pts)))
        gs = [(t, ols([math.log(w) for w, _ in
                       [(w, cells[(dom, t, w)][0]) for w in ws if (dom, t, w) in cells]],
                      [math.log(cells[(dom, t, w)][0]) for w in ws
                       if (dom, t, w) in cells])[0])
              for t in taus if sum((dom, t, w) in cells for w in ws) >= 3]
        if len(gs) >= 2:
            v = [g for _, g in gs]
            print("  -> spans %+.3f to %+.3f over tau (range %.3f)%s"
                  % (min(v), max(v), max(v) - min(v),
                     ", SIGN CHANGES" if min(v) < 0 < max(v) else ""))


def part2(pattern):
    rows = {}
    for p in sorted(glob.glob(pattern)):
        m = REPAIR.search(os.path.basename(p))
        if not m:
            continue
        alpha = float(m.group(2).replace("p", "."))
        tau = int(m.group(3))
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        sc = [r for r in j["raw"] if r.get("mode") == "stfcal"]
        if not sc:
            continue
        res = {r["mode"]: r for r in j["results"]}
        rows[(alpha, tau)] = (
            mean([float(r["cal_rho_u"]) for r in sc]),
            mean([float(r["cal_gate"]) for r in sc]),
            mean([float(r["cal_rank"]) for r in sc]),
            res.get("stfcal", {}), res.get("joint", {}), res.get("stf0", {}))
    if not rows:
        print("part 2: no rows under %s" % pattern)
        return

    print("\n" + "=" * 90)
    print("PART 2  the warm-up length sets the gate's sensitivity floor 1/rho-hat")
    print("=" * 90)
    for alpha in sorted({a for a, _ in rows}):
        print("\nalpha = %g" % alpha)
        print("  %-5s %-11s %-9s %-8s %-7s %-24s %s"
              % ("tau", "rho-hat", "floor", "gate", "rank", "stfcal met/coll/rel",
                 "joint met/coll/rel"))
        for tau in sorted({t for a, t in rows if a == alpha}):
            rho, gate, rank, sc, jo, s0 = rows[(alpha, tau)]
            print("  %-5d %-11.4g %-9.4g %-8.2f %-7.2f %-24s %s"
                  % (tau, rho, 1.0 / rho, gate, rank,
                     "%.3f / %.2f / %.2f" % (sc.get("metric_mean", float("nan")),
                                             sc.get("collapse_rate", float("nan")),
                                             sc.get("rel_scatter_mean", float("nan"))),
                     "%.3f / %.2f / %.2f" % (jo.get("metric_mean", float("nan")),
                                             jo.get("collapse_rate", float("nan")),
                                             jo.get("rel_scatter_mean", float("nan")))))
        shut = [t for t in sorted({t for a, t in rows if a == alpha})
                if rows[(alpha, t)][1] < 0.5]
        if shut:
            print("  -> gate shut in every seed at tau = %s; there stfcal reduces to"
                  " unfiltered joint" % shut)
        else:
            print("  -> gate open in every seed at every tau")


def main(argv):
    w = argv[argv.index("--warm") + 1] if "--warm" in argv else WARM
    u = argv[argv.index("--warmu") + 1] if "--warmu" in argv else WARMU
    part1(w)
    part2(u)


if __name__ == "__main__":
    main(sys.argv)
