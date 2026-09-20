# -*- coding: utf-8 -*-
"""Summarise the rotated wall-clock matrix behind tab:overhead.

Reads results/TIME/overhead_<domain>_m<width>.json (produced with --rotate) and
prints, per width, the per-row median epoch time, its overhead against plain
summation, the per-repetition CV, the position spread, and the position-paired
estimator.  The last line is a set of LaTeX rows for the table.

Usage:  python repro/scripts/time_table.py <domain> [32,64,128,256]
"""
from __future__ import annotations

import io
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repro/


def _data_dir():
    """results/TIME, falling back to results/TIME2 while a rename is pending."""
    for name in ("TIME", "TIME2"):
        d = os.path.join(ROOT, "results", name)
        if os.path.isdir(d):
            return d
    return os.path.join(ROOT, "results", "TIME")


D = _data_dir()

ROWS = ["joint+joint", "stf0+joint", "stfhard+joint", "stfsoft+joint",
        "stfcal+joint", "joint+pcgrad", "joint+cagrad", "joint+famo"]
LABEL = {
    "joint+joint": "plain summation",
    "stf0+joint": "\\textsc{stf-0} (detach)",
    "stfhard+joint": "\\textsc{stf-hard}",
    "stfsoft+joint": "\\textsc{stf-soft}",
    "stfcal+joint": "\\textsc{stfcal}",
    "joint+pcgrad": "PCGrad",
    "joint+cagrad": "CAGrad",
    "joint+famo": "FAMO",
}


def cell(dom, m):
    p = os.path.join(D, "overhead_%s_m%d.json" % (dom, m))
    return json.load(io.open(p, encoding="utf-8")) if os.path.isfile(p) else None


def med(d, key):
    return st.median([r["median_epoch_s"] for r in d["raw"][key]])


def cv(d, key):
    v = [r["median_epoch_s"] for r in d["raw"][key]]
    return 100.0 * st.pstdev(v) / st.mean(v)


def posspread(d, key):
    """max-min of the per-position mean, as % of the overall median."""
    bypos = {}
    for r in d["raw"][key]:
        bypos.setdefault(r.get("pos"), []).append(r["median_epoch_s"])
    if len(bypos) < 2:
        return None
    means = [st.mean(v) for v in bypos.values()]
    return 100.0 * (max(means) - min(means)) / st.median(means)


def paired(d, key, base_key="joint+joint"):
    """Position-paired overhead (%): inside each rotation position divide the
    mean of `key` by the mean of the reference, then take the median over
    positions.  The Latin square gives every row the same multiset of positions,
    so a ratio of medians is already fair *between* rows; pairing inside a
    position additionally cancels monotone device drift."""
    a, b = {}, {}
    for r in d["raw"][key]:
        a.setdefault(r.get("pos"), []).append(r["median_epoch_s"])
    for r in d["raw"][base_key]:
        b.setdefault(r.get("pos"), []).append(r["median_epoch_s"])
    rs = [st.mean(a[p]) / st.mean(b[p]) for p in sorted(a) if b.get(p)]
    return 100.0 * (st.median(rs) - 1.0) if rs else None


def main():
    dom = sys.argv[1] if len(sys.argv) > 1 else "har"
    ws = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                           else ["32", "64", "128", "256"])]
    data = {}
    for m in ws:
        d = cell(dom, m)
        if d is None:
            print("MISSING %s m=%d (looked in %s)" % (dom, m, D))
            return 1
        data[m] = d
    print("domain=%s  dir=%s  rotate=%s"
          % (dom, os.path.basename(D), [data[m].get("rotate") for m in ws]))
    print("gpu before run: %s" % [data[m]["gpu_before_run"] for m in ws])
    print()
    hdr = "%-22s" % "row" + "".join("%18s" % ("m=%d" % m) for m in ws)
    print(hdr)
    print("%-22s" % "" + "".join("%9s%9s" % ("s/epoch", "overhead") for m in ws))
    base = {m: med(data[m], "joint+joint") for m in ws}
    for k in ROWS:
        line = "%-22s" % LABEL[k].replace("\\textsc{", "").replace("}", "")
        for m in ws:
            line += "%9.4f%8.1f%%" % (med(data[m], k),
                                      100.0 * (med(data[m], k) / base[m] - 1.0))
        print(line)
    print()
    print("per-rep CV of the plain-summation row: %s"
          % ["%.1f%%" % cv(data[m], "joint+joint") for m in ws])
    print("max per-position spread (plain row):    %s"
          % [("%.1f%%" % posspread(data[m], "joint+joint")
              if posspread(data[m], "joint+joint") is not None else "n/a")
             for m in ws])
    print()
    print("position-paired overhead (%): median over rotation positions of "
          "mean(row@pos)/mean(plain@pos)")
    print(hdr)
    for k in ROWS:
        line = "%-22s" % LABEL[k].replace("\\textsc{", "").replace("}", "")
        for m in ws:
            pv = paired(data[m], k)
            line += "%18s" % ("%+.1f%%" % pv if pv is not None else "n/a")
        print(line)
    print()
    stf = [k for k in ROWS if k.startswith("stf")]
    mx = max((100.0 * (med(data[m], k) / base[m] - 1.0), k, m)
             for m in ws for k in stf)
    print("largest STF overhead: %.1f%% (%s at m=%d)" % mx)
    print()
    print("%% ---- LaTeX rows (seconds + overhead per width) ----")
    for k in ROWS:
        cells = ["%.4f & %+.1f\\%%" % (med(data[m], k),
                                       100.0 * (med(data[m], k) / base[m] - 1.0))
                 for m in ws]
        print("%-34s & %s \\\\" % (LABEL[k], " & ".join(cells)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
