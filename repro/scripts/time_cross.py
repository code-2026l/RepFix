# -*- coding: utf-8 -*-
"""Cross-domain check of the rotated wall-clock matrix.

The paper prints one domain (UCI-HAR) at four widths; this confirms the same
ordering holds in every domain, which is what makes that one table
representative rather than a lucky cell.  It also flags any cell whose run was
not rotated, started on a busy device, or drifted within the run.

Usage:  python repro/scripts/time_cross.py [results-dir-name]

The optional argument names the directory under ``repro/results`` to read
(``TIME`` by default); it exists so a reviewer can point the same check at a
fresh re-run without editing the script.
"""
from __future__ import annotations

import io
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repro/
DOMS = ["har", "battery", "radioml", "celeba", "nyu"]
WS = [32, 64, 128, 256]
ROWS = ["joint+joint", "stf0+joint", "stfhard+joint", "stfsoft+joint",
        "stfcal+joint", "joint+pcgrad", "joint+cagrad", "joint+famo"]
SHORT = {"joint+joint": "plain", "stf0+joint": "stf-0", "stfhard+joint": "stf-hard",
         "stfsoft+joint": "stf-soft", "stfcal+joint": "stfcal",
         "joint+pcgrad": "PCGrad", "joint+cagrad": "CAGrad", "joint+famo": "FAMO"}


def _data_dir():
    if len(sys.argv) > 1:
        return os.path.join(ROOT, "results", sys.argv[1])
    for name in ("TIME", "TIME2"):
        d = os.path.join(ROOT, "results", name)
        if os.path.isdir(d):
            return d
    return os.path.join(ROOT, "results", "TIME")


D = _data_dir()


def load(dom, m):
    p = os.path.join(D, "overhead_%s_m%d.json" % (dom, m))
    return json.load(io.open(p, encoding="utf-8")) if os.path.isfile(p) else None


def med(d, key):
    return st.median([r["median_epoch_s"] for r in d["raw"][key]])


def cv(d, key):
    v = [r["median_epoch_s"] for r in d["raw"][key]]
    return 100.0 * st.pstdev(v) / st.mean(v)


def posspread(d, key):
    bypos = {}
    for r in d["raw"][key]:
        bypos.setdefault(r.get("pos"), []).append(r["median_epoch_s"])
    if len(bypos) < 2:
        return None
    means = [st.mean(v) for v in bypos.values()]
    return 100.0 * (max(means) - min(means)) / st.median(means)


def main():
    data, missing = {}, []
    for dom in DOMS:
        for m in WS:
            d = load(dom, m)
            if d is None:
                missing.append("%s_m%d" % (dom, m))
            else:
                data[(dom, m)] = d
    print("dir=%s  loaded %d/%d cells" % (os.path.basename(D), len(data),
                                          len(DOMS) * len(WS)))
    if missing:
        print("MISSING: %s" % ", ".join(missing))
    print()

    print("%-16s %-6s %-7s %-28s %-10s %s"
          % ("cell", "rot", "plainCV", "gpu_before", "posspread", "flags"))
    for dom in DOMS:
        for m in WS:
            d = data.get((dom, m))
            if d is None:
                continue
            flags = []
            if not d.get("rotate"):
                flags.append("NO-ROTATE")
            if (d.get("gpu_before_run") or {}).get("util_pct", 0) > 5:
                flags.append("BUSY-START")
            ps = posspread(d, "joint+joint")
            if ps is not None and ps > 8.0:
                flags.append("POS-DRIFT")
            print("%-16s %-6s %6.1f%% %-28s %-10s %s"
                  % ("%s_m%d" % (dom, m), d.get("rotate"), cv(d, "joint+joint"),
                     str(d.get("gpu_before_run")),
                     "%.1f%%" % ps if ps is not None else "n/a",
                     ",".join(flags) or "ok"))
    print()

    for dom in DOMS:
        have = [m for m in WS if (dom, m) in data]
        if not have:
            continue
        print("=== %s ===" % dom)
        print("%-12s" % "method" + "".join("%11s" % ("m=%d" % m) for m in have))
        print("%-12s" % "plain s/ep" + "".join(
            "%11.4f" % med(data[(dom, m)], "joint+joint") for m in have))
        for k in ROWS[1:]:
            line = "%-12s" % SHORT[k]
            for m in have:
                d = data[(dom, m)]
                line += "%10.1f%%" % (100.0 * (med(d, k) / med(d, "joint+joint") - 1.0))
            print(line)
        print()

    print("=== cross-domain overhead range ===")
    for k in ROWS[1:]:
        vals = [100.0 * (med(data[(dom, m)], k) / med(data[(dom, m)], "joint+joint") - 1.0)
                for dom in DOMS for m in WS if (dom, m) in data]
        if vals:
            print("%-12s min %+7.1f%%  max %+7.1f%%  median %+7.1f%%  (n=%d)"
                  % (SHORT[k], min(vals), max(vals), st.median(vals), len(vals)))
    print()

    bad = []
    for (dom, m), d in data.items():
        for stf in ("stfhard+joint", "stfsoft+joint"):
            for other in ("joint+pcgrad", "joint+cagrad", "joint+famo"):
                if med(d, stf) >= med(d, other):
                    bad.append("%s_m%d: %s(%.4f) >= %s(%.4f)"
                               % (dom, m, SHORT[stf], med(d, stf),
                                  SHORT[other], med(d, other)))
    print("=== claim check: spectral variant cheaper than every surgery baseline ===")
    if bad:
        print("VIOLATIONS (%d):" % len(bad))
        for b in bad:
            print("  " + b)
    else:
        print("holds in every loaded (domain, width) cell")
    return 0


if __name__ == "__main__":
    sys.exit(main())
