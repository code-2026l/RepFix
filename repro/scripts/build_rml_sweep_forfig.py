#!/usr/bin/env python
"""Build repro/results/xcap2_radioml.json (plot schema) from cap + cap2 + cap3.

plot_unified_v2.py's capacity panel needs, per domain, a sweep file with
`results` = [{m, alpha, rel_scatter_mean, rel_scatter_std}] and optionally
`raw` = [{m, alpha, rel_scatter}].  The RadioML sweep was run in three passes
covering complementary alpha ranges --- cap: 1..1000; cap2: 0.003..1; cap3:
0.1..0.8, densified about the transition with five seeds.  No single pass spans
both the healthy branch and the collapsed floor, so the half-height rule of
Section~\\ref{sec:capacity_law} cannot be applied to any one of them; pooling
is what makes the curve continuous.  Read at the absolute half-height 0.5, the
pooled curves then cross at alpha_c = {0.17, 0.39, 0.24, 0.22} for
m = {32, 64, 128, 256}, which is what Appendix~D reports.

Precedence for duplicate (m, alpha): cap3 > cap2 > cap.  Written to
xcap2_radioml.json so it takes precedence over the legacy xcap_radioml.json
in plot_unified_v2.SWEEP.

The three source files per width live in repro/results/E2c/ (E2c_radioml_{cap,
cap2,cap3}_m<width>.json); the `REPFIX_RESULTS` environment variable overrides
the results root.  The crossing and the exponent are computed by
repro/scripts/capacity_exponent.py.
"""

import os
R = os.environ.get("REPFIX_RESULTS",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "results"))

import glob, json, os

BASE = R
OUT = os.path.join(BASE, "xcap2_radioml.json")
TAGS = ["cap3", "cap2", "cap"]          # highest precedence first


def load(p):
    d = json.load(open(p))
    return d.get("results", []), d.get("raw", [])


def main():
    seen = {}                            # (m, alpha) -> result row
    raw = {}                             # (m, alpha) -> [raw rows]
    for tag in TAGS:
        for f in sorted(glob.glob(os.path.join(BASE, "E2c", f"E2c_radioml_{tag}_m*.json"))):
            res, raws = load(f)
            for r in res:
                key = (int(r["m"]), float(r["alpha"]))
                if key not in seen:      # higher-precedence tag wins
                    seen[key] = r
            for r in raws:
                raw.setdefault((int(r["m"]), float(r["alpha"])), []).append(r)
    results = [seen[k] for k in sorted(seen)]
    raws = [x for k in sorted(raw) for x in raw[k]]
    ms = sorted(set(k[0] for k in seen))
    out = dict(domain="radioml", kind="sweep", results=results, raw=raws,
               note="pooled cap+cap2+cap3 (precedence cap3>cap2>cap)")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}: {len(results)} result points, {len(raws)} raw rows, "
          f"widths={ms}, n_alpha_per_width="
          f"{ {m: len([k for k in seen if k[0]==m]) for m in ms} }")


if __name__ == "__main__":
    main()
