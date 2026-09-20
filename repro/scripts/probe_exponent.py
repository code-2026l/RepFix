#!/usr/bin/env python
"""Probe exponent gamma from the one-probe calibration: alpha_c = 1/rho_hat.

This is the *second* estimator of alpha_c, and it is not the one the paper's
headline capacity law uses.  The law of Section 5 reads alpha_c where the
trained relative scatter crosses 0.5 (``capacity_exponent.py``); the probe reads
rho_hat once, on the healthy anchor, and takes alpha_c = 1/rho_hat.  The two
place the transition within a factor of a few of each other at a fixed width
but need not share its width-scaling, and the theory admits only gamma >= 0
(Eq. rank_dilution), so a negative probe exponent is not a capacity exponent.

What this script is for: showing that the probe's width-dependence is a
property of the *domain* and not of the encoder, by running the identical
calibration under the MLP harness and the Transformer encoder.

Input
-----
Per-(domain, width) calibration JSONs, one width per file, each carrying a
``raw`` list whose rows have ``cal_rho_u``.  Grouping is on the row fields, so
the file names do not matter.

    python repro/scripts/probe_exponent.py            # both architectures
    python repro/scripts/probe_exponent.py --mlp 'results/R3/R3_calib_*_m*.json'

The harness MLP runs are ``results/R3/R3_calib_<domain>_m<width>.json`` (five
widths, 32..512) and the Transformer runs are
``results/R4/R4_former_calib_<domain>_m<width>.json`` (three widths, 32/64/128);
both are 10-seed, 40-epoch, four detached warm-up epochs, alpha = 1e-9, i.e.
the healthy anchor.  ``former`` names the Transformer encoder, not a superseded
grid.
"""
from __future__ import annotations

import glob
import json
import math
import sys

# 95% two-sided t quantiles, by residual degrees of freedom (df = n - 2).
T95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469,
       7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788}

MLP = "results/R3/R3_calib_*_m*.json"
FORMER = "results/R4/R4_former_calib_*_m*.json"


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
    return "[%+.3f, %+.3f]" % (g - h, g + h)


def load(pattern):
    """(domain, m) -> (alpha_c, n_seeds, rho_mean, rho_sd)"""
    out = {}
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        raw = j.get("raw") or []
        vals = [r["cal_rho_u"] for r in raw
                if r.get("cal_rho_u") is not None
                and r["cal_rho_u"] == r["cal_rho_u"] and r["cal_rho_u"] > 0]
        if not vals:
            continue
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
            if len(vals) > 1 else float("nan")
        out[(raw[0]["domain"], int(raw[0]["m"]))] = (1.0 / mu, len(vals), mu, sd)
    return out


def report(tag, data):
    print("\n=== %s ===" % tag)
    if not data:
        print("  no calibration rows found")
        return {}
    fitted = {}
    for dom in sorted({d for d, _ in data}):
        pts = sorted((m, ac, n, mu, sd) for (d, m), (ac, n, mu, sd)
                     in data.items() if d == dom)
        print("  %s:" % dom)
        for m, ac, n, mu, sd in pts:
            print("    m=%-4d rho=%9.3f (sd %7.3f, n=%d)  alpha_c=%.5g"
                  % (m, mu, sd, n, ac))
        if len(pts) >= 3:
            g, r2, se, n = ols([math.log(m) for m, _, _, _, _ in pts],
                               [math.log(ac) for _, ac, _, _, _ in pts])
            fitted[dom] = (g, r2, ci(g, se, n), n)
            print("    gamma_probe = %+.4f  R2 = %.4f  95%% CI %s  (n=%d)"
                  % (g, r2, ci(g, se, n), n))
        else:
            print("    (only %d widths -- not fitted)" % len(pts))
    return fitted


def main(argv):
    mlp_pat = argv[argv.index("--mlp") + 1] if "--mlp" in argv else MLP
    tf_pat = argv[argv.index("--former") + 1] if "--former" in argv else FORMER
    a = report("MLP harness", load(mlp_pat))
    b = report("Transformer encoder", load(tf_pat))
    shared = sorted(set(a) & set(b))
    if shared:
        print("\n=== architecture invariance (same sign, overlapping CI) ===")
        for dom in shared:
            ga, r2a, cia, na = a[dom]
            gb, r2b, cib, nb = b[dom]
            same = (ga > 0) == (gb > 0)
            print("  %-9s MLP %+.3f %s   Transformer %+.3f %s   same sign=%s"
                  % (dom, ga, cia, gb, cib, same))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
