# -*- coding: utf-8 -*-
"""Joint fit of the rank-schedule family: gamma_0 from the whole (m, beta) grid.

`synth_beta_analyze.py` reads one exponent per beta, each from a width-wise OLS
of log alpha_c on log m.  That leaves the beta = 0 anchor with n = 4 and two
residual degrees of freedom, so its 95% interval is [-0.07, +4.93] -- it
contains zero, and the paper says so.

The same curves allow a conditional surface fit. The scaling hypothesis is:

    alpha_c(m, beta) = C (m / m_ref)^{gamma_0 (1 - beta)}

This borrows information across schedules but does not create independent
replicates: curves share seeds and the pivot curves are identical.

The schedule pins r = 32 at m = m_ref = 64 for *every* beta, so the generator
is literally the same object at m = 64 across the row and the family pivots
there by construction.  Both fits are therefore written about the pivot,

    x = log m - log m_ref,     xb = beta * x

Without a beta main effect, this changes the model restriction, not just its
numerical conditioning: a common intercept at m=64 differs from one at m=1.
With a beta main effect the two parameterizations span the same model space.

Two fits are reported, and the second is the honest one:

  constrained   log alpha_c = c + gamma_0 (1 - beta) x
                gamma_0 with n - 2 dof.  Valid only if the (1 - beta) form
                holds.

  unconstrained log alpha_c = c + a x + b xb   (+ optional beta main effect)
                a with n - 3 dof, and the proposition's content is b = -a.  An
                interval for a + b excluding zero rejects the linear restriction;
                containing zero means it is not rejected, not that it is proven.
                Cell-level OLS intervals are conditional descriptive intervals: the
                grid shares seeds and repeats identical m=64 curves.

Usage:  python repro/scripts/synth_beta_joint.py [--dir repro/results/SYN8]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# 95% two-sided t quantiles, by residual degrees of freedom.
T95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469,
       7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788,
       13: 2.1604, 14: 2.1448, 15: 2.1314, 16: 2.1199, 17: 2.1098, 18: 2.1009,
       19: 2.0930, 20: 2.0860, 21: 2.0796, 22: 2.0739, 23: 2.0687, 24: 2.0639,
       25: 2.0595, 26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452, 30: 2.0423}

ANCHORS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
DEFAULT_ANCHOR = 3e-4          # the published rule (start of the paper's grid)
PUBLISHED_MS = (32, 64, 128, 256)
PIVOT = 64                     # r(m_ref) = 32 for every beta, so the family pivots here


def value_at(c, a):
    for x, v in c:
        if abs(x - a) <= 1e-12 * max(1.0, abs(a)):
            return v
    return None


def cross(pts, t):
    """First log-alpha crossing of `t`, interpolated between neighbours."""
    for i in range(1, len(pts)):
        a0, s0 = pts[i - 1]
        a1, s1 = pts[i]
        if s0 >= t > s1:
            if a0 <= 0 or a1 <= 0 or s0 == s1:
                return a1
            w = (s0 - t) / (s0 - s1)
            return math.exp(math.log(a0) + w * (math.log(a1) - math.log(a0)))
    return None


def ols1(xs, ys):
    """Single-slope OLS.  Returns (slope, r2, se_slope, n)."""
    n = len(xs)
    if n < 3:
        return None, None, None, n
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    mx, my = float(x.mean()), float(y.mean())
    sxx = float(((x - mx) ** 2).sum())
    sl = float(((x - mx) * (y - my)).sum()) / sxx
    ssr = float(((y - (my + sl * (x - mx))) ** 2).sum())
    sst = float(((y - my) ** 2).sum())
    r2 = 1 - ssr / sst if sst > 0 else float("nan")
    se = math.sqrt(ssr / (n - 2) / sxx) if ssr > 0 else float("nan")
    return sl, r2, se, n


def ols_multi(cols, y):
    """OLS with an intercept prepended.  Returns (beta, r2, cov, n, dof)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.asarray(c, float) for c in cols]) if cols else np.zeros((len(y), 0))
    A = np.column_stack([np.ones(len(y)), X])
    n, p = A.shape
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ssr = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ssr / sst if sst > 0 else float("nan")
    dof = n - p
    cov = (ssr / dof) * np.linalg.inv(A.T @ A) if dof > 0 else np.full((p, p), np.nan)
    return beta, r2, cov, n, dof


def tq(dof):
    return T95.get(dof, float("nan"))


def ci_of(est, se, dof):
    t = tq(dof)
    if not (t == t) or not (se == se):
        return ""
    return "[%+.3f, %+.3f]" % (est - t * se, est + t * se)


def load(d):
    """(m, beta) -> (sorted [(alpha, sigma_h)], r)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "beta*_m*.json"))):
        j = json.load(io.open(p, encoding="utf-8"))
        key = (int(j["m"]), round(float(j["beta"]), 4))
        c = sorted((float(x["alpha"]), float(x["sigma_h"])) for x in j["records"])
        if key not in out or len(c) > len(out[key][0]):
            out[key] = (c, int(j["r"]))
    return out


def ac_table(cells, anchor):
    """(m, beta) -> alpha_c, read as the crossing of half the healthy level."""
    out = {}
    for (m, b), (c, r) in cells.items():
        v = c[0][1] if anchor == 0.0 else value_at(c, anchor)
        if v is None:
            continue
        a = cross(c, v / 2.0)
        if a:
            out[(m, b)] = (a, r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "repro", "results", "SYN8"))
    ap.add_argument("--anchor", type=float, default=DEFAULT_ANCHOR)
    ap.add_argument("--dedup-pivot", action="store_true",
                    help="retain only one identical m=64 curve as a sensitivity check")
    ap.add_argument("--drop-m", default="",
                    help="comma-separated widths to exclude (a cell whose scatter "
                         "never reaches the collapsed floor inside the swept alpha "
                         "grid carries no resolved crossing)")
    a = ap.parse_args()

    cells = load(a.dir)
    if a.dedup_pivot:
        keys = sorted(k for k in cells if k[0] == PIVOT)
        if keys:
            first = cells[keys[0]]
            for key in keys[1:]:
                if cells[key] != first:
                    raise ValueError('pivot curves differ; refusing deduplication')
                del cells[key]
            print('[deduplicated pivot] retained one of %d identical curves' % len(keys))
    if a.drop_m:
        drop = {int(x) for x in a.drop_m.split(",") if x.strip()}
        cells = {k: v for k, v in cells.items() if k[0] not in drop}
        print("[dropped widths] %s" % ",".join(str(m) for m in sorted(drop)))
    if not cells:
        print("[empty] no beta*_m*.json under %s" % a.dir)
        return 1

    ms = sorted({m for m, _ in cells})
    bs = sorted({b for _, b in cells})
    print("=" * 100)
    print("0. the realised grid under %s" % os.path.relpath(a.dir, ROOT))
    print("=" * 100)
    print("    %-8s %s" % ("beta", "  ".join("%-9s" % ("m=%d" % m) for m in ms)))
    for b in bs:
        row = []
        for m in ms:
            row.append("%-9s" % (cells[(m, b)][1] if (m, b) in cells else "--"))
        print("    %-8.2f %s" % (b, "  ".join(row)))
    print("    cells present: %d" % len(cells))
    print("    r = round(32 * (m / 64) ** beta), clipped to [1, m]")

    print("\n" + "=" * 100)
    print("1. alpha_c(m, beta) -- crossing of half the healthy level, anchor swept")
    print("=" * 100)
    tables = {}
    for anc in dict.fromkeys((a.anchor, *ANCHORS)):
        t = ac_table(cells, anc)
        tables[anc] = t
        if not t:
            continue
        print("    anchor %.0e" % anc)
        for b in bs:
            row = []
            for m in ms:
                row.append("%-11s" % ("%.4g" % t[(m, b)][0] if (m, b) in t else "--"))
            print("      beta=%-5.2f %s" % (b, "  ".join(row)))

    print("\n" + "=" * 100)
    print("2. the published reading: one exponent per beta, width-wise OLS")
    print("=" * 100)
    for anc in dict.fromkeys((a.anchor, DEFAULT_ANCHOR, 1e-4, 0.0)):
        t = tables.get(anc) or {}
        print("    anchor %.0e" % anc)
        for b in bs:
            pr = [(m, t[(m, b)][0]) for m in ms if (m, b) in t]
            if len(pr) < 3:
                continue
            g, r2, se, n = ols1([math.log(m) for m, _ in pr],
                                [math.log(v) for _, v in pr])
            print("      beta=%-5.2f  widths=%-18s n=%d  gamma=%+.3f  R2=%.3f  %s"
                  % (b, ",".join(str(m) for m, _ in pr), n, g, r2,
                     ci_of(g, se, n - 2)))
        print()

    print("=" * 100)
    print("3. the beta = 0 anchor alone, on two width sets")
    print("=" * 100)
    t = tables.get(a.anchor) or {}
    for label, mset in (("published set", PUBLISHED_MS),
                        ("this grid", tuple(ms))):
        pr = [(m, t[(m, 0.0)][0]) for m in mset if (m, 0.0) in t]
        if len(pr) < 3:
            print("    %-14s widths=%-24s -- not all present" % (label, ",".join(str(m) for m in mset)))
            continue
        g, r2, se, n = ols1([math.log(m) for m, _ in pr],
                            [math.log(v) for _, v in pr])
        lx = [math.log(m) for m, _ in pr]
        sxx = sum((x - sum(lx) / n) ** 2 for x in lx)
        print("    %-14s widths=%-24s n=%d  gamma_0=%+.3f  R2=%.3f  %s"
              % (label, ",".join(str(m) for m, _ in pr), n, g, r2,
                 ci_of(g, se, n - 2)))
        print("    %-14s t=%.3f  sqrt(Sxx)=%.3f  half-width = t*se = %.3f"
              % ("", tq(n - 2), math.sqrt(sxx), tq(n - 2) * se))

    print("\n" + "=" * 100)
    print("4. joint fits over the whole grid, written about the pivot m_ref")
    print("=" * 100)
    for anc in dict.fromkeys((a.anchor, DEFAULT_ANCHOR, 1e-4, 0.0)):
        t = tables.get(anc) or {}
        pts = [(m, b, t[(m, b)][0]) for (m, b) in sorted(t)]
        if len(pts) < 5:
            continue
        y = [math.log(v) for _, _, v in pts]
        x = [math.log(m) - math.log(PIVOT) for m, _, _ in pts]
        xb = [b * xx for (_, b, _), xx in zip(pts, x)]
        bmain = [b for _, b, _ in pts]
        print("    anchor %.0e   cells=%d   pivot m_ref=%d" % (anc, len(pts), PIVOT))

        beta_c, r2c, covc, n, dof = ols_multi([[(1.0 - b) * xx for (_, b, _), xx in zip(pts, x)]], y)
        g0, se0 = float(beta_c[1]), math.sqrt(float(covc[1, 1]))
        print("      constrained  log a_c = c + g0 (1-b) x")
        print("        gamma_0 = %+.3f   se = %.3f   dof = %d   %s   R2 = %.3f"
              % (g0, se0, dof, ci_of(g0, se0, dof), r2c))

        beta_u, r2u, covu, n, dof = ols_multi([x, xb], y)
        aa, bb = float(beta_u[1]), float(beta_u[2])
        sea, seb = math.sqrt(float(covu[1, 1])), math.sqrt(float(covu[2, 2]))
        s_ab = math.sqrt(float(covu[1, 1]) + float(covu[2, 2]) + 2 * float(covu[1, 2]))
        print("      unconstrained log a_c = c + a x + b xb")
        print("        a (= gamma_0) = %+.3f  se %.3f  %s" % (aa, sea, ci_of(aa, sea, dof)))
        print("        b (= -gamma_0 expected) = %+.3f  se %.3f  %s" % (bb, seb, ci_of(bb, seb, dof)))
        print("        proposition test  a + b = %+.3f  se %.3f  %s"
              % (aa + bb, s_ab, ci_of(aa + bb, s_ab, dof)))
        if abs(aa) > 1e-12:
            print("        implied dilution factor -b/a = %+.3f  (expected 1.000)" % (-bb / aa))
        print("        R2 = %.3f   dof = %d" % (r2u, dof))

        beta_m, r2m, covm, n, dofm = ols_multi([x, xb, bmain], y)
        dm = float(beta_m[3])
        sdm = math.sqrt(float(covm[3, 3]))
        print("      + beta main effect: d = %+.4f  se %.4f  %s   R2 = %.3f   dof = %d"
              % (dm, sdm, ci_of(dm, sdm, dofm), r2m, dofm))
        print("        (d = 0 is the pivot the schedule enforces; a non-zero d means")
        print("         the family does not pivot at m_ref after all)")
        print()

    print("=" * 100)
    print("5. what each reading buys")
    print("=" * 100)
    t = tables.get(a.anchor) or {}
    pr0 = [(m, t[(m, 0.0)][0]) for m in PUBLISHED_MS if (m, 0.0) in t]
    if len(pr0) >= 3:
        g, r2, se, n = ols1([math.log(m) for m, _ in pr0], [math.log(v) for _, v in pr0])
        h4 = tq(n - 2) * se
        print("    published anchor, %d widths        half-width %.3f" % (n, h4))
    pr1 = [(m, t[(m, 0.0)][0]) for m in ms if (m, 0.0) in t]
    if len(pr1) >= 3:
        g, r2, se, n = ols1([math.log(m) for m, _ in pr1], [math.log(v) for _, v in pr1])
        h5 = tq(n - 2) * se
        print("    anchor row of this grid, %d widths half-width %.3f" % (n, h5))
    pts = [(m, b, t[(m, b)][0]) for (m, b) in sorted(t)]
    if len(pts) >= 5:
        y = [math.log(v) for _, _, v in pts]
        beta_c, r2c, covc, n, dof = ols_multi(
            [[(1.0 - b) * (math.log(m) - math.log(PIVOT)) for m, b, _ in pts]], y)
        se0 = math.sqrt(float(covc[1, 1]))
        print("    joint over all %d cells           half-width %.3f"
              % (n, tq(dof) * se0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
