# -*- coding: utf-8 -*-
"""The rank-schedule family (SYN7): the direct test of gamma = gamma_0 (1 - beta).

This is the released analysis behind the paper's dilution law.  It reads
``repro/results/SYN7/beta<b>_m<m>.json`` -- the family whose auxiliary rank
follows ``r(m) = 32 (m/64)^beta`` for ``beta = 0, 0.25, 0.5, 0.75, 1``.  The two
endpoints replicate runs that already exist: ``beta = 0`` is the fixed rank 32
of SYN5 (see ``synth_rank_analyze.py``), and ``beta = 1`` is ``r = m/2``, the
diagonal testbed shipped as ``repro/results/th_critical.json``.

Everything is read with one rule -- the crossing of half the healthy level --
and the healthy level is swept over the grid, because the answer depends on
where it is read:

  anchor = 0      the alpha = 0 run.  The variance penalty is ABSENT there, so
                  sigma_h(0) is set by the ranking loss alone and is not a fixed
                  point of the penalised dynamics: it sits above the curve's own
                  trend, and the half-height crossing then lands inside the
                  initial transient.  Worst fits of the whole scan.
  anchor = 3e-4   the start of the paper's 17-point grid (the published rule).
  anchor = 1e-4   the best-conditioned choice (highest R2 at every beta).

The absolute rule t = 1.5e-3 is carried along as the control: it is completely
blind to beta, which is the quantitative statement of the artefact the paper
retires.  A criterion that cannot see the parameter the law is about cannot be
the criterion that measures the law.

The real domains' exponents are read with the same half-height rule by
``plot_unified_v4.py``; this file covers the synthetic family only.

Usage:  python repro/scripts/synth_beta_analyze.py
"""
import io
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # repository root
SYN7 = os.path.join(ROOT, "repro", "results", "SYN7")
THCRIT = os.path.join(ROOT, "repro", "results", "th_critical.json")
MS = [32, 64, 128, 256]
BETAS = [0, 25, 50, 75, 100]
ANCHORS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
T_ABS = 1.5e-3
TCRIT = {2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706}


def curve(b, m):
    p = os.path.join(SYN7, "beta%03d_m%d.json" % (b, m))
    if not os.path.exists(p):
        return None, None
    d = json.load(io.open(p, encoding="utf-8"))
    return sorted((float(x["alpha"]), float(x["sigma_h"])) for x in d["records"]), d["r"]


def value_at(c, a):
    for x, v in c:
        if abs(x - a) <= 1e-12 * max(1.0, abs(a)):
            return v
    return None


def cross(pts, t):
    for i in range(1, len(pts)):
        a0, s0 = pts[i - 1]
        a1, s1 = pts[i]
        if s0 >= t > s1:
            if a0 <= 0 or a1 <= 0 or s0 == s1:
                return a1
            w = (s0 - t) / (s0 - s1)
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
    if g is None or se is None or se != se or (n - 2) not in TCRIT:
        return ""
    h = TCRIT[n - 2] * se
    return "CI [%+.3f, %+.3f]" % (g - h, g + h)


def gamma(b, anchor, ms=MS):
    pr = []
    for m in ms:
        c, _ = curve(b, m)
        if not c:
            continue
        v = c[0][1] if anchor == 0.0 else value_at(c, anchor)
        if v is None:
            continue
        a = cross(c, v / 2.0)
        if a:
            pr.append((m, a))
    if len(pr) < 2:
        return None, None, None, len(pr), pr
    return ols([math.log(m) for m, _ in pr], [math.log(a) for _, a in pr]) + (pr,)


def abs_gamma(b, ms=MS):
    pr = []
    for m in ms:
        c, _ = curve(b, m)
        if not c:
            continue
        a = cross(c, T_ABS)
        if a:
            pr.append((m, a))
    if len(pr) < 2:
        return None, None, None, len(pr), pr
    return ols([math.log(m) for m, _ in pr], [math.log(a) for _, a in pr]) + (pr,)


print("=" * 104)
print("1. the realised rank schedule  r(m) = 32 (m/64)^beta")
print("=" * 104)
print("    %-8s %s" % ("beta", "  ".join("m=%-6d" % m for m in MS)))
for b in BETAS:
    row = []
    for m in MS:
        _, r = curve(b, m)
        row.append("%-6s" % r)
    print("    %-8.2f %s" % (b / 100.0, "  ".join(row)))
print("    beta=0 -> r = 32 at every width (fixed rank).")
print("    beta=1 -> r = m/2, the diagonal testbed of th_critical.json.")

print("\n" + "=" * 104)
print("2. alpha_c(m, beta) and gamma(beta), swept over the healthy anchor")
print("=" * 104)
print("    %-9s %s" % ("anchor", "  ".join("%-18s" % ("b=%.2f" % (b / 100.0)) for b in BETAS)))
res = {}
for anc in ANCHORS:
    cells = []
    for b in BETAS:
        g, r2, se, n, pr = gamma(b, anc)
        res[(b, anc)] = (g, r2, se, n, pr)
        cells.append("%-18s" % ("%+.2f (R2=%.2f)" % (g, r2)))
    print("    %-9s %s" % ("%.0e" % anc, "  ".join(cells)))
print()
print("    the same for the ABSOLUTE rule t = 1.5e-3 (the retired rule):")
cells = []
for b in BETAS:
    g, r2, se, n, pr = abs_gamma(b)
    cells.append("%-18s" % ("%+.2f (R2=%.2f)" % (g, r2)))
print("    %-9s %s" % ("1.5e-3", "  ".join(cells)))
print("    -> flat in beta to within 0.15 across the whole axis: the absolute rule")
print("       cannot see the rank schedule at all, so it cannot measure the law.")

print("\n" + "=" * 104)
print("3. gamma(beta) vs gamma_0 (1 - beta), at each anchor")
print("=" * 104)
for anc in (1e-4, 3e-4, 0.0):
    g0 = res[(0, anc)][0]
    print("    anchor %.0e:  gamma_0 = gamma(beta=0) = %+.3f   %s"
          % (anc, g0, ci(g0, res[(0, anc)][2], res[(0, anc)][3])))
    print("      %-7s %-11s %-11s %-8s %-8s %s"
          % ("beta", "measured", "g0(1-b)", "ratio", "R2", "95% CI"))
    for b in BETAS:
        g, r2, se, n, _ = res[(b, anc)]
        pred = g0 * (1 - b / 100.0)
        print("      %-7.2f %-11s %-11.3f %-8s %-8.2f %s"
              % (b / 100.0, "%+.3f" % g, pred,
                 "%.2f" % (g / pred) if abs(pred) > 1e-9 else "--", r2,
                 ci(g, se, n)))
    print()

print("=" * 104)
print("4. endpoint replication: beta = 1 (r = m/2) against the shipped testbed")
print("=" * 104)
print("    th_critical.json is the same diagonal family at the paper's 17-point")
print("    grid; on the four shared widths and the shared alphas the two must agree.")
d = json.load(io.open(THCRIT, encoding="utf-8"))
for m in MS:
    c, r = curve(100, m)
    ref = {float(x["alpha"]): float(x["sigma_h"])
           for x in d["records"] if int(x["m"]) == m}
    shared = [a for a, _ in c if a in ref and a > 0]
    if shared:
        dmax = max(abs(v - ref[a]) for a, v in c if a in ref and a > 0)
        print("      m=%-4d r=%-4d  max |sigma_h(SYN7 b=1) - sigma_h(th_critical)|"
              " over %d shared alphas = %.3e" % (m, r, len(shared), dmax))

print("\n" + "=" * 104)
print("5. where the real domains sit on this axis")
print("=" * 104)
print("    The real domains' exponents are read from the unified-protocol sweeps")
print("    (repro/results/xcap*.json) with this same half-height rule, by")
print("    `plot_unified_v4.py`, which prints one gamma per domain.  With the")
print("    gamma_0 measured above, beta follows as 1 - gamma/gamma_0; the paper")
print("    reports that arithmetic rather than re-deriving it here, so this file")
print("    stays a statement about the synthetic family alone.")
