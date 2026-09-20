# -*- coding: utf-8 -*-
"""The fixed-rank family (SYN5): the beta = 0 anchor of the capacity law.

This is the released analysis behind the paper's undiluted exponent
``gamma_0 = 2.43`` and behind the claim that a fixed toxic rank ``r <= 4`` does
not collapse at all.  It reads ``repro/results/SYN5/rank_m<m>_r<r>.json`` -- the
synthetic family whose auxiliary rank is held fixed while the width grows,
which is ``beta = 0`` by construction under the Proposition's parametrisation
``m_Pi ~ m^beta``.

Everything is read with one rule -- the crossing of half the healthy level --
and the healthy level is swept over a grid of anchors, because the answer
depends on where it is read:

  anchor = 0      the alpha = 0 run.  The variance penalty is ABSENT there, so
                  sigma_h(0) is set by the ranking loss alone and is not a fixed
                  point of the penalised dynamics: it sits above the curve's own
                  trend, and the half-height crossing then lands inside the
                  initial transient.  Worst fits of the whole scan.
  anchor = 3e-4   the start of the paper's 17-point grid (the published rule).
  anchor = 1e-4   the best-conditioned choice (highest R2 at every beta).

Three things are reported, because the first two turned out to matter:

1. THE COLLAPSE MUST EXIST.  For small r the curve never reaches the collapsed
   floor: it settles on a positive plateau (r = 1, m = 256 -> 0.0804 = 4.9% of
   sigma_h(0)).  A "half-height crossing" of such a curve is a point on the
   initial transient, not a capacity threshold, so gamma is meaningless there.
   The plateau / sigma_h(0) ratio is printed per cell.

2. ANCHOR SENSITIVITY.  gamma(r) is tabulated for a whole family of anchors.

3. sigma_h(0) IS RANK-INDEPENDENT (the aux branch does not touch the encoder
   when alpha = 0), so the healthy anchor is a single number per width -- and it
   grows with m as m^0.95, which is why a fixed ABSOLUTE level mechanically
   steepens the measured exponent.  The absolute rule is carried along as the
   control that quantifies this.

The real domains' exponents are read with the same half-height rule by
``plot_unified_v4.py``; this file covers the synthetic family only.

Usage:  python repro/scripts/synth_rank_analyze.py
"""
import io
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # repository root
SYN5 = os.path.join(ROOT, "repro", "results", "SYN5")
SYN7 = os.path.join(ROOT, "repro", "results", "SYN7")
MS = [32, 64, 128, 256]
RS = [1, 2, 4, 8, 16, 32]
ANCHORS = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
T_ABS = 1.5e-3
TCRIT = {2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706}


def curve(r, m):
    p = os.path.join(SYN5, "rank_m%d_r%d.json" % (m, r))
    if not os.path.exists(p):
        return None
    d = json.load(io.open(p, encoding="utf-8"))
    return sorted((float(x["alpha"]), float(x["sigma_h"])) for x in d["records"])


def curve_syn7(m):
    """SYN7's beta = 0 file: the same fixed rank 32 schedule, re-run."""
    p = os.path.join(SYN7, "beta000_m%d.json" % m)
    if not os.path.exists(p):
        return None, None
    d = json.load(io.open(p, encoding="utf-8"))
    return sorted((float(x["alpha"]), float(x["sigma_h"])) for x in d["records"]), d.get("r")


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


def fit(pairs):
    if len(pairs) < 2:
        return None, None, None, len(pairs)
    return ols([math.log(m) for m, _ in pairs], [math.log(a) for _, a in pairs])


def gamma(r, anchor, ms=MS):
    pr = []
    for m in ms:
        c = curve(r, m)
        if not c:
            continue
        v = c[0][1] if anchor == 0.0 else value_at(c, anchor)
        if v is None:
            continue
        a = cross(c, v / 2.0)
        if a:
            pr.append((m, a))
    return fit(pr) + (pr,)


print("=" * 100)
print("1. THE HEALTHY ANCHOR  sigma_h(0):  rank-independent, and it grows with m")
print("=" * 100)
print("    %-6s %s" % ("r", "  ".join("m=%-12d" % m for m in MS)))
for r in RS:
    cells = []
    for m in MS:
        c = curve(r, m)
        cells.append("%-12.5f" % c[0][1])
    print("    %-6d %s" % (r, "  ".join(cells)))
pairs = [(m, curve(32, m)[0][1]) for m in MS]
g, r2, se, n = fit(pairs)
print("    identical across r -> one anchor per width:  sigma_h(0) ~ m^%+.3f (R2=%.3f, n=%d)"
      % (g, r2, n))
print("    a fixed ABSOLUTE level therefore sits at a relative depth that shallows with m.")

print("\n" + "=" * 100)
print("2. DOES THE CURVE ACTUALLY COLLAPSE?  plateau = mean sigma_h over the last 5 alphas")
print("=" * 100)
print("    %-6s %s" % ("r", "  ".join("m=%-16d" % m for m in MS)))
for r in RS:
    cells = []
    for m in MS:
        c = curve(r, m)
        plateau = sum(v for _, v in c[-5:]) / 5.0
        cells.append("%-16s" % ("%.4f (%.1f%%)" % (plateau, 100 * plateau / c[0][1])))
    print("    %-6d %s" % (r, "  ".join(cells)))
print("    -> r=1..4 keep >1% of sigma_h(0): no collapse, so no capacity threshold.")

print("\n" + "=" * 100)
print("3. gamma(r) as a function of the anchor  (half = sigma_h(anchor)/2)")
print("=" * 100)
print("    %-9s %s" % ("anchor", "  ".join("%-16s" % ("r=%d" % r) for r in RS)))
for anc in ANCHORS:
    cells = []
    for r in RS:
        g, r2, se, n, _ = gamma(r, anc)
        cells.append("%-16s" % ("%+.3f (R2=%.2f)" % (g, r2)))
    print("    %-9s %s" % ("%.0e" % anc, "  ".join(cells)))

print("\n" + "=" * 100)
print("4. the same table with the ABSOLUTE rule t = 1.5e-3 (the retired rule)")
print("=" * 100)
print("    %-6s %-26s %s" % ("r", "gamma (abs 1.5e-3)", "alpha_c per width"))
for r in RS:
    pr = []
    for m in MS:
        c = curve(r, m)
        a = cross(c, T_ABS)
        if a:
            pr.append((m, a))
    g, r2, se, n = fit(pr)
    print("    %-6d %-26s %s"
          % (r, ("%+.3f (R2=%.2f n=%d)" % (g, r2, n)) if g is not None else "-- (n=%d)" % n,
             " ".join("%d:%.4g" % p for p in pr)))
print("    -> the absolute rule only resolves the ranks that collapse hard enough to")
print("       reach 1.5e-3, and it is blind to the rank schedule (see synth_beta_analyze.py).")

print("\n" + "=" * 100)
print("5. cross-check: SYN5 r=32 against SYN7 beta=0 (the same schedule, re-run)")
print("=" * 100)
print("    beta = 0 is the schedule r(m) = 32 at every width, so SYN5's r=32 column")
print("    and SYN7's beta=0 column are the same experiment run twice (independent")
print("    seeds).  The per-point differences below are the seed scatter; what must")
print("    agree is the exponent they imply.")
for m in MS:
    p = os.path.join(SYN7, "beta000_m%d.json" % m)
    if not os.path.exists(p):
        continue
    d = json.load(io.open(p, encoding="utf-8"))
    old = {float(x["alpha"]): float(x["sigma_h"]) for x in d["records"]}
    c = curve(32, m)
    dmax = max(abs(s - old[a]) for a, s in c if a in old)
    print("    m=%-4d  max |sigma_h(SYN5 r=32) - sigma_h(SYN7 beta=0)| = %.3e" % (m, dmax))
g5, r25, se5, n5, _ = gamma(32, 3e-4)
p7 = os.path.join(SYN7, "beta000_m%d.json" % MS[0])
if os.path.exists(p7):
    pr = []
    for m in MS:
        c, _ = curve_syn7(m)
        if not c:
            continue
        v = value_at(c, 3e-4)
        a = cross(c, v / 2.0) if v is not None else None
        if a:
            pr.append((m, a))
    if len(pr) >= 2:
        g7, r27, se7, n7 = ols([math.log(m) for m, _ in pr], [math.log(a) for _, a in pr])
        print("    anchor 3e-4:  SYN5 r=32 gamma = %+.3f (R2=%.2f)   vs   "
              "SYN7 beta=0 gamma = %+.3f (R2=%.2f)   ->  agree to %.3f"
              % (g5, r25, g7, r27, abs(g5 - g7)))
