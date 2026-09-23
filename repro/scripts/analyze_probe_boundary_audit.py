"""analyze_probe_boundary_audit.py -- read results/PBA/PBA_har.json and print
the tables the paper's audit paragraph needs.

Everything is recomputed from the stored primitives (r, c, d, a, nu, n_p, B,
k_opt), so the term attribution is available at the tightest scale k_opt as
well as at the canonical zero-trace scale k_tr = tr(K)/(Bm).

  eps_p(k) = r / (k*base) - d,   eps_a(k) = c / (k*base) - a,
  base     = ||u|| * n_p / B,
  Delta(k) = |d-1| + |eps_p(k)| + alpha (|a-q| + |eps_a(k)|).
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def _terms(p, tag, k):
    base = p["nu"] * p["n_p"] / p["B"]
    N = k * base
    eps_p = p["r"] / N - p["d"]
    eps_a = p["c"] / N - p["a"]
    q = p[tag]["q"]
    al = p["alpha"]
    return dict(t_d=abs(p["d"] - 1.0),
                t_ep=abs(eps_p),
                t_aq=al * abs(p["a"] - q),
                t_ea=al * abs(eps_a),
                Delta=abs(p["d"] - 1.0) + abs(eps_p) + al * (abs(p["a"] - q) + abs(eps_a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="../results/PBA/PBA_har.json")
    ap.add_argument("--tag", default="frozen", choices=["frozen", "live"])
    a = ap.parse_args()

    with open(a.json, encoding="utf-8") as f:
        D = json.load(f)

    pts = []
    for run in D["runs"]:
        for p in run["points"]:
            p = dict(p)
            p["m"] = run["m"]
            p["seed"] = run["seed"]
            p["B"] = run["audit_batch"]
            pts.append(p)

    print("== %s ==  runs=%d  points=%d  defined=%d  degenerate=%d"
          % (os.path.basename(a.json), len(D["runs"]), len(pts),
             sum(1 for p in pts if not p.get("degenerate")),
             sum(1 for p in pts if p.get("degenerate"))))
    print("identity residual  max = %.3e" % D["summary"]["max_ident_resid"])
    print("finite-difference  median rel err = %.3e   max = %.3e"
          % (D["summary"]["median_fd_rel_err"], D["summary"]["max_fd_rel_err"]))
    print()

    # ---------------- per (m, alpha) table ---------------------------------
    grp = defaultdict(list)
    for p in pts:
        if p.get("degenerate") or a.tag not in p:
            continue
        grp[(p["m"], p["alpha"])].append(p)

    hdr = ("  m    alpha   n   fired  fire@kopt  agree  |1-aq|    Delta   Dmin   "
           "margin@kopt  dom.term")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key in sorted(grp, key=lambda k: (k[0], k[1])):
        m, al = key
        g = grp[key]
        n = len(g)
        fired = np.mean([p[a.tag]["fired"] for p in g])
        fired_opt = np.mean([p[a.tag]["fired_at_kopt"] for p in g])
        agree = np.mean([p[a.tag]["agree"] for p in g])
        m1aq = np.median([abs(p[a.tag]["probe_says"]) for p in g])
        dl = np.median([p[a.tag]["Delta"] for p in g])
        dm = np.median([p[a.tag]["Delta_min"] for p in g])
        marg = np.median([p[a.tag]["margin_kopt"] for p in g])
        T = [_terms(p, a.tag, p[a.tag]["k_opt"]) for p in g]
        names = ["|d-1|", "|eps_p|", "a|a-q|", "a|eps_a|"]
        sums = np.array([[t["t_d"], t["t_ep"], t["t_aq"], t["t_ea"]] for t in T])
        med = np.median(sums, axis=0)
        dom = names[int(np.argmax(med))]
        print("  %-4d %-7g %-3d %5.2f  %8.2f  %5.2f  %7.3g  %7.3g  %6.3g  %10.3g  %s"
              % (m, al, n, fired, fired_opt, agree, m1aq, dl, dm, marg, dom))

    # ---------------- term census at k_opt ---------------------------------
    ok = [p for p in pts if not p.get("degenerate") and a.tag in p]
    T = np.array([[_terms(p, a.tag, p[a.tag]["k_opt"])[k] for k in
                   ("t_d", "t_ep", "t_aq", "t_ea")] for p in ok])
    names = ["|d-1|", "|eps_p|", "alpha|a-q|", "alpha|eps_a|"]
    print("\n== median term census at k_opt (share of Delta_min) ==")
    med = np.median(T, axis=0)
    tot = med.sum()
    for nm, v in zip(names, med):
        print("   %-14s %10.4g   %5.1f%%" % (nm, v, 100 * v / max(tot, 1e-30)))

    # ---------------- the two named failure modes --------------------------
    fn = [p for p in ok if (not p[a.tag]["agree"]) and p[a.tag]["probe_says"] > 0]
    fp = [p for p in ok if (not p[a.tag]["agree"]) and p[a.tag]["probe_says"] < 0]
    print("\n== sign disagreement vs the exact flux (probe wrong) ==")
    print("   probe says SAFE   but dV/dt<0 (false negative) : %d" % len(fn))
    print("   probe says COLLAPSE but dV/dt>0 (false positive): %d" % len(fp))
    print("   of those, certificate fired (must be 0)        : %d"
          % sum(1 for p in fn + fp if p[a.tag]["fired"]))
    if fn:
        dd = np.array([[p["d"], p["alpha"] * p[a.tag]["q"]] for p in fn])
        print("   false negatives: 0<d<alpha*q<1 in %d/%d cases"
              % (int(np.sum((dd[:, 0] > 0) & (dd[:, 0] < dd[:, 1])
                            & (dd[:, 1] < 1))), len(fn)))
    if fp:
        aa = np.array([[p["a"], p[a.tag]["q"]] for p in fp])
        print("   false positives: sign(a) != sign(q) in %d/%d cases"
              % (int(np.sum(np.sign(aa[:, 0]) != np.sign(aa[:, 1]))), len(fp)))

    # ---------------- where it does fire ----------------------------------
    fired = [p for p in ok if p[a.tag]["fired_at_kopt"]]
    print("\n== checkpoints where the certificate FIRES (at k_opt) ==")
    if not fired:
        print("   none")
    for p in fired[:20]:
        print("   m=%-4d a=%-7g seed=%d ep=%-3d  |1-aq|=%.4g  Dmin=%.4g  "
              "E=%+.4g  d=%.3f a=%.3f q=%.4g"
              % (p["m"], p["alpha"], p["seed"], p["epoch"],
                 abs(p[a.tag]["probe_says"]), p[a.tag]["Delta_min"],
                 p[a.tag]["E"], p["d"], p["a"], p[a.tag]["q"]))
    print("\n   fired count = %d / %d   (frozen=%d, live=%d)"
          % (len(fired), len(ok),
             sum(1 for p in ok if p["frozen"]["fired_at_kopt"]),
             sum(1 for p in ok if p["live"]["fired_at_kopt"])))


    # ---------------- the corrected probe ----------------------------------
    cor = [p for p in ok if p.get("corrected")]
    print("\n== corrected probe q* = a/d  (signed loading / directional primary) ==")
    if cor:
        cf = [p for p in cor if p["corrected"]["fired"]]
        ca = [p for p in cor if p["corrected"]["agree"]]
        print("   n=%d   fired=%.3f   agree=%.4f   fired_and_wrong=%d"
              % (len(cor), len(cf) / len(cor), len(ca) / len(cor),
                 sum(1 for p in cf if not p["corrected"]["agree"])))
        print("   Delta_star  median=%.3e  max=%.3e   (vs frozen Delta_min median=%.3e)"
              % (np.median([p["corrected"]["Delta_star_min"] for p in cor]),
                 max(p["corrected"]["Delta_star_min"] for p in cor),
                 np.median([p[a.tag]["Delta_min"] for p in cor])))
        ratio = [p[a.tag]["Delta_min"] / max(p["corrected"]["Delta_star_min"], 1e-300)
                 for p in cor]
        print("   envelope shrink factor (frozen Dmin / Delta_star): median=%.1fx  min=%.1fx"
              % (np.median(ratio), min(ratio)))
        grp2 = defaultdict(list)
        for p in cor:
            grp2[(p["m"], p["alpha"])].append(p)
        print("\n   per-cell agreement of the corrected probe:")
        print("     m    alpha   n   agree*   fired*   D* median")
        for key in sorted(grp2, key=lambda k: (k[0], k[1])):
            g = grp2[key]
            print("     %-4d %-7g %-3d %6.2f  %7.2f  %.3e"
                  % (key[0], key[1], len(g),
                     np.mean([p["corrected"]["agree"] for p in g]),
                     np.mean([p["corrected"]["fired"] for p in g]),
                     np.median([p["corrected"]["Delta_star_min"] for p in g])))
    else:
        print("   (no corrected records in this file)")

    # ---------------- agreement restricted to the toxic regime -------------
    for lab, sel in (("alpha = 0", lambda p: p["alpha"] == 0),
                     ("alpha > 0", lambda p: p["alpha"] > 0)):
        sub = [p for p in ok if sel(p)]
        if sub:
            print("\n== agreement, %s (n=%d) ==" % (lab, len(sub)))
            print("   frozen   %.3f" % np.mean([p["frozen"]["agree"] for p in sub]))
            print("   live     %.3f" % np.mean([p["live"]["agree"] for p in sub]))
            if any(p.get("corrected") for p in sub):
                s2 = [p for p in sub if p.get("corrected")]
                print("   corrected %.4f" % np.mean([p["corrected"]["agree"] for p in s2]))


if __name__ == "__main__":
    main()
