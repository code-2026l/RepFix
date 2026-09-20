"""The per-step probe along the training trajectory (Appendix: online probe).

Section stfcal rejects an online rho-gate.  This script prints the evidence:
the probe's time series on the unfiltered joint path, with and without the
toxifier, and what the gate alpha*rho >= 1 would have done with it.

Inputs (relative to the repository root):

  results/RHO/xcap_<domain>_trace_m<m>.json  joint path, --rho-trace on
  results/RHO/xcap_<domain>_zero_m<m>.json   the same with alpha = 0, i.e. the
                                             toxifier inert (control)
  results/BSCALE/xcap_<domain>_m<m>.json     the untraced runs the above must
                                             reproduce (monitor must not perturb)

Each raw record carries rho_trace (probe every 10 optimiser steps), the
endpoint relative scatter, and latch_step -- the step at which rho_ref*alpha
first reached 1, or -1 if it never did.  The probe itself is rho_probe() in
repro/scripts/cross_domain_mtl.py, the paper's estimator.

Run from the repository root:

    python repro/scripts/rho_trace_analyze.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RHO = os.path.join(ROOT, "results", "RHO")
BSCALE = os.path.join(ROOT, "results", "BSCALE")

CROSSING = 0.5          # the half-height crossing that defines collapse here
MATCH = ("metric_mean", "collapse_rate", "rel_scatter_mean", "rho_mean")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def by_alpha(path):
    out = {}
    if not os.path.exists(path):
        return out
    for r in load(path).get("raw") or []:
        if r.get("rho_trace"):
            out.setdefault(float(r["alpha"]), []).append(r)
    return out


def cells(domain, m):
    """Merge the traced and the zero-dose control for one (domain, width)."""
    merged = {}
    for name in ("trace", "zero"):
        for a, rs in by_alpha(os.path.join(RHO, "xcap_%s_%s_m%d.json" % (domain, name, m))).items():
            merged.setdefault(a, []).extend(rs)
    return merged


def monitor_is_inert():
    """The traced runs must reproduce the untraced ones exactly."""
    worst, matched, mismatched = 0.0, 0, 0
    for path in sorted(glob.glob(os.path.join(RHO, "xcap_*_trace_m*.json"))):
        dom = os.path.basename(path).split("_")[1]
        m = int(os.path.basename(path).split("_m")[-1].split(".")[0])
        ref = {}
        bp = os.path.join(BSCALE, "xcap_%s_m%d.json" % (dom, m))
        if not os.path.exists(bp):
            continue
        for r in load(bp).get("results") or []:
            ref[float(r["alpha"])] = r
        for r in load(path).get("results") or []:
            b = ref.get(float(r["alpha"]))
            if b is None:
                continue
            matched += 1
            d = max(abs(float(r[k]) - float(b[k])) for k in MATCH)
            worst = max(worst, d)
            if d:
                mismatched += 1
    return matched, worst, mismatched


def main():
    matched, worst, bad = monitor_is_inert()
    print("Monitor check: %d traced cells matched into BSCALE, max |delta| = %.3e,"
          " mismatching cells: %d" % (matched, worst, bad))
    print()

    print("%-8s %-5s %-9s %-5s %-9s | %-10s %-11s | %-24s %-6s"
          % ("domain", "m", "alpha", "n", "rel_s", "rho_first", "rho_last",
             "latch_step (per seed)", "coll?"))
    print("-" * 106)

    spurious = []          # gate fires although the scatter never crosses
    battery_noncollapse_seeds = 0
    battery_noncollapse_trips = 0
    for domain in ("har", "battery"):
        for m in (32, 64, 256):
            for a, rs in sorted(cells(domain, m).items()):
                first = sum(float(r["rho_trace"][0]["rho_u"]) for r in rs) / len(rs)
                last = sum(float(r["rho_trace"][-1]["rho_u"]) for r in rs) / len(rs)
                rel = sum(float(r["rel_scatter"]) for r in rs) / len(rs)
                latch = [int(r.get("latch_step", -1)) for r in rs]
                coll = "YES" if rel < CROSSING else "no"
                if coll == "no" and any(s >= 0 for s in latch):
                    spurious.append((domain, m, a, rel, latch))
                    if domain == "battery":
                        battery_noncollapse_seeds += len(rs)
                        battery_noncollapse_trips += sum(1 for s in latch if s >= 0)
                print("%-8s %-5d %-9g %-5d %-9.3f | %-10.4g %-11.4g | %-24s %-6s"
                      % (domain, m, a, len(rs), rel, first, last,
                         ",".join(str(s) for s in latch), coll))
        print()

    har = [s for s in spurious if s[0] == "har"]
    steps = [v for s in har for v in s[4] if v >= 0]
    print("Gate fires on a run whose scatter stays at or above %.1f:" % CROSSING)
    print("  HAR     : %d dose cells, all seeds (latch step %d--%d)"
          % (len(har), min(steps), max(steps)))
    print("  battery : %d trips" % battery_noncollapse_trips)


if __name__ == "__main__":
    main()
