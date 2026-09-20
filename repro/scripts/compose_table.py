#!/usr/bin/env python
"""Composition table: 'filter toxicity, then resolve conflict' (E2e).

For each domain the E2e run composes an STF mode (toxicity filter) with a
conflict-resolution operator, i.e. combo = "<stfmode>+<method>":
    joint+joint      plain summation (reference)
    joint+pcgrad     conflict resolution on a still-toxic gradient
    joint+cagrad     conflict resolution on a still-toxic gradient
    stfhard+joint    toxicity filtered, no conflict operator
    stfhard+pcgrad   filtered THEN resolved  <- the design-rule cell
    stfhard+cagrad   filtered THEN resolved
    stfsoft+*        soft gate variant

Reads results/E2e/E2e_compose_<domain>.json (kind='composition'), which stores
`results` with mode/metric_mean/metric_std/collapse_rate/recovery_rate/
rel_scatter_mean/rho_mean.
"""
import glob, json, os, sys

D = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "paper", "_hpc", "e2e_pull"))

ORDER = ["joint+joint", "joint+pcgrad", "joint+cagrad",
         "stfhard+joint", "stfhard+pcgrad", "stfhard+cagrad",
         "stfsoft+joint", "stfsoft+pcgrad", "stfsoft+cagrad"]


def f(x, p=4):
    try:
        return f"{float(x):.{p}f}"
    except (TypeError, ValueError):
        return "  nan "


def main(paths):
    for p in paths:
        j = json.load(open(p))
        dom = j.get("domain", "?")
        rs = {r["mode"]: r for r in j.get("results", [])}
        print(f"\n=== {os.path.basename(p)}  domain={dom}  kind={j.get('kind')}")
        print(f"    {'combo':<18} {'metric':>10} {'coll':>6} {'recov':>6} {'relScat':>8}")
        for k in ORDER:
            r = rs.get(k)
            if not r:
                continue
            print(f"    {k:<18} {f(r.get('metric_mean'))} {f(r.get('collapse_rate'),2):>6}"
                  f" {f(r.get('recovery_rate'),2):>6} {f(r.get('rel_scatter_mean'),3):>8}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        main(sorted(glob.glob(os.path.join(D, "*.json"))))
