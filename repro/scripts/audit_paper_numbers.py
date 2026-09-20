# -*- coding: utf-8 -*-
"""Audit: paper table numbers vs source merged JSONs (local)."""
import json, os

# Pulled JSONs live outside the repository (paper/_hpc is ignored); resolve
# them from the repository root, overridable, so no machine-specific absolute
# path travels in the released package.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.environ.get("REPFIX_PULL", os.path.join(_ROOT, "paper", "_hpc", "merged_pull"))
C = os.environ.get("REPFIX_CONFLICT_PULL",
                   os.path.join(_ROOT, "paper", "_hpc", "conflict_pull"))


def load(p):
    return json.load(open(p))


def rows(path):
    d = load(path)
    return {r["config"]: r for r in d.get("results", [])}, d.get("single_ceiling", {}), d.get("seeds")


def show(tag, path):
    if not os.path.exists(path):
        print(f"  [missing] {path}")
        return
    rs, sc, seeds = rows(path)
    print(f"--- {tag}  seeds={seeds}")
    for cfg in ("single", "joint", "stf0", "stf0_w", "toxifier"):
        r = rs.get(cfg)
        if not r:
            continue
        print(f"    {cfg:<9} soh={r['rmse_soh']:.2f}+-{r['rmse_soh_std']:.2f}"
              f"  rul={r['mae_rul']:.1f}+-{r['mae_rul_std']:.1f}"
              f"  scatter={r['scatter']:.3f}")
    if sc:
        for k, v in sc.items():
            print(f"    ceiling[{k}] soh={v.get('rmse_soh'):.2f} rul={v.get('mae_rul'):.1f}"
                  f" scatter={v.get('scatter'):.3f}")


print("=" * 70)
print("tab:bmtl  ->  MATR / NASA main")
print("=" * 70)
show("MATR_main", os.path.join(D, "merged_MATR_main.json"))
show("NASA_main", os.path.join(D, "merged_NASA_main.json"))

print()
print("=" * 70)
print("tab:cens_ablation  ->  nocens / eolonly / (main)")
print("=" * 70)
show("MATR_nocens", os.path.join(D, "merged_MATR_nocens.json"))
show("MATR_eolonly", os.path.join(D, "merged_MATR_eolonly.json"))

print()
print("=" * 70)
print("tri-weight ablations (rate/temp/degrad, gamma 0.5/2.0)")
print("=" * 70)
for f in sorted(os.listdir(D)):
    if "wterm" in f or "gamma" in f:
        show(f, os.path.join(D, f))

print()
print("=" * 70)
print("contrast (collapse/recovery) source check")
print("=" * 70)
for f in sorted(os.listdir(C)):
    if f.startswith("x_") and "contrast" in f:
        d = load(os.path.join(C, f))
        print(f"--- {f} domain={d.get('domain')}")
        for r in d.get("results", []):
            print(f"    {r.get('mode','?'):<8} metric={r.get('metric_mean', float('nan')):.4f}"
                  f" collapse={r.get('collapse_rate')} recov={r.get('recovery_rate')}"
                  f" rel={r.get('rel_scatter_mean', float('nan')):.3f}")
