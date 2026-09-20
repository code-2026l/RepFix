"""Summarize pulled conflict/baseline JSONs into one table (local analysis)."""
import json, glob, os

# Pulls live outside the repository (paper/_hpc is ignored); resolve from the
# repository root so the released script carries no absolute path.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.environ.get("REPFIX_CONFLICT_PULL",
                   os.path.join(_ROOT, "paper", "_hpc", "conflict_pull"))

print("=" * 72)
print("CONTRAST / CONFLICT files (unified protocol: metric + collapse rate)")
print("=" * 72)
for f in sorted(glob.glob(os.path.join(D, "x_*contrast*.json"))) + \
         sorted(glob.glob(os.path.join(D, "E2c_radioml_conflict.json"))):
    d = json.load(open(f))
    print(f"--- {os.path.basename(f)}  domain={d.get('domain')}")
    for r in d.get("results", []):
        print(f"   {r.get('mode','?'):<8} metric={r.get('metric_mean', float('nan')):.4f}"
              f"+-{r.get('metric_std', 0):.4f} collapse={r.get('collapse_rate')}"
              f" rel={r.get('rel_scatter_mean', float('nan')):.3f}"
              f" alpha={r.get('alpha')}")

print()
print("=" * 72)
print("E2a BatteryMTL baselines (agg = sum/pcgrad/famo/cagrad), alpha_aux=10")
print("=" * 72)
rows = {}
for f in sorted(glob.glob(os.path.join(D, "merged_E2a_*.json"))):
    d = json.load(open(f))
    ds, agg = d.get("dataset"), d.get("agg")
    rs = {r["config"]: r for r in d.get("results", [])}
    sc = d.get("single_ceiling", {})
    rows[(ds, agg)] = (d.get("seeds"), rs, sc)
for (ds, agg), (seeds, rs, sc) in sorted(rows.items()):
    j = rs.get("joint", {}); s0 = rs.get("stf0", {})
    print(f"--- {ds} agg={agg} seeds={seeds}")
    print(f"    joint soh={j.get('rmse_soh', float('nan')):.2f}+-{j.get('rmse_soh_std',0):.2f}"
          f" rul={j.get('mae_rul', float('nan')):.1f}+-{j.get('mae_rul_std',0):.1f}"
          f" scatter={j.get('scatter', float('nan')):.3f}")
    print(f"    stf0  soh={s0.get('rmse_soh', float('nan')):.2f}+-{s0.get('rmse_soh_std',0):.2f}"
          f" rul={s0.get('mae_rul', float('nan')):.1f}+-{s0.get('mae_rul_std',0):.1f}"
          f" scatter={s0.get('scatter', float('nan')):.3f}")
    if sc:
        print(f"    ceiling soh={sc.get('soh',{}).get('rmse_soh',float('nan')):.2f}"
              f" rul={sc.get('rul',{}).get('mae_rul',float('nan')):.1f}")
