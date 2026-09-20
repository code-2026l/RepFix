# -*- coding: utf-8 -*-
"""Per-condition (standard/fast-charge/extreme-T) gate benefit check."""
import json, os

# The pulled per-condition JSONs live outside the repository (paper/_hpc is
# ignored, and the pulls are produced on the cluster).  Resolve them relative
# to the repository root and allow an override, so the script never carries a
# machine-specific absolute path into the released package.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.environ.get("REPFIX_PULL", os.path.join(_ROOT, "paper", "_hpc", "merged_pull"))
COND = {"0": "standard", "1": "fast-charge", "2": "extreme-T"}


def per_cond(path, tag):
    if not os.path.exists(path):
        print(f"  [missing] {os.path.basename(path)}")
        return
    d = json.load(open(path))
    rs = {r["config"]: r for r in d.get("results", [])}
    print(f"--- {tag}")
    for cfg in ("joint", "stf0", "stf0_w"):
        r = rs.get(cfg)
        if not r or "per_cond" not in r:
            continue
        parts = []
        for g in ("0", "1", "2"):
            pc = r["per_cond"].get(g)
            if pc:
                parts.append(f"{COND[g]}: soh={pc['soh']:.2f} rul={pc['rul']:.0f} n={pc['n']}")
        print(f"    {cfg:<8} " + " | ".join(parts))


print("=== MATR (full tri-weight gate vs stf0) ===")
per_cond(os.path.join(D, "merged_MATR_main.json"), "MATR_main")
print()
print("=== MATR single-term gates ===")
for t in ("rate", "temp", "degrad"):
    per_cond(os.path.join(D, f"merged_MATR_wterm_{t}.json"), f"wterm_{t}")
print()
print("=== NASA ===")
per_cond(os.path.join(D, "merged_NASA_main.json"), "NASA_main")
