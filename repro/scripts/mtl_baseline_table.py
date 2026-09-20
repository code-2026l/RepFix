#!/usr/bin/env python
"""Build the numeric MTL-baseline table for the RepFix paper (local analysis).

Two outputs
-----------
(1) BatteryMTL numeric baseline table (E2a):
      single-task ceiling | sum(joint) | PCGrad | FAMO | CAGrad | STF-0
      x  {MATR, NASA} x {SOH rmse%, RUL MAE, rel. scatter}
    Source: _hpc/conflict_pull/merged_E2a_{DS}_{AGG}.json (seed-merged).

(2) Cross-domain conflict-method table (metric + scatter + collapse rate):
      HAR / Battery / RadioML x {joint, pcgrad, famo, cagrad, stf0}
    Source: _hpc/conflict_pull/x_*contrast*.json + E2c_radioml_conflict.json

Run:  python mtl_baseline_table.py            (uses E2a seeds present)
      python mtl_baseline_table.py --tex      (emit LaTeX rows)
"""
import argparse, glob, json, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "paper", "_hpc", "conflict_pull")
D = os.path.normpath(D)
AGGS = ["sum", "pcgrad", "famo", "cagrad"]
DSETS = ["MATR", "NASA"]


def f(x, p=2):
    try:
        return f"{float(x):.{p}f}"
    except (TypeError, ValueError):
        return "nan"


def load_e2a():
    """Return {ds: {'agg': {..}, 'stf0': {..}, 'ceiling': {..}, 'seeds': [..]}}."""
    out = {}
    for ds in DSETS:
        d = {"agg": {}, "seeds": []}
        for agg in AGGS:
            p = os.path.join(D, f"merged_E2a_{ds}_{agg}.json")
            if not os.path.exists(p):
                continue
            j = json.load(open(p))
            rs = {r["config"]: r for r in j.get("results", [])}
            d.setdefault("stf0", rs.get("stf0", {}))
            d["agg"][agg] = rs.get("joint", {})
            if j.get("seeds"):
                d["seeds"] = j["seeds"]
            sc = j.get("single_ceiling", {}) or {}
            if sc:
                d.setdefault("ceiling", sc.get("soh") or sc.get("rul") or {})
                # merged single_ceiling is a dict keyed by 'soh'/'rul'
                d["ceiling"] = {
                    "soh": (sc.get("soh") or {}).get("rmse_soh"),
                    "rul": (sc.get("rul") or {}).get("mae_rul"),
                }
        out[ds] = d
    return out


def print_e2a(e2a, tex=False):
    print("=" * 78)
    print("(1) BatteryMTL numeric MTL-baseline table (alpha_aux=10, d=128)")
    print("=" * 78)
    for ds in DSETS:
        d = e2a.get(ds) or {}
        if not d.get("agg"):
            continue
        seeds = d.get("seeds")
        n = len(seeds) if seeds else "?"
        print(f"\n--- {ds}  (seeds={seeds}, n={n})")
        c = d.get("ceiling", {})
        print(f"    {'ceiling':<22} SOH={f(c.get('soh'))}  RUL={f(c.get('rul'),1)}")
        for agg in AGGS:
            r = d["agg"].get(agg)
            if not r:
                continue
            print(f"    {agg:<22} SOH={f(r.get('rmse_soh'))}+-{f(r.get('rmse_soh_std'))}"
                  f"  RUL={f(r.get('mae_rul'),1)}+-{f(r.get('mae_rul_std'),1)}"
                  f"  scatter={f(r.get('scatter'),3)}")
        s0 = d.get("stf0") or {}
        if s0:
            print(f"    {'STF-0':<22} SOH={f(s0.get('rmse_soh'))}+-{f(s0.get('rmse_soh_std'))}"
                  f"  RUL={f(s0.get('mae_rul'),1)}+-{f(s0.get('mae_rul_std'),1)}"
                  f"  scatter={f(s0.get('scatter'),3)}")


def print_contrast(tex=False):
    print("\n" + "=" * 78)
    print("(2) Cross-domain conflict methods (metric / scatter / collapse)")
    print("=" * 78)
    files = sorted(glob.glob(os.path.join(D, "x_*contrast.json")))
    files += sorted(glob.glob(os.path.join(D, "E2c_radioml_conflict.json")))
    seen = set()
    for p in files:
        j = json.load(open(p))
        dom = j.get("domain")
        base = os.path.basename(p)
        if dom in seen:
            continue
        seen.add(dom)
        print(f"\n--- {base}  domain={dom}")
        for r in j.get("results", []):
            print(f"    {r.get('mode','?'):<8} metric={f(r.get('metric_mean'),4)}"
                  f"+-{f(r.get('metric_std'),4)} collapse={r.get('collapse_rate')}"
                  f" rel={f(r.get('rel_scatter_mean'),3)} alpha={r.get('alpha')}")


def emit_tex(e2a):
    """LaTeX body for the numeric MTL-baseline table (MATR | NASA)."""
    print("% ---- numeric MTL-baseline table (emit_tex) ----")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{\textbf{MATR}} & \multicolumn{3}{c}{\textbf{NASA}} \\")
    print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    print(r"\textbf{Mode} & \textbf{SOH rms(\%)} & \textbf{RUL MAE} & \textbf{scatter}"
          r" & \textbf{SOH rms(\%)} & \textbf{RUL MAE} & \textbf{scatter} \\")
    print(r"\midrule")
    # ceiling
    cells = []
    for ds in DSETS:
        c = (e2a.get(ds) or {}).get("ceiling", {})
        cells += [f"${f(c.get('soh'))}$", f"${f(c.get('rul'),1)}$", "--"]
    print("single-task ceiling & " + " & ".join(cells) + r" \\")
    order = ["sum", "pcgrad", "famo", "cagrad"]
    label = {"sum": r"\textsc{joint} (sum)", "pcgrad": "PCGrad",
             "famo": "FAMO", "cagrad": "CAGrad"}
    for agg in order:
        cells = []
        present = True
        for ds in DSETS:
            r = (e2a.get(ds) or {}).get("agg", {}).get(agg)
            if not r:
                present = False
                break
            cells += [f"${f(r.get('rmse_soh'))}\\pm{f(r.get('rmse_soh_std'))}$",
                      f"${f(r.get('mae_rul'),1)}\\pm{f(r.get('mae_rul_std'),1)}$",
                      f"${f(r.get('scatter'),3)}$"]
        if present:
            print(f"{label[agg]} & " + " & ".join(cells) + r" \\")
    cells = []
    for ds in DSETS:
        r = (e2a.get(ds) or {}).get("stf0") or {}
        cells += [f"\\textbf{{{f(r.get('rmse_soh'))}}}" if r else "--",
                  f"\\textbf{{{f(r.get('mae_rul'),1)}}}" if r else "--",
                  f"${f(r.get('scatter'),3)}$" if r else "--"]
    print(r"\midrule")
    print(r"\textbf{STF-0} (ours) & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", action="store_true")
    a = ap.parse_args()
    e2a = load_e2a()
    if a.tex:
        emit_tex(e2a)
    else:
        print_e2a(e2a, a.tex)
    print_contrast(a.tex)
