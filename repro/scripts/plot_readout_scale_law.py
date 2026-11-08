"""Plot the critical-scale law from results/READOUT/readout_scale_law.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "READOUT" / "readout_scale_law.json"
OUT = ROOT / "paper" / "figures" / "fig_readout_scale_law"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

d = json.load(open(SRC))
rows = d["rows"]
cs = np.array(d["c_grid"])
n = len(rows)

def stack(key, sub=None):
    out = []
    for r in rows:
        vals = []
        for rec in r["curve"]:
            v = rec[key]
            if sub is not None:
                v = v[sub]
            vals.append(v)
        out.append(vals)
    return np.array(out)

acc_f = stack("frozen", "acc")
ece_f = stack("frozen", "ece")
ece_ft = stack("frozen", "ece_temp")
acc_r = stack("recalibrated", "acc")
ece_r = stack("recalibrated", "ece")
flip = stack("flip_fraction")

cstar = np.array([r["critical"]["c_star_median"] for r in rows])
cstar_med = float(np.median(cstar))

fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55))

ax = axes[0]
for y, color, label, ls in [
    (acc_f, "#C44E52", "frozen readout", "-"),
    (acc_r, "#55A868", "input-recalibrated", "-"),
]:
    m, s = y.mean(0), y.std(0)
    ax.plot(cs, m, ls, color=color, lw=1.8, label=label)
    ax.fill_between(cs, m - s, m + s, color=color, alpha=0.14, lw=0)
ax.axvline(cstar_med, color="#555555", ls="--", lw=0.9)
ax.axhline(acc_r.mean(), color="#55A868", ls=":", lw=0.8)
ax.set_xscale("log")
ax.set_xlim(min(cs), max(cs))
ax.set_ylim(0.10, 1.02)
ax.set_xlabel(r"representation scale $c$ (fraction of healthy)")
ax.set_ylabel("accuracy")
ax.set_title("(a) frozen vs scale-invariant readout")
ax.text(0.02, 0.08, f"$c^\\star={cstar_med:.3f}$", transform=ax.transAxes,
        fontsize=9, color="#333333")

ax = axes[1]
for y, color, label, ls in [
    (ece_f, "#C44E52", "frozen readout (healthy $T$)", "-"),
    (ece_ft, "#8172B2", "frozen + temperature $T$", "--"),
    (ece_r, "#55A868", "input-recalibrated", "-"),
]:
    m, s = y.mean(0), y.std(0)
    ax.plot(cs, m, ls, color=color, lw=1.7, label=label)
    ax.fill_between(cs, m - s, m + s, color=color, alpha=0.12, lw=0)
ax.axvline(cstar_med, color="#555555", ls="--", lw=0.9)
ax.set_xscale("log")
ax.set_xlim(min(cs), max(cs))
ax.set_ylim(0, 0.55)
ax.set_xlabel(r"representation scale $c$ (fraction of healthy)")
ax.set_ylabel("ECE")
ax.set_title("(b) calibration map does not repair scale")
handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
           ncol=3, frameon=False, handlelength=1.8, columnspacing=1.8,
           fontsize=9)
fig.tight_layout(pad=0.35, w_pad=1.0, rect=(0, 0.11, 1, 1))
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
print("wrote", OUT.with_suffix(".pdf"))
print("c* mean/median", cstar.mean(), cstar_med)
print("frozen acc 1/0.01", acc_f.mean(0)[0], acc_f.mean(0)[cs == 0.01][0])
