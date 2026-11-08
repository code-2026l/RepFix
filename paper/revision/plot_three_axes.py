"""Two-panel summary: contraction vs scale-invariant information, and readout transfer."""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "fig_three_axes.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "pdf.fonttype": 42,
})

summary = json.loads((ROOT / "results" / "REPAUDIT" / "summary.json").read_text())
alphas = [0.0, 0.1, 100.0]
rel = [summary["cells"]["0.0"]["relative_scatter"]["mean"],
       summary["cells"]["0.1"]["relative_scatter"]["mean"],
       summary["cells"]["100.0"]["relative_scatter"]["mean"]]
probe = [summary["cells"]["0.0"]["probe_activity_std"]["mean"],
         summary["cells"]["0.1"]["probe_activity_std"]["mean"],
         summary["cells"]["100.0"]["probe_activity_std"]["mean"]]

readout = json.loads((ROOT / "results" / "READOUT" / "readout_har.json").read_text())
intervention = json.loads((ROOT / "results" / "READOUT" / "stf_readout_har.json").read_text())
rows = readout["rows"]
tags = ["healthy_ref", "frozen", "input_cal", "retrained"]
labels = ["healthy", "frozen", "input\nrecal.", "retrained", "STF-cal"]
acc = [st.mean([r[t]["acc"] for r in rows]) for t in tags]
ece = [st.mean([r[t]["ece"] for r in rows]) for t in tags]
acc.append(intervention["means"]["stfcal"]["acc"])
ece.append(intervention["means"]["stfcal"]["ece"])

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw={"wspace": 0.38})

ax = axes[0]
x = np.arange(len(alphas))
ax.plot(x, rel, "o-", color="#C44E52", lw=1.4, ms=4.5, label="relative scatter")
ax.set_yscale("log")
ax.axhline(0.3, color="#888888", ls="--", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["0", "0.1", "100"])
ax.set_xlabel(r"auxiliary dose $\alpha$")
ax.set_ylabel("relative scatter (log)")
ax.set_ylim(8e-3, 1.4)
ax2 = ax.twinx()
ax2.plot(x, probe, "s-", color="#4C72B0", lw=1.4, ms=4.0, label="scale-invariant probe")
ax2.set_ylim(0.90, 0.99)
ax2.set_ylabel("standardised probe accuracy")
ax.set_title("(a) contraction without information loss")
ax.annotate("0.3 criterion", xy=(2.0, 0.3), xytext=(1.05, 0.33),
            fontsize=7, color="#666666")
lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
ax.legend(lines, [l.get_label() for l in lines], loc="lower left",
          frameon=False, fontsize=7)

ax = axes[1]
colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974"]
bars = ax.bar(np.arange(5), acc, color=colors, width=0.62)
ax.set_xticks(np.arange(5))
ax.set_xticklabels(labels, fontsize=6.4)
ax.set_ylim(0, 1.12)
ax.set_ylabel("accuracy on held-out HAR")
ax.set_title("(b) a frozen readout breaks; STF-cal protects it")
for i, (b, e) in enumerate(zip(bars, ece)):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
            f"{acc[i]:.3f}\nECE {e:.3f}", ha="center", va="bottom", fontsize=5.6)
ax.axhline(1 / 6, color="#888888", ls=":", lw=0.8)
ax.text(4.45, 1 / 6 + 0.02, "chance", ha="right", fontsize=6.3, color="#666666")

fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
print("wrote", OUT)
