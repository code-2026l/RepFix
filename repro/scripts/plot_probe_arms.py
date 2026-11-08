"""Summarize the corrected probe: certified counts and margin ROC.

Panel (a): certified checkpoints for the frozen probe at the canonical and
optimal scale, and for the corrected probe q* = a/d.
Panel (b): ROC of the certificate margin for predicting sign agreement.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "repro" / "results" / "PBA" / "PBA_har.json"
OUT = ROOT / "paper" / "figures" / "fig_probe_arms"
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

D = json.load(open(SRC))
P = [p for r in D["runs"] for p in r["points"] if p.get("frozen") and p.get("corrected")]
n = len(P)

def fired(tag, field):
    return sum(1 for p in P if p[tag][field])

counts = [fired("frozen", "fired"), fired("frozen", "fired_at_kopt"),
          fired("corrected", "fired")]
labels = ["frozen\ncanonical $k$", "frozen\noptimal $k$", "corrected\n$q^\\star=a/d$"]
colors = ["#8C9BAB", "#4C72B0", "#C44E52"]

def roc(scores, labels_):
    s = np.asarray(scores, float)
    y = np.asarray(labels_, bool)
    order = np.argsort(-s)
    tpr = np.concatenate([[0], np.cumsum(y[order]) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(~y[order]) / (~y).sum()])
    return fpr, tpr

def auc(scores, labels_):
    s = np.asarray(scores, float)
    y = np.asarray(labels_, bool)
    n1, n0 = y.sum(), (~y).sum()
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

m_frozen = [abs(1 - float(p["alpha"]) * float(p["frozen"]["q"]))
            / max(float(p["frozen"]["Delta"]), 1e-12) for p in P]
y_frozen = [p["frozen"]["agree"] for p in P]
m_corr = [abs(1 - float(p["alpha"]) * float(p["corrected"]["q_star"]))
          / max(float(p["corrected"]["Delta_star_min"]), 1e-12) for p in P]
y_corr = [p["corrected"]["agree"] for p in P]

fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55))

ax = axes[0]
bars = ax.bar(range(3), counts, color=colors, width=0.62)
ax.set_xticks(range(3))
ax.set_xticklabels(labels)
ax.set_ylabel("certified checkpoints")
ax.set_ylim(0, 175)
ax.set_title("(a) certified set, zero false fires")
for i, c in enumerate(counts):
    ax.text(i, c + 4, str(c), ha="center", fontsize=7.4)
ax.text(0.02, 0.92, f"of {n} checkpoints", transform=ax.transAxes, fontsize=7)

ax = axes[1]
for m, y, c, lab in ((m_frozen, y_frozen, "#4C72B0", "frozen $q$"),
                     (m_corr, y_corr, "#C44E52", "corrected $q^\\star$")):
    f, t = roc(m, y)
    ax.plot(f, t, color=c, lw=1.2, label=f"{lab} (AUC {auc(m, y):.2f})")
ax.plot([0, 1], [0, 1], ls="--", color="#777777", lw=0.8)
ax.set_xlabel("false-positive rate")
ax.set_ylabel("true-positive rate")
ax.set_title("(b) margin predicts sign agreement")
ax.legend(loc="lower right", frameon=False)

fig.tight_layout(pad=0.35, w_pad=1.1)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
print("wrote", OUT.with_suffix(".pdf"))
print("counts", counts, "auc frozen", auc(m_frozen, y_frozen),
      "auc corrected", auc(m_corr, y_corr))
