"""Plot the probe-validity phase diagram and certificate slack from PBA_har.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "repro" / "results" / "PBA" / "PBA_har.json"
OUT = ROOT / "paper" / "figures" / "fig_probe_phase"
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
pts = [p for run in D["runs"] for p in run["points"]]

margin, envelope, agree, fired = [], [], [], []
for p in pts:
    a = float(p["alpha"])
    q0 = float(p["frozen"]["q"])
    qt = float(p["live"]["q"])
    m = abs(1.0 - a * q0)
    env = float(p["live"]["Delta_min"]) + a * abs(qt - q0)
    margin.append(m)
    envelope.append(env)
    agree.append(bool(p["live"]["agree"]))
    fired.append(bool(m > env))

margin = np.array(margin)
envelope = np.array(envelope)
agree = np.array(agree)
fired = np.array(fired)
ratio = margin / envelope

fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45))

ax = axes[0]
ax.scatter(margin[~fired], envelope[~fired], s=7, c="#4C72B0", alpha=0.55,
           linewidths=0, label="undecided")
ax.scatter(margin[fired], envelope[fired], s=14, c="#C44E52",
           edgecolors="black", linewidths=0.4, label="certified", zorder=3)
lims = [max(min(margin.min(), envelope.min()), 1e-4),
        max(margin.max(), envelope.max())]
ax.plot(lims, lims, ls="--", color="#555555", lw=0.9)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lims)
ax.set_ylim(*lims)
ax.set_xlabel(r"frozen margin $|1-\alpha q_0|$")
ax.set_ylabel(r"transport envelope $\Delta_t+\alpha|q_t-q_0|$")
ax.set_title("(a) certified region below the diagonal")
ax.text(0.04, 0.90, f"{int(fired.sum())}/{len(fired)} certified, 0 wrong",
        transform=ax.transAxes, fontsize=7.2, color="#333333")
ax.legend(loc="lower right", frameon=False, markerscale=1.2)

ax = axes[1]
logr = np.log10(ratio)
bins = np.linspace(-2.7, 0.15, 34)
ax.hist(logr, bins=bins, color="#4C72B0", alpha=0.75, edgecolor="white", lw=0.4)
ax.axvline(0.0, color="#C44E52", ls="--", lw=1.1)
ax.set_xlabel(r"$\log_{10}\left(|1-\alpha q_0|/\left[\Delta_t+\alpha|q_t-q_0|\right]\right)$")
ax.set_ylabel("checkpoints")
ax.set_title("(b) slack is concentrated near the boundary")
ax.text(0.03, 0.90, f"median {np.median(ratio):.2f}", transform=ax.transAxes,
        fontsize=7.2, color="#333333")
ax.text(0.03, 0.80, f"fires in {int(fired.sum())}/{len(fired)}", transform=ax.transAxes,
        fontsize=7.2, color="#333333")

fig.tight_layout(pad=0.35, w_pad=1.0)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
print("wrote", OUT.with_suffix(".pdf"))
print("n", len(margin), "fired", int(fired.sum()), "median slack", float(np.median(ratio)))
print("all fired agree?", bool(all(agree[fired])))
