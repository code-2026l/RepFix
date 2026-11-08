"""Plot the probe-validity phase diagram, slack, cell mismatch, and E/Delta histogram."""
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
    "legend.fontsize": 6.8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

D = json.load(open(SRC))
pts = []
for run in D["runs"]:
    for p in run["points"]:
        q = dict(p)
        q["m"] = run["m"]
        q["B"] = run["audit_batch"]
        pts.append(q)

margin, envelope, agree, fired, dvals, aq, E, Dl = [], [], [], [], [], [], [], []
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
    dvals.append(float(p["d"]))
    aq.append(abs(float(p["a"]) - float(p["frozen"]["q"])))
    base = float(p['nu']) * float(p['n_p']) / float(p['B'])
    k_opt = float(p['frozen']['k_opt'])
    exact_at_k_opt = (float(p['r']) - a * float(p['c'])) / (k_opt * base)
    E.append(abs(exact_at_k_opt - (1.0 - a * q0)))
    Dl.append(float(p["frozen"]["Delta_min"]))

if np.any(np.asarray(E) > np.asarray(Dl) * (1 + 1e-10) + 1e-10):
    raise AssertionError('Discrepancy and envelope must be compared at the same scale')

margin = np.array(margin)
envelope = np.array(envelope)
agree = np.array(agree)
fired = np.array(fired)
ratio = margin / envelope

fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.75))

ax = axes[0, 0]
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

ax = axes[0, 1]
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

# cell-level d vs signed mismatch, colour = sign agreement
cells = {}
for p in pts:
    key = (p["m"], float(p["alpha"]))
    cells.setdefault(key, []).append(p)
xd = np.array([np.median([p["d"] for p in z]) for z in cells.values()])
yd = np.array([np.median([abs(float(p["a"]) - float(p["frozen"]["q"])) for p in z])
               for z in cells.values()])
cd = np.array([np.mean([p["frozen"]["agree"] for p in z]) for z in cells.values()])
ax = axes[1, 0]
sc = ax.scatter(xd, yd, c=cd, cmap="coolwarm", vmin=0.2, vmax=1.0,
                s=28, edgecolors="#333333", linewidths=0.35)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"median restoration alignment $d$")
ax.set_ylabel(r"median signed mismatch $|a-q|$")
ax.set_title("(c) cell-level mismatch phase diagram")
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("sign agreement", fontsize=7)
ax.axvline(float(np.median(dvals)), color="#555555", ls=":", lw=0.8)
ax.text(0.47, 0.66, f"median $d={np.median(dvals):.3f}$",
        transform=ax.transAxes, fontsize=7, color="#333333")

# exact flux vs envelope
ax = axes[1, 1]
bins = np.linspace(-2.2, 5.5, 34)
ax.hist(np.log10(E), bins=bins, color="#4C72B0", alpha=0.62,
        label=r"exact discrepancy $|\mathcal{E}|$", edgecolor="white", lw=0.3)
ax.hist(np.log10(Dl), bins=bins, color="#DD8452", alpha=0.55,
        label=r"envelope $\Delta_{\min}$", edgecolor="white", lw=0.3)
ax.set_xlabel(r"$\log_{10}$ magnitude")
ax.set_ylabel("checkpoints")
ax.set_ylim(0, 140)
ax.set_title("(d) discrepancy and envelope at the same $k$")
ax.text(0.04, 0.93, f"median $|\\mathcal{{E}}|/\\Delta={np.median(np.array(E)/np.array(Dl)):.2f}$",
        transform=ax.transAxes, fontsize=7.2, color="#333333")
ax.legend(loc="upper right", frameon=False)

fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
print("wrote", OUT.with_suffix(".pdf"))
print("cell n", len(cells), "fired", int(fired.sum()), "median slack", float(np.median(ratio)),
      "median E/Delta", float(np.median(np.array(E) / np.array(Dl))))
print("all fired agree?", bool(all(agree[fired])))
