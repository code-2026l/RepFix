"""Cross-domain unified analysis & figures.

Puts the finance discovery domain and the three scientific validation domains
(HAR / RadioML / battery) on the SAME axes, using only domain-independent
quantities (S_h relative scatter, collapse rate, recovery rate, STF metric gain)
so the gradient-toxicity -> collapse -> STF-repair narrative is uniformly
demonstrated.  NeurIPS-style: Okabe-Ito palette, 7pt sans-serif, minimal ink.

Reads: repro/results/{x_battery_contrast,x_har_contrast,x_rml_contrast}.json
       repro/results/wf_stf_ablation_multiseed.json   (finance discovery)
Writes: repro/figures/cross_domain_* (PNG/SVG)
"""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

OITO = {"blue":"#0072B2","orange":"#E69F00","purple":"#CC79A7","green":"#009E73",
        "sky":"#56B4E9","yellow":"#F0E442","gray":"#999999","blk":"#000000"}
rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Helvetica","Arial"],
    "font.size":7,"axes.linewidth":0.5,"axes.labelsize":7,"axes.titlesize":7.5,
    "xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6,
    "lines.linewidth":1.2,"lines.markersize":3,"savefig.dpi":300,
    "xtick.direction":"in","ytick.direction":"in","axes.spines.top":False,
    "axes.spines.right":False,"legend.frameon":False,"figure.dpi":300})

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(F, exist_ok=True)


def load(name):
    fp = os.path.join(R, name)
    return json.load(open(fp)) if os.path.exists(fp) else None


# =============================================================================
# Build a uniform table: for each domain -> {mode -> {metric_std, rel_scatter,
# recovery_rate, metric_mean}}  (metric is the raw per-domain metric).
# =============================================================================
def domain_table(domain, contrast_json, higher_better=True):
    d = load(contrast_json)
    if not d or d.get("domain") != domain:
        return None
    out = {}
    for r in d["results"]:
        out[r["mode"]] = r
    return out


# How much each mode improves the primary-task metric relative to joint (+ => better)
def metric_delta(table):
    joint = table.get("joint") or table.get("single")
    base = joint["metric_mean"]
    deltas = {}
    for m, r in table.items():
        if m in ("joint", "single"):
            continue
        deltas[m] = (r["metric_mean"] - base) / max(abs(base), 1e-9)
    return deltas, base


def fig_cross_domain_bar():
    domains = [("battery", "x_battery_contrast.json", "SOH"),
               ("har", "x_har_contrast.json", "Activity acc"),
               ("radioml", "x_rml_contrast.json", "Modulation acc")]
    tabs = {}
    for dm, jf, _lb in domains:
        t = domain_table(dm, jf)
        if t:
            tabs[dm] = t

    # collapse / recovery matrix
    stf_modes = ["stf0", "stfhard", "stfsoft"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), dpi=300)
    for ax, (dm, jf, lb) in zip(axes, domains):
        t = tabs.get(dm)
        if not t:
            ax.set_title(f"{dm}\n(data missing)"); continue
        deltas, base = metric_delta(t)
        names = ["STF-0", "STF-hard", "STF-soft"]
        xs = np.arange(len(stf_modes))
        vals = [deltas.get(m, float("nan")) for m in stf_modes]
        cols = [OITO["blue"], OITO["orange"], OITO["purple"]]
        ax.bar(xs, [0 if math.isnan(v) else v * 100 for v in vals], color=cols, width=0.62)
        for x, v in zip(xs, vals):
            if not math.isnan(v):
                ax.text(x, v * 100 + (1.0 if v >= 0 else -3.5), f"{v*100:.0f}%",
                        ha="center", fontsize=6)
        ax.axhline(0, color="#444", linewidth=0.6)
        ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=6)
        ax.set_title(f"{dm.upper()} ({lb})", fontsize=7)
        ax.set_ylim(-15, 90)
        if ax is axes[0]:
            ax.set_ylabel("metric change vs joint (%)")
    fig.suptitle("STF repair across domains: primary-task metric vs joint training",
                 fontsize=8, y=1.02)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(F, f"cross_domain_metrics.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_cross_domain_summary():
    """One panel: relative scatter (S_h / S_single) per mode per available domain,
    with joint=1 axis.  Shows collapse (# collapse) only in weak-recovery domain."""
    domains = [("battery", "x_battery_contrast.json", "SOH (reg)"),
               ("har", "x_har_contrast.json", "Activity (cls)"),
               ("radioml", "x_rml_contrast.json", "Modulation (cls)")]
    fig, ax = plt.subplots(figsize=(4.6, 2.6), dpi=300)
    xs_all, labels = [], []
    for dm, jf, lb in domains:
        t = domain_table(dm, jf)
        if not t:
            continue
        single = t.get("single") or t.get("stf0")
        for m in ["joint", "stf0", "stfsoft"]:
            if m in t:
                r = t[m]
                rs = r.get("rel_scatter_mean") or (r["scatter_mean"] / single["scatter_mean"])
                x = len(xs_all)
                xs_all.append(x)
                labels.append(f"{dm}:{m}")
                col = OITO["blue"] if m == "joint" else (OITO["green"] if m == "stfsoft" else OITO["gray"])
                ax.scatter(x, rs, color=col, s=38, zorder=3, edgecolor="#fff", linewidth=.4)
        ax.axhline(0.3, color="#777", linewidth=.5, linestyle="--", zorder=1)
    ax.text(-0.4, 0.33, "collapse\nthreshold 0.3", fontsize=5.5, color="#777")
    ax.set_xticks(xs_all); ax.set_xticklabels(labels, fontsize=5.2, rotation=30, ha="right")
    ax.set_ylabel(r"$S_h$ / $S_{single}$")
    ax.set_ylim(-0.1, 1.8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(F, f"cross_domain_scatter.{ext}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_cross_domain_bar()
    fig_cross_domain_summary()
    print("figures written to", F)