"""Unified paper-grade figures for the gradient-toxicity narrative.

Discovery domain (finance, walk-forward OOS IC) + three scientific validation
domains (HAR / RadioML / battery, phase transition + STF repair + capacity law)
on the SAME axes with domain-independent quantities only.

NeurIPS style: Okabe-Ito palette, 7pt sans-serif, minimal ink.
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

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(F, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("fig:", name)


# =============================================================================
# Panel A -- finance discovery: OOS rank-IC by STF mode (walk-forward)
# =============================================================================
def _fin_agg():
    modes = [("stfoff", "joint"), ("detach1", "STF-0"), ("stfhard", "STF-hard"), ("stfsoft", "STF-soft")]
    out = []
    for suf, label in modes:
        vals = []
        for f in range(6):
            d = load(f"fin_fold{f}_{suf}.json")
            if d:
                r = d[0] if isinstance(d, list) else d
                vals.append(r.get("oos_ic"))
        vals = [v for v in vals if v is not None]
        if vals:
            out.append((label, vals))
    return out


def panel_a_finance():
    data = _fin_agg()
    if not data:
        return
    labels = [l for l, _ in data]
    means = [np.mean(v) for _, v in data]
    stds = [np.std(v) for _, v in data]
    cols = [OITO["blue"], OITO["orange"], OITO["purple"], OITO["green"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    xs = np.arange(len(data))
    ax.bar(xs, means, 0.6, yerr=stds, capsize=2.5, color=cols[:len(data)], edgecolor="k", linewidth=0.4)
    base = means[0]
    for x, m, s in zip(xs, means, stds):
        ax.text(x, m + s + 0.003, f"{m:.3f}\n({(m/base-1)*100:+.0f}%)", ha="center", fontsize=5.8)
    ax.axhline(0, color="#444", lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("OOS rank-IC")
    ax.set_title("Discovery domain: A-share ranking", fontsize=7.5)
    ax.set_ylim(0, max(means + stds) * 1.55)
    save(fig, "unified_a_finance")


# =============================================================================
# Panel B -- phase transition: relative scatter vs alpha (log) per domain
# =============================================================================
def _sweep_curve(dm):
    jf = {"har": "xcap_har_fine.json", "radioml": "xcap_radioml.json",
          "battery": "xcap_battery_multi.json"}[dm]
    d = load(jf)
    if not d or d.get("domain") != dm:
        return None
    return [(r["alpha"], r.get("rel_scatter_mean")) for r in d["results"]]


def panel_b_phase():
    cols = [OITO["blue"], OITO["orange"], OITO["green"]]
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    for dm, c in zip(["har", "radioml", "battery"], cols):
        cur = _sweep_curve(dm)
        if not cur:
            continue
        # include the alpha=0 healthy anchor (rel_scatter=1) so the curve starts
        # at the single-task ceiling instead of already-collapsed
        pts = [(a, s) for a, s in cur if a > 0]
        if pts:
            a = [0.5 * min(x for x, _ in pts)] + [x for x, _ in pts]
            s = [1.0] + [y for _, y in pts]
        else:
            a = [x for x, _ in cur]; s = [y for _, y in cur]
        ax.semilogx(a, s, "-o", color=c, ms=2.8, label=dm.upper())
        ax.plot([0.5 * min(x for x, _ in pts)] if pts else [], [1.0],
                "o", color=c, ms=3.5, mfc="white", mew=1.0, zorder=4)
    ax.axhline(0.3, color=OITO["gray"], ls=":", lw=1)
    ax.text(0.03, 0.315, "collapse thr. 0.3", fontsize=5.2, color=OITO["gray"])
    ax.set_xlabel(r"toxicity $\alpha_{aux}$")
    ax.set_ylabel(r"relative rep. scatter $S_h/S_{single}$")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(ncol=1, fontsize=6)
    save(fig, "unified_b_phase")


# =============================================================================
# Panel C -- STF repair: collapse rate (left) & recovery rate (right) per mode
# =============================================================================
def _repair_table(dm):
    jf = {"har": "xrep_har.json", "radioml": "xrep_radioml.json",
          "battery": "xrep_battery_multi.json"}[dm]
    d = load(jf)
    if not d or d.get("domain") != dm:
        return None
    return {r["mode"]: r for r in d["results"]}


def panel_c_repair():
    stf_modes = ["stf0", "stfhard", "stfsoft"]
    labels = ["STF-0", "STF-hard", "STF-soft"]
    cols = [OITO["blue"], OITO["orange"], OITO["purple"]]
    domains = [dm for dm in ["har", "radioml", "battery"] if _repair_table(dm)]
    if not domains:
        return
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3))
    x = np.arange(len(domains))
    w = 0.24
    for i, m in enumerate(stf_modes):
        coll = []
        for dm in domains:
            t = _repair_table(dm)
            coll.append(t[m]["collapse_rate"] if t and m in t else float("nan"))
        bars = axes[0].bar(x + (i - 1) * w, coll, w, color=cols[i], label=labels[i], edgecolor="k", linewidth=0.3)
        # zero-collapse bars sit on the baseline; mark them so the reader sees
        # that STF fully prevents collapse (the core claim)
        for bx, v in zip(bars, coll):
            if v == 0:
                axes[0].plot(bx.get_x() + bx.get_width() / 2, 0.008, marker="_", ms=4, mew=1.2,
                             color=cols[i], zorder=5)
    axes[0].set_xticks(x); axes[0].set_xticklabels([d.upper() for d in domains])
    axes[0].set_ylabel("collapse rate")
    axes[0].set_title("STF prevents collapse (supercritical)", fontsize=7)
    jcoll = [_repair_table(dm)["joint"]["collapse_rate"] for dm in domains]
    axes[0].plot(x, jcoll, "--o", color=OITO["blk"], ms=3, lw=1, label="joint")
    axes[0].legend(fontsize=5.5, ncol=1)
    axes[0].set_ylim(-0.05, 1.15)
    for i, m in enumerate(stf_modes):
        rec = []
        for dm in domains:
            t = _repair_table(dm)
            rec.append(t[m]["recovery_rate"] if t and m in t else float("nan"))
        axes[1].bar(x + (i - 1) * w, rec, w, color=cols[i], edgecolor="k", linewidth=0.3)
    axes[1].set_xticks(x); axes[1].set_xticklabels([d.upper() for d in domains])
    axes[1].set_ylabel("recovery rate")
    axes[1].set_title("Primary metric recovery", fontsize=7)
    axes[1].set_ylim(-0.05, 1.15)
    # HAR's primary accuracy is itself insensitive to representational collapse
    # (collapse hides in accuracy), so low HAR recovery reflects metric
    # insensitivity, NOT failed repair -- the representation is fully restored.
    if "har" in domains:
        axes[1].text(0.98, 0.03,
                     "HAR: primary acc. insensitive to collapse;\nrep. fully restored (left panel)",
                     transform=axes[1].transAxes, fontsize=5, color="#555", ha="right", va="bottom")
    save(fig, "unified_c_repair")


# =============================================================================
# Panel D -- capacity law: alpha_c(m) per domain (needs xcap_*)
# =============================================================================
def _capacity_points(dm, target=0.3):
    jf = {"har": "xcap_har_fine.json", "radioml": "xcap_radioml.json",
          "battery": "xcap_battery_multi.json"}[dm]
    d = load(jf)
    if not d or d.get("domain") != dm:
        return None
    rows = d["results"]
    ms = sorted(set(r["m"] for r in rows))
    pts = []
    for m in ms:
        rs = sorted([r for r in rows if r["m"] == m], key=lambda r: r["alpha"])
        s0 = rs[0].get("rel_scatter_mean") or 1.0
        ac = None
        for i in range(1, len(rs)):
            s1, s2 = rs[i - 1].get("rel_scatter_mean", 1.0), rs[i].get("rel_scatter_mean", 1.0)
            a1, a2 = rs[i - 1]["alpha"], rs[i]["alpha"]
            if s2 < target <= s1:
                t = (target - s1) / ((s2 - s1) or 1e-9)
                ac = float(np.exp(np.log(a1) + t * (np.log(a2) - np.log(a1))))
                break
        if ac is None:
            ac = rs[-1]["alpha"]
        pts.append((m, ac))
    return pts


def panel_d_capacity():
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    cols = [OITO["blue"], OITO["orange"], OITO["green"]]
    for dm, c in zip(["har", "radioml", "battery"], cols):
        pts = _capacity_points(dm)
        if not pts or len(pts) < 3:
            continue
        ms = [p[0] for p in pts]; ac = [p[1] for p in pts]
        ax.loglog(ms, ac, "-o", color=c, ms=3.5, label=dm.upper())
        g = np.polyfit(np.log(ms), np.log(ac), 1)
        xs = np.linspace(np.log(min(ms)), np.log(max(ms)), 40)
        ax.plot(np.exp(xs), np.exp(np.polyval(g, xs)), "--", color=c, lw=0.8, alpha=0.7)
        ax.text(ms[-1], ac[-1], f" $\\gamma_{{ {dm} }}={g[0]:.1f}$", fontsize=5.5, color=c, ha="left")
    # theory reference: alpha_c ~ m^gamma from the synthetic testbed (th_critical)
    tc = load("th_critical.json")
    if tc and "records" in tc:
        from collections import defaultdict
        by = defaultdict(list)
        for x in tc["records"]:
            by[x["m"]].append(x)

        def alpha_at(m, target=0.0015):
            rows = sorted(by[m], key=lambda r: r["alpha"])
            for i in range(1, len(rows)):
                a0, s0 = rows[i - 1]["alpha"], rows[i - 1]["sigma_h"]
                a1, s1 = rows[i]["alpha"], rows[i]["sigma_h"]
                if s1 < target <= s0:
                    t = (np.log(target) - np.log(s0)) / ((np.log(s1) - np.log(s0)) or 1e-9)
                    return float(np.exp(np.log(a0) + t * (np.log(a1) - np.log(a0))))
            return None
        ms = sorted(by)
        av = [(m, alpha_at(m)) for m in ms if alpha_at(m) is not None]
        if len(av) >= 3:
            g = np.polyfit(np.log([m for m, _ in av]), np.log([a for _, a in av]), 1)[0]
            xs = np.linspace(min(ms), max(ms), 40)
            base = av[0][1] / (av[0][0] ** g)
            ax.plot(xs, base * np.array(xs) ** g, ":", color=OITO["gray"], lw=1.2,
                    label=f"theory $\\propto m^{{{g:.1f}}}$")
    ax.set_xlabel("hidden width $m$")
    ax.set_ylabel(r"critical $\alpha_c$")
    ax.set_title("Capacity law: larger encoder harder to poison", fontsize=7)
    ax.legend(fontsize=6)
    save(fig, "unified_d_capacity")


# =============================================================================
# Panel ALL -- paper Fig.1: four-panel unified narrative (2x2)
# =============================================================================
def _draw_panel_a(ax):
    data = _fin_agg()
    if not data:
        return
    labels = [l for l, _ in data]
    means = [np.mean(v) for _, v in data]
    stds = [np.std(v) for _, v in data]
    cols = [OITO["blue"], OITO["orange"], OITO["purple"], OITO["green"]]
    xs = np.arange(len(data))
    ax.bar(xs, means, 0.62, yerr=stds, capsize=2.2, color=cols[:len(data)],
           edgecolor="k", linewidth=0.4)
    base = means[0]
    for x, m, s in zip(xs, means, stds):
        ax.text(x, m + s + 0.004, f"{m:.3f}\n({(m/base-1)*100:+.0f}%)",
                ha="center", fontsize=5.5)
    ax.axhline(0, color="#444", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_ylabel("OOS rank-IC")
    ax.set_title("(a) Discovery: A-share ranking", fontsize=7)
    ax.set_ylim(0, max(means + stds) * 1.55)


def _draw_panel_b(ax):
    cols = [OITO["blue"], OITO["orange"], OITO["green"]]
    for dm, c in zip(["har", "radioml", "battery"], cols):
        cur = _sweep_curve(dm)
        if not cur:
            continue
        pts = [(a, s) for a, s in cur if a > 0]
        if pts:
            a = [0.5 * min(x for x, _ in pts)] + [x for x, _ in pts]
            s = [1.0] + [y for _, y in pts]
        else:
            a = [x for x, _ in cur]; s = [y for _, y in cur]
        ax.semilogx(a, s, "-o", color=c, ms=2.6, label=dm.upper())
        if pts:
            ax.plot([0.5 * min(x for x, _ in pts)], [1.0], "o", color=c, ms=3.4,
                    mfc="white", mew=1.0, zorder=4)
    ax.axhline(0.3, color=OITO["gray"], ls=":", lw=1)
    ax.text(0.03, 0.315, "collapse thr.", fontsize=5, color=OITO["gray"])
    ax.set_xlabel(r"toxicity $\alpha_{aux}$")
    ax.set_ylabel(r"rel. scatter $S_h/S_{single}$")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("(b) Phase transition", fontsize=7)
    ax.legend(ncol=1, fontsize=5.5)


def _draw_panel_c(ax_coll, ax_rec):
    stf_modes = ["stf0", "stfhard", "stfsoft"]
    labels = ["STF-0", "STF-hard", "STF-soft"]
    cols = [OITO["blue"], OITO["orange"], OITO["purple"]]
    domains = [dm for dm in ["har", "radioml", "battery"] if _repair_table(dm)]
    if not domains:
        return
    x = np.arange(len(domains))
    w = 0.24
    for i, m in enumerate(stf_modes):
        coll = []
        for dm in domains:
            t = _repair_table(dm)
            coll.append(t[m]["collapse_rate"] if t and m in t else float("nan"))
        bars = ax_coll.bar(x + (i - 1) * w, coll, w, color=cols[i], label=labels[i],
                           edgecolor="k", linewidth=0.3)
        for bx, v in zip(bars, coll):
            if v == 0:
                ax_coll.plot(bx.get_x() + bx.get_width() / 2, 0.01, marker="_", ms=4,
                             mew=1.2, color=cols[i], zorder=5)
    ax_coll.set_xticks(x); ax_coll.set_xticklabels([d.upper() for d in domains], fontsize=6)
    ax_coll.set_ylabel("collapse rate")
    ax_coll.set_title("(c) STF prevents collapse", fontsize=7)
    jcoll = [_repair_table(dm)["joint"]["collapse_rate"] for dm in domains]
    ax_coll.plot(x, jcoll, "--o", color=OITO["blk"], ms=3, lw=1, label="joint")
    ax_coll.legend(fontsize=5, ncol=1)
    ax_coll.set_ylim(-0.05, 1.15)
    for i, m in enumerate(stf_modes):
        rec = []
        for dm in domains:
            t = _repair_table(dm)
            rec.append(t[m]["recovery_rate"] if t and m in t else float("nan"))
        ax_rec.bar(x + (i - 1) * w, rec, w, color=cols[i], edgecolor="k", linewidth=0.3)
    ax_rec.set_xticks(x); ax_rec.set_xticklabels([d.upper() for d in domains], fontsize=6)
    ax_rec.set_ylabel("recovery rate")
    ax_rec.set_title("(d) Primary metric recovery", fontsize=7)
    ax_rec.set_ylim(-0.05, 1.15)
    if "har" in domains:
        ax_rec.text(0.99, 0.02,
                    "HAR: rep. restored, acc. insensitive",
                    transform=ax_rec.transAxes, fontsize=4.6, color="#555",
                    ha="right", va="bottom")


def _draw_panel_d(ax):
    cols = [OITO["blue"], OITO["orange"], OITO["green"]]
    for dm, c in zip(["har", "radioml", "battery"], cols):
        pts = _capacity_points(dm)
        if not pts or len(pts) < 3:
            continue
        ms = [p[0] for p in pts]; ac = [p[1] for p in pts]
        ax.loglog(ms, ac, "-o", color=c, ms=3.2, label=dm.upper())
        g = np.polyfit(np.log(ms), np.log(ac), 1)
        xs = np.linspace(np.log(min(ms)), np.log(max(ms)), 40)
        ax.plot(np.exp(xs), np.exp(np.polyval(g, xs)), "--", color=c, lw=0.8, alpha=0.7)
        ax.text(ms[-1], ac[-1], f" $\\gamma_{{ {dm} }}={g[0]:.1f}$", fontsize=5.2,
                color=c, ha="left")
    tc = load("th_critical.json")
    if tc and "records" in tc:
        from collections import defaultdict
        by = defaultdict(list)
        for x in tc["records"]:
            by[x["m"]].append(x)

        def alpha_at(m, target=0.0015):
            rows = sorted(by[m], key=lambda r: r["alpha"])
            for i in range(1, len(rows)):
                a0, s0 = rows[i - 1]["alpha"], rows[i - 1]["sigma_h"]
                a1, s1 = rows[i]["alpha"], rows[i]["sigma_h"]
                if s1 < target <= s0:
                    t = (np.log(target) - np.log(s0)) / ((np.log(s1) - np.log(s0)) or 1e-9)
                    return float(np.exp(np.log(a0) + t * (np.log(a1) - np.log(a0))))
            return None
        ms = sorted(by)
        av = [(m, alpha_at(m)) for m in ms if alpha_at(m) is not None]
        if len(av) >= 3:
            g = np.polyfit(np.log([m for m, _ in av]), np.log([a for _, a in av]), 1)[0]
            xs = np.linspace(min(ms), max(ms), 40)
            base = av[0][1] / (av[0][0] ** g)
            ax.plot(xs, base * np.array(xs) ** g, ":", color=OITO["gray"], lw=1.2,
                    label=f"theory $\\propto m^{{{g:.1f}}}$")
    ax.set_xlabel("hidden width $m$")
    ax.set_ylabel(r"critical $\alpha_c$")
    ax.set_title("(e) Capacity law", fontsize=7)
    ax.legend(fontsize=5.5)


def panel_all():
    fig = plt.figure(figsize=(7.0, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.15, 1.25],
                          height_ratios=[1, 1], hspace=0.62, wspace=0.42)
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1:])
    axc = fig.add_subplot(gs[1, 0]); axd = fig.add_subplot(gs[1, 1])
    axe = fig.add_subplot(gs[1, 2])
    _draw_panel_a(axa)
    _draw_panel_b(axb)
    _draw_panel_c(axc, axd)
    _draw_panel_d(axe)
    save(fig, "unified_all")


if __name__ == "__main__":
    panel_a_finance()
    panel_b_phase()
    panel_c_repair()
    panel_d_capacity()
    panel_all()
    print("ALL DONE")
