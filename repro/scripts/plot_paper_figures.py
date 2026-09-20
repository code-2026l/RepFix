"""Paper-grade figures for the Gradient-Toxicity / Representational-Collapse /
Spectral-Toxicity-Filtering narrative.  NeurIPS-style: Okabe-Ito colorblind-safe
palette, 7pt sans-serif, minimal ink.  Reads the aggregated result JSONs + the
critical-scaling log produced on the HPC.
"""
import json, os, re, glob, math
import numpy as np
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---- style: NeurIPS-ish, 7pt sans-serif, colorblind-safe (Okabe-Ito) ----
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


def _pop(p):  # load json (may not exist)
    fp = os.path.join(R, p)
    return json.load(open(fp)) if os.path.exists(fp) else None


# ===========================================================================
# Figure 2: phase transition (theory, order parameter sigma_h + test IC)
# ===========================================================================
def fig_phase_transition():
    d = _pop("th_alpha.json")
    if not d:
        return
    alpha = np.array([x["alpha"] for x in d], dtype=float)
    sig = np.array([x["sigma_h"] for x in d], dtype=float)
    ic = np.array([x["test_ic"] for x in d], dtype=float)
    coll = np.array([x["collapse_rate"] for x in d], dtype=float)
    fig, axs = plt.subplots(1, 3, figsize=(6.0, 1.95))
    # predicted alpha_c = 1/rho from rho_v2 (mean init rho)
    rv = _pop("th_rho_v2.json")
    rho_pred = rv["summary"]["rho_proj_init"]["mean"] if rv else None
    for ax in axs:
        ax.set_xscale("log")
    axs[0].plot(alpha, sig, "-o", color=OITO["blue"], ms=3, label=r"$\sigma_h$")
    if rho_pred and rho_pred > 0:
        axs[0].axvline(1.0/rho_pred, color=OITO["orange"], ls="--", lw=1,
                       label=r"pred. $\alpha_c=1/\rho$")
    axs[0].set_xlabel(r"toxicity control $\alpha_{aux}$")
    axs[0].set_ylabel(r"order param. $\sigma_h$")
    axs[0].legend()
    axs[1].plot(alpha, ic, "-o", color=OITO["purple"], ms=3)
    axs[1].set_xlabel(r"$\alpha_{aux}$"); axs[1].set_ylabel("test rank-IC")
    axs[2].plot(alpha, coll, "-s", color=OITO["green"], ms=3)
    axs[2].set_xlabel(r"$\alpha_{aux}$"); axs[2].set_ylabel("collapse rate")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig_phase_transition.pdf")); fig.savefig(os.path.join(F, "fig_phase_transition.png"))
    plt.close(fig); print("fig_phase_transition")


# ===========================================================================
# Figure 3: capacity law (alpha_c vs m) from critical-scaling log + coarse sweep
# ===========================================================================
def _parse_critical_log():
    fp = os.path.join(R, "th_critical.log")
    if not os.path.exists(fp):
        return {}
    rows = {}
    for ln in open(fp):
        m = re.match(r"\s*m=\s*(\d+)\s*a=\s*([\d.eE+-]+)\s*->\s*sigma_h=([\d.eE]+)[+-][\d.eE]+ ic=([+-]?[\d.eE]+).*?coll=([\d.eE]+)", ln)
        if m:
            rows.setdefault(int(m.group(1)), []).append(
                dict(a=float(m.group(2)), sig=float(m.group(3)), ic=float(m.group(4)), c=float(m.group(5))))
    return rows


def _alpha_c_from_rows(rows, thresh_frac=0.3):
    # first alpha where scatter drops below thresh_frac * healthy scatter (a=0)
    rows = sorted(rows, key=lambda r: r["a"])
    if not rows:
        return None
    s0 = rows[0]["sig"] if rows[0]["a"] == 0 else None
    for r in rows:
        if s0 is not None and r["sig"] < thresh_frac * s0:
            return r["a"]
    # fallback: extrapolate crossing
    return rows[-1]["a"]


def fig_capacity_law():
    from collections import defaultdict
    cr = _pop("th_critical.json")
    recs = cr["records"] if cr and "records" in cr else None
    widths = cr["widths"] if cr else [8, 16, 32, 64, 128, 256]

    def _interp_alpha_c(m, target=0.0015):
        # alpha needed for the (absolute) representation scatter to fall to `target`
        rows = sorted([r for r in recs if r["m"] == m], key=lambda r: r["alpha"])
        if len(rows) < 2:
            return None
        for i in range(1, len(rows)):
            a0, s0 = rows[i - 1]["alpha"], rows[i - 1]["sigma_h"]
            a1, s1 = rows[i]["alpha"], rows[i]["sigma_h"]
            if s1 < target <= s0:
                t = (np.log(target) - np.log(s0)) / ((np.log(s1) - np.log(s0)) or 1e-9)
                return float(np.exp(np.log(a0) + t * (np.log(a1) - np.log(a0))))
        return None

    mvals, ac = [], []
    for m in widths:
        x = _interp_alpha_c(m) if recs else None
        if x:
            mvals.append(m); ac.append(x)

    fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.2))
    # left: normalized scatter vs alpha for each width
    cols = [OITO["blue"], OITO["orange"], OITO["green"], OITO["purple"], OITO["sky"], OITO["yellow"]]
    if recs:
        for i, m in enumerate(widths):
            rows = sorted([r for r in recs if r["m"] == m], key=lambda r: r["alpha"])
            s0 = rows[0]["sigma_h"]
            ax[0].semilogx([r["alpha"] for r in rows], [r["sigma_h"] / s0 for r in rows],
                           "-o", color=cols[i % 6], ms=2.2, label=f"$m={m}$")
    ax[0].axhline(0.3, color=OITO["gray"], ls=":", lw=1)
    ax[0].set_xlabel(r"$\alpha_{aux}$"); ax[0].set_ylabel(r"$\sigma_h / \sigma_h(0)$")
    ax[0].legend(ncol=2, fontsize=5.5)
    if len(mvals) >= 3:
        ax[1].loglog(mvals, ac, "-o", color=OITO["blk"], ms=4)
        lx, ly = np.log(mvals), np.log(ac)
        g = np.polyfit(lx, ly, 1)
        xs = np.linspace(min(lx), max(lx), 50)
        ax[1].plot(np.exp(xs), np.exp(np.polyval(g, xs)), "--", color=OITO["orange"], lw=1)
        ax[1].text(0.03, 0.10, r"$\alpha_c \propto m^{%.2f}$" % g[0], transform=ax[1].transAxes, fontsize=7.5)
    ax[1].set_xlabel("hidden width $m$"); ax[1].set_ylabel(r"critical $\alpha_c$")
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig_capacity_law.pdf")); fig.savefig(os.path.join(F, "fig_capacity_law.png"))
    plt.close(fig)
    print("fig_capacity_law", dict(zip(mvals, [round(a, 4) for a in ac])), "gamma=", g[0] if len(mvals) >= 3 else "nan")


# ===========================================================================
# Figure 4: finance STF contrast (OOS IC) -- real discovery domain
# ===========================================================================
def _wf_fold_ic(p, mode):
    d = _pop(p)
    if not d:
        return None, []
    seen = {}
    for r in d:
        seen[r["fold"]] = r["oos_ic"]
    return mode, [seen[k] for k in sorted(seen)]


def fig_finance_stf():
    data = []
    for p, label in [("wf_all_detach0.json", "joint"),
                     ("wf_all_stfhard.json", "STF-hard"),
                     ("wf_all_stfsoft.json", "STF-soft")]:
        m, vals = _wf_fold_ic(p, label)
        if m:
            data.append((label, vals))
    if not data:
        print("no finance"); return
    fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.1))
    pos = np.arange(len(data))
    means = [np.mean(v) for _, v in data]
    stds = [np.std(v) for _, v in data]
    ax[0].bar(pos, means, 0.5, yerr=stds, capsize=2, color=[OITO["blue"], OITO["green"], OITO["orange"]],
              edgecolor="k", linewidth=0.4)
    ax[0].set_xticks(pos); ax[0].set_xticklabels([l.split("-")[0] if l!="STF-hard" else "STF-h" for l,_ in data])
    ax[0].set_ylabel("OOS rank-IC")
    ax[0].set_ylim(0, max(means)*1.5)
    # show relative gain over joint
    jm = means[0]
    for i, v in enumerate(means):
        ax[0].text(i, v + stds[i] + 0.004, f"{(v/jm-1)*100:+.0f}%", ha="center", fontsize=6)
    for m_, v in data:
        ax[1].plot(np.arange(len(v)), v, "-o", ms=2.5, label=m_, color=OITO[[ "blue","green","orange"][[l for l,_ in data].index(m_)]])
    ax[1].set_xlabel("walk-forward fold"); ax[1].set_ylabel("OOS rank-IC")
    ax[1].legend(loc="lower left", fontsize=5.5)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig_finance_stf.pdf")); fig.savefig(os.path.join(F, "fig_finance_stf.png"))
    plt.close(fig); 
    print("fig_finance_stf", {l: round(np.mean(v),4) for l,v in data})


if __name__ == "__main__":
    fig_phase_transition()
    fig_capacity_law()
    fig_finance_stf()
    print("DONE")