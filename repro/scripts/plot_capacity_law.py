"""Capacity scaling law figure (A results) — publication-quality.

Reads results/capacity_out_capacity.json (m x alpha grid, 8 seeds) and renders:
  (a) sigma_h vs alpha (log-x) per capacity m  — collapse transition curves
  (b) alpha_c vs m on log-log axes              — capacity scaling law + power fit
  (c) test_ic vs alpha (log-x) per capacity m   — health degradation

Style: NeurIPS-grade, consistent palette, clean spines, tight legends.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "capacity_out_capacity.json")
OUT = os.path.join(HERE, "..", "figures", "fig_capacity_law.png")
OUT_PDF = os.path.join(HERE, "..", "figures", "fig_capacity_law.pdf")

# ---- palette (colorblind-safe, ordered by capacity) -------------------------
CMAP = plt.get_cmap("viridis")
MS = [8, 16, 32, 64, 128, 256]
COLORS = {m: CMAP(0.08 + 0.84 * i / (len(MS) - 1)) for i, m in enumerate(MS)}
ALPHA_C = {"8": 0.1, "16": 1.0, "32": 1.0, "64": 3.0, "128": None, "256": None}


def main():
    with open(SRC) as f:
        data = json.load(f)
    recs = data["records"]

    by_m = {}
    for r in recs:
        by_m.setdefault(r["m"], []).append(r)
    for m in by_m:
        by_m[m].sort(key=lambda r: r["alpha"])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.16,
                        wspace=0.34)

    # ---- (a) sigma_h vs alpha ------------------------------------------------
    ax = axes[0]
    for m in MS:
        rs = by_m[m]
        a = [r["alpha"] for r in rs]
        s = [r["sigma_h"] for r in rs]
        se = [r["sigma_h_std"] for r in rs]
        ax.errorbar(a, s, yerr=se, fmt="o-", ms=4.5, lw=1.6, capsize=2.5,
                    color=COLORS[m], label=f"$m={m}$",
                    markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Auxiliary weight  $\alpha$", fontsize=12)
    ax.set_ylabel(r"Representation variance  $\sigma_h$", fontsize=12)
    ax.set_title(r"(a)  Collapse transition vs capacity", fontsize=12.5)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(fontsize=9, ncol=2, frameon=False, loc="lower left")
    ax.set_ylim(5e-4, 3.5)

    # annotate alpha_c on each curve
    for m in MS:
        ac = ALPHA_C[str(m)]
        if ac is None:
            continue
        ax.axvline(ac, color=COLORS[m], ls="--", lw=1.0, alpha=0.55)
    ax.text(0.045, 0.06, r"$\alpha_c(m)$", fontsize=10, color="#333333")

    # ---- (b) alpha_c vs m (log-log) + power-law fit --------------------------
    ax = axes[1]
    ms_fit = [m for m in MS if ALPHA_C[str(m)] is not None]
    ac_fit = [ALPHA_C[str(m)] for m in ms_fit]
    ax.plot(ms_fit, ac_fit, "o-", color="#c0392b", ms=7, lw=2,
            markeredgecolor="white", markeredgewidth=0.6,
            label=r"measured $\alpha_c(m)$")
    # power-law fit: alpha_c = C * m^gamma
    logm = np.log(np.asarray(ms_fit, dtype=float))
    loga = np.log(np.asarray(ac_fit, dtype=float))
    g, b = np.polyfit(logm, loga, 1)
    C = np.exp(b)
    m_grid = np.logspace(np.log10(8), np.log10(256), 200)
    ax.plot(m_grid, C * m_grid ** g, "--", color="#7f8c8d", lw=1.6,
            label=rf"$\alpha_c = {C:.2f}\,m^{{{g:.2f}}}$")
    # extrapolate: m* where alpha_c crosses 10 (no collapse in range)
    m_star = (10.0 / C) ** (1.0 / g)
    ax.axhline(10.0, color="#95a5a6", ls=":", lw=1.2)
    ax.annotate(r"$\alpha_c>10$ (no collapse)", xy=(256, 10.5),
                fontsize=9, color="#555555", ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Encoder capacity  $m$", fontsize=12)
    ax.set_ylabel(r"Critical auxiliary weight  $\alpha_c$", fontsize=12)
    ax.set_title(r"(b)  Capacity scaling law", fontsize=12.5)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.set_xlim(7, 300)
    ax.set_ylim(0.05, 30)
    ax.text(0.03, 0.03,
            rf"slope $\gamma={g:.2f}$" + "\n" + rf"$m^*\approx{m_star:.0f}$",
            transform=ax.transAxes, fontsize=10, va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fdf6ec", ec="#e0c9a6"))

    # ---- (c) test_ic vs alpha ------------------------------------------------
    ax = axes[2]
    for m in MS:
        rs = by_m[m]
        a = [r["alpha"] for r in rs]
        ic = [r["test_ic"] for r in rs]
        ax.plot(a, ic, "o-", ms=4.5, lw=1.6, color=COLORS[m],
                markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"Auxiliary weight  $\alpha$", fontsize=12)
    ax.set_ylabel(r"Test rank-IC", fontsize=12)
    ax.set_title(r"(c)  Predictive health degradation", fontsize=12.5)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.set_ylim(-0.15, 1.05)
    # shade the collapsed region
    ax.axvspan(1.0, 12, color="#c0392b", alpha=0.06)
    ax.text(2.2, 0.90, "collapsed\nregime", fontsize=9, color="#a93226",
            ha="center")
    ax.axhline(0.0, color="#555555", lw=0.8, ls="--", alpha=0.6)

    fig.suptitle("Capacity scaling of the auxiliary-induced collapse transition",
                 fontsize=14, y=0.985)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"[saved] {OUT}")
    print(f"[saved] {OUT_PDF}")
    print(f"power-law fit: alpha_c = {C:.2f} * m^{g:.2f} ; m*={m_star:.0f}")


if __name__ == "__main__":
    main()
