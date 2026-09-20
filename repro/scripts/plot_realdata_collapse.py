"""Real-data collapse transition figure (B, deepened bridge) — v4, clean labels.

v4 fixes v3 panel (b): every IC value label was stacked ABOVE its point
(`v + 0.010`, `va="bottom"`), so the negative cluster (c_aux = 5/20/100,
IC ~ -0.05) collided vertically and the c_aux=100 label drifted over the
collapse-basin text.  Here each IC label sits on a SHORT VERTICAL LEADER LINE
positioned on the point's signed side (positive above, negative below) with
automatic vertical de-collision, and the explanatory notes are pinned to the
axis corners / white boxes so no glyph ever overlaps another.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "wf_collapse_v2.json")
OUT_PNG = os.path.join(HERE, "..", "figures", "fig_realdata_collapse.png")
OUT_PDF = os.path.join(HERE, "..", "figures", "fig_realdata_collapse.pdf")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "grid.color": "#cfcfcf",
    "grid.alpha": 0.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def state_color(s, thresh=1e-3):
    """Blue when the representation is healthy, red when collapsed."""
    return ("#2b6ca8" if s > thresh else "#c0392b")


def signed_ic_labels(x, y, base=0.024, tol=0.030, xgap=0.8):
    """Collision-free label y-positions for panel (b).

    Labels go on the point's signed side (positive above, negative below).
    A per-pair guard only pushes the later point further out when two
    same-side, x-close labels would sit closer than ``tol`` in value units,
    so genuinely different values (e.g. c=100 near zero) keep their natural
    short leader instead of being shoved away.
    """
    xs = np.asarray(x, float)
    ys = np.asarray(y, float)
    n = len(xs)
    side = np.where(ys >= 0, 1.0, -1.0)
    ly = ys + side * base
    idx = np.argsort(np.log10(np.maximum(xs, 1e-6)))
    for a in range(1, n):
        i = idx[a]
        for b in range(a):
            j = idx[b]
            if side[i] != side[j]:
                continue
            if np.log10(xs[i]) - np.log10(xs[j]) > xgap:
                continue
            if abs(ly[i] - ly[j]) < tol:
                if side[i] > 0:
                    ly[i] = max(ly[i], ly[j] + tol)
                else:
                    ly[i] = min(ly[i], ly[j] - tol)
    return side, ly


def main():
    data = json.load(open(SRC))
    data.sort(key=lambda r: r["fixed_w"])
    ws = np.array([r["fixed_w"] for r in data], dtype=float)
    scatter = np.array([r["collapse"] for r in data], dtype=float)
    fv = np.array([r["fixed_var"] for r in data], dtype=float)
    ic = np.array([r["oos_ic"] for r in data], dtype=float)

    THRESH = 1e-3
    xs = ws + 0.0
    xs[xs == 0.0] = 0.35   # slide the healthy anchor just off 0

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.35))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.70, bottom=0.20,
                        wspace=0.28)

    # =================== (a) order parameter ===================
    ax = axes[0]
    mtox = ws >= 1.0          # toxic ladder runs
    idx = np.where(mtox)[0]
    for j in range(len(idx) - 1):
        a, b = idx[j], idx[j + 1]
        col = state_color(scatter[a])
        ax.plot([xs[a], xs[b]], [scatter[a], scatter[b]], "-", color=col,
                lw=1.8, alpha=0.80, zorder=1, solid_capstyle="round")
    ax.axhline(THRESH, color="#c0392b", lw=1.6, ls="--", zorder=0)
    ax.text(0.015, 0.05,
            "collapse threshold:  $\\mathcal{S}_h \\leq 10^{-3}$",
            transform=ax.transAxes, fontsize=10, color="#c0392b",
            ha="left", va="bottom", fontweight="bold",
            bbox=dict(fc="#fdeaea", ec="#c0392b", lw=0.8, pad=3))
    for x, w, s in zip(xs, ws, scatter):
        col = state_color(s)
        ax.plot([x], [s], "o", ms=10, color=col, zorder=4,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.text(x, s * 2.2, f"{s:.2g}", ha="center", va="bottom", fontsize=11,
                color=col, fontweight="bold")
    ax.annotate("", xy=(xs[-1], scatter[-1] * 1.4),
                xytext=(xs[-2], scatter[-2] * 3.2),
                arrowprops=dict(arrowstyle="<->", color="#34495e", lw=1.6))
    # place the scale note at the double-arrow's geometric midpoint
    midx = float(np.sqrt(xs[-1] * xs[-2]))
    midy = float(10 ** (0.5 * (np.log10(scatter[-1] * 1.4) +
                               np.log10(scatter[-2] * 3.2))))
    ax.text(midx * 1.28, midy * 1.18, "$\\times 10^3$",
            fontsize=11, color="#34495e", ha="center", va="bottom",
            fontweight="bold")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.15, 400); ax.set_ylim(7e-5, 9.0)
    ax.set_xticks([0.35, 1, 5, 20, 100])
    ax.set_xticklabels([r"$0$", r"$1$", r"$5$", r"$20$", r"$100$"])
    ax.set_xlabel("Toxicity weight  $c_{\\mathrm{aux}}$")
    ax.set_ylabel("Cross-sample scatter  $\\mathcal{S}_h$  (log)")
    ax.set_title(r"(a)  void order parameter $\to$ 0", pad=10)
    ax.grid(True, which="both", ls=":", alpha=0.35)

    # =================== (b) predictive health ===================
    ax = axes[1]
    ax.axhline(0.0, color="#7f8c8d", lw=1.1, ls="--", alpha=0.8)
    side, ly = signed_ic_labels(xs, ic)

    # connecting line by state (positive run blue, negative run red)
    for j in range(len(idx) - 1):
        a, b = idx[j], idx[j + 1]
        col = "#c0504d" if ic[a] < 0 else "#2c7fb8"
        ax.plot([xs[a], xs[b]], [ic[a], ic[b]], "-", color=col, lw=1.8,
                alpha=0.80, zorder=1)

    # points + leader line + value label (signed side)
    for x, v, s, sgn, lv in zip(xs, ic, scatter, side, ly):
        col = state_color(s)
        ax.plot([x], [v], "o", ms=10, color=col, zorder=4,
                markeredgecolor="white", markeredgewidth=1.2)
        tip = v + sgn * 0.006          # end of the leader just off the dot
        ax.plot([x, x], [tip, lv - sgn * 0.010], "-", color=col,
                lw=1.0, alpha=0.85, zorder=2)
        ax.text(x, lv, f"{v:+.3f}", ha="center",
                va="bottom" if sgn > 0 else "top",
                fontsize=11, color=col, fontweight="bold")

    # healthy-region note (top-left, outside any data label)
    ax.text(0.030, 0.965, "healthy multi-task band  ($\\approx$ detach-free)",
            transform=ax.transAxes, fontsize=9.5, color="#2c7fb8",
            ha="left", va="top")
    # beneficial sweet spot (arrow into open top-centre)
    ax.annotate("", xy=(xs[1], ic[1] + 0.012), xytext=(4.5, 0.135),
                arrowprops=dict(arrowstyle="->", color="#1f6fb2", lw=1.6))
    ax.text(4.5, 0.120, "light toxicity beneficial  (IC $+0.087$)",
            fontsize=10, color="#1f6fb2", ha="left", va="top",
            bbox=dict(fc="#e8f1f9", ec="#1f6fb2", lw=0.8, pad=3))
    # collapse basin (axvspan + bottom-right text)
    ax.axvspan(5, 400, color="#c0392b", alpha=0.06, zorder=0)
    ax.text(320, -0.112, "collapse basin", fontsize=10, color="#a93226",
            ha="right", va="center", fontweight="bold")
    # c_aux=100 state box (shifted left into the open gap, clear of the point)
    ax.text(45, 0.052,
            r"$\mathcal{S}_h{\approx}1.7{\times}10^{-4},\ \; "
            r"f_v{\approx}1.1{\times}10^{-7}$",
            fontsize=10, color="#333333", va="center", ha="center",
            bbox=dict(fc="white", ec="#bbbbbb", lw=0.8, pad=3))

    ax.set_xscale("log")
    ax.set_xlim(0.15, 400); ax.set_ylim(-0.135, 0.17)
    ax.set_xticks([0.35, 1, 5, 20, 100])
    ax.set_xticklabels([r"$0$", r"$1$", r"$5$", r"$20$", r"$100$"])
    ax.set_xlabel("Toxicity weight  $c_{\\mathrm{aux}}$")
    ax.set_ylabel("OOS rank-IC")
    ax.set_title(r"(b)  predictive collapse", pad=10)
    ax.grid(True, which="both", ls=":", alpha=0.35)

    fig.suptitle(
        "Auxiliary-induced collapse on real market data "
        "(full-market StudentModel, walk-forward fold 0, rank-loss primary)",
        fontsize=12.5, y=0.97, color="#222222")

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG.replace(".png", "_hi.png"), dpi=600,
                bbox_inches="tight")
    print(f"[saved] {OUT_PNG}")
    print(f"[saved] {OUT_PNG.replace('.png', '_hi.png')} (600 dpi)")
    print(f"[saved] {OUT_PDF}")
    print("\nlabel layout (panel b):")
    for x, v, sgn, lv in zip(xs, ic, side, ly):
        print(f"  x={x:.2f}  ic={v:+.4f}  side={'+' if sgn>0 else '-'}  "
              f"label_y={lv:+.4f}")


if __name__ == "__main__":
    main()