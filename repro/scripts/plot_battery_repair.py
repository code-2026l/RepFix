"""Battery focus figure: primary-metric-visible collapse and repair.

Revision 2 fixes
----------------
* Two-line tick labels (``STF\\nhard``) instead of 12-degree rotation: at the
  inclusion size the four mode names used to run together into
  ``STF-0STF-hardSTF-soft``.
* No per-bar numbers on panels (b)/(c): every STF rate is exactly 0 or 1 and the
  y axis already carries the scale, so the crowded 1.0/0.0 tags only obscured
  the joint line they sat on.
* Legends moved into regions that are provably empty --- (b) gains headroom
  above the 1.0 bars, (c) puts its single entry over the collapsed joint bar.
* Authored at exactly the width it is included at (``\\columnwidth`` = 5.5 in)
  so glyphs print at their nominal size; revision 1 was authored 7.2 in wide and
  included at 5.06 in, shrinking 6.5 pt text to ~4.5 pt.
* A text-overlap report runs before saving (see ``report_overlaps``).

Run:  python plot_battery_repair.py
Env:  REPFIX_RESULTS overrides the results root.
"""
import json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

OITO = {"blue": "#0072B2", "orange": "#E69F00", "purple": "#CC79A7",
        "green": "#009E73", "gray": "#999999", "blk": "#000000"}
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 6.5, "axes.linewidth": 0.5, "axes.labelsize": 6.8,
    "axes.titlesize": 7.0, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
    "legend.fontsize": 5.5, "lines.linewidth": 1.1, "lines.markersize": 3,
    "xtick.direction": "in", "ytick.direction": "in", "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 300,
})

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOTS = [os.environ.get("REPFIX_RESULTS"),
          os.path.join(HERE, "..", "results"),
          os.path.join(HERE, "..", "..", "results"),
          os.path.join(HERE, "..", "..", "repro", "results")]
ROOTS = [os.path.abspath(p) for p in _ROOTS if p]
F = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(F, exist_ok=True)


def load_rep():
    for name in ("xrep2_battery.json", "xrep_battery_multi.json",
                 "xrep_battery.json"):
        for root in ROOTS:
            p = os.path.join(root, name)
            if os.path.exists(p):
                with open(p) as fh:
                    return json.load(fh)
    return None


def report_overlaps(fig, tag=""):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    items, seen = [], set()
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in seen:
            continue
        seen.add(id(t))
        try:
            items.append((t.get_text().replace("\n", "\\n"),
                          t.get_window_extent(renderer=rend)))
        except Exception:
            pass
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 1.0 and oy > 1.0:
                bad.append(f"    {tag} {ox:.0f}x{oy:.0f}px {items[i][0]!r} x "
                           f"{items[j][0]!r}")
    print(f"  [{tag}] text artists={len(items)} overlaps={len(bad)}")
    for line in bad:
        print(line)
    return bad


def save(fig, name):
    fig.savefig(os.path.join(F, f"{name}.pdf"), facecolor="white")
    fig.savefig(os.path.join(F, f"{name}.png"), facecolor="white", dpi=400)
    plt.close(fig)
    print("fig:", name)


ORDER = [("joint", "joint", OITO["gray"]),
         ("stf0", "STF-0", OITO["blue"]),
         ("stfhard", "STF\nhard", OITO["orange"]),
         ("stfsoft", "STF\nsoft", OITO["green"])]
CEIL = 0.613          # constant-map SOH-RMSE of the collapsed joint model


def main():
    d = load_rep()
    if not d:
        print("no battery repair json found in", ROOTS)
        return
    res = {r["mode"]: r for r in d["results"]}
    lab = [o[1] for o in ORDER]
    col = [o[2] for o in ORDER]
    xs = np.arange(len(ORDER))

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95),
                             constrained_layout=True)
    axa, axb, axc = axes

    # ---- (a) SOH-RMSE with per-seed points
    ms = [res[o[0]]["metric_mean"] for o in ORDER]
    sd = [res[o[0]]["metric_std"] for o in ORDER]
    axa.bar(xs, ms, 0.6, yerr=sd, capsize=2.0, color=col, edgecolor="k",
            linewidth=0.4, error_kw=dict(elinewidth=0.8, capthick=0.8))
    per = {}
    for r in d.get("raw", []):
        per.setdefault(r["mode"], []).append(r["metric"])
    for i, (mode, _, c) in enumerate(ORDER):
        v = np.asarray(per.get(mode, []))
        if v.size:
            jit = np.random.default_rng(i).uniform(-0.17, 0.17, v.size)
            axa.plot(i + jit, v, ".", color="k", ms=1.5, alpha=0.40, zorder=3)
    for x, m, s in zip(xs, ms, sd):
        axa.text(x, m + s + 0.012, f"{m:.3f}", ha="center", va="bottom",
                 fontsize=5.6, zorder=6)
    axa.axhline(CEIL, ls="--", lw=0.8, color=OITO["blk"], alpha=0.55)
    axa.set_xlim(-0.62, 3.75)
    axa.text(3.72, CEIL + 0.012, "constant map", ha="right", va="bottom",
             fontsize=5.2, color="#555555")
    axa.set_ylim(0, max(m + s for m, s in zip(ms, sd)) * 1.42)
    axa.set_xticks(xs)
    axa.set_xticklabels(lab)
    axa.set_ylabel("SOH-RMSE")
    axa.set_title("(a) Collapse vs repair", fontsize=7)

    # ---- (b) collapse / recovery rates: every STF bar is 0 or 1
    coll = [res[o[0]]["collapse_rate"] for o in ORDER]
    rec = [res[o[0]]["recovery_rate"] for o in ORDER]
    w = 0.34
    axb.bar(xs - w / 2, coll, w, color=OITO["gray"], label="collapse",
            edgecolor="k", lw=0.3)
    axb.bar(xs + w / 2, rec, w, color=OITO["green"], label="recovery",
            edgecolor="k", lw=0.3)
    axb.set_xticks(xs)
    axb.set_xticklabels(lab)
    axb.set_ylim(0, 1.52)
    axb.set_yticks([0, 0.5, 1.0])
    axb.set_ylabel("rate")
    axb.set_title("(b) Collapse / recovery", fontsize=7)
    axb.legend(ncol=2, loc="upper center", handlelength=1.2,
               columnspacing=0.9, borderaxespad=0.15, fontsize=5.5)

    # ---- (c) relative scatter restored to the single-task ceiling
    rel = [res[o[0]].get("rel_scatter_mean", np.nan) for o in ORDER]
    axc.bar(xs, rel, 0.6, color=col, edgecolor="k", linewidth=0.4)
    for x, v in zip(xs, rel):
        if np.isfinite(v):
            axc.text(x, v + 0.035, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=5.6)
    axc.axhline(1.0, color=OITO["green"], ls="--", lw=0.9,
                label="single-task ceiling")
    axc.set_xticks(xs)
    axc.set_xticklabels(lab)
    axc.set_ylim(0, 1.48)
    axc.set_yticks([0, 0.5, 1.0])
    axc.set_ylabel(r"$S_h/S_{\mathrm{single}}$")
    axc.set_title("(c) Scatter restored", fontsize=7)
    axc.legend(loc="upper left", borderaxespad=0.2, fontsize=5.5)

    bad = report_overlaps(fig, "battery")
    save(fig, "battery_repair")
    return bad


if __name__ == "__main__":
    main()
