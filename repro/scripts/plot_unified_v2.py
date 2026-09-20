"""Unified gradient-toxicity figure (panels a-e) -- revision 3.

Changes vs revision 2
---------------------
* Panel (a) now reads the A-share *100 fold-cell* sweep
  (``R5fin_<mode>_s<seed>.json``: 10 walk-forward folds x 10 seeds), i.e. the
  exact JSON family behind Table~\\ref{tab:finance_repair}.  Revision 2 drew an
  independent 6-fold subset whose STF-0 bar was *higher* than joint, flatly
  contradicting the table that the caption cites; the two are now the same
  experiment.  The panel is drawn as a strip plot (all 100 cells) with a
  mean+/-std summary, which suits a comparison whose within-cell spread is
  comparable to the effect size.
* Two-row layout, authored at exactly the width it is included at
  (``\\columnwidth`` = 5.5 in), so every glyph prints at its nominal point
  size.  Revision 2 was authored 10 in wide and included at 4.3 in, which
  shrank 7 pt text to ~3 pt and made the panel (c)/(d) tick labels merge.
* No ``bbox_inches='tight'``: constrained_layout already owns the margins, and
  an exact bounding box keeps the 1:1 inclusion guarantee above.

Run:  python plot_unified_v2.py
Env:  REPFIX_RESULTS overrides the results root.
"""
import glob, json, os
from collections import defaultdict

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
    "legend.fontsize": 5.6, "lines.linewidth": 1.1, "lines.markersize": 3,
    "xtick.direction": "in", "ytick.direction": "in", "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

# ---------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
_ROOTS = [os.environ.get("REPFIX_RESULTS"),
          os.path.join(HERE, "..", "results"),
          os.path.join(HERE, "..", "..", "results"),
          os.path.join(HERE, "..", "..", "repro", "results")]
R = None
for _p in _ROOTS:
    if not _p:
        continue
    _p = os.path.abspath(_p)
    if os.path.isdir(_p) and os.listdir(_p):
        R = R or _p
if R is None:
    R = os.path.abspath(os.path.join(HERE, "..", "results"))
F = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(F, exist_ok=True)
ROOTS = [os.path.abspath(p) for p in _ROOTS if p]


def load(name):
    """First existing copy of *name* across the candidate results roots."""
    for root in ROOTS:
        fp = os.path.join(root, name)
        if os.path.exists(fp):
            with open(fp) as fh:
                return json.load(fh)
    return None


def load_first(cands):
    for c in cands:
        d = load(c)
        if d:
            return d
    return None


def load_glob(pattern):
    """All matches of a relative glob, first root that yields anything wins."""
    for root in ROOTS:
        hit = sorted(glob.glob(os.path.join(root, pattern)))
        if hit:
            return hit
    return []


SWEEP = {"har": ("xcap2_har_fine.json", "xcap_har_fine.json"),
         "radioml": ("xcap2_radioml.json", "xcap_radioml.json"),
         "battery": ("xcap2_battery.json", "xcap_battery_multi.json")}
REPAIR = {"har": ("xrep2_har.json", "xrep_har.json"),
          "radioml": ("xrep2_radioml.json", "xrep_radioml.json"),
          "battery": ("xrep2_battery.json", "xrep_battery_multi.json")}
DOM = {"har": "HAR", "radioml": "RadioML", "battery": "Battery"}
DCOL = {"har": OITO["blue"], "radioml": OITO["orange"], "battery": OITO["green"]}

FIN_MODES = [("joint", "joint", OITO["blue"]),
             ("stf0", "STF-0", OITO["orange"]),
             ("stfhard", "STF\nhard", OITO["purple"]),
             ("stfsoft", "STF\nsoft", OITO["green"])]


def save(fig, name):
    fig.savefig(os.path.join(F, f"{name}.pdf"), facecolor="white")
    fig.savefig(os.path.join(F, f"{name}.png"), facecolor="white", dpi=400)
    plt.close(fig)
    print("fig:", name)


# ---------------------------------------------------------------- QA
def report_overlaps(fig, tag=""):
    """Print any pair of visible text artists whose boxes intersect.

    Cheap and objective substitute for eyeballing a 5-panel figure: any
    overlapping tick label / annotation shows up here before it reaches the
    PDF."""
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
                bad.append(f"    {tag} overlap {ox:.0f}x{oy:.0f}px  "
                           f"{items[i][0]!r}@{items[i][1].x0:.0f},{items[i][1].y0:.0f}"
                           f"  x  {items[j][0]!r}@{items[j][1].x0:.0f},{items[j][1].y0:.0f}")
    print(f"  [{tag}] text artists={len(items)} overlaps={len(bad)}")
    for line in bad:
        print(line)
    return bad


# ---------------------------------------------------------------- (a) finance
def _fin_cells():
    out = []
    for mk, lab, col in FIN_MODES:
        vals = []
        for fp in load_glob(f"R5fin/R5fin_{mk}_s*.json"):
            with open(fp) as fh:
                d = json.load(fh)
            for r in d:
                v = r.get("oos_ic")
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
        if vals:
            out.append((lab, col, np.asarray(vals)))
    return out


def _panel_fin(ax):
    data = _fin_cells()
    if not data:
        ax.text(0.5, 0.5, "R5fin/ not found", ha="center", va="center",
                transform=ax.transAxes, fontsize=6)
        return
    rng = np.random.default_rng(7)
    base = data[0][2].mean()
    means, stds, cols = [], [], []
    for i, (lab, col, v) in enumerate(data):
        jit = rng.uniform(-0.26, 0.26, v.size)
        ax.plot(i + jit, v, ".", color=col, ms=1.05, alpha=0.30,
                zorder=2, rasterized=True)
    for i, (lab, col, v) in enumerate(data):
        means.append(float(v.mean()))
        stds.append(float(v.std()))
        cols.append(col)
    ax.errorbar(np.arange(len(data)), means, yerr=stds, fmt="o", ms=3.0,
                color="k", ecolor="k", elinewidth=0.9, capsize=2.0,
                capthick=0.9, zorder=4)
    ax.axhline(base, ls="--", lw=0.7, color=OITO["gray"], zorder=1)
    lo = min(m - s for m, s in zip(means, stds))
    hi = max(m + s for m, s in zip(means, stds))
    span = hi - lo
    for i, (m, s) in enumerate(zip(means, stds)):
        pct = "(ref)" if i == 0 else f"{(m/base-1)*100:+.0f}%"
        ax.text(i, m + s + 0.03 * span, f"{m:.3f}\n{pct}", ha="center",
                va="bottom", fontsize=5.4, linespacing=1.15, zorder=6)
    ax.set_ylim(lo - 0.06 * span, hi + 0.34 * span)
    ax.set_xlim(-0.55, len(data) - 0.45)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels([l for l, _, _ in data])
    ax.set_ylabel("OOS rank-IC")
    ax.set_title("(a) A-share (discovery)", fontsize=7)


# ---------------------------------------------------------------- (b) phase
def _sweep(dm):
    return load_first(SWEEP[dm])


def _panel_phase(ax):
    added = False
    for dm in ("har", "radioml", "battery"):
        d = _sweep(dm)
        if not d:
            continue
        rows = d["results"]
        ms = sorted(set(r["m"] for r in rows))
        m = ms[len(ms) // 2] if ms else None
        rs = sorted([r for r in rows if r["m"] == m
                     and r.get("rel_scatter_mean") is not None],
                    key=lambda r: r["alpha"])
        if not rs:
            continue
        a = [0.5 * rs[0]["alpha"]] + [r["alpha"] for r in rs]
        s = [1.0] + [r["rel_scatter_mean"] for r in rs]
        sd = [0.0] + [r.get("rel_scatter_std") or 0.0 for r in rs]
        c = DCOL[dm]
        ax.semilogx(a, s, "-o", color=c, ms=2.4, label=f"{DOM[dm]} ($m={m}$)")
        ax.fill_between(a, [y - z for y, z in zip(s, sd)],
                        [min(1.0, y + z) for y, z in zip(s, sd)],
                        color=c, alpha=0.12, lw=0)
        if "raw" in d:
            ra = [r for r in d["raw"] if r.get("m") == m
                  and r.get("rel_scatter") is not None]
            if ra:
                by = defaultdict(list)
                for r in ra:
                    by[r["alpha"]].append(r["rel_scatter"])
                aa = sorted(by)
                if len(aa) >= 3:
                    lo = [min(by[z]) for z in aa]
                    hi2 = [max(by[z]) for z in aa]
                    ax.fill_between([0.5 * aa[0]] + aa, [1.0] + lo,
                                    [1.0] + hi2, color=c, alpha=0.15, lw=0)
        ax.axhline(s[0] / 2.0, ls=":", lw=0.8, color=c, alpha=0.6)
        added = True
    if added:
        ax.set_xlabel(r"toxicity  $\alpha$  (aux scale)")
        ax.set_ylabel(r"$S_h/S_{\mathrm{single}}$")
        ax.set_ylim(0, 1.28)
        ax.set_title(r"(b) Universal collapse: relative scatter", fontsize=7)
        ax.legend(ncol=1, fontsize=5.6, loc="upper right",
                  handlelength=1.6, borderaxespad=0.3)


# ---------------------------------------------------------------- (c,d) repair
def _rep(dm):
    d = load_first(REPAIR[dm])
    return {r["mode"]: r for r in d["results"]} if d else None


def _group_labels(ax, gi, vals, w):
    """Annotate one domain group.

    Three bars 9 px apart cannot each carry a two-decimal label, so the three
    regimes are handled explicitly: all-zero groups get a single group label,
    groups whose three modes agree get a single in-bar label, and the remaining
    case (HAR recovery, where the modes differ) gets staggered labels that clear
    each other vertically."""
    if max(vals) < 1e-9:
        ax.text(gi, 0.06, "0.0", ha="center", va="bottom", fontsize=5.6)
    elif len({round(v, 4) for v in vals}) == 1:
        ax.text(gi, vals[0] * 0.5, f"{vals[0]:.1f}", ha="center", va="center",
                fontsize=5.8, color="white")
    else:
        for k, (v, off) in enumerate(zip(vals, (0.05, 0.16, 0.05))):
            ax.text(gi + (k - 1) * w, v + off,
                    f"{v:.2f}".rstrip("0").rstrip("."), ha="center",
                    va="bottom", fontsize=5.4)


def _panel_repair(axl, axr):
    stf = [("stf0", "STF-0", OITO["blue"]),
           ("stfhard", "STF-hard", OITO["orange"]),
           ("stfsoft", "STF-soft", OITO["purple"])]
    dms = [dm for dm in ("har", "radioml", "battery") if _rep(dm)]
    if not dms:
        return
    x = np.arange(len(dms))
    w = 0.25
    for ax, key, ttl, ylab in ((axl, "collapse_rate", "(c) STF prevents collapse",
                                "collapse rate"),
                               (axr, "recovery_rate", "(d) Primary-metric recovery",
                                "recovery rate")):
        for i, (mk, lab, col) in enumerate(stf):
            vals = [(_rep(dm)[mk][key] if mk in _rep(dm) else np.nan)
                    for dm in dms]
            ax.bar(x + (i - 1) * w, vals, w, color=col, edgecolor="k", lw=0.3,
                   label=lab)
        for gi, dm in enumerate(dms):
            vals = [_rep(dm)[mk][key] for mk, _, _ in stf]
            if np.all(np.isfinite(vals)):
                _group_labels(ax, gi, vals, w)
        ax.plot(x, [_rep(dm)["joint"][key] for dm in dms], "--o",
                color=OITO["blk"], ms=2.6, lw=0.9, label="joint")
        ax.set_xticks(x)
        ax.set_xticklabels([DOM[d] for d in dms], fontsize=5.9)
        ax.set_ylim(-0.05, 1.30)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=7)
    # one legend for the (c)/(d) pair; in (c) every STF bar sits at zero, so the
    # middle of the panel is free and the joint line at 1.0 stays clear
    h, lb = axl.get_legend_handles_labels()
    order = ["joint", "STF-0", "STF-hard", "STF-soft"]
    idx = [lb.index(o) for o in order if o in lb]
    axl.legend([h[i] for i in idx], [lb[i] for i in idx], fontsize=5.4,
               loc="center", ncol=2, handlelength=1.3, columnspacing=0.9,
               borderaxespad=0.2)


# ---------------------------------------------------------------- (e) capacity
def _across(rows, m):
    rs = sorted([r for r in rows if r["m"] == m
                 and r.get("rel_scatter_mean") is not None],
                key=lambda r: r["alpha"])
    if len(rs) < 2:
        return None
    health = rs[0]["rel_scatter_mean"]
    if health < 0.6:
        return None
    half = health / 2.0
    if rs[-1]["rel_scatter_mean"] >= half:
        return None
    for i in range(1, len(rs)):
        s1, s2 = rs[i - 1]["rel_scatter_mean"], rs[i]["rel_scatter_mean"]
        if s2 < half <= s1:
            t = (half - s1) / ((s2 - s1) or 1e-9)
            return float(np.exp(np.log(rs[i - 1]["alpha"])
                                + t * (np.log(rs[i]["alpha"])
                                       - np.log(rs[i - 1]["alpha"]))))
    return None


def _panel_capacity(ax):
    for dm in ("har", "radioml", "battery"):
        d = _sweep(dm)
        if not d:
            continue
        rows = d["results"]
        ms = sorted(set(r["m"] for r in rows))
        pts = [(m, _across(rows, m)) for m in ms]
        pts = [(m, a) for m, a in pts if a is not None and a > 0]
        if len(pts) < 3:
            continue
        ms_ = [p[0] for p in pts]
        ac = [p[1] for p in pts]
        c = DCOL[dm]
        ax.loglog(ms_, ac, "-s", color=c, ms=3.0, label=DOM[dm])
        g = np.polyfit(np.log(ms_), np.log(ac), 1)
        xs = np.linspace(np.log(min(ms_)), np.log(max(ms_)), 40)
        ax.plot(np.exp(xs), np.exp(np.polyval(g, xs)), "--", color=c, lw=0.8,
                alpha=0.7)
        dy = {"har": 5, "radioml": -7, "battery": -7}.get(dm, 5)
        ax.annotate(f"$\\gamma_{{{dm[:3]}}}={g[0]:.2f}$", (ms_[-1], ac[-1]),
                    textcoords="offset points", xytext=(-4, dy), fontsize=5.6,
                    color=c, ha="right", va="bottom")
    ax.set_xlabel("encoder width $m$")
    ax.set_ylabel(r"critical  $\alpha_c$")
    ax.set_xticks([32, 64, 128, 256])
    ax.set_xticklabels(["32", "64", "128", "256"])
    ax.minorticks_off()
    ax.margins(y=0.24)
    ax.set_title(r"(e) Capacity law $\alpha_c\!\propto\!m^{\gamma}$",
                 fontsize=6.4)
    ax.legend(fontsize=5.6, loc="lower left", handlelength=1.5,
              borderaxespad=0.3)


# ---------------------------------------------------------------- main
def main():
    print("results roots:", ROOTS)
    # Authored at exactly the inclusion width (\columnwidth = 5.5 in) so every
    # glyph prints at its nominal point size.  Two rows: a single row of five
    # panels collapses -- the panel titles alone exceed a 1.1 in column, which
    # matplotlib reports as "axes sizes collapsed to zero".
    fig = plt.figure(figsize=(5.5, 2.45), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 1],
                          hspace=0.16, wspace=0.55)
    axa = fig.add_subplot(gs[0, 0:2])
    axb = fig.add_subplot(gs[0, 2:6])
    axc = fig.add_subplot(gs[1, 0:2])
    axd = fig.add_subplot(gs[1, 2:4])
    axe = fig.add_subplot(gs[1, 4:6])
    _panel_fin(axa)
    _panel_phase(axb)
    _panel_repair(axc, axd)
    _panel_capacity(axe)
    bad = report_overlaps(fig, "unified")
    save(fig, "unified_all")
    return bad


if __name__ == "__main__":
    main()
