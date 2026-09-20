"""Unified gradient-toxicity figure -- revision 5: six panels, seed-level gate panel.

Changes vs revision 4 (plot_unified_v3.py)
-------------------------------------------
* Layout 2x3 (six panels, each 2 grid columns): the calibration story now has
  its own panel.  Panel (f) shows the NYUv2 seed-level gate experiment at the
  calibrated dose alpha=0.5 (m=64, ten seeds): joint training collapses in
  every seed, while STFcal survives exactly in the seeds whose one-probe
  rho_hat exceeds 1/alpha = 2 -- the gate implements the prescription seed by
  seed.  Per-seed probe values come from the STFcal runs; the detached warm-up
  is seed-identical across modes, so the same probe applies to the joint runs
  of the same seed.
* Panel (b) narrows from four to two grid columns; its legend is dropped (the
  domain colours are already identified by panel (e)'s legend) and the xlabel
  shortened.  Panel data and the figure box (5.5 x 2.45 in) are unchanged.
* Same QA as v4: text-overlap counter must report 0; the saved PDF must carry
  TrueType programs (/FontFile2) and no Type 3 fonts.

Run:  python plot_unified_v4.py
Env:  REPFIX_RESULTS overrides the results root.
"""
import glob, json, os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D

OITO = {"blue": "#0072B2", "sky": "#56B4E9", "orange": "#E69F00",
        "green": "#009E73", "gray": "#999999", "blk": "#000000",
        "ink": "#3A3A3A", "verm": "#D55E00"}
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 6.5, "axes.linewidth": 0.5, "axes.labelsize": 6.4,
    "axes.titlesize": 7.0, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
    "legend.fontsize": 5.4, "lines.linewidth": 1.1, "lines.markersize": 3,
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
    for root in ROOTS:
        hit = sorted(glob.glob(os.path.join(root, pattern)))
        if hit:
            return hit
    return []


SWEEP = {"har": ("xcap2_har_fine.json", "xcap_har_fine.json"),
         "radioml": ("xcap2_radioml.json", "xcap_radioml.json"),
         # xcap3 first: its grid reaches the healthy branch (rel_scatter ~ 1.00 at
         # alpha <= 1e-3), so the battery curve survives the health guard in _across
         # and the panel finally carries a measured battery exponent.
         "battery": ("xcap3_battery.json", "xcap2_battery.json",
                     "xcap_battery_multi.json")}
REPAIR = {"har": ("xrep2_har.json", "xrep_har.json"),
          "radioml": ("xrep2_radioml.json", "xrep_radioml.json"),
          "battery": ("xrep2_battery.json", "xrep_battery_multi.json")}
DOM = {"har": "HAR", "radioml": "RadioML", "battery": "Battery"}
DCOL = {"har": OITO["blue"], "radioml": OITO["orange"], "battery": OITO["green"]}

FIN_MODES = [("joint", "joint"), ("stf0", "STF-0"),
             ("stfhard", "STF\nhard"), ("stfsoft", "STF\nsoft")]


def save(fig, name):
    fp = os.path.join(F, f"{name}.pdf")
    fig.savefig(fp, facecolor="white")
    fig.savefig(os.path.join(F, f"{name}.png"), facecolor="white", dpi=400)
    plt.close(fig)
    raw = open(fp, "rb").read()
    type3 = b"/Type3" in raw
    ttf = b"/FontFile2" in raw
    print(f"fig: {name}  ({len(raw)} bytes)  Type3={type3}  TrueType={ttf}")
    assert not type3, "Type 3 fonts leaked into the PDF"
    assert ttf, "TrueType programs missing from the PDF"


# ---------------------------------------------------------------- QA
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
                bad.append(f"    {tag} overlap {ox:.0f}x{oy:.0f}px  "
                           f"{items[i][0]!r}@({items[i][1].x0:.0f},{items[i][1].y0:.0f})"
                           f"  x  {items[j][0]!r}@({items[j][1].x0:.0f},{items[j][1].y0:.0f})")
    print(f"  [{tag}] text artists={len(items)} overlaps={len(bad)}")
    for line in bad:
        print(line)
    return bad


def letters(ax, s, centre):
    ax.set_title(s, loc="left", fontweight="bold", fontsize=8, pad=3)
    ax.text(0.21, 1.015, centre, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=7)


# ---------------------------------------------------------------- (a) finance
def _fin_cells():
    out = []
    for mk, lab in FIN_MODES:
        vals = []
        for fp in load_glob(f"R5fin/R5fin_{mk}_s*.json"):
            with open(fp) as fh:
                d = json.load(fh)
            for r in d:
                v = r.get("oos_ic")
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
        if vals:
            out.append((lab, np.asarray(vals)))
    return out


def _panel_fin(ax):
    data = _fin_cells()
    if not data:
        ax.text(0.5, 0.5, "R5fin/ not found", ha="center", va="center",
                transform=ax.transAxes, fontsize=6)
        return
    rng = np.random.default_rng(7)
    base = data[0][1].mean()
    means, stds = [], []
    for i, (lab, v) in enumerate(data):
        jit = rng.uniform(-0.26, 0.26, v.size)
        ax.plot(i + jit, v, ".", color=OITO["ink"], ms=1.05, alpha=0.30,
                zorder=2, rasterized=True)
    for i, (lab, v) in enumerate(data):
        means.append(float(v.mean()))
        stds.append(float(v.std()))
    ax.errorbar(np.arange(len(data)), means, yerr=stds, fmt="o", ms=3.0,
                color="k", ecolor="k", elinewidth=0.9, capsize=2.0,
                capthick=0.9, zorder=4)
    ax.axhline(base, ls="--", lw=0.7, color=OITO["gray"], zorder=1)
    lo = min(m - s for m, s in zip(means, stds))
    hi = max(m + s for m, s in zip(means, stds))
    span = hi - lo
    for i, (m, s) in enumerate(zip(means, stds)):
        pct = "(ref)" if i == 0 else f"{(m/base-1)*100:+.0f}%"
        # stf0's 100-cell mean over R5fin is 0.03649 -> 0.036 (not 0.037);
        # Table 10 is kept in sync with this panel.
        ax.text(i, m + s + 0.03 * span, f"{m:.3f}\n{pct}", ha="center",
                va="bottom", fontsize=5.4, linespacing=1.15, zorder=6)
    ax.set_ylim(lo - 0.06 * span, hi + 0.44 * span)
    ax.set_xlim(-0.55, len(data) - 0.45)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels([l for l, _ in data])
    ax.set_ylabel("OOS rank-IC")
    letters(ax, "(a)", "A-share")


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
        ax.semilogx(a, s, "-o", color=c, ms=2.2, lw=0.9)
        ax.fill_between(a, [y - z for y, z in zip(s, sd)],
                        [min(1.0, y + z) for y, z in zip(s, sd)],
                        color=c, alpha=0.13, lw=0)
        added = True
    if added:
        ax.axhline(0.5, ls="--", lw=0.7, color=OITO["gray"], alpha=0.85)
        tb = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.97, 0.5, "half height", transform=tb, ha="right",
                va="bottom", fontsize=5.0, color="#777777")
        ax.set_xlabel(r"dose $\alpha$", labelpad=1.5)
        ax.set_ylabel(r"$S_h/S_{\mathrm{single}}$")
        ax.set_ylim(0, 1.25)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xticks([1e-3, 1e-1, 1e1, 1e3])
        ax.set_xticklabels([r"$10^{-3}$", r"$10^{-1}$", r"$10^{1}$",
                            r"$10^{3}$"])
        ax.minorticks_off()
        letters(ax, "(b)", "Universal collapse")


# ---------------------------------------------------------------- (c,d) repair
def _rep(dm):
    d = load_first(REPAIR[dm])
    return {r["mode"]: r for r in d["results"]} if d else None


def _group_labels(ax, gi, vals, w):
    if max(vals) < 1e-9:
        ax.text(gi, 0.05, "0.0", ha="center", va="bottom", fontsize=5.4)
    elif len({round(v, 4) for v in vals}) == 1:
        ax.text(gi, vals[0] * 0.5, f"{vals[0]:.1f}", ha="center", va="center",
                fontsize=5.6, color="white")
    else:
        for k, (v, off) in enumerate(zip(vals, (0.05, 0.16, 0.05))):
            ax.text(gi + (k - 1) * w, v + off,
                    f"{v:.2f}".rstrip("0").rstrip("."), ha="center",
                    va="bottom", fontsize=5.2)


def _panel_repair(axl, axr):
    stf = [("stf0", "STF-0", OITO["gray"]),
           ("stfhard", "STF-hard", OITO["blue"]),
           ("stfsoft", "STF-soft", OITO["sky"])]
    dms = [dm for dm in ("har", "radioml", "battery") if _rep(dm)]
    if not dms:
        return
    x = np.arange(len(dms))
    w = 0.25
    for ax, key, ylab in ((axl, "collapse_rate", "collapse rate"),
                          (axr, "recovery_rate", "recovery rate")):
        for i, (mk, lab, col) in enumerate(stf):
            vals = [(_rep(dm)[mk][key] if mk in _rep(dm) else np.nan)
                    for dm in dms]
            ax.bar(x + (i - 1) * w, vals, w, color=col, lw=0, label=lab)
        for gi, dm in enumerate(dms):
            vals = [_rep(dm)[mk][key] for mk, _, _ in stf]
            if np.all(np.isfinite(vals)):
                _group_labels(ax, gi, vals, w)
        ax.plot(x, [_rep(dm)["joint"][key] for dm in dms], "--o",
                color=OITO["blk"], ms=2.4, lw=0.9, label="joint")
        ax.set_xticks(x)
        ax.set_xticklabels([DOM[d] for d in dms], fontsize=5.7)
        ax.set_ylim(-0.04, 1.16)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel(ylab)
    letters(axl, "(c)", "Collapse rate")
    letters(axr, "(d)", "Recovery rate")
    h, lb = axl.get_legend_handles_labels()
    order = ["joint", "STF-0", "STF-hard", "STF-soft"]
    idx = [lb.index(o) for o in order if o in lb]
    axl.legend([h[i] for i in idx], [lb[i] for i in idx], fontsize=5.2,
               loc="center", bbox_to_anchor=(0.5, 0.56), ncol=1,
               handlelength=1.2, columnspacing=0.8, borderaxespad=0.2,
               handletextpad=0.5, labelspacing=0.25)


# ---------------------------------------------------------------- (e) capacity
def _across(rows, m, level=0.5):
    """Critical alpha: the first downward crossing of rel_scatter through level.

    rel_scatter is normalised by the single-task (unpoisoned) model -- see
    cross_domain_mtl.sweep_alpha, which divides by the per-seed single-task
    scatter -- so 1.0 IS the healthy anchor by construction and the half-height
    threshold is 0.5 for every domain and every grid.

    An earlier version crossed half of the value at the SMALLEST alpha present
    in the file, which equals 0.5 only when the grid starts at alpha = 0.  The
    three grids start at different alphas, so that version read the domains at
    different depths: HAR (grid from 3e-4, rel_scatter 0.69-0.91 there) crossed
    0.34-0.45 while battery and RadioML crossed ~0.50, and HAR's exponent came
    out at 0.30 instead of 0.67.
    """
    rs = sorted([r for r in rows if r["m"] == m
                 and r.get("rel_scatter_mean") is not None],
                key=lambda r: r["alpha"])
    if len(rs) < 2:
        return None
    if rs[0]["rel_scatter_mean"] < level:
        return None          # grid never reaches the healthy side of the level
    if rs[-1]["rel_scatter_mean"] >= level:
        return None          # curve never falls below the level
    for i in range(1, len(rs)):
        s1, s2 = rs[i - 1]["rel_scatter_mean"], rs[i]["rel_scatter_mean"]
        if s2 < level <= s1:
            t = (level - s1) / ((s2 - s1) or 1e-9)
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
        ax.loglog(ms_, ac, "-s", color=c, ms=2.8, label=DOM[dm])
        g = np.polyfit(np.log(ms_), np.log(ac), 1)
        print("[capacity] %-8s gamma=%+.3f  alpha_c=%s"
              % (dm, g[0], " ".join("%d:%.4g" % (m, a) for m, a in zip(ms_, ac))),
              flush=True)
        xs = np.linspace(np.log(min(ms_)), np.log(max(ms_)), 40)
        ax.plot(np.exp(xs), np.exp(np.polyval(g, xs)), "--", color=c, lw=0.8,
                alpha=0.7)
        # Anchor each gamma callout to its OWN curve endpoint and push it away
        # from the neighbouring curve.  The three curves stack vertically
        # (battery ~1.0, radioml ~0.25, har ~0.004), so the two upper labels
        # take opposite sides of their endpoints: battery's above, radioml's
        # below.  A shared sign would stack them across curves and read as a
        # mispairing -- battery's callout landing next to radioml's line.
        off = {"har": (2, -8), "radioml": (2, -8), "battery": (2, 6)}.get(dm, (2, -8))
        ha = "left" if dm in ("har", "radioml", "battery") else "right"
        ax.annotate(f"$\\gamma_{{{dm[:3]}}}={g[0]:.2f}$", (ms_[-1], ac[-1]),
                    textcoords="offset points", xytext=off, fontsize=5.4,
                    color=c, ha=ha, va="bottom")
    ax.set_xlabel("width $m$", labelpad=1.5)
    ax.set_ylabel(r"critical $\alpha_c$")
    ax.set_xticks([32, 64, 128, 256])
    ax.set_xticklabels(["32", "64", "128", "256"])
    ax.minorticks_off()
    ax.margins(y=0.24)
    letters(ax, "(e)", "Capacity law")
    # Three curves stack at ~1.0 / ~0.25 / ~0.005, so the only free band is the
    # one between har and radioml (axes-fraction 0.24..0.63) -- too short for a
    # three-row legend.  Lay the legend out as a single row and keep it low in
    # that band, clear of the HAR curve below and radioml's gamma callout above.
    ax.legend(fontsize=5.2, loc="lower left", bbox_to_anchor=(0.02, 0.25),
              ncol=3, handlelength=1.2, columnspacing=0.7,
              borderaxespad=0.2, labelspacing=0.25)


# ---------------------------------------------------------------- (f) gate
def _panel_gate(ax):
    d = load("R7/R7_dose_nyu_a0.5.json")
    if not d:
        ax.text(0.5, 0.5, "R7 a0.5 not found", ha="center", va="center",
                transform=ax.transAxes, fontsize=6)
        return
    raw = d["raw"]
    stf = {r["seed"]: r for r in raw
           if r.get("mode") == "stfcal" and r.get("cal_rho_u") is not None
           and not (isinstance(r.get("cal_rho_u"), float) and np.isnan(r["cal_rho_u"]))}
    jnt = {r["seed"]: r for r in raw if r.get("mode") == "joint"}
    inv = 1.0 / 0.5
    for s, r in sorted(jnt.items()):
        rho = stf[s]["cal_rho_u"] if s in stf else None
        if rho is None:
            continue
        ax.plot(rho * 0.965, r["rel_scatter"], "o", mfc="none", mec="#555555",
                ms=3.0, mew=0.8, zorder=3)
    for s, r in sorted(stf.items()):
        ax.plot(r["cal_rho_u"] * 1.035, r["rel_scatter"], "s", color=OITO["verm"],
                ms=3.0, zorder=4)
    ax.axvline(inv, ls="--", lw=0.7, color=OITO["gray"])
    ax.set_xscale("log")
    ax.set_xlim(1.82, 2.34)
    ax.set_ylim(-0.08, 1.42)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([1.9, 2.0, 2.2])
    ax.set_xticklabels(["1.9", "2.0", "2.2"])
    ax.minorticks_off()
    ax.set_xlabel(r"probe $\hat\rho$ (per seed)", labelpad=1.5)
    ax.set_ylabel("relative scatter")
    letters(ax, "(f)", "Seed-level gate")
    handles = [Line2D([], [], linestyle="none", marker="o", mfc="none",
                      mec="#555555", ms=3.0, label="joint"),
               Line2D([], [], linestyle="none", marker="s", color=OITO["verm"],
                      ms=3.0, label=r"STFcal")]
    ax.legend(handles=handles, fontsize=5.2, loc="upper left",
              borderaxespad=0.25, labelspacing=0.25, handletextpad=0.4)
    tb = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.01, 0.03, r"$1/\alpha{=}2$", transform=tb, ha="left", va="bottom",
            fontsize=5.0, color="#555555", rotation=90)


# ---------------------------------------------------------------- main
def main():
    print("results roots:", ROOTS)
    fig = plt.figure(figsize=(5.5, 2.45), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 1],
                          hspace=0.30, wspace=0.52)
    axa = fig.add_subplot(gs[0, 0:2])
    axb = fig.add_subplot(gs[0, 2:4])
    axc = fig.add_subplot(gs[0, 4:6])
    axd = fig.add_subplot(gs[1, 0:2])
    axe = fig.add_subplot(gs[1, 2:4])
    axf = fig.add_subplot(gs[1, 4:6])
    _panel_fin(axa)
    _panel_phase(axb)
    _panel_repair(axc, axd)
    _panel_capacity(axe)
    _panel_gate(axf)
    bad = report_overlaps(fig, "unified")
    save(fig, "unified_all")
    return bad


if __name__ == "__main__":
    main()
