"""P0-3b: sweep the rank-growth exponent beta and measure gamma(beta).

Why this experiment
-------------------
Proposition `rank_dilution` states  gamma = gamma_0 (1 - beta)  with the toxic
rank growing as m_Pi ~ m^beta.  The paper's synthetic testbed, however, is NOT a
fixed-rank anchor: `collapse_phase.CollapseModel` (used by `critical_scaling.py`
to produce results/th_critical.json) builds

        W_aux = randn(k_aux, m, m // 2) / sqrt(m // 2)

so the aux projection has rank floor(m/2) -- a fixed *fraction* of the width,
i.e. beta = 1 in the parametrisation of the proposition, at the edge of its
stated range 0 <= beta < 1.  This script makes beta an explicit knob:

        r(m) = clip(round(r0 * (m / m_ref) ** beta), 1, m)

so beta = 0 is a genuinely width-independent toxic rank (the proposition's
fixed-rank branch) and beta = 1 reproduces the paper's construction.  Holding
everything else identical to the paper's testbed (RankModel is a bit-exact port
of CollapseModel: same seed-1234 projection, same 1/sqrt(r) normalisation, same
pairwise quadratic-margin rank primary, same batch-variance auxiliary, same
17-point log alpha grid over [3e-4, 10], 800 epochs, lr 3e-2, k_aux = 2,
d_in = 16, 1500/1500 samples).

Read-out: for each beta, alpha_c is taken as the crossing of the absolute
scatter level 1.5e-3 -- the criterion `repro/scripts/_cap_analysis.py` uses and
the one that reproduces the paper's printed alpha_c = 0.025, 0.26, 0.23, 0.73,
3.0 (gamma = 1.53, R^2 = 0.91) from results/th_critical.json -- and gamma(beta)
is the log-log OLS slope of alpha_c against m.

Outcome A: gamma decreases with beta  -> the dilution law holds and the beta
           axis is realisable with this generator; the testbed is simply
           mislabelled in the text (its rank is m/2, not fixed).
Outcome B: gamma does not decrease   -> the (1 - beta) form is not supported by
           this generator and the proposition's empirical anchor must be
           dropped or restated.

One JSON per (m, beta): resumable via a non-empty out file.
"""
import argparse
import json
import math
import os

import numpy as np

from synth_rank_scan import ALPHAS, run_one


def rank_schedule(m, beta, r0, m_ref, cap=True):
    r = int(round(r0 * (m / float(m_ref)) ** beta))
    r = max(1, r)
    if cap:
        r = min(r, m)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--r0", type=int, default=32,
                    help="toxic rank pinned at m = --m-ref")
    ap.add_argument("--m-ref", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--epochs", type=int, default=800)
    # Optional alpha grid; default is the paper's 17-point grid, so omitting the
    # flag reproduces the original runs.  Pass a grid starting at alpha = 0 to
    # anchor the crossing on the healthy branch (sigma_h(0)) rather than on
    # sigma_h(3e-4), which is what left-censored the first beta sweep.
    ap.add_argument("--alphas", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    alphas = ([float(x) for x in a.alphas.split(",")] if a.alphas
              else list(ALPHAS))
    r = rank_schedule(a.m, a.beta, a.r0, a.m_ref)

    print("[sched] beta=%.3f m=%d m_ref=%d r0=%d -> r=%d"
          % (a.beta, a.m, a.m_ref, a.r0, r), flush=True)

    recs = []
    for alpha in alphas:
        sh, ics, cols = [], [], []
        for s in seeds:
            out = run_one(alpha, a.m, r, s, epochs=a.epochs)
            sh.append(out["sigma_h"])
            ics.append(out["test_ic"])
            cols.append(out["collapsed"])
        rec = dict(m=a.m, r=r, beta=a.beta, alpha=float(alpha),
                   sigma_h=float(np.mean(sh)), sigma_h_std=float(np.std(sh)),
                   test_ic=float(np.mean(ics)), collapse_rate=float(np.mean(cols)))
        recs.append(rec)
        print("  m=%3d r=%2d b=%.2f a=%9.3e sigma_h=%.5f+-%.5f ic=%+.4f coll=%.2f"
              % (a.m, r, a.beta, alpha, rec["sigma_h"], rec["sigma_h_std"],
                 rec["test_ic"], rec["collapse_rate"]), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(m=a.m, r=r, beta=a.beta, r0=a.r0, m_ref=a.m_ref,
                       alphas=[float(x) for x in alphas], records=recs),
                  f, indent=1)
    print("[saved] %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
