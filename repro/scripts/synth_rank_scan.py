"""P0-3: does the capacity exponent gamma depend on the auxiliary toxic rank?

Why this experiment
-------------------
Proposition `rank_dilution` states gamma = gamma_0 (1 - beta) with m_Pi ~ m^beta,
i.e. a *trained* head whose toxic subspace grows with width gives sub-mean-field
exponents.  The paper anchors this with a synthetic testbed of FIXED random
projections (beta = 0, gamma_0 = 1.53).  Before one can scan a beta axis one must
know whether this generator can express beta at all: the aux projection here is
W in R^{k x m x r} normalised by 1/sqrt(r), and the collapse-axis projection
u_c^T g_a ~ 2||h_c|| is then rank-independent -- so the toxic strength may not
respond to r.

This script measures gamma as a function of r at FIXED r (r does not scale with
m), holding everything else identical to the paper's testbed (pairwise
quadratic-margin rank primary, batch-variance auxiliary, 17-point log alpha grid
over [3e-4, 10], 800 epochs, lr 3e-2, k_aux = 2, d_in = 16, 1500/1500 samples).

Outcome A: gamma invariant in r  -> the fixed-rank testbed is beta = 0 by
           construction and cannot host a beta scan; the paper's beta values stay
           an inference from the real domains (which is how the text words them).
Outcome B: gamma varies with r   -> beta is realisable as a rank schedule
           r(m) = round(m^beta) with this normalisation, and the scan follows.

One JSON per (m, r): resumable via a non-empty out file.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

ALPHAS = list(np.logspace(math.log10(3e-4), 1.0, 17))   # the paper's 17-point grid


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def _rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    sa = np.asarray(a, dtype=np.float64)
    sorted_a = sa[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def rank_ic(pred, y):
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if pred.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(_rankdata(pred), _rankdata(y))[0, 1])


def rank_margin_loss(s, y):
    D = s.unsqueeze(1) - s.unsqueeze(0)
    P = (y.unsqueeze(1) > y.unsqueeze(0)).float()
    margin = torch.relu(-D)
    npos = P.sum().clamp_min(1.0)
    return (P * margin * margin).sum() / npos


def var_loss(a):
    return ((a - a.mean(0)) ** 2).mean()


class RankModel(nn.Module):
    """collapse_phase.CollapseModel with the aux output rank r exposed.

    W_aux ~ N(0, 1/r) keeps the projection norm-preserving: ||W h|| ~ ||h||
    for every r, so any r-dependence of gamma is purely a rank/subspace effect,
    not a gradient-magnitude artefact.
    """

    def __init__(self, d_in, m, k_aux, r):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, m), nn.ReLU(), nn.Linear(m, m))
        self.primary = nn.Linear(m, 1)
        g = torch.Generator().manual_seed(1234)
        W = torch.randn(k_aux, m, r, generator=g) / math.sqrt(r)
        self.register_buffer("W_aux", W)

    def forward(self, x):
        h = self.encoder(x)
        y_hat = self.primary(h).squeeze(-1)
        a = torch.einsum("bm,kmn->bkn", h, self.W_aux)
        return h, y_hat, a


def run_one(alpha, m, r, seed, epochs=800, lr=3e-2, d_in=16, n_tr=1500, n_te=1500,
            k_aux=2):
    set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    w_true = torch.randn(d_in, generator=g)
    w_true[torch.abs(w_true) < 1.0] = 0.0
    X = torch.randn(n_tr + n_te, d_in, generator=g)
    Y = (X @ w_true) + 0.4 * torch.randn(n_tr + n_te, generator=g)
    Xtr, Ytr = X[:n_tr].to(dev), Y[:n_tr].to(dev)
    Xte, Yte = X[n_tr:].to(dev), Y[n_tr:].to(dev)

    model = RankModel(d_in, m, k_aux, r).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        h, y_hat, a = model(Xtr)
        loss = rank_margin_loss(y_hat, Ytr) + alpha * var_loss(a)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        h, y_hat, _ = model(Xte)
        sigma_h = float(h.std(0).mean())
        ic = rank_ic(y_hat.cpu().numpy(), Yte.cpu().numpy())
    return dict(alpha=float(alpha), m=m, r=r, seed=seed, sigma_h=sigma_h,
                test_ic=ic, collapsed=1.0 if sigma_h < 1e-3 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--epochs", type=int, default=800)
    # Optional alpha grid.  Default is the paper's 17-point grid, so omitting
    # this flag reproduces the original runs bit for bit; pass a grid that
    # starts at alpha = 0 to anchor the crossing on the healthy branch.
    ap.add_argument("--alphas", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    alphas = ([float(x) for x in a.alphas.split(",")] if a.alphas
              else list(ALPHAS))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "cuda:" + torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    print("[device] %s  m=%d r=%d seeds=%d alphas=%d" % (tag, a.m, a.r, len(seeds), len(alphas)), flush=True)

    recs = []
    for alpha in alphas:
        sh, ics, cols = [], [], []
        for s in seeds:
            r = run_one(alpha, a.m, a.r, s, epochs=a.epochs)
            sh.append(r["sigma_h"])
            ics.append(r["test_ic"])
            cols.append(r["collapsed"])
        rec = dict(m=a.m, r=a.r, alpha=float(alpha),
                   sigma_h=float(np.mean(sh)), sigma_h_std=float(np.std(sh)),
                   test_ic=float(np.mean(ics)), collapse_rate=float(np.mean(cols)))
        recs.append(rec)
        print("  m=%3d r=%2d a=%9.3e sigma_h=%.5f+-%.5f ic=%+.4f coll=%.2f"
              % (a.m, a.r, alpha, rec["sigma_h"], rec["sigma_h_std"],
                 rec["test_ic"], rec["collapse_rate"]), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(m=a.m, r=a.r, alphas=[float(x) for x in alphas],
                       records=recs), f, indent=1)
    print("[saved] %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
