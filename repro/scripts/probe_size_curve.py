"""P2-6: empirical probe-size curve for the calibrated gate.

Proposition `calibration` gives a sample-size requirement of the form
B >~ 2 rho^2 c_0^2 eps^{-2} ln(2/delta) but the paper has no B-versus-accuracy
curve.  This script builds one on the synthetic testbed used for the toxicity
order parameter.

Protocol
--------
1. Train the generator to its healthy fixed point (alpha = 0) exactly as in
   collapse_phase/critical_scaling (pairwise quadratic-margin rank primary,
   batch-variance auxiliary, 800 epochs, lr 3e-2, k_aux = 2, d_in = 16).
2. On one large batch (N = 16384) form the per-sample collapse projections
       p_i = |u_c^T g_p|,  q_i = |u_c^T g_a|,   u_c = (h_i - mean h)/||h_i - mean h||
   so that the population toxicity ratio is rho = mean(q)/mean(p).
3. The gate opens iff alpha * rho_hat > 1.  Fix the operating point
   alpha = (1 + delta)/rho, i.e. truly supercritical for delta > 0 and truly
   subcritical for delta < 0, and re-estimate rho_hat from B re-drawn samples
   (T trials).  Report
       power        = P(gate opens | delta > 0)     -- must approach 1
       false_alarm  = P(gate opens | delta < 0)     -- must approach 0
   against B, and the empirical B needed for power 0.95.

The theorem's scale is B ~ rho^2, so B95/rho^2 should be roughly constant across
seeds; the script prints that ratio, which is the falsifiable part.

One JSON per seed: resumable via a non-empty out file.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

B_GRID = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
          1536, 2048, 3072, 4096, 6144, 8192]
DELTAS = [-0.5, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.5]
N_POOL = 16384
T_TRIALS = 4000


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def rank_margin_loss(s, y):
    D = s.unsqueeze(1) - s.unsqueeze(0)
    P = (y.unsqueeze(1) > y.unsqueeze(0)).float()
    return (P * torch.relu(-D) ** 2).sum() / P.sum().clamp_min(1.0)


def var_loss(a):
    return ((a - a.mean(0)) ** 2).mean()


class Sync(nn.Module):
    """collapse_phase.CollapseModel, verbatim geometry."""

    def __init__(self, d_in, m, k_aux):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, m), nn.ReLU(), nn.Linear(m, m))
        self.primary = nn.Linear(m, 1)
        g = torch.Generator().manual_seed(1234)
        W = torch.randn(k_aux, m, m // 2, generator=g) / math.sqrt(m // 2)
        self.register_buffer("W_aux", W)

    def forward(self, x):
        h = self.encoder(x)
        y_hat = self.primary(h).squeeze(-1)
        a = torch.einsum("bm,kmn->bkn", h, self.W_aux)
        return h, y_hat, a


def per_sample_projections(seed, m, d_in=16, n_tr=1500, epochs=800, lr=3e-2):
    set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    w_true = torch.randn(d_in, generator=g)
    w_true[torch.abs(w_true) < 1.0] = 0.0
    X = torch.randn(n_tr + N_POOL, d_in, generator=g)
    Y = (X @ w_true) + 0.4 * torch.randn(n_tr + N_POOL, generator=g)
    Xtr, Ytr = X[:n_tr].to(dev), Y[:n_tr].to(dev)
    Xp, Yp = X[n_tr:].to(dev), Y[n_tr:].to(dev)

    model = Sync(d_in, m, 2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        h, y_hat, a = model(Xtr)
        loss = rank_margin_loss(y_hat, Ytr)      # alpha = 0: healthy fixed point
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.train()
    h, y_hat, a = model(Xp)
    loss_p = rank_margin_loss(y_hat, Yp)
    loss_a = var_loss(a)
    gp = torch.autograd.grad(loss_p, h, retain_graph=True)[0]
    ga = torch.autograd.grad(loss_a, h, retain_graph=True)[0]
    uc = h - h.mean(0)
    uhat = uc / uc.norm(dim=1, keepdim=True).clamp_min(1e-8)
    with torch.no_grad():
        p = (gp * uhat).sum(-1).abs().cpu().numpy().astype(np.float64)
        q = (ga * uhat).sum(-1).abs().cpu().numpy().astype(np.float64)
    return p, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device] %s seed=%d m=%d" % (dev, a.seed, a.m), flush=True)

    p, q = per_sample_projections(a.seed, a.m)
    rho = float(q.mean() / (p.mean() + 1e-30))
    print("[pool] rho_true=%.5g  mean(p)=%.4g mean(q)=%.4g" % (rho, p.mean(), q.mean()),
          flush=True)

    rng = np.random.default_rng(1000 + a.seed)
    n = len(p)
    # B-outer / delta-inner so each B draws its resample indices exactly once
    # (the naive delta-outer loop redoes identical sampling 8x).  Trials are
    # capped so that trials*B stays bounded -> constant work per grid point.
    rows = {d: {"delta": d, "alpha": (1.0 + d) / rho, "curve": [], "B95": None}
            for d in DELTAS}
    for B in B_GRID:
        if B > n:
            break
        T = max(400, min(T_TRIALS, 4_000_000 // B))
        idx = rng.integers(0, n, size=(T, B))
        qh = q[idx].mean(axis=1)
        ph = p[idx].mean(axis=1)
        rho_hat = qh / (ph + 1e-30)
        qh = ph = None
        for d in DELTAS:
            alpha = (1.0 + d) / rho
            rate = float((alpha * rho_hat > 1.0).mean())
            rows[d]["curve"].append({"B": B, "trials": T, "open_rate": rate})
            if d > 0 and rows[d]["B95"] is None and rate >= 0.95:
                rows[d]["B95"] = B
        print("  B=%-6d T=%-5d " % (B, T)
              + " ".join("d%+0.1f:%.3f" % (d, rows[d]["curve"][-1]["open_rate"])
                         for d in DELTAS), flush=True)

    out_rows = []
    for d in DELTAS:
        rec = rows[d]
        rec["B95_over_rho2"] = (rec["B95"] / rho ** 2) if rec["B95"] else None
        out_rows.append(rec)
        print("  delta=%+0.2f alpha=%.4g B95=%s B95/rho^2=%s"
              % (d, rec["alpha"], rec["B95"],
                 ("%.5g" % rec["B95_over_rho2"]) if rec["B95"] else "-"), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(seed=a.seed, m=a.m, rho_true=rho, n_pool=n,
                       trials_cap=T_TRIALS, rows=out_rows), f, indent=1)
    print("[saved] %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
