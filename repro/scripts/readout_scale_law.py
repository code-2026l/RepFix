"""Critical-scale law for a frozen affine readout (CPU/GPU).

This driver isolates the scale channel of the readout failure.  A linear head
(plus fixed input statistics and a fixed temperature map) is fitted on the
healthy HAR encoder.  We then evaluate the same readout on c * H for a grid of
c, where H is the healthy representation.  At c = 1 the readout is healthy; as
c falls, the bias term dominates and the prediction becomes constant.

For an affine readout z_c = W(c h) + b with fixed statistics (mu, sd), define

    z_0 = b - W mu / sd,          s_i = W h_i / sd,
    c_i^* = min_{j != top(z_0)} (z_0[top] - z_0[j]) / (s_i[j] - s_i[top]),

over competitors with a positive denominator.  The prediction for example i
first changes at c_i^*.  The median c_i^* is the scale at which half the batch
flips, and 1/median(c_i^*) is the bias-to-signal ratio of the readout.

The script reports, per seed:
  * the measured frozen-readout accuracy/NLL/ECE on the c grid;
  * the input-recalibrated readout (statistics refitted at each c), which is
    scale-invariant by construction;
  * the exact per-example critical scale c_i^*, its median/quartiles, and the
    fraction of examples already flipped at each c;
  * the bias-to-signal ratio ||z_0|| / median_i ||s_i||.

Output: results/READOUT/readout_scale_law.json and the figure
paper/figures/fig_readout_scale_law.pdf.

USAGE
-----
  export REPFIX_DATA_X=/path/to/data_x
  python repro/scripts/readout_scale_law.py --seeds 0,1,2,3,4 \
      --out results/READOUT/readout_scale_law.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "repro" / "scripts"))
import cross_domain_mtl as cdm  # noqa: E402
from readout_transfer_audit import (  # noqa: E402
    apply_stats, evaluate, fit_head, fit_temperature, stats,
    stratified_split,
)


@torch.no_grad()
def embed_model(model, X, batch=1024):
    """Embed on the model's device, return CPU double features."""
    dev = next(model.parameters()).device
    return torch.cat([model(X[b0:b0 + batch].to(dev)).cpu()
                      for b0 in range(0, len(X), batch)]).double()


def train_encoder(D, alpha, seed, m, epochs, lr, batch, device):
    """Same objective as readout_transfer_audit.train_encoder, device-aware."""
    cdm.set_seed(seed)
    model = cdm.MTLModel(D["d_in"], D["n_classes"], m=m, k_aux=3,
                         seed=seed, arch="mlp").to(device)
    X = torch.from_numpy(D["X_tr"]).float().to(device)
    y = torch.from_numpy(D["y_tr"]).long().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            h = model(X[idx])
            loss = F.cross_entropy(model.primary(h), y[idx])
            loss = loss + alpha * ((h - h.mean(0, keepdim=True)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def ece(logits, y, bins=15):
    with torch.no_grad():
        probs = logits.softmax(1)
        conf, pred = probs.max(1)
    e = 0.0
    for lo in torch.linspace(0, 1, bins + 1)[:-1]:
        hi = lo + 1 / bins
        m = (conf >= lo) & (conf < hi)
        if m.any():
            e += float(m.float().mean()) * abs(
                float((pred[m] == y[m]).float().mean()) - float(conf[m].mean()))
    return e


def evaluate_logits(logits, y, temperature=1.0):
    with torch.no_grad():
        raw = logits
        temp = logits / float(temperature)
    return dict(acc=float((raw.argmax(1) == y).float().mean()),
                nll=float(F.cross_entropy(raw, y)),
                ece=ece(raw, y),
                nll_temp=float(F.cross_entropy(temp, y)),
                ece_temp=ece(temp, y))


def critical_scales(clf, mu, sd, X):
    """Bias-class exit scale while increasing c from zero (legacy output keys).

    This is not generally the first flip while shrinking from c=1. The stored
    c_star_median is conditional on a finite bias-exit threshold.
    """
    W = clf.weight.detach().double()
    b = clf.bias.detach().double()
    z0 = b - (W @ (mu / sd).squeeze(0))          # logits at c = 0
    S = (X / sd) @ W.t()                        # per-example signal, c-scaled
    top = z0.argmax().item()
    n, k = S.shape
    cstar = torch.full((n,), float("inf"), dtype=torch.float64)
    for j in range(k):
        if j == top:
            continue
        gap = z0[top] - z0[j]
        den = S[:, j] - S[:, top]
        # z_c[top] - z_c[j] = gap - c * den, because den = S_j - S_top.
        # A positive crossing exists only when the competitor grows faster.
        cand = torch.where(den > 0, gap / den.clamp_min(1e-12),
                           torch.full_like(den, float("inf")))
        cand = torch.where(cand > 0, cand, torch.full_like(cand, float("inf")))
        cstar = torch.minimum(cstar, cand)
    finite = cstar[torch.isfinite(cstar)]
    if len(finite) == 0:
        med = float("inf")
        q25 = q75 = float("inf")
    else:
        med = float(finite.median())
        q25 = float(finite.quantile(0.25))
        q75 = float(finite.quantile(0.75))
    bsr = float(z0.norm() / S.norm(dim=1).median().clamp_min(1e-12))
    return dict(c_star_median=med, c_star_q25=q25, c_star_q75=q75,
                bsr=bsr, bias_top=int(top), n_finite=int(len(finite)),
                n=int(n), z0=z0.tolist())


C_GRID = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07,
          0.05, 0.03, 0.02, 0.01, 0.005, 0.001]


def run_seed(D, seed, m, epochs, lr, batch, device):
    Xtr = torch.from_numpy(D["X_tr"]).float()
    ytr = torch.from_numpy(D["y_tr"]).long()
    Xte = torch.from_numpy(D["X_te"]).float()
    yte = torch.from_numpy(D["y_te"]).long()
    ht, cal = stratified_split(ytr, 0.2, seed)

    enc = train_encoder(D, 0.0, seed, m, epochs, lr, batch, device)
    htr = embed_model(enc, Xtr)
    hte = embed_model(enc, Xte)

    clf, mu, sd = fit_head(htr[ht], ytr[ht])
    # healthy temperature map, fitted on the calibration split
    with torch.no_grad():
        logits_cal = clf(apply_stats(htr[cal], mu, sd))
        T = fit_temperature(logits_cal, ytr[cal])

    base = evaluate(clf, hte, yte, mu, sd)
    crit = critical_scales(clf, mu, sd, hte)
    crit["bias_top_class_freq"] = float((yte == crit["bias_top"]).float().mean())

    curve = []
    for c in C_GRID:
        # frozen readout: head, statistics, and temperature all fixed
        logits = clf(apply_stats(c * hte, mu, sd))
        frozen = evaluate_logits(logits, yte, temperature=T)
        # input recalibration: refit statistics on the scaled training features
        mu_c, sd_c = stats(c * htr[ht])
        logits_r = clf(apply_stats(c * hte, mu_c, sd_c))
        recal = evaluate_logits(logits_r, yte, temperature=T)
        with torch.no_grad():
            z = logits
            flip = float(((z.argmax(1) != int(crit["bias_top"]))).float().mean())
        curve.append(dict(c=float(c), frozen=frozen, recalibrated=recal,
                          flip_fraction=flip))

    return dict(seed=int(seed), base=base, temperature=T,
                critical=crit, curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/READOUT/readout_scale_law.json")
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    D = cdm.load_domain("har")
    rows = [run_seed(D, int(s), a.m, a.epochs, a.lr, a.batch, device)
            for s in a.seeds.split(",")]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(domain="har", protocol="healthy readout on c * H, fixed statistics",
                   c_grid=C_GRID, seeds=[int(s) for s in a.seeds.split(",")],
                   device=device, rows=rows)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps([dict(seed=r["seed"], c_star=r["critical"]["c_star_median"],
                           bsr=r["critical"]["bsr"],
                           acc_at_1=r["curve"][0]["frozen"]["acc"],
                           acc_at_0p01=r["curve"][-3]["frozen"]["acc"])
                      for r in rows], indent=2))


if __name__ == "__main__":
    main()
