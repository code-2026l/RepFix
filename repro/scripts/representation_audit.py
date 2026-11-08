"""Scale-invariant audit of the operational scatter criterion (CPU).

Trains the released HAR toxifier protocol at three doses (healthy alpha=0,
near-threshold alpha=0.1, strongly supercritical alpha=100) and audits the
frozen representation with:

  * covariance geometry: trace, participation rank, entropy (effective) rank,
    stable rank, top-1 eigenvalue fraction, condition number;
  * linear probes on raw vs per-dimension-standardised frozen features for the
    primary activity task;
  * a low-label (10%) linear probe, i.e. a new classifier trained on the
    representation;
  * a probe for a new label space (static vs dynamic activities).

The point of the audit is the distinction the reviews asked for: a fall in
cross-sample scatter can be a global scale change rather than a loss of
linearly decodable information.  The script reports both, so the two are not
conflated.

Protocol mirrors code/cross_domain_mtl.py::train_one: Adam (lr 1e-3, batch
256), m=64, 60 epochs, seeds 0-9.  Each (alpha, seed) cell is one process so the
sweep can be parallelised with xargs.

USAGE
-----
  export REPFIX_DATA_X=/path/to/data_x   # containing uci_har/extracted/...
  python repro/scripts/representation_audit.py --alpha 100 --seed 0 \
      --out results/REPAUDIT/rec_a100_s0.json
  python repro/scripts/representation_audit.py --summary results/REPAUDIT
"""
import argparse
import glob
import json
import math
import os
import statistics as stats
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import cross_domain_mtl as cdm  # noqa: E402


def train_model(D, alpha, m, seed, epochs, lr, batch):
    torch.set_num_threads(1)
    cdm.set_seed(seed)
    model = cdm.MTLModel(D["d_in"], D["n_classes"], m=m, k_aux=3, seed=seed,
                         regress=(D["n_classes"] == 1),
                         y_dim=int(D.get("y_dim", 1))).to("cpu")
    Xtr = torch.from_numpy(D["X_tr"]).float()
    ytr = torch.from_numpy(D["y_tr"]).long()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            h = model(xb)
            mu = h.mean(0, keepdim=True)
            L_tox = ((h - mu) ** 2).mean()
            L = criterion(model.primary(h), yb) + alpha * L_tox
            L.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def embed(model, X, batch=1024):
    return torch.cat([model(X[b0:b0 + batch]) for b0 in range(0, len(X), batch)]).double()


def geometry(h):
    n, d = h.shape
    hc = h - h.mean(0, keepdim=True)
    cov = (hc.T @ hc) / max(n - 1, 1)
    eig = torch.linalg.eigvalsh((cov + cov.T) / 2).clamp_min(0)
    tr = float(eig.sum())
    pr = float(eig.sum().square() / eig.square().sum().clamp_min(1e-30))
    p = eig / eig.sum().clamp_min(1e-30)
    ent = float((-(p * p.clamp_min(1e-30).log()).sum()).exp())
    srank = float(eig.sum().square() / eig.max().square().clamp_min(1e-30)) if tr > 0 else 0.0
    top1 = float(eig.max() / eig.sum().clamp_min(1e-30)) if tr > 0 else 0.0
    cond = float(eig.max() / eig[eig > 0].min()) if bool((eig > 0).any()) else float("inf")
    return dict(cov_trace=tr, mean_feature_variance=tr / d,
                participation_rank=pr, entropy_rank=ent, stable_rank=srank,
                top1_fraction=top1, condition_number=cond,
                mean_norm=float(h.norm(dim=1).mean()),
                std_norm=float(hc.norm(dim=1).mean()))


def fit_probe(Xtr, ytr, Xte, yte, n_classes, standardize=True, whiten=False,
              max_iter=200, l2=1e-4):
    Xtr, Xte = Xtr.double(), Xte.double()
    if standardize:
        mu = Xtr.mean(0, keepdim=True)
        sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if whiten:
        cov = Xtr.T @ Xtr / max(len(Xtr) - 1, 1)
        eig, vec = torch.linalg.eigh((cov + cov.T) / 2)
        keep = eig > max(float(eig.max()) * 1e-3, 1e-8)
        W = vec[:, keep] / eig[keep].clamp_min(1e-8).sqrt()
        Xtr, Xte = Xtr @ W, Xte @ W
    clf = nn.Linear(Xtr.shape[1], n_classes).double()
    opt = torch.optim.LBFGS(clf.parameters(), lr=1.0, max_iter=max_iter,
                            history_size=20, tolerance_grad=1e-9,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(clf(Xtr), ytr) + l2 * sum(
            p.pow(2).sum() for p in clf.parameters())
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float((clf(Xte).argmax(1) == yte).float().mean())


def fit_mlp_probe(Xtr, ytr, Xte, yte, n_classes, hidden=64, epochs=120,
                  lr=1e-3, batch=256, seed=0, l2=1e-4):
    """One-hidden-layer MLP probe on the same frozen features.

    The linear probe establishes that the primary label is linearly decodable;
    this probe checks whether a mild nonlinear readout finds more.  Inputs are
    per-dimension standardised with train statistics, as in the linear probe.
    """
    Xtr, Xte = Xtr.float(), Xte.float()
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    mlp = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, n_classes))
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=l2)
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(mlp(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        return float((mlp(Xte).argmax(1) == yte).float().mean())


def stratified_subsample(y, frac, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    keep = []
    for c in y.unique():
        idx = torch.where(y == c)[0]
        k = max(1, int(round(len(idx) * frac)))
        keep.append(idx[torch.randperm(len(idx), generator=g)[:k]])
    return torch.cat(keep)


def audit(model, D, seed, frac=0.1):
    Xtr = torch.from_numpy(D["X_tr"]).float()
    ytr = torch.from_numpy(D["y_tr"]).long()
    Xte = torch.from_numpy(D["X_te"]).float()
    yte = torch.from_numpy(D["y_te"]).long()
    htr, hte = embed(model, Xtr), embed(model, Xte)
    with torch.no_grad():
        primary_acc = float((model.primary(hte.float()).argmax(1) == yte).float().mean())

    out = dict(geom=geometry(hte), activity_acc=primary_acc, d=int(hte.shape[1]))
    out["probe_activity_raw"] = fit_probe(htr, ytr, hte, yte, 6, standardize=False)
    out["probe_activity_std"] = fit_probe(htr, ytr, hte, yte, 6, standardize=True)
    out["probe_activity_whiten"] = fit_probe(htr, ytr, hte, yte, 6, standardize=True, whiten=True)
    out["probe_activity_mlp"] = fit_mlp_probe(htr, ytr, hte, yte, 6, seed=seed)
    sub = stratified_subsample(ytr, frac, seed)
    out["probe_activity_10pct_std"] = fit_probe(htr[sub], ytr[sub], hte, yte, 6,
                                                standardize=True, max_iter=150)
    ycoarse_tr, ycoarse_te = (ytr >= 3).long(), (yte >= 3).long()
    out["probe_coarse_raw"] = fit_probe(htr, ycoarse_tr, hte, ycoarse_te, 2, standardize=False)
    out["probe_coarse_std"] = fit_probe(htr, ycoarse_tr, hte, ycoarse_te, 2, standardize=True)
    return out


def run_one(a):
    D = cdm.load_domain("har")
    t0 = time.time()
    model = train_model(D, a.alpha, a.m, a.seed, a.epochs, a.lr, a.batch)
    rec = dict(alpha=a.alpha, seed=a.seed, m=a.m, epochs=a.epochs,
               train_seconds=round(time.time() - t0, 2))
    rec.update(audit(model, D, a.seed))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec.items() if not isinstance(v, dict)}, sort_keys=True), flush=True)


def summarise(outdir):
    recs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(Path(outdir) / "rec_*.json")))]
    if not recs:
        raise SystemExit("no rec_*.json in %s" % outdir)
    by = {}
    for r in recs:
        by.setdefault(float(r["alpha"]), []).append(r)
    healthy = {r["seed"]: r["geom"]["std_norm"] / math.sqrt(r["d"]) for r in by.get(0.0, [])}
    fields = ["cov_trace", "mean_feature_variance", "participation_rank", "entropy_rank",
              "stable_rank", "top1_fraction", "mean_norm", "std_norm"]
    probes = ["activity_acc", "probe_activity_raw", "probe_activity_std",
              "probe_activity_whiten", "probe_activity_mlp",
              "probe_activity_10pct_std",
              "probe_coarse_raw", "probe_coarse_std"]
    ms = lambda v: dict(mean=stats.mean(v), std=stats.pstdev(v))
    summary = {"n_seeds": {str(a): len(rs) for a, rs in by.items()}, "cells": {}}
    for a, rs in sorted(by.items()):
        cell = {f: ms([r["geom"][f] for r in rs]) for f in fields}
        cell.update({p: ms([r[p] for r in rs]) for p in probes})
        rel = [(r["geom"]["std_norm"] / math.sqrt(r["d"])) / healthy[r["seed"]]
               for r in rs] if a != 0.0 and healthy else [1.0 for _ in rs]
        cell["relative_scatter"] = ms(rel)
        summary["cells"][str(a)] = cell
    (Path(outdir) / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(Path(outdir) / "summary.txt", "w") as fh:
        fh.write("alpha n  rel_scatter  cov_trace  PR  erank  probe_std  probe_raw  new_task  primary\n")
        for a, rs in sorted(by.items()):
            c = summary["cells"][str(a)]
            fh.write("%-5g %d  %.4f  %.4g  %.2f  %.2f  %.4f  %.4f  %.4f  %.4f\n" % (
                a, len(rs), c["relative_scatter"]["mean"], c["cov_trace"]["mean"],
                c["participation_rank"]["mean"], c["entropy_rank"]["mean"],
                c["probe_activity_std"]["mean"], c["probe_activity_raw"]["mean"],
                c["probe_coarse_std"]["mean"], c["activity_acc"]["mean"]))
    print((Path(outdir) / "summary.txt").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.summary:
        summarise(a.summary)
        return
    if a.alpha is None or a.seed is None or not a.out:
        ap.error("--alpha, --seed and --out are required unless --summary is used")
    run_one(a)


if __name__ == "__main__":
    main()
