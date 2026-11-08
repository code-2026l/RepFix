"""Does variance contraction break a readout calibrated before it? (CPU)

The scale-invariant audit shows that contraction need not erase linearly
decodable information. This driver asks the operational follow-up: what happens
to a readout that was trained and calibrated on the healthy representation and
then applied, unchanged, to a contracted one?

For each seed we train a healthy encoder (alpha=0) and a contracted encoder
(alpha=100), then compare readouts on the held-out test split:

  healthy-ref : head + input statistics fitted on the healthy encoder;
  frozen      : the same head and the same healthy input statistics, applied
                unchanged to the contracted features;
  input-cal   : the same head weights, but input statistics refitted on the
                contracted features (cheap recalibration, no new labels);
  retrained   : a new head trained on the contracted features.

Reported: accuracy, NLL, expected calibration error (ECE, 15 bins), and
split-conformal coverage at the 90% target. The point is not that the
information is gone; it is that a scale-dependent readout calibrated before the
contraction is no longer valid, an operational cost task accuracy alone does
not report.

USAGE
-----
  export REPFIX_DATA_X=/path/to/data_x
  python repro/scripts/readout_transfer_audit.py --seeds 0,1,2,3,4 \
      --out results/READOUT/readout_har.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import cross_domain_mtl as cdm  # noqa: E402


def train_encoder(D, alpha, seed, m, epochs, lr, batch, arch="mlp"):
    torch.set_num_threads(1)
    cdm.set_seed(seed)
    model = cdm.MTLModel(D["d_in"], D["n_classes"], m=m, k_aux=3, seed=seed, arch=arch).to("cpu")
    X = torch.from_numpy(D["X_tr"]).float()
    y = torch.from_numpy(D["y_tr"]).long()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            h = model(X[idx])
            loss = F.cross_entropy(model.primary(h), y[idx])
            loss = loss + alpha * ((h - h.mean(0, keepdim=True)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def embed(model, X, batch=1024):
    return torch.cat([model(X[b0:b0 + batch]) for b0 in range(0, len(X), batch)]).double()


def stats(X):
    return X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)


def apply_stats(X, mu, sd):
    return (X - mu) / sd


def fit_head(Xtr, ytr, max_iter=200, l2=1e-4, bias=True):
    mu, sd = stats(Xtr)
    Xs = apply_stats(Xtr, mu, sd)
    clf = nn.Linear(Xs.shape[1], int(ytr.max()) + 1, bias=bias).double()
    opt = torch.optim.LBFGS(clf.parameters(), lr=1.0, max_iter=max_iter,
                            history_size=20, tolerance_grad=1e-9,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(clf(Xs), ytr) + l2 * sum(p.pow(2).sum() for p in clf.parameters())
        loss.backward()
        return loss
    opt.step(closure)
    return clf, mu, sd


def evaluate(clf, X, y, mu, sd):
    with torch.no_grad():
        logits = clf(apply_stats(X, mu, sd))
        acc = float((logits.argmax(1) == y).float().mean())
        nll = float(F.cross_entropy(logits, y))
        probs = logits.softmax(1)
        conf, pred = probs.max(1)
        ece = 0.0
        for lo in torch.linspace(0, 1, 16)[:-1]:
            hi = lo + 1 / 15
            m = (conf >= lo) & (conf < hi)
            if m.any():
                ece += float(m.float().mean()) * abs(float((pred[m] == y[m]).float().mean()) - float(conf[m].mean()))
        return dict(acc=acc, nll=nll, ece=ece)


def fit_temperature(logits, y):
    T = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp_min(1e-6), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach())


def q_conformal(logits, y, target=0.90):
    probs = logits.softmax(1)
    return float(torch.quantile(1 - probs[torch.arange(len(y)), y], target))


def coverage(logits, y, q):
    probs = logits.softmax(1)
    return float(((1 - probs[torch.arange(len(y)), y]) <= q).float().mean())


def stratified_split(y, frac, seed):
    g = torch.Generator().manual_seed(seed + 77)
    head, cal = [], []
    for c in y.unique():
        idx = torch.where(y == c)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        k = max(1, int(round(len(idx) * frac)))
        cal.append(idx[:k]); head.append(idx[k:])
    return torch.cat(head), torch.cat(cal)


def run_seed(D, seed, m, epochs, lr, batch, arch="mlp"):
    Xtr = torch.from_numpy(D["X_tr"]).float()
    ytr = torch.from_numpy(D["y_tr"]).long()
    Xte = torch.from_numpy(D["X_te"]).float()
    yte = torch.from_numpy(D["y_te"]).long()
    ht, cal = stratified_split(ytr, 0.2, seed)

    enc_h = train_encoder(D, 0.0, seed, m, epochs, lr, batch, arch=arch)
    enc_c = train_encoder(D, 100.0, seed, m, epochs, lr, batch, arch=arch)
    htr_h, hte_h = embed(enc_h, Xtr), embed(enc_h, Xte)
    htr_c, hte_c = embed(enc_c, Xtr), embed(enc_c, Xte)

    clf_h, mu_h, sd_h = fit_head(htr_h[ht], ytr[ht])
    clf_c, mu_c, sd_c = fit_head(htr_c[ht], ytr[ht])

    healthy_ref = evaluate(clf_h, hte_h, yte, mu_h, sd_h)
    frozen = evaluate(clf_h, hte_c, yte, mu_h, sd_h)
    input_cal = evaluate(clf_h, hte_c, yte, mu_c, sd_c)
    retrained = evaluate(clf_c, hte_c, yte, mu_c, sd_c)

    with torch.no_grad():
        logits_cal_h = clf_h(apply_stats(htr_h[cal], mu_h, sd_h))
        logits_cal_c = clf_h(apply_stats(htr_c[cal], mu_c, sd_c))
        logits_te_h = clf_h(apply_stats(hte_h, mu_h, sd_h))
        logits_te_c = clf_h(apply_stats(hte_c, mu_h, sd_h))
        logits_te_c_recal = clf_h(apply_stats(hte_c, mu_c, sd_c))
        logits_cal_c_retrained = clf_c(apply_stats(htr_c[cal], mu_c, sd_c))
        logits_te_c_retrained = clf_c(apply_stats(hte_c, mu_c, sd_c))
        T_h = fit_temperature(logits_cal_h, ytr[cal])
        T_c = fit_temperature(logits_cal_c, ytr[cal])
        q_h = q_conformal(logits_cal_h, ytr[cal])
        q_c = q_conformal(logits_cal_c, ytr[cal])
        q_retrained = q_conformal(logits_cal_c_retrained, ytr[cal])

    def ece_temp(logits, T, y):
        with torch.no_grad():
            probs = (logits / T).softmax(1)
            conf, pred = probs.max(1)
            e = 0.0
            for lo in torch.linspace(0, 1, 16)[:-1]:
                hi = lo + 1 / 15
                m = (conf >= lo) & (conf < hi)
                if m.any():
                    e += float(m.float().mean()) * abs(float((pred[m] == y[m]).float().mean()) - float(conf[m].mean()))
            return e

    rec = dict(seed=seed,
               healthy_ref=healthy_ref,
               frozen=frozen,
               input_cal=input_cal,
               retrained=retrained,
               calibration=dict(
                   temperature_healthy=T_h,
                   nll_frozen_with_healthy_T=float(F.cross_entropy(logits_te_c / T_h, yte)),
                   nll_recalibrated_with_contracted_T=float(F.cross_entropy(logits_te_c_recal / T_c, yte)),
                   ece_frozen_with_healthy_T=ece_temp(logits_te_c, T_h, yte),
                   ece_recalibrated_with_contracted_T=ece_temp(logits_te_c_recal, T_c, yte),
                   conformal_90_healthy_ref=coverage(logits_te_h, yte, q_h),
                   conformal_90_frozen=coverage(logits_te_c, yte, q_h),
                   conformal_90_recalibrated=coverage(logits_te_c_recal, yte, q_c),
                   conformal_90_retrained=coverage(logits_te_c_retrained, yte, q_retrained),
               ))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="results/READOUT/readout_har.json")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "transformer"])
    a = ap.parse_args()
    D = cdm.load_domain("har")
    rows = [run_seed(D, int(s), a.m, a.epochs, a.lr, a.batch, arch=a.arch)
            for s in a.seeds.split(",")]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(domain="har", protocol="healthy readout transferred to a contracted encoder",
                                   seeds=[int(s) for s in a.seeds.split(",")], rows=rows), indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
