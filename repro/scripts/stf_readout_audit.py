"""Does the survival intervention protect a readout calibrated before training? (CPU)

The readout-transfer audit shows that a head fitted on the healthy encoder
fails after joint contraction. This driver asks the intervention question: if
the same head is fitted on the healthy encoder and then applied to an encoder
trained with STF-0 (full detachment), STF-cal (one-probe gate and radial
deletion), or joint training, does the readout survive?

Protocol: HAR, m=64, 60 epochs, five seeds, four detached warm-up epochs for
STF-cal. The readout is fitted on the healthy encoder and applied unchanged
to every arm; accuracy and ECE are reported.

USAGE
-----
  export REPFIX_DATA_X=/path/to/data_x
  python repro/scripts/stf_readout_audit.py --seeds 0,1,2,3,4 \
      --out results/READOUT/stf_readout_har.json
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "repro" / "scripts"))
import cross_domain_mtl as cdm  # noqa: E402
from readout_transfer_audit import (  # noqa: E402
    apply_stats, embed, evaluate, fit_head, stats,
)


def train_encoder(D, alpha, seed, m, epochs, lr, batch, mode="joint", arch="mlp"):
    torch.set_num_threads(1)
    cdm.set_seed(seed)
    model = cdm.MTLModel(D["d_in"], D["n_classes"], m=m, k_aux=3, seed=seed, arch=arch).to("cpu")
    X = torch.from_numpy(D["X_tr"]).float()
    y = torch.from_numpy(D["y_tr"]).long()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    n_steps = (n + batch - 1) // batch
    cdm.STF_RHO_CFG.update(cal_warm_epochs=4.0, rank_max=3, basis="pca")
    cal = None
    if mode == "stfcal":
        cal = cdm.CalController(alpha, warm_steps=int(4.0 * n_steps), rank_max=3,
                                ema=0.9, basis="pca")
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            xb, yb = X[idx], y[idx]
            opt.zero_grad()
            h = model(xb)
            if cal is not None:
                if not cal.calibrated:
                    if cal.step >= cal.warm_steps:
                        cal.calibrate(cdm.rho_probe(model, xb, yb, criterion, False,
                                                    R=3, basis="pca"))
                    else:
                        with torch.no_grad():
                            cal.observe_scatter(cdm.scatter_t(h.detach()))
                else:
                    with torch.no_grad():
                        cal.observe_scatter(cdm.scatter_t(h.detach()))
                aux, _, _ = cal.step_h(h, "stfcal", warm=(not cal.calibrated))
            elif mode == "stf0":
                aux = h.detach()
            else:
                aux = h
            L_tox = ((aux - aux.mean(0, keepdim=True)) ** 2).mean()
            loss = criterion(model.primary(h), yb) + alpha * L_tox
            loss.backward()
            opt.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--arch", default="mlp", choices=["mlp", "transformer"])
    ap.add_argument("--out", default="results/READOUT/stf_readout_har.json")
    a = ap.parse_args()

    D = cdm.load_domain("har")
    Xtr = torch.from_numpy(D["X_tr"]).float()
    ytr = torch.from_numpy(D["y_tr"]).long()
    Xte = torch.from_numpy(D["X_te"]).float()
    yte = torch.from_numpy(D["y_te"]).long()
    rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        enc_healthy = train_encoder(D, 0.0, seed, a.m, a.epochs, a.lr, a.batch,
                                    mode="joint", arch=a.arch)
        h_healthy = embed(enc_healthy, Xtr)
        hte_healthy = embed(enc_healthy, Xte)
        enc_joint = train_encoder(D, 100.0, seed, a.m, a.epochs, a.lr, a.batch,
                                  mode="joint", arch=a.arch)
        enc_stf0 = train_encoder(D, 100.0, seed, a.m, a.epochs, a.lr, a.batch,
                                 mode="stf0", arch=a.arch)
        enc_stfcal = train_encoder(D, 100.0, seed, a.m, a.epochs, a.lr, a.batch,
                                   mode="stfcal", arch=a.arch)
        mu, sd = stats(h_healthy)
        clf, _, _ = fit_head(h_healthy, ytr)
        rec = {"seed": seed}
        for tag, hte in (("healthy", hte_healthy),
                         ("joint", embed(enc_joint, Xte)),
                         ("stf0", embed(enc_stf0, Xte)),
                         ("stfcal", embed(enc_stfcal, Xte))):
            rec[tag] = evaluate(clf, hte, yte, mu, sd)
        rows.append(rec)

    means = {k: dict(acc=st.mean([r[k]["acc"] for r in rows]),
                     nll=st.mean([r[k]["nll"] for r in rows]),
                     ece=st.mean([r[k]["ece"] for r in rows]))
             for k in ("healthy", "joint", "stf0", "stfcal")}
    payload = dict(domain="har", protocol="healthy readout applied to each training arm",
                   seeds=[int(s) for s in a.seeds.split(",")], rows=rows, means=means)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
