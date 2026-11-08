"""Does a real auxiliary head break a readout fitted on the healthy encoder?

This is the real-head counterpart of readout_scale_law.py / readout_har.json.
It trains three HAR encoders under the true subject-ID auxiliary (alpha=10):

  healthy  : alpha=0, no auxiliary gradient
  joint    : alpha=10, auxiliary gradient reaches the encoder
  detached : alpha=10, auxiliary branch detached (STF-0)

A primary readout is fitted on the healthy encoder (head/cal split).  We then
apply it unchanged to the joint and detached representations, refit only the
input statistics (cheap recalibration, no new labels), or retrain a head.  We
also report the encoders' own primary accuracy and relative scatter.

The point: subject-ID loses primary accuracy without contraction.  If the
frozen readout survives there, readout failure is specifically a scale-channel
failure, not a generic consequence of negative transfer.

USAGE
-----
  export REPFIX_DATA_X=/path/to/data_x
  python repro/scripts/real_head_readout_transfer.py --aux clf --alpha 10 \
      --seeds 0,1,2,3,4 --out results/READOUT/har_subject_readout.json
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
import real_aux_mtl as ram  # noqa: E402
from readout_transfer_audit import (  # noqa: E402
    apply_stats, evaluate, fit_head, stats, stratified_split,
)


@torch.no_grad()
def embed_model(model, X, batch=1024):
    dev = next(model.parameters()).device
    return torch.cat([model(X[b0:b0 + batch].to(dev)).cpu()
                      for b0 in range(0, len(X), batch)]).double()


def train_real(D, aux_kind, alpha, mode, seed, m, epochs, lr, batch, device):
    cdm.set_seed(seed)
    model = ram.RealAuxModel(D["d_in"], D["n_classes"], m, aux_kind,
                             D["aux_dim"], regress=False).to(device)
    X = torch.from_numpy(D["X_tr"]).float().to(device)
    y = torch.from_numpy(D["y_tr"]).long().to(device)
    A = (torch.from_numpy(np.asarray(D["aux_tr"])).float().to(device)
         if aux_kind in ("recon", "reg")
         else torch.from_numpy(np.asarray(D["aux_tr"])).long().to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            h = model(X[idx])
            aux_in = h.detach() if mode == "stf0" else h
            loss = F.cross_entropy(model.primary(h), y[idx])
            loss = loss + alpha * ram._aux_loss(model, aux_in, A[idx], aux_kind)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="har", choices=["har", "radioml"])
    ap.add_argument("--aux", default="clf", choices=["clf", "recon", "reg"])
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/READOUT/har_subject_readout.json")
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    D = ram.load_real_aux(a.domain, a.aux)
    Xtr = torch.from_numpy(D["X_tr"]).float()
    ytr = torch.from_numpy(D["y_tr"]).long()
    Xte = torch.from_numpy(D["X_te"]).float()
    yte = torch.from_numpy(D["y_te"]).long()
    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        print(f"seed {s}: training healthy/joint/detached", flush=True)
        models = {}
        for mode, alpha in [("healthy", 0.0), ("joint", a.alpha), ("stf0", a.alpha)]:
            models[mode] = train_real(D, a.aux, alpha, mode, s, a.m,
                                      a.epochs, a.lr, a.batch, device)
        ht, cal = stratified_split(ytr, 0.2, s)
        htr = {k: embed_model(v, Xtr) for k, v in models.items()}
        hte = {k: embed_model(v, Xte) for k, v in models.items()}
        clf, mu, sd = fit_head(htr["healthy"][ht], ytr[ht])
        row = dict(seed=s, aux=a.aux, alpha=a.alpha, m=a.m,
                   base_scatter=float(cdm.scatter(htr["healthy"].float())),
                   encoders={}, readouts={})
        for name in ["healthy", "joint", "stf0"]:
            own = evaluate(clf, hte[name], yte, mu, sd)
            sc = float(cdm.scatter(hte[name].float()))
            with torch.no_grad():
                hdev = hte[name].float().to(device)
                ydev = yte.to(device)
                acc_own = float((models[name].primary(hdev).argmax(1) == ydev).float().mean())
            row["encoders"][name] = dict(
                acc_own=acc_own,
                scatter=sc,
                rel_scatter=sc / row["base_scatter"] if row["base_scatter"] else sc)
            if name == "healthy":
                frozen, recal, retrained = own, own, own
            else:
                frozen = evaluate(clf, hte[name], yte, mu, sd)
                mu_c, sd_c = stats(htr[name][ht])
                recal = evaluate(clf, hte[name], yte, mu_c, sd_c)
                clf_c, mu_c2, sd_c2 = fit_head(htr[name][ht], ytr[ht])
                retrained = evaluate(clf_c, hte[name], yte, mu_c2, sd_c2)
            row["readouts"][name] = dict(frozen=frozen, recalibrated=recal,
                                         retrained=retrained)
        rows.append(row)
        print(json.dumps({k: v for k, v in row["readouts"].items()}, indent=2), flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(domain=a.domain, aux=a.aux, alpha=a.alpha,
                                   protocol="healthy readout transferred to a real auxiliary encoder",
                                   rows=rows), indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
