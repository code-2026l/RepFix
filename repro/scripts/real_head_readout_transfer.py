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


def _flat_grad(grads, params):
    parts = []
    for g, p in zip(grads, params):
        parts.append((g if g is not None else torch.zeros_like(p)).reshape(-1))
    return torch.cat(parts)


def flux_terms(model, Xq, yq, aq, aux_kind, mode):
    """Signed variance flux on a fixed diagnostic batch.

    V = ||C_B H||_F^2 / (2B); r = -v^T G_p and c = v^T G_a with v = dV/dtheta,
    matching eq. (signed_variance) of the paper.  The gradients are taken over
    all model parameters; v is zero on the heads, so c only sees the encoder.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    B = Xq.shape[0]
    h = model(Xq)
    hc = h - h.mean(0, keepdim=True)
    V = hc.pow(2).sum() / (2 * B)
    v = _flat_grad(torch.autograd.grad(V, params, retain_graph=True,
                                       allow_unused=True), params)
    Lp = F.cross_entropy(model.primary(h), yq)
    aux_in = h.detach() if mode == "stf0" else h
    La = ram._aux_loss(model, aux_in, aq, aux_kind)
    gp = _flat_grad(torch.autograd.grad(Lp, params, retain_graph=True,
                                        allow_unused=True), params)
    ga = _flat_grad(torch.autograd.grad(La, params, allow_unused=True), params)
    vn, gpn, gan = float(v.norm()), float(gp.norm()), float(ga.norm())
    r = -float((v * gp).sum())
    c = float((v * ga).sum())
    return dict(r=r, c=c, c_over_r=(c / r if abs(r) > 1e-12 else float("nan")),
                cos_p=float((v * gp).sum() / (vn * gpn + 1e-12)),
                cos_a=float((v * ga).sum() / (vn * gan + 1e-12)),
                v_norm=vn, gp_norm=gpn, ga_norm=gan,
                V=float(V), scatter=float(hc.std(dim=1).mean()))


def train_real(D, aux_kind, alpha, mode, seed, m, epochs, lr, batch, device,
               flux_trace=False, flux_batch=384, flux_every=25,
               flux_epochs=2, cal_warm=3):
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
    cal = None
    if mode == "stfcal":
        n_steps = (n + batch - 1) // batch
        cal = ram.CalController(alpha, warm_steps=int(cal_warm * n_steps),
                                rank_max=3)
    flux_rows = []
    if flux_trace:
        Bq = min(flux_batch, len(D["X_te"]))
        Xq = torch.from_numpy(D["X_te"][:Bq]).float().to(device)
        yq = torch.from_numpy(D["y_te"][:Bq]).long().to(device)
        aq = torch.from_numpy(np.asarray(D["aux_te"])[:Bq])
        if aux_kind in ("recon", "reg"):
            aq = aq.float().to(device)
        else:
            aq = aq.long().clamp_max(D["aux_dim"] - 1).to(device)
    gstep = 0
    if flux_trace:
        model.eval()
        ft = flux_terms(model, Xq, yq, aq, aux_kind, mode)
        ft.update(epoch=0, step=0, phase="init")
        flux_rows.append(ft)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            h = model(X[idx])
            if cal is not None:
                if not cal.calibrated and cal.step >= cal.warm_steps:
                    pb = ram.real_aux_probe(model, X[idx], y[idx], A[idx],
                                            aux_kind, D["aux_dim"],
                                            F.cross_entropy, False, R=3,
                                            basis="pca")
                    cal.calibrate(pb)
                aux_in, _, _ = cal.step_h(h, "stfcal",
                                         warm=(not cal.calibrated))
            else:
                aux_in = h.detach() if mode == "stf0" else h
            loss = F.cross_entropy(model.primary(h), y[idx])
            loss = loss + alpha * ram._aux_loss(model, aux_in, A[idx], aux_kind)
            opt.zero_grad()
            loss.backward()
            opt.step()
            gstep += 1
            # Resolve the pre-collapse window: sample the signed flux inside the
            # first epochs, where the representation moves fastest.
            if flux_trace and mode == "joint" and ep < flux_epochs:
                if gstep % max(int(flux_every), 1) == 0:
                    model.eval()
                    ft = flux_terms(model, Xq, yq, aq, aux_kind, mode)
                    ft.update(epoch=ep, step=gstep, phase="step")
                    flux_rows.append(ft)
                    model.train()
        if flux_trace:
            model.eval()
            ft = flux_terms(model, Xq, yq, aq, aux_kind, mode)
            ft.update(epoch=ep + 1, step=gstep, phase="epoch")
            flux_rows.append(ft)
    model.eval()
    cal_info = (dict(gate=cal.gate_val, rank=cal.r_use, margin=cal.margin,
                     rho_cal=cal.rho_u) if cal is not None else {})
    return model, flux_rows, cal_info


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
    ap.add_argument("--flux-trace", action="store_true",
                    help="record the per-epoch signed flux r_t=-v^T G_p, "
                         "c_t=v^T G_a on a fixed diagnostic batch")
    ap.add_argument("--flux-batch", type=int, default=384)
    ap.add_argument("--flux-every", type=int, default=25,
                    help="sample the joint-mode flux every N minibatches inside "
                         "the first --flux-epochs epochs")
    ap.add_argument("--flux-epochs", type=int, default=2)
    ap.add_argument("--stfcal", action="store_true",
                    help="also train a calibrated STF arm and audit its readout")
    ap.add_argument("--cal-warm-epochs", type=int, default=3)
    ap.add_argument("--out", default="results/READOUT/har_subject_readout.json")
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    D = ram.load_real_aux(a.domain, a.aux)
    Xtr = torch.from_numpy(D["X_tr"]).float()
    ytr = torch.from_numpy(D["y_tr"]).long()
    Xte = torch.from_numpy(D["X_te"]).float()
    yte = torch.from_numpy(D["y_te"]).long()
    rows = []
    modes = [("healthy", 0.0), ("joint", a.alpha), ("stf0", a.alpha)]
    if a.stfcal:
        modes.append(("stfcal", a.alpha))
    for s in [int(x) for x in a.seeds.split(",")]:
        print(f"seed {s}: training {','.join(m for m, _ in modes)}", flush=True)
        models = {}
        flux = {}
        calinfo = {}
        for mode, alpha in modes:
            models[mode], flux[mode], calinfo[mode] = train_real(
                D, a.aux, alpha, mode, s, a.m, a.epochs, a.lr, a.batch, device,
                flux_trace=a.flux_trace, flux_batch=a.flux_batch,
                flux_every=a.flux_every, flux_epochs=a.flux_epochs,
                cal_warm=a.cal_warm_epochs)
        ht, cal = stratified_split(ytr, 0.2, s)
        htr = {k: embed_model(v, Xtr) for k, v in models.items()}
        hte = {k: embed_model(v, Xte) for k, v in models.items()}
        clf, mu, sd = fit_head(htr["healthy"][ht], ytr[ht])
        row = dict(seed=s, aux=a.aux, alpha=a.alpha, m=a.m,
                   base_scatter=float(cdm.scatter(htr["healthy"].float())),
                   encoders={}, readouts={})
        if a.stfcal:
            row["stfcal"] = calinfo["stfcal"]
        for name in models.keys():
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
        if a.flux_trace:
            row["flux"] = flux
        rows.append(row)
        print(json.dumps({k: v for k, v in row["readouts"].items()}, indent=2), flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(domain=a.domain, aux=a.aux, alpha=a.alpha,
                                   flux_trace=a.flux_trace, flux_batch=a.flux_batch,
                                   flux_every=a.flux_every, flux_epochs=a.flux_epochs,
                                   stfcal=a.stfcal,
                                   cal_warm_epochs=a.cal_warm_epochs,
                                   protocol="healthy readout transferred to a real auxiliary encoder",
                                   rows=rows), indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
