"""Finance-domain conflict-resolution ablation: PCGrad / FAMO / CAGrad in place
of plain summation at supercritical auxiliary strength, on the wf_repro
walk-forward pipeline (pairwise-rank primary + MDN/AE/contrastive aux heads).

Duplicates wf_repro.train_fold verbatim except at the single backward point,
where the primary and the auxiliary-block gradients are formed separately and
combined by the same conflict-resolution operators used for HAR/battery in
cross_domain_mtl.py (imported from there so all domains share one
implementation).  Writes one JSON per (fold, method) compatible with the
existing fin_fold*_*.json artefacts.

Usage:
  python wf_conflict_ablate.py --fold 0 --method pcgrad --aux-scale 1.0 \
      --out results/E2d/fin_fold0_pcgrad.json
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # wf_repro.py lives in repro/, not repro/scripts/
sys.path.insert(0, HERE)                    # scripts/ wins: current cross_domain_mtl, not the stale repro/ copy
import wf_repro as W                        # noqa: E402
import cross_domain_mtl as cdm              # noqa: E402  (shared harness helpers)
import moo_combiners as moo                 # noqa: E402  (reference MOO operators)

FOLD_PATTERN = W.FOLD_PATTERN


def train_fold_ablate(fold, method, epochs, batch, device, lr, patience, seed,
                      aux_scale=1.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    path = FOLD_PATTERN.format(n=fold)
    print(f"[fold {fold}] method={method} load {path}", flush=True)
    d = np.load(path, mmap_mode="r")
    Xtr, ytr, Atr, MStr = d["train_X"], d["train_y"], d["train_aux"], d["train_ms"]
    Xva, yva, MSva = d["val_X"], d["val_y"], d["val_ms"]
    Xte, yte, dte = d["test_X"], d["test_y"], d["test_d"]

    model = W.StudentModel(detach_aux_heads=False, fixed_aux_weight=0.0,
                           fixed_aux_k=2, stf_mode=None).to(device)
    moo_state = moo.make_state(method, 2, device)
    opt = torch.optim.Adam(list(model.parameters())
                           + moo.state_parameters(moo_state), lr=lr)
    n = len(Xtr)
    ema = {"lmain": torch.as_tensor(1.0, device=device),
           "ltox": torch.as_tensor(1.0, device=device)}

    best_ic, best_sd, bad = -999.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss, cnt = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = torch.as_tensor(Xtr[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(ytr[idx], dtype=torch.float32, device=device)
            msb = torch.as_tensor(MStr[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            p, o = model(xb, msb)
            loss_p = W.rank_margin_loss(p, yb, beta=1e-3)
            loss_mdn = (o["mdn_log_sigma"]
                        + (yb - o["mdn_mu"]) ** 2 / (2 * torch.exp(2 * o["mdn_log_sigma"]) + 1e-8)).mean()
            loss_ae = F.mse_loss(o["recon"], o["h_aux"])
            loss_cont = W._nt_xent(o["z1"], o["z2"], model.contrast_tau)
            loss_aux = aux_scale * (0.3 * loss_mdn + 0.2 * loss_ae + 0.1 * loss_cont)
            with torch.no_grad():
                ema["lmain"] = 0.9 * ema["lmain"] + 0.1 * loss_p.detach()
                ema["ltox"] = 0.9 * ema["ltox"] + 0.1 * loss_aux.detach()
            # ---- conflict-resolution combination at the backward point ----
            if method in moo.WEIGHTING:
                # Loss-weighting families (UW / IMTL-L / GradNorm / FAMO) learn the
                # scalarisation instead of editing the gradient; the two tasks are
                # the same instance every other row sees, (primary, scaled aux).
                losses = torch.stack([loss_p, loss_aux])
                if method == "gradnorm":
                    params = [p for p in model.parameters() if p.requires_grad]
                    gL = list(torch.autograd.grad(loss_p, params, retain_graph=True,
                                                  allow_unused=True))
                    gT = list(torch.autograd.grad(loss_aux, params, retain_graph=True,
                                                  allow_unused=True))
                    moo_state.weighted_loss(losses).backward()
                    moo_state.update_w(losses, [gL, gT], params, epoch=ep)
                else:
                    moo_state.weighted_loss(losses).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                if method == "famo":
                    # the reference recomputes the post-step losses on the same
                    # batch to form its first-order progress term
                    with torch.no_grad():
                        p2, o2 = model(xb, msb)
                        lp2 = W.rank_margin_loss(p2, yb, beta=1e-3)
                        lmdn2 = (o2["mdn_log_sigma"] + (yb - o2["mdn_mu"]) ** 2
                                 / (2 * torch.exp(2 * o2["mdn_log_sigma"]) + 1e-8)).mean()
                        lae2 = F.mse_loss(o2["recon"], o2["h_aux"])
                        lcont2 = W._nt_xent(o2["z1"], o2["z2"], model.contrast_tau)
                    moo_state.update_w(torch.stack(
                        [lp2, aux_scale * (0.3 * lmdn2 + 0.2 * lae2 + 0.1 * lcont2)]))
            else:
                gp = cdm._param_grads(model, opt, loss_p, retain=True)
                gt = cdm._param_grads(model, opt, loss_aux, retain=True)
                gc = moo.combine_flat(method, [gp, gt], moo_state)[0]
                for pp, g in zip(model.parameters(), gc):
                    pp.grad = g
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            ep_loss += float((loss_p + loss_aux).item()) * len(idx)
            cnt += len(idx)
        val_ic = W.compute_rank_ic(W._predict(model, Xva, MSva, device), yva, d["val_d"])
        if val_ic > best_ic + 1e-5:
            best_ic, bad = val_ic, 0
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"    ep {ep + 1:2d}/{epochs} loss={ep_loss / max(cnt, 1):.5f} "
              f"val_ic={val_ic:+.5f}", flush=True)
        if bad >= patience:
            print(f"    [early-stop] {patience} no-improve", flush=True)
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    oos_ic = W.compute_rank_ic(W._predict(model, Xte, d["test_ms"], device), yte, dte)
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(Xte[:8192], dtype=torch.float32, device=device)
        hb = model.encoder(xb)
        hb = hb - hb.mean(0, keepdim=True)
        cross_std = float(hb.std(dim=0).mean().cpu().numpy())
        sigma_h = float(hb.norm().cpu().numpy())
    print(f"[fold {fold}] method={method} val_ic={best_ic:+.5f} oos_ic={oos_ic:+.5f} "
          f"sigma_h={sigma_h:.4f} collapse={cross_std:.4f}", flush=True)
    return {"fold": fold, "method": method, "aux_scale": aux_scale,
            "val_ic": best_ic, "oos_ic": oos_ic, "sigma_h": sigma_h,
            "collapse": cross_std}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--method", required=True, choices=sorted(moo.FAMILY))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux-scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rec = train_fold_ablate(a.fold, a.method, a.epochs, a.batch, device, a.lr,
                            a.patience, a.seed, a.aux_scale)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
