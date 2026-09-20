"""Direct measurement of gradient alignment on the real auxiliary heads.

Section 7.3 of the paper argues that toxic auxiliary gradients are *aligned*
(each head's gradient is dominated by one shared variance-degenerating axis u_c)
rather than *opposing* the primary gradient, which is why pairwise conflict
resolution has nothing to act on.  This script measures that claim directly,
on the same four real heads and the same doses as Table `real_aux`
(har: alpha=0.1 for recon/clf, radioml: alpha=3000 for recon/reg, m=64).

Per seed we train the unfiltered joint configuration, then on a held-out probe
batch in fp32 take gradients -- with respect to the shared representation h --
of the primary loss and of each real auxiliary loss:

  cos_pa    <g_p, g_a> / (|g_p| |g_a|)          averaged over probe samples
  frac_uc   |P_{u_c} g_a| / |g_a|               share of g_a lying on the collapse axis
  cos_a_uc  <g_a, u_c> / |g_a|                  signed, averaged over samples
  sign_cons share of samples agreeing with the majority sign of g_a . u_c
  cos_p_uc  <g_p, u_c> / |g_p|
  neg_frac  share of samples with g_p . g_a < 0  (where PCGrad would act)
  rho       |P_{u_c} g_a| / |g_p|                the paper's order parameter

With --dual the two HAR heads are co-trained, which also gives the cross-head
cosine cos(g_a1, g_a2): the quantity a conflict operator needs to be negative
to have any opposed pair to resolve.

The reported `scatter` should reproduce the joint row of R5aux for the same
seed and dose; the aggregator joins the two files and uses that agreement as a
consistency gate.

Usage (one shard per seed, so the runner can resume):
  python align_probe.py --domain har --heads recon --alpha 0.1 --m 64 \
      --seeds 0 --epochs 60 --out results/A1/A1_har_recon_s0.json
  python align_probe.py --domain har --heads recon,clf --alpha 0.1 --m 64 \
      --seeds 0 --epochs 60 --dual --out results/A1/A1_har_dual_s0.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_domain_mtl as cdm          # noqa: E402
import real_aux_mtl as ram             # noqa: E402


class MultiAuxModel(nn.Module):
    """Shared encoder + primary head + one or more REAL auxiliary heads."""

    def __init__(self, d_in, n_classes, m, specs, regress):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, m), nn.ReLU(), nn.Linear(m, m), nn.ReLU(), nn.Linear(m, m))
        self.primary = nn.Linear(m, 1 if regress else n_classes)
        self.kinds = [s["kind"] for s in specs]
        heads = []
        for s in specs:
            if s["kind"] == "recon":
                heads.append(nn.Sequential(nn.Linear(m, 256), nn.ReLU(),
                                           nn.Linear(256, s["dim"])))
            else:
                heads.append(nn.Linear(m, s["dim"]))
        self.aux = nn.ModuleList(heads)

    def forward(self, x):
        return self.encoder(x)


def _head_loss(head, h, target, kind):
    if kind == "recon":
        return nn.functional.mse_loss(head(h), target)
    if kind == "clf":
        return nn.functional.cross_entropy(head(h), target)
    if kind == "reg":
        return nn.functional.mse_loss(head(h).squeeze(-1), target)
    raise ValueError(kind)


@torch.no_grad()
def _head_metric(head, h, target, kind):
    if kind == "recon":
        return float(nn.functional.mse_loss(head(h), target).item())
    if kind == "clf":
        return float((head(h).argmax(1) == target).float().mean().item())
    if kind == "reg":
        return float(nn.functional.mse_loss(head(h).squeeze(-1), target).item())
    raise ValueError(kind)


def _cos(a, b, eps=1e-8):
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + eps
    return num / den


def build_specs(domain, heads):
    """One spec per requested head, sharing the primary dataset of the domain."""
    specs = []
    for hk in heads:
        D = ram.load_real_aux(domain, hk)
        specs.append(dict(kind=hk, dim=int(D["aux_dim"]), tr=D["aux_tr"],
                          te=D["aux_te"], eval_split=D.get("aux_eval_split", "test")))
    return specs


def run_seed(domain, heads, alpha, m, seed, epochs, lr, batch, rho_c, beta,
             device, probe_n=384):
    D = cdm.load_domain(domain)
    regress = D["n_classes"] == 1
    specs = build_specs(domain, heads)
    cdm.set_seed(seed)
    torch.cuda.manual_seed(seed)
    cdm.enable_max_perf()

    model = MultiAuxModel(D["d_in"], D["n_classes"], m, specs, regress).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr = torch.from_numpy(D["X_tr"]).float().to(device)
    Xte = torch.from_numpy(D["X_te"]).float().to(device)
    ytr = (torch.from_numpy(D["y_tr"]).float() if regress
           else torch.from_numpy(D["y_tr"]).long()).to(device)
    yte = (torch.from_numpy(D["y_te"]).float() if regress
           else torch.from_numpy(D["y_te"]).long()).to(device)
    Atr, Ate = [], []
    for s in specs:
        if s["kind"] in ("recon", "reg"):
            Atr.append(torch.from_numpy(np.asarray(s["tr"])).float().to(device))
            Ate.append(torch.from_numpy(np.asarray(s["te"])).float().to(device))
        else:
            Atr.append(torch.from_numpy(np.asarray(s["tr"])).long().to(device))
            Ate.append(torch.from_numpy(np.asarray(s["te"])).long().to(device))
    criterion = nn.MSELoss() if regress else nn.CrossEntropyLoss()
    use_amp = device == "cuda"
    n = Xtr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                h = model(xb)
                pk = model.primary(h)
                L = criterion(pk.squeeze(-1) if regress else pk, yb)
                for k, s in enumerate(specs):
                    L = L + alpha * _head_loss(model.aux[k], h, Atr[k][idx], s["kind"])
            L.backward()
            opt.step()

    # ---- test metrics -------------------------------------------------------
    model.eval()
    with torch.no_grad():
        hp = model(Xte)
        sh = cdm.scatter(hp)
        pred = model.primary(hp)
        metric = (float((pred.squeeze(-1) - yte).pow(2).mean().sqrt().item()) if regress
                  else float((pred.argmax(1) == yte).float().mean().item()))
        aux_metrics = []
        for k, s in enumerate(specs):
            if s["eval_split"] == "train":
                hq = model(Xtr[:8192])
                aux_metrics.append(_head_metric(model.aux[k], hq, Atr[k][:8192], s["kind"]))
            else:
                aux_metrics.append(_head_metric(model.aux[k], hp, Ate[k], s["kind"]))

    # ---- alignment probe (fp32, outside autocast) --------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type=dev, enabled=False), torch.enable_grad():
        Xq = Xte[:min(len(Xte), probe_n)].clone()
        yq = yte[:min(len(yte), probe_n)]
        h = model(Xq).detach().float().requires_grad_(True)
        pk = model.primary(h)
        pk = pk.squeeze(-1) if regress else pk
        gp = torch.autograd.grad(criterion(pk, yq), h, retain_graph=True,
                                 allow_unused=True)[0]
        ga = []
        for k, s in enumerate(specs):
            aq = Ate[k][:min(len(Ate[k]), probe_n)]
            if aq.dtype == torch.long:
                aq = aq.clamp_max(s["dim"] - 1)
            ga.append(torch.autograd.grad(
                _head_loss(model.aux[k], h, aq, s["kind"]), h,
                retain_graph=(k < len(specs) - 1), allow_unused=True)[0])

    out = dict(domain=domain, heads=list(heads), alpha=float(alpha), m=m, seed=seed,
               mode="joint", metric=metric, aux_metric=aux_metrics, scatter=sh)
    if gp is None or any(g is None for g in ga):
        out["align_ok"] = False
        return out

    with torch.no_grad():
        hd = h.detach()
        hc = hd - hd.mean(0, keepdim=True)
        uc = hc / hc.norm(dim=1, keepdim=True).clamp_min(1e-8)   # per-sample u_c
        gp_n = gp.norm(dim=1) + 1e-8
        per_head = []
        for k, g in enumerate(ga):
            g_uc = (g * uc).sum(-1)
            per_head.append(dict(
                cos_pa=float(_cos(gp, g).mean().item()),
                cos_pa_std=float(_cos(gp, g).std().item()),
                cos_a_uc=float(_cos(g, uc).mean().item()),
                frac_uc=float((g_uc.abs() / (g.norm(dim=1) + 1e-8)).mean().item()),
                cos_p_uc=float(_cos(gp, uc).mean().item()),
                neg_frac=float(((_cos(gp, g) < 0).float().mean()).item()),
                sign_cons=float(max((g_uc > 0).float().mean().item(),
                                    (g_uc <= 0).float().mean().item())),
                rho=float(((g_uc.abs() / gp_n).mean()).item()),
            ))
        pairwise = []
        for i in range(len(ga)):
            for j in range(i + 1, len(ga)):
                pairwise.append(dict(pair="%s|%s" % (specs[i]["kind"], specs[j]["kind"]),
                                     cos=float(_cos(ga[i], ga[j]).mean().item()),
                                     neg_frac=float(((_cos(ga[i], ga[j]) < 0)
                                                     .float().mean()).item())))
    out.update(align_ok=True, align=per_head, pairwise=pairwise, metric_probe=metric)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["har", "radioml"])
    ap.add_argument("--heads", required=True, help="comma list, e.g. recon or recon,clf")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--rho-c", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--probe-n", type=int, default=384)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    heads = [h for h in a.heads.split(",") if h]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        r = run_seed(a.domain, heads, a.alpha, a.m, s, a.epochs, a.lr, a.batch,
                     a.rho_c, a.beta, device, probe_n=a.probe_n)
        rows.append(r)
        if r.get("align_ok"):
            h0 = r["align"][0]
            print("[align] %s %s seed=%d scatter=%.4f metric=%.4f | cos_pa=%+.3f "
                  "frac_uc=%.3f cos_a_uc=%+.3f neg_frac=%.3f sign_cons=%.2f rho=%.3f"
                  % (a.domain, "+".join(heads), s, r["scatter"], r["metric"],
                     h0["cos_pa"], h0["frac_uc"], h0["cos_a_uc"], h0["neg_frac"],
                     h0["sign_cons"], h0["rho"]), flush=True)
            for pw in r["pairwise"]:
                print("        pairwise %s cos=%+.3f neg_frac=%.3f"
                      % (pw["pair"], pw["cos"], pw["neg_frac"]), flush=True)
        else:
            print("[align] %s %s seed=%d FAILED probe" % (a.domain, "+".join(heads), s),
                  flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(domain=a.domain, heads=heads, alpha=a.alpha, m=a.m,
                       kind="align_probe", raw=rows), f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
