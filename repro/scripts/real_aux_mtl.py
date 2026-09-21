"""Real auxiliary heads on HAR and RadioML: does gradient toxicity (and its STF
repair) transfer from the synthetic variance-shrinkage toxifier to the practical
auxiliary tasks used in production?

For each domain we attach ONE real auxiliary head at a time (clean attribution):

  HAR (UCI-HAR, 561-d features):
    * recon : sensor-reconstruction autoencoder head  (561-d -> m -> 561-d, MSE)
    * clf   : subject-identification head             (21 train subjects, CE)
  RadioML2016.10A (256-d flattened I/Q):
    * recon : IQ-reconstruction head                  (256-d -> m -> 256-d, MSE)
    * reg   : SNR-regression head                     (linear, MSE on z-scored SNR)

All of these heads contain a constant attractor (reconstruction is minimised by
a constant output that decodes to the batch mean; subject-ID collapses to the
subject prior; SNR regression degenerates to the mean SNR), so Definition 1
(degenerate minimizer) applies to every one of them.

Protocol mirrors cross_domain_mtl.train_one: per seed, first train the
single-task health anchor (alpha=0) to get the scatter ceiling and the spectral
rho anchor; then train joint / stf0 / stfhard / stfsoft / stfcal / stfcalrank at
super-critical alpha.

STF-cal (zero-tuning calibration, mirrors cross_domain_mtl.CalController):
  (1) WARM-UP for `cal_warm_epochs` epochs with the aux branch detached, so the
      encoder follows the healthy primary-task trajectory (scatter anchor S*);
  (2) one probe on that healthy model yields rho_u and the residual family
      rho_res[0..R] for the REAL auxiliary gradient, and the saddle-node
      theorem fixes everything: gate = 1[alpha*rho_u >= 1],
      rank = smallest r with alpha*rho_res[r] < 1;
  (3) the configuration is latched for the rest of training, and a sub-critical
      calibration leaves the auxiliary branch completely untouched.

Metrics per run: primary test metric, auxiliary test metric, relative scatter,
collapse flag, geometric toxicity ratio rho, and the latched gate/rank/margin.
"""
import os
import sys
import json
import math
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_domain_mtl as cdm  # noqa: E402  (reuses loaders, STF, metrics)


# ----------------------------------------------------------------- model
class RealAuxModel(nn.Module):
    """Same 3-layer MLP encoder + primary head as MTLModel, but the auxiliary
    branch is a REAL task head instead of the fixed random projections."""

    def __init__(self, d_in, n_classes, m, aux_kind, aux_dim, regress=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, m), nn.ReLU(), nn.Linear(m, m), nn.ReLU(), nn.Linear(m, m))
        self.primary = nn.Linear(m, 1 if regress else n_classes)
        self.aux_kind = aux_kind
        if aux_kind == "recon":
            self.aux = nn.Sequential(nn.Linear(m, 256), nn.ReLU(), nn.Linear(256, aux_dim))
        elif aux_kind in ("clf", "reg"):
            self.aux = nn.Linear(m, aux_dim)
        else:
            raise ValueError(aux_kind)

    def forward(self, x):
        return self.encoder(x)


def _aux_loss(model, aux_in, target, aux_kind):
    if aux_kind == "recon":
        return nn.functional.mse_loss(model.aux(aux_in), target)
    if aux_kind == "clf":
        return nn.functional.cross_entropy(model.aux(aux_in), target)
    if aux_kind == "reg":
        return nn.functional.mse_loss(model.aux(aux_in).squeeze(-1), target)
    raise ValueError(aux_kind)


@torch.no_grad()
def _aux_metric(model, aux_in, target, aux_kind):
    if aux_kind == "recon":
        return float(nn.functional.mse_loss(model.aux(aux_in), target).item())
    if aux_kind == "clf":
        return float((model.aux(aux_in).argmax(1) == target).float().mean().item())
    if aux_kind == "reg":
        return float(nn.functional.mse_loss(model.aux(aux_in).squeeze(-1), target).item())
    raise ValueError(aux_kind)


# ----------------------------------------------------------------- data
def load_real_aux(domain, aux_kind):
    """Returns dict with X/y (primary) + aux targets for train/test."""
    D = cdm.load_domain(domain)
    out = dict(D)
    if domain == "har":
        if aux_kind == "clf":
            base = cdm._path("uci_har", "extracted", "UCI HAR Dataset")
            s_tr = np.loadtxt(os.path.join(base, "train", "subject_train.txt")).astype(np.int64)
            s_te = np.loadtxt(os.path.join(base, "test", "subject_test.txt")).astype(np.int64)
            ids = sorted(set(s_tr.tolist()))            # 21 train subjects
            remap = {s: i for i, s in enumerate(ids)}
            out["aux_tr"] = np.array([remap[s] for s in s_tr], dtype=np.int64)
            out["aux_te"] = np.array([remap.get(s, len(ids)) for s in s_te], dtype=np.int64)
            out["aux_dim"] = len(ids)                   # unseen test subjects -> last class
            out["aux_eval_split"] = "train"             # test subjects are unseen by design
        elif aux_kind == "recon":
            out["aux_tr"] = D["X_tr"]; out["aux_te"] = D["X_te"]; out["aux_dim"] = D["d_in"]
            out["aux_eval_split"] = "test"
    elif domain == "radioml":
        if aux_kind == "reg":
            mu, sd = float(D["aux_tr"].mean()), float(D["aux_tr"].std() + 1e-8)
            out["aux_tr"] = (D["aux_tr"] - mu) / sd
            out["aux_te"] = (D["aux_te"] - mu) / sd
            out["aux_dim"] = 1
            out["aux_eval_split"] = "test"
        elif aux_kind == "recon":
            out["aux_tr"] = D["X_tr"]; out["aux_te"] = D["X_te"]; out["aux_dim"] = D["d_in"]
            out["aux_eval_split"] = "test"
    else:
        raise ValueError(domain)
    return out


# ------------------------------------------------------------- calibration
CAL_MODES = ("stfcal", "stfcalrank")
RANK_MAX = 3


class CalController:
    """Verbatim port of cross_domain_mtl.CalController (one-shot calibration)."""

    def __init__(self, alpha, warm_steps, rank_max=RANK_MAX, ema=0.9):
        self.alpha = float(alpha); self.warm_steps = int(warm_steps)
        self.rank_max = int(rank_max); self.ema = float(ema)
        self.step = 0
        self.S_ema = None; self.S_star = None
        self.calibrated = False
        self.gate_val = 0.0; self.r_use = 1
        self.rho_u = float("nan"); self.rho_res = []; self.U = None
        self.w_sum = 0.0; self.w_n = 0; self.r_sum = 0

    def observe_scatter(self, s):
        # GPU-resident EMA (0-d tensor on device): no D2H sync in the step loop.
        s = s.detach().float() if torch.is_tensor(s) else torch.as_tensor(float(s))
        self.S_ema = s.clone() if self.S_ema is None else \
            self.ema * self.S_ema + (1.0 - self.ema) * s

    def calibrate(self, probe):
        self.S_star = self.S_ema
        if probe is None:
            self.calibrated = True
            return
        self.rho_u = float(probe["rho_u"])
        self.rho_res = [float(x) for x in probe["rho_res"]]
        self.U = probe["U"].clone()
        self.gate_val = 1.0 if self.alpha * self.rho_u >= 1.0 else 0.0
        r_use = self.rank_max
        for r in range(0, min(len(self.rho_res) - 1, self.rank_max) + 1):
            if self.alpha * self.rho_res[r] < 1.0:
                r_use = r
                break
        self.r_use = max(1, r_use)
        self.calibrated = True

    @property
    def margin(self):
        return self.alpha * self.rho_u

    def step_h(self, h, mode, warm):
        """warm=True: healthy warm-up (aux branch detached).  warm=False and a
        sub-critical calibration: the auxiliary branch is left intact."""
        self.step += 1
        self.w_n += 1
        if warm:
            return h.detach(), 0.0, 0
        if self.gate_val <= 0.0:
            return h, 0.0, 0
        hc = h - h.mean(0, keepdim=True)
        if mode == "stfcal":
            u = hc / hc.norm(dim=1, keepdim=True).clamp_min(1e-8)
            self.w_sum += 1.0; self.r_sum += 1
            return h - ((hc * u).sum(-1, keepdim=True) * u), 1.0, 1
        r = min(self.r_use, self.U.shape[1])
        U = self.U[:, :r].to(device=h.device, dtype=h.dtype)
        self.w_sum += 1.0; self.r_sum += r
        return h - ((hc @ U) @ U.t()), 1.0, r


def real_aux_probe(model, xb, yb, ab, aux_kind, aux_dim, criterion, regress,
                   R=RANK_MAX, basis="pca"):
    """Order-parameter family for a REAL auxiliary gradient, on the UNFILTERED
    branch: rho_u is the mean per-example regularized projected norm ratio;
    rho_res[0..R] uses the full residual gradient norm without another P_c,
    after removing the top-r directions of the depth-R basis.
    fp32, gradients w.r.t. h only -- training graph untouched.

    basis='pca': top-R principal directions of the centered representation, the
    paper's surrogate. A PSD compression theorem applies to its specified
    operator, not automatically to these covariance directions. Even a shared
    eigenbasis does not establish agreement of the leading-eigenvalue ordering.
    basis='aux': top-R eigenvectors of Sigma_a = E[g_a g_a^T].  The denominator
    of rho_res does not depend on the basis, this span maximizes captured squared auxiliary-gradient
    energy. It need not minimize the mean-of-norm-ratios residual used here;
    minimality for the theoretical toxic operator is a separate statement.  Mirrors `_toxic_basis` in code/cross_domain_mtl.py.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type=dev, enabled=False), torch.enable_grad():
        h = model(xb).detach().float().requires_grad_(True)
        pk = model.primary(h)
        pk = pk.squeeze(-1) if regress else pk
        gp = torch.autograd.grad(criterion(pk, yb), h, retain_graph=True,
                                 allow_unused=True)[0]
        aq = ab
        if aq.dtype == torch.long:
            aq = aq.clamp_max(aux_dim - 1)
        ga = torch.autograd.grad(_aux_loss(model, h, aq, aux_kind), h,
                                 retain_graph=False, allow_unused=True)[0]
    if gp is None or ga is None:
        return None
    with torch.no_grad(), torch.autocast(device_type=dev, enabled=False):
        hc = h.detach() - h.detach().mean(0, keepdim=True)
        d = hc.shape[1]
        if basis == "aux":
            n = max(int(ga.shape[0]), 1)
            ga32 = ga.float()
            Sa = (ga32.t() @ ga32) / n
            ev_a, U_a = torch.linalg.eigh(0.5 * (Sa + Sa.t()))
            order = torch.argsort(ev_a, descending=True)
            # re-orthonormalise the retained span so (I - U U^T) is a projector
            U, _ = torch.linalg.qr(U_a[:, order][:, :max(int(R), 1)])
        else:
            cov = (hc.t() @ hc) / max(hc.shape[0], 1) + 1e-6 * torch.eye(d, device=hc.device)
            evals, evecs = torch.linalg.eigh(cov)
            U = evecs[:, torch.argsort(evals, descending=True)][:, :max(int(R), 1)]
        gp_n = gp.norm(dim=1) + 1e-8
        u = hc / hc.norm(dim=1, keepdim=True).clamp_min(1e-8)
        rho_u = float(((((ga * u).sum(-1, keepdim=True)) * u).norm(dim=1) / gp_n).mean().item())
        rho_res = []
        for r in range(0, U.shape[1] + 1):
            ga_r = ga if r == 0 else ga - (ga @ U[:, :r]) @ U[:, :r].t()
            rho_res.append(float((ga_r.norm(dim=1) / gp_n).mean().item()))
    return dict(rho_u=rho_u, rho_res=rho_res, U=U.detach())


# ----------------------------------------------------------------- train
MODES = ("joint", "stf0", "stfhard", "stfsoft", "stfcal", "stfcalrank")


def train_once(domain, D, mode, alpha, m, seed, epochs, lr, batch, rho_c, beta,
               base_scatter, base_rho, aux_kind, device, cal_warm=3, label_frac=1.0,
               rank_max=RANK_MAX, basis="pca"):
    cdm.set_seed(seed)
    torch.cuda.manual_seed(seed)
    cdm.enable_max_perf()
    model = RealAuxModel(D["d_in"], D["n_classes"], m, aux_kind, D["aux_dim"],
                         regress=(D["n_classes"] == 1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    regress = D["n_classes"] == 1
    Xtr = torch.from_numpy(D["X_tr"]).float().to(device)
    Xte = torch.from_numpy(D["X_te"]).float().to(device)
    ytr = (torch.from_numpy(D["y_tr"]).float() if regress else torch.from_numpy(D["y_tr"]).long()).to(device)
    yte = (torch.from_numpy(D["y_te"]).float() if regress else torch.from_numpy(D["y_te"]).long()).to(device)
    atr = torch.from_numpy(np.asarray(D["aux_tr"])).float().to(device) if aux_kind in ("recon", "reg") \
        else torch.from_numpy(np.asarray(D["aux_tr"])).long().to(device)
    ate = torch.from_numpy(np.asarray(D["aux_te"])).float().to(device) if aux_kind in ("recon", "reg") \
        else torch.from_numpy(np.asarray(D["aux_te"])).long().to(device)
    criterion = nn.MSELoss() if regress else nn.CrossEntropyLoss()
    rc = rho_c * base_rho if base_rho else rho_c
    use_amp = device == "cuda"
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    cal = None
    if mode in CAL_MODES:
        n_steps = max(1, math.ceil(Xtr.shape[0] / batch))
        cal = CalController(alpha, warm_steps=int(cal_warm * n_steps),
                            rank_max=rank_max)
    n = Xtr.shape[0]
    # --label-frac: semi-supervised regime.  A fixed random fraction of the
    # training set carries a primary-task label; the auxiliary objective is
    # supervised by the input itself (reconstruction), so it keeps using every
    # sample.  This is the setting in which a benign auxiliary head can pay for
    # itself, and therefore the setting that separates STF-cal -- which leaves a
    # benign head alone when the gate stays closed -- from STF-0, which erases it
    # unconditionally.  label_frac = 1.0 reproduces the released behaviour.
    lab = torch.ones(n, dtype=torch.bool, device=device)
    lab_idx = None
    if label_frac < 1.0:
        _g = torch.Generator().manual_seed(1234 + seed)
        _n_lab = max(1, int(round(float(label_frac) * n)))
        lab[:] = False
        lab[torch.randperm(n, generator=_g)[: _n_lab]] = True
        # Dedicated labelled probe set for the calibration read-out.  The order
        # parameter is a mean of per-cell ratios over B cells whose centred
        # representation spans d dimensions, so B must stay well above d: a
        # 16-cell probe of a 64-d subspace is rank-deficient and reads a
        # spuriously small rho.  Draw a fixed subset of the labelled pool
        # (>= 2d cells) and probe on that, never on unlabelled cells.
        _g2 = torch.Generator().manual_seed(4321 + seed)
        lab_idx = torch.nonzero(lab, as_tuple=False).squeeze(1)
        if lab_idx.numel() > 512:
            lab_idx = lab_idx[torch.randperm(lab_idx.numel(), generator=_g2)[:512]]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            xb, yb, ab = Xtr[idx], ytr[idx], atr[idx]
            opt.zero_grad()
            with amp_ctx:
                h = model(xb)
                if cal is not None:
                    if not cal.calibrated:
                        if cal.step >= cal.warm_steps:
                            # one probe on the healthy model fixes gate AND rank.
                            # Under a label fraction the order parameter must be
                            # read on labelled cells only, or the primary gradient
                            # would be taken against labels the run never sees.
                            if lab_idx is not None and lab_idx.numel() >= 2 * m:
                                _pb = real_aux_probe(model, Xtr[lab_idx], ytr[lab_idx],
                                                     atr[lab_idx], aux_kind, D["aux_dim"],
                                                     criterion, regress, R=rank_max,
                                                     basis=basis)
                            else:
                                _pb = real_aux_probe(model, xb, yb, ab, aux_kind,
                                                     D["aux_dim"], criterion, regress,
                                                     R=rank_max, basis=basis)
                            cal.calibrate(_pb)
                        if not cal.calibrated:
                            with torch.no_grad():
                                cal.observe_scatter(cdm.scatter_t(h.detach()))
                    else:
                        with torch.no_grad():
                            cal.observe_scatter(cdm.scatter_t(h.detach()))
                    aux_in, _w, _r = cal.step_h(h, mode, warm=(not cal.calibrated))
                else:
                    aux_in = cdm.stf_filter(h, mode, rc, beta)[0]
                _pk = model.primary(h)
                _pk = _pk.squeeze(-1) if regress else _pk
                if label_frac < 1.0:
                    # primary loss over the labelled cells of this batch only;
                    # the auxiliary loss below still sees every cell
                    _mk = lab[idx]
                    _per = (nn.functional.mse_loss(_pk, yb, reduction="none") if regress
                            else nn.functional.cross_entropy(_pk, yb, reduction="none"))
                    L_main = _per[_mk].mean() if bool(_mk.any()) else _pk.sum() * 0.0
                else:
                    L_main = criterion(_pk, yb)
                L_aux = _aux_loss(model, aux_in, ab, aux_kind)
                (L_main + alpha * L_aux).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        hp = model(Xte)
        sh = cdm.scatter(hp)
        pred = model.primary(hp)
        metric = float((pred.squeeze(-1) - yte).pow(2).mean().sqrt().item()) if regress \
            else float((pred.argmax(1) == yte).float().mean().item())
        # aux task quality (train slice if test subjects are unseen, else test)
        if D.get("aux_eval_split") == "train":
            hq = model(Xtr[:8192]); aux_eval = _aux_metric(model, hq, atr[:8192], aux_kind)
        else:
            hq = model(Xte); aux_eval = _aux_metric(model, hq, ate, aux_kind)
    # geometric order parameter rho on a probe batch (needs grad -> OUTSIDE no_grad)
    rho = float("nan")
    try:
        Xq = Xte[:min(len(Xte), 384)].clone()
        yq = yte[:min(len(yte), 384)]
        with amp_ctx:
            hqq = model(Xq)
            pk = model.primary(hqq)
            pk = pk.squeeze(-1) if regress else pk
            gp = torch.autograd.grad(criterion(pk, yq), hqq, retain_graph=True,
                                     allow_unused=True)[0]
            # clamp clf targets into the valid class range (unseen test subjects
            # map to the sentinel class n_classes, which CE cannot consume)
            aq = ate[:min(len(ate), 384)]
            if aq.dtype == torch.long:
                aq = aq.clamp_max(D["aux_dim"] - 1)
            ga = torch.autograd.grad(_aux_loss(model, hqq, aq, aux_kind),
                                     hqq, retain_graph=False, allow_unused=True)[0]
        if gp is not None and ga is not None:
            hcq = hqq - hqq.mean(0)
            u = hcq / hcq.norm(dim=1, keepdim=True).clamp_min(1e-8)
            ga_c = (ga * u).sum(-1, keepdim=True) * u
            rho = float(((ga_c.norm(dim=1) / (gp.norm(dim=1) + 1e-8)).mean()).item())
    except Exception as _e:
        import traceback
        print(f"    [rho probe failed] hqq.requires_grad={hqq.requires_grad if 'hqq' in dir() else 'NA'}\n"
              + traceback.format_exc(), flush=True)
    collapse = bool(sh < 0.3 * base_scatter) if base_scatter > 0 else False
    row = dict(domain=domain, aux=aux_kind, mode=mode, alpha=float(alpha), m=m,
               seed=seed, metric=metric, aux_metric=aux_eval, scatter=sh,
               rel_scatter=sh / base_scatter if base_scatter else sh,
               collapse=collapse, rho_mean=rho)
    if cal is not None:
        row.update(gate=cal.gate_val, margin=cal.margin, rank=cal.r_use,
                   rho_cal=cal.rho_u, rank_max=rank_max, cal_basis=basis,
                   cal_warm_epochs=cal_warm)
    return row


def run(domain, aux_kind, alpha, m, seeds, epochs, lr, batch, rho_c, beta, cal_warm=3,
        label_frac=1.0, modes=None, rank_max=RANK_MAX, basis="pca"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = load_real_aux(domain, aux_kind)
    modes = tuple(modes) if modes else MODES
    rows = []
    for s in seeds:
        # single-task health anchor (no aux head in the graph).  Under a label
        # fraction the anchor must be label-starved too, or it would set a
        # ceiling the compared runs cannot reach; train_once at alpha = 0 is
        # exactly a primary-only run and inherits the same mask.
        if label_frac < 1.0:
            anc = train_once(domain, D, "joint", 0.0, m, s, epochs, lr, batch,
                             rho_c, beta, 0.0, None, aux_kind, device,
                             cal_warm=cal_warm, label_frac=label_frac,
                             rank_max=rank_max, basis=basis)
            base_scatter, base_rho = anc["scatter"], anc["rho_mean"]
        else:
            single = cdm.train_one(domain, "single", 0.0, m, s, epochs, lr, batch,
                                   rho_c, beta)
            base_scatter, base_rho = single["scatter"], single["srho"]
        for mode in modes:
            r = train_once(domain, D, mode, alpha, m, s, epochs, lr, batch, rho_c, beta,
                           base_scatter, base_rho, aux_kind, device, cal_warm=cal_warm,
                           label_frac=label_frac, rank_max=rank_max, basis=basis)
            r["base_scatter"] = base_scatter; r["base_rho"] = base_rho
            rows.append(r)
            extra = (f" gate={r.get('gate', float('nan')):.0f}"
                     f" rank={r.get('rank', float('nan')):.0f}"
                     f" margin={r.get('margin', float('nan')):.3f}"
                     if mode in CAL_MODES else "")
            print(f"[{domain}/{aux_kind}] mode={mode} seed={s} metric={r['metric']:.4f} "
                  f"aux={r['aux_metric']:.4f} rel_scatter={r['rel_scatter']:.3f} "
                  f"collapse={r['collapse']} rho={r['rho_mean']:.4f}{extra}", flush=True)
    # aggregate
    agg = []
    for mode in modes:
        rs = [r for r in rows if r["mode"] == mode]
        if not rs:
            continue
        joint_by_seed = {x["seed"]: x["metric"] for x in rows if x["mode"] == "joint"}
        agg.append(dict(domain=domain, aux=aux_kind, mode=mode, alpha=float(alpha), m=m,
                        n_seeds=len(rs),
                        metric_mean=float(np.mean([r["metric"] for r in rs])),
                        metric_std=float(np.std([r["metric"] for r in rs])),
                        aux_metric_mean=float(np.mean([r["aux_metric"] for r in rs])),
                        collapse_rate=float(np.mean([r["collapse"] for r in rs])),
                        rel_scatter_mean=float(np.mean([r["rel_scatter"] for r in rs])),
                        rho_mean=float(np.nanmean([r["rho_mean"] for r in rs])),
                        gate_rate=float(np.mean([r.get("gate", 0.0) for r in rs])),
                        rank_mean=float(np.mean([r.get("rank", 0.0) for r in rs])),
                        margin_mean=float(np.mean([r.get("margin", float("nan")) for r in rs])),
                        recovery_rate=float(np.mean([r["metric"] > joint_by_seed[r["seed"]]
                                                     for r in rs])) if mode != "joint" else 0.0))
    return rows, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["har", "radioml"])
    ap.add_argument("--aux", required=True, choices=["recon", "clf", "reg"])
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--rho-c", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--cal-warm-epochs", type=int, default=3)
    ap.add_argument("--label-frac", type=float, default=1.0,
                    help="fraction of the training set carrying a primary-task "
                         "label; the auxiliary objective still uses every sample "
                         "(semi-supervised regime). 1.0 = released behaviour.")
    ap.add_argument("--modes", default="",
                    help="comma-separated subset of MODES; empty = all")
    ap.add_argument("--rho-rank-max", type=int, default=RANK_MAX,
                    help="probe depth R of the calibrated rank rule (default 3, "
                         "the value every released number was produced with)")
    ap.add_argument("--cal-basis", default="pca", choices=["pca", "aux"],
                    help="basis the residual family is read in: 'pca' (released) "
                         "or 'aux' (auxiliary-gradient eigenvectors)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    modes = tuple(x for x in a.modes.split(",") if x) or None
    rows, agg = run(a.domain, a.aux, a.alpha, a.m, seeds, a.epochs, a.lr, a.batch,
                    a.rho_c, a.beta, cal_warm=a.cal_warm_epochs,
                    label_frac=a.label_frac, modes=modes,
                    rank_max=a.rho_rank_max, basis=a.cal_basis)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(domain=a.domain, aux=a.aux, alpha=a.alpha, m=a.m,
                       kind="real_aux", cal_warm_epochs=a.cal_warm_epochs,
                       label_frac=a.label_frac, rho_rank_max=a.rho_rank_max,
                       cal_basis=a.cal_basis,
                       modes=list(modes or MODES),
                       results=agg, raw=rows), f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
