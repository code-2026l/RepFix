"""NYUv2 dense benchmark: 3 real heads + calibrated STF on a conv encoder.

Standard-benchmark validation of the toxicity mechanism and STF-cal, mirroring
the battery flagship protocol (battery_bmtl_v3.py):

    x (B,3,144,192) 0-1 RGB
      -> conv encoder (4 stride-2 blocks, BN+ReLU)  (B,256,9,12)
      -> flatten + Linear + ReLU + Linear           z (B,d)
    real heads on z (linear, per-cell coarse grids at 9x12):
      seg    : Linear(d, 13*108)  CE(ignore_index=-1)
      depth  : Linear(d, 108)     L1 (metres)
      normal : Linear(d, 3*108)   cosine, invalid (zero-norm) cells masked
    auxiliary toxicity, two flavors (--aux):
      proj   : k_aux FIXED random variance projections on z (the harness
               toxifier -- the theory's exact constant attractor)
      ae     : trainable reconstruction decoder z -> image (the battery
               flagship's production aux; carries real aux signal so the
               STF-cal rank filter can be contrasted with STF-0 erasure on
               the aux-reconstruction metric)
    STF-cal: warm-up epochs with the aux branch detached + healthy scatter
    anchor S*, ONE rho probe on the healthy encoder, then
    gate = 1[alpha*rho_u >= 1], rank = min{r: alpha*rho_res[r] < 1}, latched.

Configs: single_seg / single_dep / single_nor (health ceilings), joint3 (real
MTL, no aux), joint3_aux, joint3_aux_stf0, joint3_aux_stfcal,
joint3_aux_stfcalrank.  Metrics are FINAL-epoch (no test-based selection).

Run:
  python nyu_bmtl.py --configs single,joint3,joint3_aux,joint3_aux_stf0,joint3_aux_stfcal \
      --aux proj --alpha-aux 4.0 --seeds 0,1,2,3,4,5,6,7,8,9 --out results/R7/nyu_bmtl.json
"""
from __future__ import annotations
import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cross_domain_mtl import set_seed, enable_max_perf

N_SEG = 13          # valid seg classes 0..12 (-1 = unlabeled / ignore)
GH, GW = 9, 12      # coarse grid
N_CELL = GH * GW


def _path(*parts):
    root = os.environ.get("REPFIX_DATA_X",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_x"))
    return os.path.join(root, *parts)


# =============================================================================
#  Model
# =============================================================================
class NyuMTL(nn.Module):
    def __init__(self, d=128, k_aux=3, aux_seed=11):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        self.enc = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        self.flat = nn.Linear(256 * GH * GW, d)
        self.out = nn.Linear(d, d)
        self.head_seg = nn.Linear(d, N_SEG * N_CELL)
        self.head_dep = nn.Linear(d, N_CELL)
        self.head_nor = nn.Linear(d, 3 * N_CELL)
        self.head_ae = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(),
                                     nn.Linear(2 * d, 2 * d), nn.ReLU(),
                                     nn.Linear(2 * d, 3 * 144 * 192))
        # fixed random variance projections (the harness toxifier)
        g = torch.Generator().manual_seed(aux_seed)
        d_proj = max(d // 4, 4)
        W = torch.randn(k_aux, d, d_proj, generator=g) / math.sqrt(d_proj)
        self.register_buffer("W_aux", W)
        self.d = d

    def encode(self, x):                       # (B,3,144,192) -> (B,d)
        f = self.enc(x)
        z = self.flat(f.reshape(f.shape[0], -1))
        return self.out(F.relu(z))

    def proj_aux(self, z):
        return torch.einsum("bm,kmn->bkn", z, self.W_aux)

    def seg_logits(self, z):                   # (B,13,GH*GW)
        return self.head_seg(z).view(z.shape[0], N_SEG, N_CELL)

    def normal_cos_loss(self, z, tgt_nor):
        B = z.shape[0]
        nor = self.head_nor(z).view(B, 3, N_CELL)
        nor = nor / nor.norm(dim=1, keepdim=True).clamp_min(1e-6)
        vm = tgt_nor.norm(dim=1) > 0.5
        if vm.any():
            return (1.0 - (nor * tgt_nor).sum(1))[vm].mean()
        return z.new_zeros(())

    def primary_losses(self, z, tgt):
        """Sum of the three real-task losses (each mean-reduced)."""
        l_seg = F.cross_entropy(self.seg_logits(z), tgt["seg"], ignore_index=-1)
        l_dep = F.l1_loss(self.head_dep(z), tgt["dep"])
        l_nor = self.normal_cos_loss(z, tgt["nor"])
        return l_seg + l_dep + l_nor


def make_aux_fn(model, kind):
    """Auxiliary loss as a function of the (possibly filtered) representation
    and the raw input batch (the AE flavor reconstructs x)."""
    if kind == "proj":
        def aux_fn(z, x=None):
            a = model.proj_aux(z)
            return ((a - a.mean(0)) ** 2).mean()
        return aux_fn
    if kind == "ae":
        def aux_fn(z, x=None):
            return F.mse_loss(model.head_ae(z).view_as(x), x)
        return aux_fn
    raise ValueError(kind)


# =============================================================================
#  STF-cal (battery-flagship controller, adapted)
# =============================================================================
def rho_probe(model, xb, tgt, aux_fn, R=3):
    """Order-parameter family rho_u / rho_res[0..R] on the UNFILTERED aux branch.
    fp32 outside autocast; gradients w.r.t. z only (no parameter grads)."""
    with torch.autocast("cuda", enabled=False), torch.enable_grad():
        z = model.encode(xb).detach().float().requires_grad_(True)
        lp = model.primary_losses(z, tgt)
        gp = torch.autograd.grad(lp, z, retain_graph=True, allow_unused=True)[0]
        ga = torch.autograd.grad(aux_fn(z, xb), z, retain_graph=False, allow_unused=True)[0]
    if gp is None or ga is None:
        return None
    with torch.no_grad():
        hc = z.detach() - z.detach().mean(0, keepdim=True)
        dz = hc.shape[1]
        cov = (hc.t() @ hc) / max(hc.shape[0], 1) + 1e-6 * torch.eye(dz, device=hc.device)
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


class Cal:
    def __init__(self, alpha_aux, warm_epochs, rank_max=3, ema=0.9):
        self.alpha_aux = float(alpha_aux); self.warm_epochs = int(warm_epochs)
        self.rank_max = int(rank_max); self.ema = float(ema)
        self.epoch = 1
        self.S_ema = None; self.S_star = None
        self.calibrated = False
        self.gate = 0.0; self.rank = 1
        self.rho_u = float("nan"); self.rho_res = []; self.U = None

    def warm_phase(self):
        return self.epoch <= self.warm_epochs

    def observe_scatter(self, s):
        self.S_ema = float(s) if self.S_ema is None else \
            self.ema * self.S_ema + (1.0 - self.ema) * float(s)

    def calibrate(self, probe):
        self.S_star = self.S_ema
        if probe is None:
            self.calibrated = True
            return
        self.rho_u = float(probe["rho_u"])
        self.rho_res = [float(x) for x in probe["rho_res"]]
        self.U = probe["U"].clone()
        self.gate = 1.0 if self.alpha_aux * self.rho_u >= 1.0 else 0.0
        r_use = self.rank_max
        for r in range(0, min(len(self.rho_res) - 1, self.rank_max) + 1):
            if self.alpha_aux * self.rho_res[r] < 1.0:
                r_use = r
                break
        self.rank = max(1, r_use)
        self.calibrated = True

    @property
    def margin(self):
        return self.alpha_aux * self.rho_u

    def filter_z(self, z, mode):
        if self.gate <= 0.0 or self.U is None:
            return z
        hc = z - z.mean(0, keepdim=True)
        if mode == "cal":
            u = hc / hc.norm(dim=1, keepdim=True).clamp_min(1e-8)
            return z - ((hc * u).sum(-1, keepdim=True) * u)
        r = min(self.rank, self.U.shape[1])
        U = self.U[:, :r].to(device=z.device, dtype=z.dtype)
        return z - ((hc @ U) @ U.t())

    def diag(self):
        return dict(cal_rho_u=self.rho_u, cal_margin=self.margin, cal_gate=self.gate,
                    cal_rank=self.rank,
                    cal_S_star=(self.S_star if self.S_star is not None else float("nan")),
                    cal_scatter_ratio=((self.S_ema / self.S_star)
                                       if (self.S_ema is not None and self.S_star) else float("nan")))


# =============================================================================
#  Data
# =============================================================================
def load_tensors(dev):
    p = _path("nyu", "nyu_nyu.npz")
    if not os.path.exists(p):
        print("ERROR: " + p + " missing -- run _hpc/_nyu_prep.py first")
        sys.exit(1)
    Z = np.load(p)
    def T(a, dt):
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev, dtype=dt)
    tr = dict(X=T(Z["X_tr"], torch.float32),
              seg=T(Z["sgr_tr"].reshape(len(Z["sgr_tr"]), -1), torch.long),
              dep=T(Z["y_tr"].reshape(len(Z["y_tr"]), -1), torch.float32),
              nor=T(Z["ngr_tr"].reshape(len(Z["ngr_tr"]), 3, -1), torch.float32))
    te = dict(X=T(Z["X_te"], torch.float32),
              seg=T(Z["sgr_te"].reshape(len(Z["sgr_te"]), -1), torch.long),
              dep=T(Z["y_te"].reshape(len(Z["y_te"]), -1), torch.float32),
              nor=T(Z["ngr_te"].reshape(len(Z["ngr_te"]), 3, -1), torch.float32))
    return tr, te


# =============================================================================
#  Train / eval
# =============================================================================
CFG_AUX = {"joint3": False, "joint3_aux": True, "joint3_aux_stf0": True,
           "joint3_aux_stfcal": True, "joint3_aux_stfcalrank": True}
SINGLES = ("single_seg", "single_dep", "single_nor")


@torch.no_grad()
def eval_metrics(model, te, dev, batch=256):
    model.eval()
    n = te["X"].shape[0]
    conf = torch.zeros(N_SEG * N_SEG, dtype=torch.long, device=dev)
    sq = 0.0; ad = 0.0; ang = 0.0; nvalid_nor = 0; npx = 0; correct = 0
    var_sum = 0.0
    with torch.autocast("cuda", enabled=False):
        for b0 in range(0, n, batch):
            sl = slice(b0, min(b0 + batch, n))
            x = te["X"][sl]
            z = model.encode(x)
            var_sum += float(z.var(dim=1).mean().item()) * x.shape[0]
            logits = model.seg_logits(z)
            pred = logits.argmax(1)                       # (B,cells)
            t = te["seg"][sl]
            valid = t >= 0
            correct += int((pred[valid] == t[valid]).sum().item())
            npx += int(valid.sum().item())
            idx = (t[valid] * N_SEG + pred[valid])
            conf += torch.bincount(idx, minlength=N_SEG * N_SEG)
            dep = model.head_dep(z)
            sq += float(F.mse_loss(dep, te["dep"][sl], reduction="sum").item())
            ad += float(F.l1_loss(dep, te["dep"][sl], reduction="sum").item())
            B = x.shape[0]
            nor = model.head_nor(z).view(B, 3, N_CELL)
            nor = nor / nor.norm(dim=1, keepdim=True).clamp_min(1e-6)
            vm = te["nor"][sl].norm(dim=1) > 0.5
            if vm.any():
                cos = (nor * te["nor"][sl]).sum(1)[vm]
                ang += float(torch.acos(cos.clamp(-1, 1)).sum().item())
                nvalid_nor += int(vm.sum().item())
    conf = conf.view(N_SEG, N_SEG).float()
    union = conf.sum(0) + conf.sum(1) - conf.diag()
    iou = [float(conf[c, c] / union[c]) for c in range(N_SEG) if union[c] > 0]
    # depth losses are summed over samples x grid cells -- normalise by BOTH
    return dict(seg_pxacc=correct / max(npx, 1),
                seg_miou=float(np.mean(iou)),
                dep_rmse=math.sqrt(sq / (n * N_CELL)),
                dep_mae=ad / (n * N_CELL),
                nor_angle=float(ang / max(nvalid_nor, 1)),
                scatter=var_sum / n)


def run_one(cfg, tr, te, dev, d, alpha_aux, epochs, lr, batch, seed,
            aux_kind, cal_warm_epochs, cal_rank_max, probe_only=False):
    torch.set_num_threads(1)
    torch.cuda.manual_seed(seed)
    set_seed(seed)
    enable_max_perf()
    model = NyuMTL(d=d).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    aux_fn = make_aux_fn(model, aux_kind)
    single = None
    if cfg in SINGLES:
        single = cfg.split("_")[1]                    # seg / dep / nor
    cal = None
    if cfg.endswith("stfcal") or cfg.endswith("stfcalrank"):
        cal = Cal(alpha_aux, cal_warm_epochs, rank_max=cal_rank_max)
    n = tr["X"].shape[0]
    probe_info = None
    for ep in range(1, epochs + 1):
        if cal is not None:
            cal.epoch = ep
        model.train()
        perm = torch.randperm(n, device=dev)
        for b0 in range(0, n, batch):
            idx = perm[b0:b0 + batch]
            x = tr["X"][idx]
            tgt = dict(seg=tr["seg"][idx], dep=tr["dep"][idx], nor=tr["nor"][idx])
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(dev == "cuda")):
                z = model.encode(x)
                if single == "seg":
                    loss = F.cross_entropy(model.seg_logits(z), tgt["seg"], ignore_index=-1)
                elif single == "dep":
                    loss = F.l1_loss(model.head_dep(z), tgt["dep"])
                elif single == "nor":
                    loss = model.normal_cos_loss(z, tgt["nor"])
                else:
                    loss = model.primary_losses(z, tgt)
                    # NOTE: probe_only keeps the trajectory aux-free (pure joint3)
                    # so the probe measures the aux pull on a HEALTHY encoder.
                    if CFG_AUX.get(cfg, False) and not probe_only:
                        if cal is not None:
                            with torch.no_grad():
                                cal.observe_scatter(float(z.detach().var(dim=1).mean().item()))
                            if cal.warm_phase():
                                aux_src = z.detach()
                            else:
                                if not cal.calibrated:
                                    cal.calibrate(rho_probe(model, x, tgt, aux_fn,
                                                            R=cal_rank_max))
                                aux_src = cal.filter_z(z, "cal" if cfg.endswith("stfcal")
                                                       else "calrank")
                        elif cfg == "joint3_aux_stf0":
                            aux_src = z.detach()
                        else:
                            aux_src = z
                        loss = loss + alpha_aux * aux_fn(aux_src, x)
            loss.backward()
            opt.step()
        if probe_only and ep == cal_warm_epochs:
            # calibration probe on the healthy (aux-free) encoder
            probe_info = rho_probe(model, tr["X"][:512],
                                   dict(seg=tr["seg"][:512], dep=tr["dep"][:512],
                                        nor=tr["nor"][:512]),
                                   aux_fn, R=cal_rank_max)
            break
    if probe_only:
        return dict(seed=seed, d=d, aux=aux_kind,
                    rho_u=(probe_info["rho_u"] if probe_info else float("nan")),
                    rho_res=([float(v) for v in probe_info["rho_res"]] if probe_info else []))
    m = eval_metrics(model, te, dev)
    if cal is not None:
        m.update(cal.diag())
    m.update(dict(seed=seed, config=cfg))
    return m


def agg(rows, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        out[k + "_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[k + "_std"] = float(np.std(vals)) if vals else float("nan")
    return out


METRICS = ("seg_pxacc", "seg_miou", "dep_rmse", "dep_mae", "nor_angle", "scatter",
           "cal_rho_u", "cal_margin", "cal_gate", "cal_rank",
           "cal_S_star", "cal_scatter_ratio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="single,joint3,joint3_aux,joint3_aux_stf0,joint3_aux_stfcal",
                    help="comma list; 'single' expands to single_seg/dep/nor")
    ap.add_argument("--aux", default="proj", choices=["proj", "ae"])
    ap.add_argument("--alpha-aux", type=float, default=1.0)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--cal-warm-epochs", type=int, default=5)
    ap.add_argument("--cal-rank-max", type=int, default=3)
    ap.add_argument("--probe-only", action="store_true",
                    help="train joint3 for --cal-warm-epochs (aux-free trajectory), "
                         "run the calibration probe, dump rho; no full training")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    enable_max_perf()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, te = load_tensors(dev)
    print("data: train", tuple(tr["X"].shape), "val", tuple(te["X"].shape), flush=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    cfgs = []
    for c in a.configs.split(","):
        c = c.strip()
        if c == "single":
            cfgs.extend(SINGLES)
        elif c:
            cfgs.append(c)
    print("configs:", cfgs, "aux:", a.aux, "alpha:", a.alpha_aux, flush=True)

    if a.probe_only:
        rows = [run_one("joint3_aux", tr, te, dev, a.d, a.alpha_aux,
                        a.epochs, a.lr, a.batch, sd, a.aux,
                        a.cal_warm_epochs, a.cal_rank_max, probe_only=True)
                for sd in seeds]
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(dict(kind="probe", aux=a.aux, d=a.d,
                           warm_epochs=a.cal_warm_epochs, seeds=seeds,
                           results=rows), f, indent=2)
        print("wrote", a.out)
        for r in rows:
            print("seed", r["seed"], "rho_u=%.3f" % r["rho_u"],
                  "rho_res=", ["%.3f" % v for v in r["rho_res"]])
        return

    single_ceiling = {}
    for cfg in SINGLES:
        if cfg not in cfgs:
            continue
        rows = [run_one(cfg, tr, te, dev, a.d, a.alpha_aux, a.epochs, a.lr,
                        a.batch, sd, a.aux, a.cal_warm_epochs, a.cal_rank_max)
                for sd in seeds]
        single_ceiling[cfg] = agg(rows, METRICS)
        print(cfg, single_ceiling[cfg], flush=True)

    results = []
    for cfg in cfgs:
        if cfg in SINGLES:
            continue
        rows = [run_one(cfg, tr, te, dev, a.d, a.alpha_aux, a.epochs, a.lr,
                        a.batch, sd, a.aux, a.cal_warm_epochs, a.cal_rank_max)
                for sd in seeds]
        row = dict(config=cfg)
        row.update(agg(rows, METRICS))
        row["per_seed"] = [dict({k: r[k] for k in METRICS if k in r}, seed=r["seed"])
                           for r in rows]
        results.append(row)
        print(cfg, {k: v for k, v in row.items() if k.endswith("_mean")}, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(dataset="nyu", kind="bmtl", d=a.d, aux=a.aux,
                       alpha_aux=a.alpha_aux, epochs=a.epochs, lr=a.lr,
                       batch=a.batch, seeds=seeds, selection="final",
                       single_ceiling=single_ceiling, results=results), f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
