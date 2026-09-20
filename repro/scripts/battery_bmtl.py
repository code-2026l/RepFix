"""BMTL: Battery Multi-Task Learning — gradient toxicity + STF repair
specialized for SOH & RUL prediction.  Built on the RepFix theory.

 Architecture
   per-cycle window  x:(B,T,D_in)
     -> TSMixer encoder (keep T)              (B,T,d)
     -> GatedDeltaNet encoder (keep T)        (B,T,d)
     -> mean-pool over time  z:(B,d)
   primary heads : SOH MSE head, RUL MAE head     (drive the encoder)
   toxic aux heads: AE reconstruction head, MDN RUL-distribution head
                   (constant attractors; their grads collapse the encoder)

 Configs
   joint    : no isolation, no weighting            -> expected collapse
   joint_w  : no isolation, tri-weighting           -> weights help but collapse persists
   stf0     : STF-0 (detach aux) isolation          -> repaired, no weighting
   stf0_w   : STF-0 + tri-weighting  (ours, full)   -> best

 Tri-weighting (MTG-style, multiplicative)
   w_i = w_rate(rate_i) * w_temp(temp_i) * w_degrad(|dSOH|_i)
   up-weighs fast-charging / extreme-temp / rapid-degradation windows.

 Run (after battery_parsed/<ds>_cycles.json exists):
   python battery_bmtl.py --dataset CALCE --out results/battery_bmtl_CALCE.json
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from cross_domain_mtl import set_seed, enable_max_perf


# =============================================================================
#  TSMixer block/encoder (ICDM2023) -- feature + temporal mixing, keeps T
# =============================================================================
class TSMixerBlock(nn.Module):
    def __init__(self, d: int, max_ts: int, dropout=0.05):
        super().__init__()
        self.norm_f = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, 4*d); self.f2 = nn.Linear(4*d, d)
        self.norm_t = nn.LayerNorm(d)
        # time mixing: after feature mixing, apply Linear along time axis
        # since all sequences already padded/cut to max_ts, fixed shape.
        self.t1 = nn.Linear(max_ts, 4*max_ts); self.t2 = nn.Linear(4*max_ts, max_ts)
        self.drop = dropout
        self.max_ts = max_ts

    def forward(self, x):                     # x:(B,T,d)
        r = x; x = self.norm_f(x)
        x = F.gelu(self.f1(x)); x = F.dropout(x, self.drop, training=self.training)
        x = r + self.f2(x)
        r = x; x = self.norm_t(x)
        xt = x.transpose(1, 2)                                 # (B,d,T)
        h = F.gelu(self.t1(xt)); h = F.dropout(h, self.drop, training=self.training)
        x = r + self.t2(h).transpose(1, 2)                     # back to (B,T,d)
        return x


class TSMixerEncoder(nn.Module):
    def __init__(self, d_in, d, max_ts, n_layer=2, dropout=0.05):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList([TSMixerBlock(d, max_ts, dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                     # (B,T,d_in)->(B,T,d)
        x = F.silu(self.proj(x))
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


# =============================================================================
#  GatedDeltaNet block/encoder -- gated MLP with depthwise channel mixing, keeps T
# =============================================================================
class GatedDeltaBlock(nn.Module):
    def __init__(self, d, kernel=3, expansion=4, dropout=0.05):
        super().__init__()
        self.dw = nn.Conv1d(d, d, kernel, groups=d, padding=kernel//2)
        self.norm = nn.LayerNorm(d)
        self.a = nn.Linear(d, expansion*d); self.b = nn.Linear(expansion*d, d)
        self.g = nn.Linear(d, expansion*d)
        self.drop = dropout

    def forward(self, x):                     # (B,T,d)
        r = x
        x = F.silu(self.dw(x.transpose(1, 2)).transpose(1, 2))
        h = F.gelu(self.a(x)) * F.silu(self.g(x))
        h = F.dropout(h, self.drop, training=self.training)
        return r + self.norm(self.b(h) + x)


class GatedDeltaEncoder(nn.Module):
    def __init__(self, d_in, d, n_layer=2, dropout=0.05):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList([GatedDeltaBlock(d, dropout=dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):                     # (B,T,d_in)->(B,T,d)
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


# =============================================================================
#  Toxic auxiliary heads (constant attractors)
# =============================================================================
class AEToxicHead(nn.Module):
    def __init__(self, d_emb, d_in, n_ts):
        super().__init__()
        self.dec = nn.Sequential(nn.Linear(d_emb, 2*d_emb), nn.ReLU(),
                                 nn.Linear(2*d_emb, 2*d_emb), nn.ReLU(),
                                 nn.Linear(2*d_emb, n_ts*d_in))
        self.n_ts, self.d_in = n_ts, d_in

    def forward(self, z):                      # (B,d_emb)->(B,T,D_in)
        return self.dec(z).reshape(z.shape[0], self.n_ts, self.d_in)


class MDNToxicHead(nn.Module):
    def __init__(self, d_emb, k=3):
        super().__init__()
        self.pi = nn.Linear(d_emb, k); self.mu = nn.Linear(d_emb, k)
        self.sig = nn.Linear(d_emb, k)

    def forward(self, z):
        return (F.softmax(self.pi(z), -1), self.mu(z), F.softplus(self.sig(z)) + 1e-3)

    @staticmethod
    def nll(pi, mu, sig, y):
        y = y.unsqueeze(-1)
        comp = pi / (sig * math.sqrt(2*math.pi)) * torch.exp(-0.5*((y-mu)/sig).pow(2))
        return -torch.log(comp.sum(-1).clamp_min(1e-10))


# =============================================================================
#  Full model
# =============================================================================
class BMTL(nn.Module):
    def __init__(self, d_in, d=128, n_ts=128, use_tsmixer=True, use_gatedelta=True, k_mdn=3):
        super().__init__()
        self.use_t, self.use_g, self.n_ts, self.d_in = use_tsmixer, use_gatedelta, n_ts, d_in
        self.proj = nn.Linear(d_in, d)
        self.ts = TSMixerEncoder(d, d, n_ts) if use_tsmixer else None
        self.gd = GatedDeltaEncoder(d, d) if use_gatedelta else None
        self.norm = nn.LayerNorm(d)
        self.head_soh = nn.Linear(d, 1); self.head_rul = nn.Linear(d, 1)
        self.head_ae = AEToxicHead(d, d_in, n_ts); self.head_mdn = MDNToxicHead(d, k_mdn)

    def encode_seq(self, x):                  # (B,T,D_in)->(B,T,d)
        h = F.silu(self.proj(x))
        if self.ts is not None: h = self.ts(h)
        if self.gd is not None: h = self.gd(h)
        return self.norm(h)

    def embed(self, x):
        return self.encode_seq(x).mean(dim=1)  # (B,d)

    def forward(self, x):
        z = self.embed(x)
        soh = self.head_soh(z).squeeze(-1); rul = self.head_rul(z).squeeze(-1)
        return soh, rul, self.head_ae(z), self.head_mdn(z)


# =============================================================================
#  MTG tri-weighting
# =============================================================================
def tri_weights(rate, temp, dsoh, gamma=1.0):
    wr = np.where(rate > 1.5, 1.0 + gamma, 1.0).astype(np.float32)
    wt = np.where((temp < 0) | (temp > 40), 1.0 + gamma, 1.0).astype(np.float32)
    md = np.maximum(np.abs(dsoh).max(), 1e-8)
    wd = 1.0 + gamma * (np.abs(dsoh) / md).astype(np.float32)
    return wr * wt * wd


# =============================================================================
#  Dataset & collate
# =============================================================================
FEATS = ("voltage", "current", "temperature", "resistance")

def _norm(a, lo, hi):
    a = np.asarray(a, np.float32)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def build_dataset(cycles, max_ts=128, gamma_w=1.0):
    """cycles: list of dicts. Returns
    (X, S_norm, R_norm, W, cids, S_scale, R_scale).
    Features are globally min-max normalised per channel; both targets are
    normalised to [0,1] so the two primary-task gradients stay balanced and the
    toxic aux heads can drive collapse; the caller maps back to physical units."""
    cycles = [c for c in cycles if c.get("SOH", 0) > 0]   # drop missing-capacity rows
    rate = np.array([c["C_rate"] for c in cycles], np.float32)
    temp = np.array([c["temp"] for c in cycles], np.float32)
    dsoh = np.array([c["dSOH"] for c in cycles], np.float32)
    w = tri_weights(rate, temp, dsoh, gamma_w)
    # condition group for per-condition eval: 0=standard, 1=fast-charge, 2=extreme-temp
    cond = np.where(rate > 1.5, 1, np.where((temp < 0.0) | (temp > 40.0), 2, 0)).astype(np.int64)
    X = np.zeros((len(cycles), max_ts, len(FEATS)), np.float32)
    SOH = np.array([c["SOH"] for c in cycles], np.float32)      # 0-100 %
    RUL = np.array([c["RUL"] for c in cycles], np.float32)      # cycles
    cids = np.array([c.get("_cid", "cell") for c in cycles])
    rul_max = max(float(RUL.max()), 1.0)
    S_n = SOH / 100.0
    R_n = RUL / rul_max
    # per-channel global min/max for normalisation
    for j, f in enumerate(FEATS):
        lo, hi = float("inf"), float("-inf")
        for c in cycles:
            v = np.asarray(c.get(f, np.zeros(0)), np.float32)
            if v.size:
                lo = min(lo, float(v.min()))
                hi = max(hi, float(v.max()))
        for i, c in enumerate(cycles):
            v = np.asarray(c.get(f, np.zeros(0)), np.float32)
            v = _norm(v, lo, hi)
            if len(v) > max_ts: v = v[-max_ts:]
            elif len(v) < max_ts: v = np.concatenate([np.zeros(max_ts - len(v)), v])
            X[i, :, j] = v
    return X, S_n, R_n, w, cond, cids, (100.0, rul_max)


class CycDataset(Dataset):
    def __init__(self, X, SOH, RUL, W, Cond):
        self.X, self.S, self.R, self.W, self.C = X, SOH, RUL, W, Cond
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (torch.tensor(self.X[i]), torch.tensor(self.S[i]),
                torch.tensor(float(self.R[i])), torch.tensor(self.W[i]),
                torch.tensor(int(self.C[i])))   # RUL as float for reg


def collate(b):
    xs, ss, rs, ws, cs = zip(*b)
    return (torch.stack(xs), torch.stack(ss), torch.stack(rs), torch.stack(ws),
            torch.stack(cs))


# =============================================================================
#  Train / eval
# =============================================================================
def run(model, dl_tr, dl_te, opt, epochs, dev, alpha_aux, stf, wattr, wgt, scale,
        tox_var=False, single=False):
    """single: 'soh'/'rul' -> train only that head (single-task ceiling, no aux).
    Returns best {soh_rmse_%, rul_mae_cycles} in physical units."""
    s_scale, r_scale = scale
    best = {"soh": float("inf"), "rul": float("inf")}
    for ep in range(1, epochs+1):
        model.train(); tot = 0.0; nt = 0
        for xb, sb, rb, wb, _ in dl_tr:
            xb, sb, rb, wb = xb.to(dev), sb.to(dev), rb.to(dev), wb.to(dev)
            opt.zero_grad()
            z = model.embed(xb)
            soh = model.head_soh(z).squeeze(-1); rul = model.head_rul(z).squeeze(-1)
            l_soh = F.mse_loss(soh, sb, reduction='none')
            l_rul = F.l1_loss(rul, rb, reduction='none')
            # weight multiplication on the primary task(s) only
            w = wb if wattr else torch.ones_like(wb)
            if single == "soh":
                prim = l_soh * w
            elif single == "rul":
                prim = l_rul * w
            else:
                prim = (l_soh + l_rul) * w
            if single:
                loss = prim.mean()
            elif tox_var:
                # pure variance-shrinkage: pulls every rep toward the batch mean,
                # i.e. along the exact theoretical collapse direction u_c = h-E_b[h].
                aux_src = z.detach() if stf else z          # z:(B,d)
                mu = aux_src.mean(0, keepdim=True)
                L_aux = ((aux_src - mu) ** 2).mean(dim=1)   # per-sample batch variance
                aux = L_aux
            elif stf:
                zt = z.detach()
                l_ae = F.mse_loss(model.head_ae(zt), xb, reduction='none').mean(dim=(1, 2))
                pi, mu, sig = model.head_mdn(zt)
                l_mdn = MDNToxicHead.nll(pi, mu, sig, rb)
                aux = l_ae + l_mdn
            else:
                recon = model.head_ae(z)
                l_ae = F.mse_loss(recon, xb, reduction='none').mean(dim=(1, 2))
                pi, mu, sig = model.head_mdn(z)
                l_mdn = MDNToxicHead.nll(pi, mu, sig, rb)
                aux = l_ae + l_mdn
            if not single:
                loss = prim.mean() + alpha_aux * aux.mean()
            loss.backward(); opt.step()
            tot += loss.item() * xb.shape[0]; nt += xb.shape[0]
        # eval (overall + per-condition)
        model.eval()
        with torch.no_grad():
            mse_s = 0.0; mae_r = 0.0; nb = 0
            acc = {g: [0.0, 0.0, 0] for g in (0, 1, 2)}   # g -> [mse, mae, n]
            for xb, sb, rb, _, cb in dl_te:
                xb, sb, rb, cb = xb.to(dev), sb.to(dev), rb.to(dev), cb.to(dev)
                soh, rul, _, _ = model(xb)
                mse_s += F.mse_loss(soh, sb).item() * xb.shape[0]
                mae_r += F.l1_loss(rul, rb).item() * xb.shape[0]
                nb += xb.shape[0]
                for g in (0, 1, 2):
                    m = cb == g
                    if m.any():
                        acc[g][0] += F.mse_loss(soh[m], sb[m]).item() * m.sum().item()
                        acc[g][1] += F.l1_loss(rul[m], rb[m]).item() * m.sum().item()
                        acc[g][2] += m.sum().item()
            rmse_pct = math.sqrt(mse_s / nb) * s_scale   # SOH RMSE in %
            mae_cyc = (mae_r / nb) * r_scale             # RUL MAE in cycles
            per_cond = {str(g): dict(soh=(math.sqrt(acc[g][0] / acc[g][2]) * s_scale
                                            if acc[g][2] else None),
                                     rul=(acc[g][1] / acc[g][2] * r_scale
                                          if acc[g][2] else None),
                                     n=acc[g][2]) for g in (0, 1, 2)}
        if rmse_pct < best["soh"]:
            best = dict(soh=rmse_pct, rul=mae_cyc, scatter=rep_scatter(model, dl_te, dev),
                        per_cond=per_cond)
            torch.save(model.state_dict(), f"/tmp/_bmtl_best_{id(model)}.pt")
    # load best weights for final scatter
    bf = f"/tmp/_bmtl_best_{id(model)}.pt"
    if os.path.exists(bf):
        model.load_state_dict(torch.load(bf, map_location=dev))
        best["scatter"] = rep_scatter(model, dl_te, dev)
        os.remove(bf)
    return best


@torch.no_grad()
def rep_scatter(model, dl, dev):
    model.eval(); sc = []
    for xb, *_ in dl:
        z = model.embed(xb.to(dev))
        sc.append(float(z.var(dim=1).mean().item()))
    return float(np.mean(sc))


def _path(*parts):
    root = os.environ.get("REPFIX_DATA_X",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_x"))
    return os.path.join(root, *parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CALCE", choices=["CALCE", "MATR", "NASA"])
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--max-ts", type=int, default=128)
    ap.add_argument("--alpha-aux", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--seed-cell", type=int, default=0)
    ap.add_argument("--no-tsmixer", action="store_true")
    ap.add_argument("--no-gatedelta", action="store_true")
    ap.add_argument("--tox-var", action="store_true",
                    help="theory-consistent variance-shrinkage toxifier (u_c=h-E[h]) "
                         "instead of AE+MDN distributed heads")
    ap.add_argument("--single-only", action="store_true",
                    help="train only single-task SOH/RUL health ceilings (no MT configs)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    enable_max_perf()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = _path("battery_parsed", f"{a.dataset.lower()}_cycles.json")
    if not os.path.exists(p):
        print(f"ERROR: parsed data not found at {p}\nRun battery_preprocess.py first.")
        sys.exit(1)
    with open(p) as f:
        cycles = json.load(f)
    X, S, R, W, C, cids, scale = build_dataset(cycles, a.max_ts, a.gamma)
    # leave-cell-out split: partition unique cells, keep 85% of cells in train
    uniq = sorted(set(cids.tolist()))
    random.Random(a.seed_cell).shuffle(uniq)
    tr_cells = set(uniq[:max(1, int(0.85 * len(uniq)))])
    tr_mask = np.isin(cids, list(tr_cells))
    te_mask = ~tr_mask
    Xtr, Str, Rtr, Wtr, Ctr = X[tr_mask], S[tr_mask], R[tr_mask], W[tr_mask], C[tr_mask]
    Xte, Ste, Rte, Wte, Cte = X[te_mask], S[te_mask], R[te_mask], W[te_mask], C[te_mask]
    print(f"  cells train={len(tr_cells)} test={len(uniq)-len(tr_cells)} "
          f"samples train={Xtr.shape[0]} test={Xte.shape[0]}")
    dl_tr = DataLoader(CycDataset(Xtr, Str, Rtr, Wtr, Ctr), batch_size=a.batch, shuffle=True, collate_fn=collate)
    dl_te = DataLoader(CycDataset(Xte, Ste, Rte, Wte, Cte), batch_size=a.batch, shuffle=False, collate_fn=collate)

    configs = [("joint", False, 0), ("joint_w", False, 1),
               ("stf0", True, 0), ("stf0_w", True, 1)]   # ours = stf0_w
    seeds = [int(x) for x in a.seeds.split(",")]
    rows = []
    # single-task health ceilings (SOH-only, RUL-only) -- no auxiliary loss
    if not a.single_only:
        st_ceiling = {}
        for tk in ("soh", "rul"):
            per = []
            for sd in seeds:
                set_seed(sd)
                m = BMTL(d_in=len(FEATS), d=a.d, n_ts=a.max_ts,
                         use_tsmixer=not a.no_tsmixer, use_gatedelta=not a.no_gatedelta).to(dev)
                opt = torch.optim.Adam(m.parameters(), lr=a.lr)
                per.append(run(m, dl_tr, dl_te, opt, a.epochs, dev, a.alpha_aux,
                               False, 0, a.gamma, scale, single=tk))
            st_ceiling[tk] = dict(
                rmse_soh=float(np.mean([r["soh"] for r in per])),
                mae_rul=float(np.mean([r["rul"] for r in per])),
                scatter=float(np.mean([r["scatter"] for r in per])))
            print(f"single[{tk}]", st_ceiling[tk])
    else:
        st_ceiling = {}
    for cname, stf, wattr in configs:
        per_seed = []
        for sd in seeds:
            set_seed(sd)
            m = BMTL(d_in=len(FEATS), d=a.d, n_ts=a.max_ts,
                     use_tsmixer=not a.no_tsmixer, use_gatedelta=not a.no_gatedelta).to(dev)
            opt = torch.optim.Adam(m.parameters(), lr=a.lr)
            b = run(m, dl_tr, dl_te, opt, a.epochs, dev, a.alpha_aux, stf, wattr, a.gamma,
                    scale, tox_var=a.tox_var)
            per_seed.append(dict(**b, seed=sd))
        rows.append(dict(config=cname,
                         rmse_soh=float(np.mean([r["soh"] for r in per_seed])),
                         rmse_soh_std=float(np.std([r["soh"] for r in per_seed])),
                         mae_rul=float(np.mean([r["rul"] for r in per_seed])),
                         scatter=float(np.mean([r["scatter"] for r in per_seed])),
                         per_cond={g: dict(
                             soh=float(np.nanmean([r["per_cond"][g]["soh"]
                                                   for r in per_seed
                                                   if r["per_cond"][g]["soh"] is not None])),
                             rul=float(np.nanmean([r["per_cond"][g]["rul"]
                                                   for r in per_seed
                                                   if r["per_cond"][g]["rul"] is not None])),
                             n=int(np.sum([r["per_cond"][g]["n"] for r in per_seed])))
                                   for g in ("0", "1", "2")}))
        print(cname, rows[-1])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(dataset=a.dataset, d=a.d, gamma=a.gamma, alpha_aux=a.alpha_aux,
                       tox_var=a.tox_var, seeds=seeds,
                       single_ceiling=st_ceiling, results=rows), f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()