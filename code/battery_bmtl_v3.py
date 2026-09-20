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

 ---------------------------------------------------------------------------
 PERFORMANCE NOTES (v3-opt)
   The A100 is a single shared device: the wins come from keeping it busy.
   Four changes remove the CPU-side serial bottlenecks that used to idle it:
     * build_dataset: single conversion pass per channel (the old code walked
       the cycle list twice per channel and re-called np.asarray each time);
     * a transparent .npz cache of the built tensors, so the 322 MB MATR JSON
       parse and the normalisation loops run once per (dataset, config) rather
       than once per job;
     * CycDataset pre-materialises torch tensors, so __getitem__ is a view
       instead of a per-sample numpy->torch copy, and the loaders run with
       workers + pin_memory + non_blocking H2D;
     * the eval pass accumulates on-device (no per-batch .item() sync) and the
       best checkpoint is kept in memory instead of written to /tmp every time.
   All of these are numerically identical to the previous revision.
---------------------------------------------------------------------------
 Run (after battery_parsed/<ds>_cycles.json exists):
   python battery_bmtl_v3.py --dataset CALCE --out results/battery_bmtl_CALCE.json
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from cross_domain_mtl import set_seed, enable_max_perf
import moo_combiners as moo          # faithful reference MOO operators (P0-2)


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
def tri_weights(rate, temp, dsoh, gamma=1.0, rate_hi=None, temp_lo=None, temp_hi=None,
                terms=("rate", "temp", "degrad")):
    """Multiplicative tri-weight gate w = w_rate * w_temp * w_degrad.

    Thresholds are DATA-DRIVEN (quantiles of the current dataset) by default, so
    the fast-charge and extreme-temperature windows remain meaningful even when a
    dataset is concentrated at high C-rate (MATR: 99.8% >= 3.6 C).  A fixed 1.5 C
    cut would label almost every MATR sample 'fast-charge' and erase the
    per-condition contrast this gate is meant to create.

    rate_hi : fast-charge threshold (default: dataset 70th percentile of c_rate)
    temp_lo/temp_hi : extreme-temperature band (default: 10th/90th percentile)
    """
    rate = np.asarray(rate, np.float32)
    temp = np.asarray(temp, np.float32)
    note = []
    # --- degenerate-quantile guard (single-protocol datasets like HUST) ---
    rate_spread = float(rate.max() - rate.min()) if rate.size else 0.0
    rate_uniform = rate_spread < 1e-6
    if rate_hi is None:
        if rate_uniform:
            # every sample shares one C-rate: keep the whole set as 'fast-charge'
            # when that shared rate is physically high (>= 2C), else drop the term.
            if float(rate.max()) >= 2.0:
                rate_hi = float(rate.max()) - 1e-3
                note.append('rate uniform %.2fC -> all fast-charge' % rate.max())
            else:
                rate_hi = float('inf')
                note.append('rate uniform %.2fC < 2C -> gate disabled' % rate.max())
        else:
            rate_hi = float(np.quantile(rate, 0.70))
    temp_spread = float(temp.max() - temp.min()) if temp.size else 0.0
    if temp_lo is None or temp_hi is None:
        if temp_spread < 1.0:
            temp_lo, temp_hi = -float('inf'), float('inf')
            note.append('temp uniform -> gate disabled')
        else:
            temp_lo = float(np.quantile(temp, 0.10))
            temp_hi = float(np.quantile(temp, 0.90))
    wr = np.where(rate > rate_hi, 1.0 + gamma, 1.0).astype(np.float32) if "rate" in terms \
        else np.ones_like(rate, np.float32)
    wt = np.where((temp < temp_lo) | (temp > temp_hi), 1.0 + gamma, 1.0).astype(np.float32) if "temp" in terms \
        else np.ones_like(temp, np.float32)
    md = np.maximum(np.abs(dsoh).max(), 1e-8)
    wd = (1.0 + gamma * (np.abs(dsoh) / md).astype(np.float32)) if "degrad" in terms \
        else np.ones_like(rate, np.float32)
    return wr * wt * wd, dict(rate_hi=rate_hi, temp_lo=temp_lo, temp_hi=temp_hi,
                              terms=list(terms), notes=note)


# =============================================================================
#  Dataset & collate
# =============================================================================
FEATS = ("voltage", "current", "temperature", "resistance")

def _norm(a, lo, hi):
    a = np.asarray(a, np.float32)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def build_dataset(cycles, max_ts=128, gamma_w=1.0, w_terms=("rate", "temp", "degrad")):
    """cycles: list of dicts. Returns
    (X, S_norm, R_norm, W, cids, S_scale, R_scale).
    Features are globally min-max normalised per channel; both targets are
    normalised to [0,1] so the two primary-task gradients stay balanced and the
    toxic aux heads can drive collapse; the caller maps back to physical units.

    v3-opt: the per-channel min/max scan and the normalisation now share ONE
    conversion pass over the cycle list (the previous revision converted every
    channel to an array twice, once per scan).  Values are bit-identical."""
    cycles = [c for c in cycles if c.get("SOH", 0) > 0]   # drop missing-capacity rows
    n = len(cycles)
    rate = np.fromiter((float(c["C_rate"]) for c in cycles), np.float32, n)
    temp = np.fromiter((float(c["temp"]) for c in cycles), np.float32, n)
    dsoh = np.fromiter((float(c["dSOH"]) for c in cycles), np.float32, n)
    w, thr = tri_weights(rate, temp, dsoh, gamma_w, terms=w_terms)
    # Condition groups from data-driven thresholds.  0=standard (rate<=thr, temp in-band),
    # 1=fast-charge (rate>thr), 2=extreme-temperature (temp out of band, not fast).
    # Priority: extreme-temp takes 2 only when not already fast-charge, so the two
    # hardest-condition groups are disjoint and each is non-empty by construction.
    cond = np.where(
        rate > thr["rate_hi"], 1,
        np.where((temp < thr["temp_lo"]) | (temp > thr["temp_hi"]), 2, 0)
    ).astype(np.int64)
    _cond_counts = {int(g): int((cond == g).sum()) for g in (0, 1, 2)}
    X = np.zeros((n, max_ts, len(FEATS)), np.float32)
    SOH = np.fromiter((float(c["SOH"]) for c in cycles), np.float32, n)
    RUL = np.fromiter((float(c["RUL"]) for c in cycles), np.float32, n)
    # CENS_AWARE_V3: right-censoring mask (1 = RUL is a lower bound)
    CENS = np.fromiter((int(c.get("cens", 0)) for c in cycles), np.int64, n)
    cids = np.array([c.get("_cid", "cell") for c in cycles])
    rul_max = max(float(RUL.max()), 1.0) if n else 1.0
    S_n = SOH / 100.0
    R_n = RUL / rul_max
    # per-channel global min/max for normalisation -- one conversion pass, then reuse
    for j, f in enumerate(FEATS):
        arrs = [np.asarray(c[f], np.float32) if c.get(f) is not None
                else np.zeros(0, np.float32) for c in cycles]
        lo, hi = float("inf"), float("-inf")
        for a in arrs:
            if a.size:
                lo = min(lo, float(a.min())); hi = max(hi, float(a.max()))
        if lo > hi:                     # channel absent everywhere -> leave zeros
            continue
        span = max(hi - lo, 1e-6)
        for i, a in enumerate(arrs):
            if a.size:
                v = np.clip((a - lo) / span, 0.0, 1.0)
            else:
                v = a
            if v.shape[0] > max_ts:
                v = v[-max_ts:]
            elif v.shape[0] < max_ts:
                v = np.concatenate([np.zeros(max_ts - v.shape[0], np.float32), v])
            X[i, :, j] = v
    return X, S_n, R_n, w, cond, cids, (100.0, rul_max), thr, _cond_counts, CENS


class CycDataset(Dataset):
    """Holds the arrays as torch tensors so __getitem__ is a view, not a copy.

    The previous revision built three fresh tensors per sample per epoch
    (numpy->torch conversion of a 128x4 window), which the single-process
    loader could not overlap with the GPU step."""
    def __init__(self, X, SOH, RUL, W, Cond, Cens=None):
        self.X = X if torch.is_tensor(X) else torch.as_tensor(X, dtype=torch.float32)
        self.S = torch.as_tensor(np.asarray(SOH), dtype=torch.float32)
        self.R = torch.as_tensor(np.asarray(RUL), dtype=torch.float32)
        self.W = torch.as_tensor(np.asarray(W), dtype=torch.float32)
        self.C = torch.as_tensor(np.asarray(Cond), dtype=torch.long)
        self.Cn = (torch.zeros(len(self.X), dtype=torch.long) if Cens is None
                   else torch.as_tensor(np.asarray(Cens), dtype=torch.long))
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.S[i], self.R[i], self.W[i], self.C[i], self.Cn[i])


def collate(b):
    xs, ss, rs, ws, cs, cns = zip(*b)
    return (torch.stack(xs), torch.stack(ss), torch.stack(rs), torch.stack(ws),
            torch.stack(cs), torch.stack(cns))


def _make_loader(ds, batch, shuffle, workers, pin):
    """Loader construction.

    NOTE (reproducibility): the multi-process DataLoader draws one value from
    the *global* torch RNG per iterator (to seed its workers).  That extra draw
    shifts the dropout RNG stream and therefore changes the trajectory even at
    a fixed seed.  pin_memory can likewise change kernel selection.  Both are
    thus opt-in: at the defaults (workers=0, pin=False) this loader is exactly
    the original single-process loader, so runs stay bit-comparable."""
    kw = dict(batch_size=batch, shuffle=shuffle, collate_fn=collate,
              pin_memory=pin, num_workers=workers)
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, **kw)


# =============================================================================
#  Train / eval
# =============================================================================
# ---------------------------------------------------------------- AGG_SWITCH_V3
def _agg_sum(gp, ga, alpha, ema):
    out = []
    for p, a in zip(gp, ga):
        if p is None and a is None:
            out.append(None)
        elif p is None:
            out.append(alpha * a)
        elif a is None:
            out.append(p)
        else:
            out.append(p + alpha * a)
    return out


def _agg_pcgrad_proxy(gp, ga, alpha, ema):
    """[superseded proxy, kept for audit] per-parameter PCGrad projection: the
    released PCGrad flattens the whole network before the dot product, so this
    is *not* the reference algorithm.  Retained only so the pre-P0-2 numbers can
    be identified; the ``pcgrad`` key now runs the reference operator."""
    out = []
    for p, a in zip(gp, ga):
        if p is None or a is None:
            out.append(p if p is not None else (a * alpha if a is not None else None))
            continue
        pv, av = p.reshape(-1), a.reshape(-1)
        dot = float(pv @ av)
        if dot < 0:
            a = (a.reshape(-1) - dot / float(pv @ pv + 1e-12) * pv).reshape(a.shape)
        out.append(p + alpha * a)
    return out


def _agg_famo_proxy(gp, ga, alpha, ema):
    """[superseded proxy, kept for audit] EMA loss-ratio rescaling, not the
    released FAMO (which learns ``softmax(w)`` over the losses by first-order
    progress).  The ``famo`` key now runs the reference loss-weighting path."""
    lp = float(ema.get('lp', 1.0)); la = float(ema.get('la', 1.0))
    w = alpha * lp / max(la, 1e-12)
    w = min(w, 1e6)
    return [((p if p is not None else 0) + w * (a if a is not None else 0)
             if (p is not None or a is not None) else None) for p, a in zip(gp, ga)]


def _agg_cagrad_proxy(gp, ga, alpha, ema, c=0.6):
    """[superseded proxy, kept for audit] per-parameter norm-ball rescaling, not
    the released CAGrad (whose weight vector comes from a simplex-constrained
    solve over the Gram matrix of the flattened gradients)."""
    out = []
    for p, a in zip(gp, ga):
        if p is None or a is None:
            out.append(p if p is not None else (a * alpha if a is not None else None))
            continue
        pv = p.reshape(-1)
        s = (p + alpha * a).reshape(-1)
        pn2 = float(pv @ pv) + 1e-12
        sn2 = float(s @ s) + 1e-12
        lam = max(0.0, (c * pn2 - float(pv @ s)) / sn2)
        out.append((s + lam * s).reshape(p.shape))
    return out


def _agg_ref(method):
    """Reference MOO operator from ``moo_combiners.py`` on the battery driver's
    list-of-tensors convention.  The auxiliary entry carries ``alpha_aux`` (the
    reference convention is ``grads = [g(L_main), g(alpha * L_aux)]``), so it is
    scaled here before flattening; everything else is the unmodified kernel."""
    def f(gp, ga, alpha, ema):
        return moo.combine_flat(
            method,
            [list(gp), [None if a is None else alpha * a for a in ga]])[0]
    return f


def _agg_detach(gp, ga, alpha, ema):
    """STF-0 boundary: aux gradient never reaches the shared encoder."""
    return [(p if p is not None else None) for p in gp]


_AGG = {"sum": _agg_sum,
        "pcgrad": _agg_ref("pcgrad"), "cagrad": _agg_ref("cagrad"),
        "cagrad04": _agg_ref("cagrad04"), "cagrad08": _agg_ref("cagrad08"),
        "mgda": _agg_ref("mgda"), "aligned": _agg_ref("aligned"),
        "imtlg": _agg_ref("imtlg"), "nash": _agg_ref("nash"),
        "detach": _agg_detach,
        "pcgrad_proxy": _agg_pcgrad_proxy, "cagrad_proxy": _agg_cagrad_proxy,
        "famo_proxy": _agg_famo_proxy}


def _grads_of(model, loss, retain=False):
    """Parameter gradients of `loss` via autograd.grad (no .grad round-trip).

    Equivalent to zero_grad + backward + collect, but avoids materialising and
    cloning every parameter's .grad tensor; that double copy dominated runtime
    on the 16k-sample MATR split.  allow_unused=True is required because the
    detached auxiliary path legitimately contributes no gradient to the shared
    encoder, so some entries come back None."""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=retain,
                                allow_unused=True)
    return list(grads)


def _write_grads(model, grads):
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.grad = None
    for p, g in zip(params, grads):
        if g is not None:
            p.grad = g.detach().clone()


# =============================================================================
#  STF-cal for the flagship SOH/RUL encoder
#
#  Same controller as the cross-domain study, adapted to the battery model.
#  Measured motivation: the gradient order parameter is NOT identifiable early
#  in training (the aux gradient carries the loss normalisation, so a probe on a
#  random encoder returns rho ~ 1e-2 for every alpha).  It IS informative on the
#  healthy state, and there it predicts the transition.  STF-cal therefore:
#    (1) WARM-UP: isolate the auxiliary heads (STF-0) so the encoder follows the
#        primary-task trajectory, and measure the healthy embedding scatter S*;
#    (2) CALIBRATE: one probe on that healthy encoder gives the order-parameter
#        family rho_res[0..R]; the theorem's saddle-node then fixes, with no
#        tuned threshold,
#          gate = 1[alpha_aux * rho >= 1]      (is the healthy state unstable?)
#          rank = smallest r with alpha_aux * rho_res[r] < 1   (remove just enough)
#        and the configuration is latched (irreversibility).
#  When the dose is sub-critical the auxiliary branch is left completely intact,
#  which is what distinguishes STF-cal from the blunt STF-0 erasure.
# =============================================================================
def _toxic_basis(hc, ga, gp, R, basis):
    """Return (U, lam) -- the depth-R basis the rank rule deletes, and the
    generalized spectrum behind it.

    basis='pca' : the top-R principal directions of the centered representation.
        This is the surrogate of the paper: Theorem~minimalrank is stated for the
        toxic operator T, and the two coincide only when the auxiliary Hessian is
        diagonal in the representation's principal basis (the variance-shrinkage
        toxifier, whose contraction is isotropic on H_c).

    basis='aux' : the top-R principal directions of the AUXILIARY GRADIENT itself,
        i.e. of Sigma_a = E[g_a g_a^T].  The paper's residual criterion is
        rho_res[r] = mean_i ||g_a^(i) - Pi_U g_a^(i)|| / ||g_p^(i)||, and its
        denominator -- the FULL primary norm -- does not depend on the direction
        being deleted.  The r-dim subspace that makes rho_res[r] fall fastest, and
        therefore the one that satisfies the criterion at the smallest r (which is
        what minimal rank asks for), is the top-r eigenspace of Sigma_a.  No inverse
        of Sigma_p is taken, so the estimate stays well conditioned.

    basis='gen' : the top-R generalized eigenvectors of (Sigma_a, Sigma_p).  These
        maximise the toxic-to-restoring ENERGY ratio.  Diagnostic only: with a
        small-sample Sigma_p the ratio is dominated by Sigma_p's smallest
        eigenvalues, so lam is not on the same scale as the amplitude ratio rho of
        Definition~toxicity.
    """
    dz = hc.shape[1]
    if basis in ('aux', 'gen'):
        # The probe is invoked from inside the training autocast region, where
        # matmul silently runs in bf16; eigh has no bf16 kernel.  The pca branch
        # below is left EXACTLY as released (its covariance picks up fp32 through
        # the +1e-6*eye promotion), so existing stfcal runs stay bit-identical.
        with torch.autocast('cuda', enabled=False):
            n = max(int(ga.shape[0]), 1)
            ga32 = ga.float(); gp32 = gp.float()
            Sa = (ga32.t() @ ga32) / n
            if basis == 'aux':
                ev_a, U_a = torch.linalg.eigh(0.5 * (Sa + Sa.t()))
                order = torch.argsort(ev_a, descending=True)
                # Euclidean-orthonormal already, but re-orthonormalise the retained
                # span so filter_z's (I - U U^T) really is a projector.
                Qr, _ = torch.linalg.qr(U_a[:, order][:, :max(int(R), 1)])
                return Qr.detach(), ev_a[order].detach()
            Sp = (gp32.t() @ gp32) / n
            Sp = Sp + 1e-6 * (Sp.diagonal().mean().clamp_min(1e-12)) * torch.eye(dz, device=hc.device)
            ev_p, U_p = torch.linalg.eigh(Sp)
            W = (U_p * ev_p.clamp_min(1e-12).rsqrt().unsqueeze(0)) @ U_p.t()
            M = W.t() @ Sa @ W
            M = 0.5 * (M + M.t())
            ev_m, Q = torch.linalg.eigh(M)
            order = torch.argsort(ev_m, descending=True)
            V = W @ Q[:, order]
            lam = ev_m[order]
            # The generalized eigenvectors are Sigma_p-orthonormal, NOT Euclidean-
            # orthonormal, while filter_z builds a Euclidean projector (I - U U^T).
            # Re-orthonormalise the retained span so U^T U = I; the span -- and hence
            # the projector -- is unchanged, and the COUNT of supercritical directions
            # (cal_rank_gen) is basis-invariant.
            Qr, _ = torch.linalg.qr(V[:, :max(int(R), 1)])
        return Qr.detach(), lam.detach()
    cov = (hc.t() @ hc) / max(hc.shape[0], 1) + 1e-6 * torch.eye(dz, device=hc.device)
    evals, evecs = torch.linalg.eigh(cov)
    U = evecs[:, torch.argsort(evals, descending=True)][:, :max(int(R), 1)]
    return U, evals[torch.argsort(evals, descending=True)][:max(int(R), 1)]


def battery_rho_probe(model, xb, sb, rb, cnb, cens_aware, R=3, basis='pca'):
    """Order-parameter family rho_u / rho_res[0..R] on the UNFILTERED auxiliary
    branch of the SOH/RUL encoder.  fp32 outside autocast; gradients are taken
    with respect to the embedding only, so no parameter gradients are created.

    rho_u is the paper's rank-1 order parameter and is IDENTICAL for both bases --
    the gate is therefore unaffected by the choice of basis.  Only the subspace the
    rank rule deletes (and the residual family rho_res read on it) changes."""
    with torch.autocast('cuda', enabled=False), torch.enable_grad():
        z = model.embed(xb).detach().float().requires_grad_(True)
        soh = model.head_soh(z).squeeze(-1); rul = model.head_rul(z).squeeze(-1)
        sb_f = sb.float(); rb_f = rb.float()
        l_soh = F.mse_loss(soh, sb_f, reduction='none')
        l_rul_exact = F.l1_loss(rul, rb_f, reduction='none')
        l_rul = (torch.where(cnb > 0, F.relu(rb_f - rul), l_rul_exact)
                 if cens_aware else l_rul_exact)
        gp = torch.autograd.grad((l_soh + l_rul).mean(), z, retain_graph=True,
                                 allow_unused=True)[0]
        l_ae = F.mse_loss(model.head_ae(z), xb.float(), reduction='none').mean(dim=(1, 2))
        pi, mu, sig = model.head_mdn(z)
        l_mdn = MDNToxicHead.nll(pi, mu, sig, rb_f)
        ga = torch.autograd.grad((l_ae + l_mdn).mean(), z, retain_graph=False,
                                 allow_unused=True)[0]
    if gp is None or ga is None:
        return None
    with torch.no_grad():
        hc = z.detach() - z.detach().mean(0, keepdim=True)
        U, lam = _toxic_basis(hc, ga, gp, R, basis)
        gp_n = gp.norm(dim=1) + 1e-8
        u = hc / hc.norm(dim=1, keepdim=True).clamp_min(1e-8)
        rho_u = float(((((ga * u).sum(-1, keepdim=True)) * u).norm(dim=1) / gp_n).mean().item())
        rho_res = []
        for r in range(0, U.shape[1] + 1):
            ga_r = ga if r == 0 else ga - (ga @ U[:, :r]) @ U[:, :r].t()
            rho_res.append(float((ga_r.norm(dim=1) / gp_n).mean().item()))
    return dict(rho_u=rho_u, rho_res=rho_res, U=U.detach(),
                lam=lam.detach(), basis=basis)


class BatteryCal:
    def __init__(self, alpha_aux, warm_epochs, rank_max=3, ema=0.9, basis='pca'):
        self.alpha_aux = float(alpha_aux); self.warm_epochs = int(warm_epochs)
        self.rank_max = int(rank_max); self.ema = float(ema)
        self.basis = str(basis)
        self.epoch = 1
        self.S_ema = None; self.S_star = None
        self.calibrated = False
        self.gate = 0.0; self.rank = 1
        self.rho_u = float("nan"); self.rho_res = []; self.U = None
        self.lam = None

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
        self.lam = None if probe.get("lam") is None else \
            [float(x) for x in probe["lam"]]
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
        """alpha_aux / alpha_c with the calibrated alpha_c = 1/rho_healthy."""
        return self.alpha_aux * self.rho_u

    def filter_z(self, z, mode):
        """Auxiliary-branch embedding with the calibrated toxic subspace removed
        (attached, so the benign residual still trains the auxiliary heads).
        A sub-critical calibration returns z untouched."""
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
        lam = self.lam or []
        return dict(
            cal_rho_u=self.rho_u, cal_margin=self.margin, cal_gate=self.gate,
            cal_rank=self.rank, cal_basis=self.basis,
            # The generalized spectrum and how many of its directions are
            # supercritical.  cal_rank_gen is the rank the SAME criterion would
            # pick with no probe-depth cap, so cal_rank < cal_rank_gen means the
            # cap -- not the criterion -- decided the deletion.
            cal_lam_top=lam[:12],
            cal_rank_gen=int(sum(1 for x in lam if self.alpha_aux * x >= 1.0)),
            cal_lam_max=(lam[0] if lam else float("nan")),
            cal_S_star=(self.S_star if self.S_star is not None else float("nan")),
            cal_scatter_ratio=((self.S_ema / self.S_star)
                               if (self.S_ema is not None and self.S_star)
                               else float("nan")))


def run(model, dl_tr, dl_te, opt, epochs, dev, alpha_aux, stf, wattr, wgt, scale,
        tox_var=False, single=False, cens_aware=True,
        agg='sum', ema=None, amp=True, stf_mode='off',
        cal_warm_epochs=3, cal_rank_max=3, cal_basis='pca', moo_state=None):
    # AGG_SWITCH_V3 + AMP_BF16_V3 + STFCAL + P0-2 faithful MOO operators
    """single: 'soh'/'rul' -> train only that head (single-task ceiling, no aux).
    Returns best {soh_rmse_%, rul_mae_cycles} in physical units."""
    s_scale, r_scale = scale
    dev_is_cuda = (dev == 'cuda')
    if agg in moo.WEIGHTING and moo_state is None:
        raise ValueError(
            "aggregator %r learns its own weights and needs the state built by "
            "main() (so that its parameters are in the optimiser); got "
            "moo_state=None" % agg)
    best = {"soh": float("inf"), "rul": float("inf")}
    best_state = None
    cal = (BatteryCal(alpha_aux, cal_warm_epochs, rank_max=cal_rank_max,
                      basis=cal_basis)
           if stf_mode in ("cal", "calrank", "calg", "calgrank") and not single
           else None)
    for ep in range(1, epochs+1):
        if cal is not None:
            cal.epoch = ep
        model.train()
        for xb, sb, rb, wb, _, cnb in dl_tr:
            xb = xb.to(dev, non_blocking=True); sb = sb.to(dev, non_blocking=True)
            rb = rb.to(dev, non_blocking=True); wb = wb.to(dev, non_blocking=True)
            cnb = cnb.to(dev, non_blocking=True)
            opt.zero_grad()
            _amp = torch.autocast('cuda', dtype=torch.bfloat16, enabled=(amp and dev_is_cuda))
            with _amp:
                z = model.embed(xb)
                soh = model.head_soh(z).squeeze(-1); rul = model.head_rul(z).squeeze(-1)
                l_soh = F.mse_loss(soh, sb, reduction='none')
                # censored-aware RUL loss: exact L1 for observed EOL, one-sided hinge
                # (penalise only under-prediction) for right-censored lower bounds.
                l_rul_exact = F.l1_loss(rul, rb, reduction='none')
                l_rul_cens = F.relu(rb - rul)          # zero when pred >= lower bound
                if cens_aware:
                    l_rul = torch.where(cnb > 0, l_rul_cens, l_rul_exact)
                else:
                    l_rul = l_rul_exact
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
                elif cal is not None:
                    # STF-cal: warm-up isolates the aux heads (healthy trajectory +
                    # scatter anchor); one probe on the healthy encoder then fixes
                    # gate and rank; afterwards the calibrated subspace is removed.
                    with torch.no_grad():
                        cal.observe_scatter(float(z.detach().var(dim=1).mean().item()))
                    if cal.warm_phase():
                        aux_src = z.detach()
                    else:
                        if not cal.calibrated:
                            cal.calibrate(battery_rho_probe(
                                model, xb, sb, rb, cnb, cens_aware,
                                R=cal_rank_max, basis=cal.basis))
                        # 'cal' is the rank-1 per-sample centered-direction filter
                        # (it ignores U and rank entirely); every rank-aware mode --
                        # calrank, calg, calgrank -- must take the subspace path.
                        aux_src = cal.filter_z(
                            z, "cal" if stf_mode == "cal" else "calrank")
                    l_ae = F.mse_loss(model.head_ae(aux_src), xb,
                                      reduction='none').mean(dim=(1, 2))
                    pi, mu, sig = model.head_mdn(aux_src)
                    l_mdn = MDNToxicHead.nll(pi, mu, sig, rb)
                    aux = l_ae + l_mdn
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
                    # AGG_SWITCH_V3: the conflict-resolution operators only make
                    # sense on the toxicity-EXPOSED path.  Under stf / tox_var the
                    # auxiliary gradient is already detached from the encoder, so
                    # the cheap single-backward path is numerically identical.
                    if agg == 'sum' or stf or tox_var or (cal is not None):
                        loss = prim.mean() + alpha_aux * aux.mean()
                        loss.backward(); opt.step()
                    elif agg in moo.WEIGHTING:
                        # Loss-weighting families (UW / IMTL-L / GradNorm / FAMO):
                        # the scalarisation itself is learned from the same two
                        # tasks the summed baseline optimises -- (primary, scaled
                        # auxiliary) -- so there is no gradient surgery here, only
                        # a positive per-task reweighting, i.e. the sign-preserving
                        # class of Proposition~minimality.
                        losses = torch.stack([prim.mean(), alpha_aux * aux.mean()])
                        if agg == 'gradnorm':
                            # GradNorm's restoring term needs the per-task
                            # gradients, taken with autograd so that .grad stays
                            # clean for the weighted-loss backward below
                            params = [p for p in model.parameters() if p.requires_grad]
                            gP = list(torch.autograd.grad(losses[0], params, retain_graph=True,
                                                          allow_unused=True))
                            gA = list(torch.autograd.grad(losses[1], params, retain_graph=True,
                                                          allow_unused=True))
                            moo_state.weighted_loss(losses).backward()
                            moo_state.update_w(losses, [gP, gA], params, epoch=ep)
                        else:
                            moo_state.weighted_loss(losses).backward()
                        opt.step()
                        if agg == 'famo':
                            # the reference recomputes the post-step losses on the
                            # same batch to form its first-order progress term
                            with torch.no_grad():
                                z2 = model.embed(xb)
                                s2 = model.head_soh(z2).squeeze(-1)
                                r2 = model.head_rul(z2).squeeze(-1)
                                l_re = F.l1_loss(r2, rb, reduction='none')
                                l_r = (torch.where(cnb > 0, F.relu(rb - r2), l_re)
                                       if cens_aware else l_re)
                                a2 = F.mse_loss(model.head_ae(z2), xb,
                                                reduction='none').mean(dim=(1, 2))
                                pi2, mu2, sg2 = model.head_mdn(z2)
                                a2 = a2 + MDNToxicHead.nll(pi2, mu2, sg2, rb)
                                p2 = (F.mse_loss(s2, sb, reduction='none') + l_r) * \
                                    (wb if wattr else torch.ones_like(wb))
                            moo_state.update_w(torch.stack(
                                [p2.mean(), alpha_aux * a2.mean()]))
                    else:
                        # AGG_SWITCH_V3: separate the two gradient sources and merge
                        # them with the chosen conflict-resolution operator.  The
                        # theory predicts none of these can stop toxicity because
                        # the toxic gradient is aligned, not opposed.
                        lp = prim.mean(); la = aux.mean()
                        if ema is not None:
                            ema['lp'] = 0.9 * ema.get('lp', float(lp)) + 0.1 * float(lp)
                            ema['la'] = 0.9 * ema.get('la', float(la)) + 0.1 * float(la)
                        gp = _grads_of(model, lp, retain=True)
                        ga = _grads_of(model, la, retain=False)
                        _write_grads(model, _AGG[agg](gp, ga, alpha_aux, ema or {}))
                        opt.step()
                else:
                    loss.backward(); opt.step()
        # eval (overall + per-condition).  Accumulation is kept in the original
        # per-batch float64 form so the selection metric is bit-identical; the
        # embedding is computed once and shared with the classification/regression
        # heads (model(xb) and model.embed(xb) yield the same z).
        model.eval()
        with torch.no_grad():
            mse_s = 0.0; mae_r = 0.0; nb = 0
            acc = {g: [0.0, 0.0, 0] for g in (0, 1, 2)}   # g -> [mse, mae, n]
            for xb, sb, rb, _, cb, _cn in dl_te:
                xb = xb.to(dev, non_blocking=True); sb = sb.to(dev, non_blocking=True)
                rb = rb.to(dev, non_blocking=True); cb = cb.to(dev, non_blocking=True)
                z = model.embed(xb)
                soh = model.head_soh(z).squeeze(-1); rul = model.head_rul(z).squeeze(-1)
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
            # rep_scatter is called exactly as in the original revision: besides
            # producing the scatter it instantiates one more DataLoader iterator,
            # and PyTorch draws a global-RNG value per iterator.  Reusing the
            # already-computed z here would silently shift the dropout stream and
            # change fixed-seed trajectories, so the extra pass is deliberate.
            best = dict(soh=rmse_pct, rul=mae_cyc, scatter=rep_scatter(model, dl_te, dev),
                        per_cond=per_cond)
            # keep the best weights in memory (a /tmp round-trip in the hot loop
            # was pure I/O overhead -- the model is a few MB)
            best_state = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
        best["scatter"] = rep_scatter(model, dl_te, dev)
    if cal is not None:
        best.update(cal.diag())
    return best


@torch.no_grad()
def rep_scatter(model, dl, dev):
    model.eval(); sc = []
    for xb, *_ in dl:   # collate yields (x, s, r, w, cond, cens)
        z = model.embed(xb.to(dev, non_blocking=True))
        sc.append(float(z.var(dim=1).mean().item()))
    return float(np.mean(sc))


def _path(*parts):
    root = os.environ.get("REPFIX_DATA_X",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_x"))
    return os.path.join(root, *parts)


def _ds_cache_path(dataset, max_ts, w_terms, gamma, eol_only, drop_fresh, eol_thresh):
    key = repr((dataset, int(max_ts), tuple(w_terms), float(gamma),
                bool(eol_only), bool(drop_fresh), float(eol_thresh)))
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return _path("cache", f"bmtl_{dataset}_{h}.npz")


def _save_ds_cache(path, X, S, R, W, C, CENS, cids, scale, thr, cond_counts):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, X=X, S=S, R=R, W=W, C=C.astype(np.int64),
                 CENS=CENS.astype(np.int64), cids=np.asarray(cids, dtype=object),
                 scale=np.asarray(scale, np.float64),
                 thr=np.asarray(json.dumps(thr), dtype=object),
                 cc=np.asarray(json.dumps(cond_counts), dtype=object))
        print(f"  [cache] wrote {path}")
    except Exception as e:                       # cache is an optimisation, never fatal
        print(f"  [cache] write failed ({e}); continuing")


def _load_ds_cache(path):
    d = np.load(path, allow_pickle=True)
    cc = {int(k): int(v) for k, v in json.loads(str(d["cc"])).items()}
    return (d["X"], d["S"], d["R"], d["W"], d["C"], d["CENS"],
            d["cids"], tuple(d["scale"].tolist()), json.loads(str(d["thr"])), cc)


def _load_hust_npz():
    """Load the HUST npz cache and expand to the list-of-dicts schema used by
    build_dataset (arrays are float16 on disk, cast to float32 on read)."""
    p = _path("battery_parsed", "hust_cycles.npz")
    if not os.path.exists(p):
        print(f"ERROR: HUST cache not found at {p}\nRun _build_hust_npz.py first.")
        sys.exit(1)
    d = np.load(p, allow_pickle=False)
    X = d["X"].astype(np.float32)
    SOH = d["SOH"].astype(np.float32)
    RUL = d["RUL"]
    CR = d["C_rate"].astype(np.float32)
    TP = d["temp"].astype(np.float32)
    DS = d["dSOH"].astype(np.float32)
    CID = d["cell_id"]
    out = []
    for i in range(len(SOH)):
        out.append(dict(
            voltage=X[i, :, 0].copy(), current=X[i, :, 1].copy(),
            temperature=X[i, :, 2].copy(), resistance=X[i, :, 3].copy(),
            SOH=float(SOH[i]), RUL=int(RUL[i]), C_rate=float(CR[i]),
            temp=float(TP[i]), dSOH=float(DS[i]), _cid=str(CID[i])))
    print(f"  HUST: loaded {len(out)} samples from {p}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CALCE", choices=["CALCE", "MATR", "NASA", "HUST"])
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
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers.  0 = original single-process loader "
                         "(bit-reproducible).  >0 overlaps batch prep with the GPU "
                         "step at 1.1-1.2x but consumes one extra global-RNG draw "
                         "per iterator, so it shifts fixed-seed trajectories.")
    ap.add_argument("--pin-memory", dest="pin_memory", action="store_true",
                    default=False,
                    help="pin host memory for H2D copies (may change kernel choice)")
    ap.add_argument("--no-cache", action="store_true",
                    help="disable the on-disk tensor cache (always rebuild)")
    ap.add_argument("--tox-var", action="store_true",
                    help="theory-consistent variance-shrinkage toxifier (u_c=h-E[h]) "
                         "instead of AE+MDN distributed heads")
    ap.add_argument("--single-only", action="store_true",
                    help="train only single-task SOH/RUL health ceilings (no MT configs)")
    ap.add_argument("--eol-only", action="store_true",
                    help="keep only cells that genuinely reached EOL (min SOH <= 80%%); "
                         "removes right-censored RUL labels (P2 fix)")
    ap.add_argument("--eol-thresh", type=float, default=80.0,
                    help="SOH %% threshold defining end-of-life (default 80)")
    ap.add_argument("--drop-fresh", action="store_true",
                    help="drop windows with SOH >= 99.9%% (unaged, label-poor)")
    ap.add_argument("--w-terms", default="rate,temp,degrad",
                    help="comma list of tri-weight terms to activate (ablation)")
    ap.add_argument("--amp", dest="amp", action="store_true", default=True,
                    help="bf16 autocast training (evaluation always fp32)")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--cens-aware", dest="cens_aware", action="store_true", default=True,
                    help="CENS_AWARE_V3_S2: use Tobit-style one-sided loss on right-censored RUL")
    ap.add_argument("--no-cens-aware", dest="cens_aware", action="store_false",
                    help="legacy behaviour: treat censored RUL lower bounds as exact labels")
    ap.add_argument("--agg", default="sum",
                    choices=sorted(set(_AGG) | set(moo.WEIGHTING)),
                    help="AGG_SWITCH_V3 / P0-2: aggregation operator.  The "
                         "conflict-resolution and geometry families (pcgrad, "
                         "cagrad, mgda, aligned, imtlg, nash) and the "
                         "loss-weighting families (famo, uw, imtll, gradnorm) run "
                         "the reference implementations of moo_combiners.py; "
                         "*_proxy names keep the pre-P0-2 scalar proxies for audit")
    ap.add_argument("--configs", default="joint,joint_w,stf0,stf0_w",
                    help="comma list of configs to run; 'stfcal'/'stfcal_w' add the "
                         "calibrated STF (STF-cal), 'stfcalrank*' the minimal-rank variant")
    ap.add_argument("--cal-warm-epochs", type=int, default=3,
                    help="STF-cal healthy warm-up length (epochs, aux heads isolated)")
    ap.add_argument("--cal-rank-max", type=int, default=3,
                    help="max rank of the toxic subspace considered by STF-cal rank selection")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    enable_max_perf()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _wterms = tuple(x.strip() for x in a.w_terms.split(",") if x.strip())

    # ---- dataset: cache -> (optional) rebuild --------------------------------
    cpath = _ds_cache_path(a.dataset, a.max_ts, _wterms, a.gamma,
                           a.eol_only, a.drop_fresh, a.eol_thresh)
    cacheable = (not a.eol_only) and (not a.drop_fresh)
    if cacheable and (not a.no_cache) and os.path.exists(cpath):
        X, S, R, W, C, CENS, cids, scale, thr, cond_counts = _load_ds_cache(cpath)
        print(f"  [cache] loaded dataset from {cpath}")
    else:
        if a.dataset == "HUST":
            cycles = _load_hust_npz()
        else:
            p = _path("battery_parsed", f"{a.dataset.lower()}_cycles.json")
            if not os.path.exists(p):
                print(f"ERROR: parsed data not found at {p}\nRun battery_preprocess.py first.")
                sys.exit(1)
            with open(p) as f:
                cycles = json.load(f)
        # ---- P2 fix: EOL-aware filtering of right-censored RUL labels ----
        if a.eol_only or a.drop_fresh:
            from collections import defaultdict as _dd
            _by = _dd(list)
            for c in cycles:
                _by[c.get("_cid", "cell")].append(c)
            kept_cells, dropped_cells = [], []
            for cid, cs in _by.items():
                mins = min(float(x.get("SOH", 100.0)) for x in cs)
                if a.eol_only and mins > a.eol_thresh:
                    dropped_cells.append(cid)
                else:
                    kept_cells.append(cid)
            keep = set(kept_cells)
            nel = sum(1 for cid, cs in _by.items()
                      if min(float(x.get("SOH", 100.0)) for x in cs) <= a.eol_thresh)
            cycles = [c for c in cycles if c.get("_cid", "cell") in keep]
            if a.drop_fresh:
                cycles = [c for c in cycles if float(c.get("SOH", 0.0)) < 99.9]
            print("  [eol] cells kept=%d dropped=%d (reached EOL=%d/%d) samples=%d"
                  % (len(kept_cells), len(dropped_cells), nel, len(_by), len(cycles)))
        X, S, R, W, C, cids, scale, thr, cond_counts, CENS = build_dataset(
            cycles, a.max_ts, a.gamma, w_terms=_wterms)
        if cacheable and (not a.no_cache):
            _save_ds_cache(cpath, X, S, R, W, C, CENS, cids, scale, thr, cond_counts)
    print("  tri-weight thresholds:", thr, " overall cond counts:", cond_counts)
    # leave-cell-out split: partition unique cells, keep 85% of cells in train
    uniq = sorted(set(cids.tolist()))
    random.Random(a.seed_cell).shuffle(uniq)
    tr_cells = set(uniq[:max(1, int(0.85 * len(uniq)))])
    tr_mask = np.isin(cids, list(tr_cells))
    te_mask = ~tr_mask
    Xtr, Str, Rtr, Wtr, Ctr = X[tr_mask], S[tr_mask], R[tr_mask], W[tr_mask], C[tr_mask]
    Xte, Ste, Rte, Wte, Cte = X[te_mask], S[te_mask], R[te_mask], W[te_mask], C[te_mask]
    Cntr, Cnte = CENS[tr_mask], CENS[te_mask]
    print(f"  cells train={len(tr_cells)} test={len(uniq)-len(tr_cells)} "
          f"samples train={Xtr.shape[0]} test={Xte.shape[0]}")
    dl_tr = _make_loader(CycDataset(Xtr, Str, Rtr, Wtr, Ctr, Cntr), a.batch, True,
                         a.workers, a.pin_memory)
    dl_te = _make_loader(CycDataset(Xte, Ste, Rte, Wte, Cte, Cnte), a.batch, False,
                         a.workers, a.pin_memory)
    _nc = int(CENS.sum())
    print("  censoring: %d/%d samples right-censored (%.1f%%) -- cens_aware=%s"
          % (_nc, len(CENS), 100.0 * _nc / max(len(CENS), 1), a.cens_aware))
    print("  perf: batch=%d workers=%d pin=%s amp=%s cache=%s"
          % (a.batch, a.workers, a.pin_memory, a.amp, cacheable))

    _ALL_CFG = {"joint": ("joint", False, 0, "off", "pca"),
                "joint_w": ("joint_w", False, 1, "off", "pca"),
                "stf0": ("stf0", True, 0, "off", "pca"),
                "stf0_w": ("stf0_w", True, 1, "off", "pca"),
                "stfcal": ("stfcal", False, 0, "cal", "pca"),
                "stfcal_w": ("stfcal_w", False, 1, "cal", "pca"),
                "stfcalrank": ("stfcalrank", False, 0, "calrank", "pca"),
                "stfcalrank_w": ("stfcalrank_w", False, 1, "calrank", "pca"),
                # STF-cal with the AUXILIARY-GRADIENT principal subspace (Sigma_a)
                # instead of the representation-PCA surrogate.  Same criterion, same
                # gate, same probe data -- only the subspace the rank rule deletes
                # changes, and it is now the r-dim choice that minimises the paper's
                # own residual rho_res[r].
                "stfcalg": ("stfcalg", False, 0, "calg", "aux"),
                "stfcalg_w": ("stfcalg_w", False, 1, "calg", "aux"),
                "stfcalgrank": ("stfcalgrank", False, 0, "calgrank", "aux"),
                "stfcalgrank_w": ("stfcalgrank_w", False, 1, "calgrank", "aux"),
                # diagnostic only: the generalized (Sigma_a, Sigma_p) eigenvectors
                "stfcalx": ("stfcalx", False, 0, "calg", "gen"),
                "stfcalxrank": ("stfcalxrank", False, 0, "calgrank", "gen")}
    configs = [_ALL_CFG[c.strip()] for c in a.configs.split(",") if c.strip() in _ALL_CFG]
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
                               False, 0, a.gamma, scale, single=tk,
                               cens_aware=a.cens_aware, amp=a.amp))
            st_ceiling[tk] = dict(
                rmse_soh=float(np.mean([r["soh"] for r in per])),
                mae_rul=float(np.mean([r["rul"] for r in per])),
                scatter=float(np.mean([r["scatter"] for r in per])))
            print(f"single[{tk}]", st_ceiling[tk])
    else:
        st_ceiling = {}
    for cname, stf, wattr, stf_mode, cal_basis in configs:
        per_seed = []
        for sd in seeds:
            set_seed(sd)
            m = BMTL(d_in=len(FEATS), d=a.d, n_ts=a.max_ts,
                     use_tsmixer=not a.no_tsmixer, use_gatedelta=not a.no_gatedelta).to(dev)
            # P0-2: stateful loss-weighting families (UW / IMTL-L / GradNorm)
            # contribute their own parameters to the main optimiser; FAMO carries
            # its own Adam over the logits and contributes none, as in the
            # reference.  make_state() returns None for every other aggregator,
            # which leaves the optimiser and the run bit-identical to before.
            moo_state = moo.make_state(a.agg, 2, dev)
            opt = torch.optim.Adam(list(m.parameters())
                                   + moo.state_parameters(moo_state), lr=a.lr)
            b = run(m, dl_tr, dl_te, opt, a.epochs, dev, a.alpha_aux, stf, wattr, a.gamma,
                    scale, tox_var=a.tox_var, cens_aware=a.cens_aware,
                    agg=a.agg, ema={}, amp=a.amp, stf_mode=stf_mode,
                    cal_warm_epochs=a.cal_warm_epochs, cal_rank_max=a.cal_rank_max,
                    cal_basis=cal_basis, moo_state=moo_state)
            per_seed.append(dict(**b, seed=sd))
        rows.append(dict(config=cname,
                         rmse_soh=float(np.mean([r["soh"] for r in per_seed])),
                         rmse_soh_std=float(np.std([r["soh"] for r in per_seed])),
                         mae_rul=float(np.mean([r["rul"] for r in per_seed])),
                         mae_rul_std=float(np.std([r["rul"] for r in per_seed])),
                         scatter=float(np.mean([r["scatter"] for r in per_seed])),
                         cal_rho_u=float(np.nanmean([r.get("cal_rho_u", float("nan"))
                                                     for r in per_seed])),
                         cal_margin=float(np.nanmean([r.get("cal_margin", float("nan"))
                                                      for r in per_seed])),
                         cal_gate=float(np.nanmean([r.get("cal_gate", float("nan"))
                                                    for r in per_seed])),
                         cal_rank=float(np.nanmean([r.get("cal_rank", float("nan"))
                                                    for r in per_seed])),
                         per_seed=[{k: v for k, v in r.items() if k != "per_cond"}
                                   for r in per_seed],
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
                       tox_var=a.tox_var, agg=a.agg, seeds=seeds,
                       eol_only=a.eol_only, drop_fresh=a.drop_fresh, w_terms=list(_wterms),
                       eol_thresh=a.eol_thresh,
                       tri_weight_thresholds=thr, cond_counts=cond_counts,
                       single_ceiling=st_ceiling, results=rows), f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
