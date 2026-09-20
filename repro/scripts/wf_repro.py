"""RepFix walk-forward reproduction (t4-B) — full-market multitask collapse.

Rebuilds the missing train_v11_wf / train_v7_lite_wf / distill_loss pipeline:

  data      : stock_data_cache.pkl (4,993 stocks, full market)
  features  : 28 (20 base + 8 CS-rank) + 16 aux + 3 market_state
  labels    : forward-5d return, per-day z-score
  model     : self-contained StudentModel (shared encoder + point/MDN/AE/SCARF heads)
  training  : point MSE + MDN NLL + AE recon + SCARF contrast; detach_aux_heads on/off
  eval      : rank_ic (per-day Spearman mean) + sigma_h (collapse indicator)

The aux / market_state / rank_ic pieces that used to live in train_v11_wf /
train_v7_lite_wf / distill_loss (source lost with the old cluster) are re-derived
here from first principles and the surviving production references
(train_v6.py::build_aux_features, _diffusion_core.py::build_market_state).

Usage:
  python wf_repro.py --stage data   --folds 0            # build fold0 npz
  python wf_repro.py --stage train  --folds 0 --detach 0 # train w/o detach
  python wf_repro.py --stage train  --folds 0 --detach 1 # train w/ detach (STF-0)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# ---------------------------------------------------------------- paths
DATA_DIR = os.environ.get("REPFIX_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
PKL_PATH = os.path.join(DATA_DIR, "stock_data_cache.pkl")
FOLD_PATTERN = os.path.join(DATA_DIR, "wf_fold{n}.npz")

SEQ_LEN = 60
FEAT_PER_DAY = 28
INPUT_DIM = SEQ_LEN * FEAT_PER_DAY  # 1680
AUX_DIM = 16
MARKET_DIM = 3
LABEL_HORIZON = 5
N_BASE = 20
N_CS = 8

TRAIN_CAP = 100_000
VAL_CAP = 40_000
TEST_CAP = 40_000


# ---------------------------------------------------------------- rolling helpers
def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    cs = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    n = len(x)
    k = np.arange(n)
    lo = np.maximum(k - w + 1, 0)
    return (cs[k + 1] - cs[lo]) / (k - lo + 1)


def _roll_std(x: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        lo = max(i - w + 1, 0)
        out[i] = np.std(x[lo:i + 1]) if i > lo else 0.0
    return out + 1e-8


def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = _roll_mean(gain, n)
    avg_l = _roll_mean(loss, n)
    rs = avg_g / (avg_l + 1e-8)
    return 100.0 - 100.0 / (1.0 + rs)


def _cs_rank_norm(values: np.ndarray) -> np.ndarray:
    from scipy.stats import norm
    r = np.argsort(np.argsort(values, kind="mergesort")).astype(np.float64) + 1.0
    pct = np.clip(r / (len(values) + 1.0), 1e-8, 1 - 1e-8)
    return norm.ppf(pct).astype(np.float32)


# ---------------------------------------------------------------- features (from train_v6.py)
def _base_20_features(stock: dict) -> np.ndarray:
    """(T, 20) base temporal features — recipe verbatim from train_v6.py::build_features."""
    eps = 1e-8
    close = stock["close"].astype(np.float64)
    open_ = stock["open"].astype(np.float64)
    high = stock["high"].astype(np.float64)
    low = stock["low"].astype(np.float64)
    vol = stock["vol"].astype(np.float64)
    amount = stock["amount"].astype(np.float64)
    pct_chg = stock["pct_chg"].astype(np.float64)

    ret1 = np.diff(close, prepend=close[0]) / (np.abs(close) + eps)
    ma5 = _roll_mean(close, 5)
    ma20 = _roll_mean(close, 20)
    vma5 = _roll_mean(vol, 5)
    ama5 = _roll_mean(amount, 5)

    def _past_ret(p: int) -> np.ndarray:
        out = np.zeros_like(close)
        out[p:] = (close[p:] - close[:-p]) / (np.abs(close[:-p]) + eps)
        return out

    ret5 = _past_ret(5)
    ret10 = _past_ret(10)
    mom20 = _past_ret(20)

    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = _roll_mean(gain, 14)
    avg_l = _roll_mean(loss, 14)
    rs = avg_g / (avg_l + eps)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    vol20 = _roll_std(ret1, 20)
    cz20 = (close - _roll_mean(close, 20)) / _roll_std(close, 20)

    body = (close - open_) / (open_ + eps)
    hl_range = (high - low) / (close + eps)
    ma5_diff = close / (ma5 + eps) - 1.0
    ma20_diff = close / (ma20 + eps) - 1.0
    vol_ratio = vol / (vma5 + eps)
    amount_ratio = amount / (ama5 + eps)
    pct_norm = pct_chg / 100.0

    feat20 = np.column_stack([
        close / (ma20 + eps), open_ / (close + eps), high / (close + eps),
        low / (close + eps), np.log1p(np.clip(vol, 0, None)) / 20.0,
        np.log1p(np.clip(amount, 0, None)) / 25.0, pct_norm,
        ret1, ret5, ret10, mom20, ma5_diff, ma20_diff, vol20,
        vol_ratio, amount_ratio, hl_range, body, rsi / 100.0, cz20,
    ])
    return np.nan_to_num(feat20, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _stock_raw_8(stock: dict) -> np.ndarray:
    """8 raw factors per stock-day (cs_rank_cache.py recipe)."""
    eps = 1e-8
    close = np.maximum(stock["close"].astype(np.float64), eps)
    vol = np.maximum(stock["vol"].astype(np.float64), 0.0)
    amount = np.maximum(stock["amount"].astype(np.float64), 0.0)
    turnover = np.maximum(stock["turnover"].astype(np.float64), 0.0)
    ma20 = _roll_mean(close, 20)
    close_norm = close / (ma20 + eps)
    log_vol = np.log1p(np.clip(vol, 0, None)) / 20.0
    log_amount = np.log1p(np.clip(amount, 0, None)) / 25.0
    mom20 = np.zeros_like(close)
    mom20[20:] = (close[20:] - close[:-20]) / (np.abs(close[:-20]) + eps)
    ret5 = np.zeros_like(close)
    ret5[5:] = (close[5:] - close[:-5]) / (np.abs(close[:-5]) + eps)
    ret1 = np.diff(close, prepend=close[0]) / (np.abs(close) + eps)
    vol20 = _roll_std(ret1, 20)
    rsi = _rsi(close, 14) / 100.0
    vma5 = _roll_mean(vol, 5)
    turnover_ratio = turnover / (vma5 + eps)
    return np.column_stack(
        [close_norm, log_vol, log_amount, mom20, ret5, vol20, rsi, turnover_ratio]
    ).astype(np.float32)


def build_cs_rank_cache(stock_data: Dict) -> Dict[str, np.ndarray]:
    """{ts_code: (T, 8)} CSRankNorm, cross-sectionally ranked per date, date-aligned."""
    all_dates = sorted({d for s in stock_data.values() for d in s["dates"]})
    date_pos = {d: i for i, d in enumerate(all_dates)}
    n_dates = len(all_dates)
    codes = list(stock_data.keys())
    raw = {c: _stock_raw_8(stock_data[c]) for c in codes}

    panel = np.full((n_dates, len(codes), 8), np.nan, dtype=np.float32)
    for ci, c in enumerate(codes):
        s = stock_data[c]
        idx = np.array([date_pos[d] for d in s["dates"]], dtype=np.int64)
        panel[idx, ci] = raw[c]

    ranked = np.full_like(panel, np.nan)
    for t in range(n_dates):
        slab = panel[t]
        valid = np.isfinite(slab).all(1)
        if valid.sum() < 5:
            continue
        for f in range(8):
            ranked[t, valid, f] = _cs_rank_norm(slab[valid, f])

    out: Dict[str, np.ndarray] = {}
    for ci, c in enumerate(codes):
        s = stock_data[c]
        idx = np.array([date_pos[d] for d in s["dates"]], dtype=np.int64)
        out[c] = np.nan_to_num(ranked[idx, ci], nan=0.0).astype(np.float32)
    return out


def build_aux_features(stock: dict) -> np.ndarray:
    """(T, 16) aux features — recipe verbatim from train_v6.py::build_aux_features.

    Derived from the trailing OHLCV window ending at each day t (proxy for the
    'market state / auxiliary target' the aux heads are trained to predict).
    """
    eps = 1e-8
    close = stock["close"].astype(np.float64)
    vol = stock["vol"].astype(np.float64)
    n = len(close)
    ret1 = np.diff(close, prepend=close[0]) / (np.abs(close) + eps)

    vol20_mean = _roll_mean(vol, min(20, n))
    vol20_std = _roll_std(vol, min(20, n))
    ret5_mean = _roll_mean(ret1, min(5, n))
    ret10_mean = _roll_mean(ret1, min(10, n))

    panic_proxy = np.maximum(-ret1, 0) * (vol / (vol20_mean + eps))
    greed_proxy = np.maximum(ret1, 0) * (vol20_mean / (vol + eps))
    news_vol_spike = (vol - vol20_mean) / vol20_std
    sentiment = (greed_proxy - panic_proxy) / 2 + news_vol_spike * 0.3

    aux = np.column_stack([
        sentiment,
        greed_proxy / (panic_proxy + 1.0),
        news_vol_spike,
        np.maximum(ret1, 0),
        np.maximum(-ret1, 0),
        ret5_mean,
        ret10_mean,
        np.log1p(np.clip(vol, 0, None)) / 20.0,
        np.abs(ret1),
        ret1,
        ret5_mean - panic_proxy,
        sentiment / 2.0,
        np.sign(ret1) * np.minimum(np.abs(ret1), 0.1),
        ret10_mean,
        ret10_mean - ret5_mean,
        ret1 * vol / (vol20_mean + eps),
    ])
    return np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ---------------------------------------------------------------- data assembly
def _forward_return(close: np.ndarray, horizon: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n > horizon:
        out[: n - horizon] = (close[horizon:] - close[:-horizon]) / (np.abs(close[:-horizon]) + 1e-8)
    return out


def _zscore_by_date(y: np.ndarray, d: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=np.float32)
    for t in np.unique(d):
        m = d == t
        seg = y[m]
        mu, sd = seg.mean(), seg.std()
        out[m] = (seg - mu) / (sd + 1e-8) if sd > 1e-8 else np.zeros_like(seg)
    return out


def _market_state_per_day(y_raw: np.ndarray, d: np.ndarray) -> Dict[int, np.ndarray]:
    """3-dim market state per date: mean ret, cross-section std, up-ratio."""
    out: Dict[int, np.ndarray] = {}
    for t in np.unique(d):
        m = d == t
        seg = y_raw[m]
        if len(seg) < 5:
            out[int(t)] = np.zeros(3, dtype=np.float32)
            continue
        out[int(t)] = np.array([
            float(np.tanh(np.mean(seg) * 50)),
            float(np.clip(np.std(seg) * 10, 0, 3)),
            float(np.mean(seg > 0)),
        ], dtype=np.float32)
    return out


def generate_walk_forward_splits(all_dates: List[str], train_days: int = 252,
                                 step_days: int = 21,
                                 max_folds: Optional[int] = None) -> List[dict]:
    n = len(all_dates)
    splits = []
    start = 0
    while start + train_days < n:
        tr_s, tr_e = start, start + train_days
        va_s, va_e = tr_e, min(tr_e + step_days, n)
        te_s, te_e = va_e, min(va_e + step_days, n)
        if te_e >= n:
            te_e = min(n - 1, te_e)
        splits.append(dict(
            train_start=all_dates[tr_s], train_end=all_dates[tr_e - 1],
            val_start=all_dates[va_s], val_end=all_dates[va_e - 1],
            test_start=all_dates[te_s], test_end=all_dates[te_e - 1],
        ))
        start += step_days
        if max_folds is not None and len(splits) >= max_folds:
            break
    return splits


def _precompute_all(stock_data, cs_cache, horizon: int = LABEL_HORIZON):
    """Per-stock feature caches, computed once for all folds."""
    feat_cache, aux_cache, fwd_cache, date_cache, cs_cache2 = {}, {}, {}, {}, {}
    for code, s in stock_data.items():
        T = len(s["close"])
        if T <= SEQ_LEN + horizon:
            continue
        feat_cache[code] = _base_20_features(s)
        aux_cache[code] = build_aux_features(s)
        fwd_cache[code] = _forward_return(s["close"].astype(np.float64), horizon)
        date_cache[code] = s["dates"]
        cs = cs_cache.get(code)
        if cs is None:
            cs_r = np.zeros((T, N_CS), dtype=np.float32)
        elif cs.shape[0] >= T:
            cs_r = cs[:T]
        else:
            cs_r = np.zeros((T, N_CS), dtype=np.float32)
            cs_r[-cs.shape[0]:] = cs
        cs_cache2[code] = cs_r
    return feat_cache, aux_cache, fwd_cache, date_cache, cs_cache2


def build_fold_npz(stock_data, cs_cache, fold: int, max_folds: int = 10,
                   horizon: int = LABEL_HORIZON, precomp=None):
    """Build one fold's (X, y, aux, market_state, dates) and save npz.

    X is (n, 1680) flattened (60, 28); y is per-day z-scored forward return;
    aux is (n, 16); ms is (n, 3) market state broadcast per date.
    """
    all_dates = sorted({d for s in stock_data.values() for d in s["dates"]})
    date_idx = {d: i for i, d in enumerate(all_dates)}
    splits = generate_walk_forward_splits(all_dates, max_folds=max_folds)
    info = splits[fold]

    if precomp is not None:
        feat_cache, aux_cache, fwd_cache, date_cache, cs_cache2 = precomp
    else:
        feat_cache, aux_cache, fwd_cache, date_cache, cs_cache2 = \
            _precompute_all(stock_data, cs_cache, horizon)

    # assemble samples within this fold's date range
    lo = date_idx[info["train_start"]]
    hi = date_idx[info["test_end"]]
    xs, ys, auxs, dts = [], [], [], []
    for code, s in stock_data.items():
        T = len(s["close"])
        if code not in feat_cache:
            continue
        feat = feat_cache[code]
        aux = aux_cache[code]
        fwd = fwd_cache[code]
        dates = date_cache[code]
        cs_r = cs_cache2[code]
        feat28 = np.concatenate([feat, cs_r], axis=1)
        for t in range(SEQ_LEN - 1, T - horizon):
            di = date_idx[dates[t]]
            if di < lo or di > hi:
                continue
            y = fwd[t]
            if not np.isfinite(y):
                continue
            xs.append(feat28[t - SEQ_LEN + 1: t + 1].reshape(-1))
            ys.append(np.float32(y))
            auxs.append(aux[t])
            dts.append(di)

    X = np.stack(xs).astype(np.float32)
    y_raw = np.asarray(ys, dtype=np.float32)
    d = np.asarray(dts, dtype=np.int64)
    A = np.stack(auxs).astype(np.float32)
    y = _zscore_by_date(y_raw, d)
    ms_map = _market_state_per_day(y_raw, d)
    MS = np.stack([ms_map[int(t)] for t in d]).astype(np.float32)

    # date-range masks
    seg = {
        "train": (d >= date_idx[info["train_start"]]) & (d <= date_idx[info["train_end"]]),
        "val": (d >= date_idx[info["val_start"]]) & (d <= date_idx[info["val_end"]]),
        "test": (d >= date_idx[info["test_start"]]) & (d <= date_idx[info["test_end"]]),
    }
    # subsample to caps (deterministic)
    rng = np.random.default_rng(42 + fold)
    out = {}
    for k in ("train", "val", "test"):
        m = seg[k]
        idx = np.where(m)[0]
        cap = {"train": TRAIN_CAP, "val": VAL_CAP, "test": TEST_CAP}[k]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        out[f"{k}_X"] = X[idx]
        out[f"{k}_y"] = y[idx]
        out[f"{k}_aux"] = A[idx]
        out[f"{k}_ms"] = MS[idx]
        out[f"{k}_d"] = d[idx]
    out["info"] = info
    out["n_stocks"] = len(feat_cache)

    path = FOLD_PATTERN.format(n=fold)
    np.savez(path, **out)
    print(f"[fold {fold}] saved {path}: train={len(out['train_X'])} "
          f"val={len(out['val_X'])} test={len(out['test_X'])} "
          f"n_stocks={out['n_stocks']} info={info}", flush=True)
    return path


# ---------------------------------------------------------------- model
class StudentModel(nn.Module):
    """Self-contained multitask student: shared encoder + point/MDN/AE/SCARF heads.

    Mirrors V11_6Student's collapse-relevant structure: a shared encoder whose
    representation h drives both the primary point head and the auxiliary heads
    (MDN distribution head, AE reconstruction, SCARF contrastive).  With
    detach_aux_heads=True the auxiliary gradients are cut from the shared encoder
    (STF-0 / Detach-Aux-Heads), leaving the encoder driven only by the primary task.
    """

    def __init__(self, input_dim: int = INPUT_DIM, d_model: int = 64,
                 aux_dim: int = AUX_DIM, market_dim: int = MARKET_DIM,
                 ae_latent: int = 16, detach_aux_heads: bool = False,
                 dropout: float = 0.1, fixed_aux_weight: float = 0.0,
                 fixed_aux_k: int = 2, stf_mode: Optional[str] = None,
                 stf_beta: float = 8.0, stf_rho_c: float = 0.02):
        super().__init__()
        self.d_model = d_model
        self.ae_latent = ae_latent
        self.detach_aux_heads = detach_aux_heads
        self.stf_mode = stf_mode          # None/'off' | 'hard' | 'soft'
        self.stf_beta = stf_beta          # smoothness of the soft gate
        self.stf_rho_c = stf_rho_c        # critical health ratio (collapse threshold)
        self.last_rho: float = float("nan")
        self.last_w: float = float("nan")
        self.market_dim = market_dim
        self.fixed_aux_weight = fixed_aux_weight
        if fixed_aux_weight > 0:
            # non-learnable random projection (toy-testbed mechanism): the only
            # way to shrink Var_b(W h) is to collapse h toward a constant.
            self.register_buffer("W_fixed",
                                 torch.randn(fixed_aux_k, d_model) / (d_model ** 0.5))

        # shared encoder (the representation whose variance can collapse)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.LayerNorm(d_model),
        )
        # primary head: point prediction (z-scored target)
        self.point_head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(),
                                        nn.Linear(32, 1))
        # aux head 1: MDN (Gaussian NLL on the same target) — variance-contracting
        self.mdn_head = nn.Linear(d_model, 2)
        # aux head 2: AE reconstruction of h
        self.ae_enc = nn.Linear(d_model, ae_latent)
        self.ae_dec = nn.Linear(ae_latent, d_model)
        # aux head 3: SCARF contrastive projection
        self.contrast_head = nn.Linear(ae_latent, ae_latent)
        self.contrast_tau = 0.1
        # market-state conditioned gate (small, keeps ms in the loop)
        self.ms_gate = nn.Sequential(nn.Linear(market_dim, 8), nn.ReLU(),
                                     nn.Linear(8, d_model))

    def _corrupt(self, h: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(h) < 0.3
        noise = torch.randn_like(h) * 0.1
        return torch.where(mask, noise, h)

    def _toxic_projection(self, h: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Estimate the collapse ('toxic') subspace direction of a batch.

        Collapse = the representation converges to a low-dimensional constant, so
        the batch scatter concentrates along few axes. The most toxic direction is
        the *smallest-variance* eigenvector v of the batch covariance: moving h
        along v is exactly what the variance-shrinking aux head wants to do, and it
        is the direction along which collapse actually happens. We return the
        per-sample projection onto v and a scalar health ratio rho = lam_min/tr(C)
        (high = spread out across dims; ~1/d) which decays to ~0 as collapse sets in.
        """
        B = h.size(0)
        d = h.size(1)
        hc = h - h.mean(0, keepdim=True)
        cov = (hc.t() @ hc) / max(B, 1) + 1e-6 * torch.eye(d, device=h.device)
        evals, evecs = torch.linalg.eigh(cov)
        lam_min = evals[0].clamp_min(0.0)
        lam_tot = evals.clamp_min(0.0).sum().clamp_min(1e-12)
        v = evecs[:, 0].detach()                 # collapse (smallest-variance) direction
        proj = (hc @ v).unsqueeze(1) * v         # (B, d) component along v
        rho = (lam_min / lam_tot).detach().clamp_max(1.0)
        return proj, rho

    def _stf_filter(self, h: torch.Tensor) -> torch.Tensor:
        """Spectral Toxicity Filtering: drop the collapse-subspace component before
        the aux heads, scaling removal by the health ratio rho.
          hard : fully remove the projection once rho < rho_c (threshold trigger)
          soft : smooth gate  w = sigmoid(beta*(rho_c - rho)) in [0,1]
        Removing the component also zeros that direction of the auxiliary gradient
        w.r.t. the encoder, i.e. the encoder is shielded from the toxic push without
        a full .detach() (which is the more aggressive STF-0).
        """
        proj, rho = self._toxic_projection(h)
        self.last_rho = rho
        self.last_w = torch.zeros((), device=h.device)
        if self.stf_mode == "hard":
            w = (rho < self.stf_rho_c).float()   # on-device gate, no D2H sync
        else:  # soft
            w = torch.sigmoid(self.stf_beta * (self.stf_rho_c - rho))
        self.last_w = w
        return h - w * proj

    def forward(self, x, market_state=None):
        h = self.encoder(x)
        if market_state is not None:
            h = h + self.ms_gate(market_state)
        point = self.point_head(h).squeeze(-1)

        if self.stf_mode in ("hard", "soft"):
            _h_aux = self._stf_filter(h)
        elif self.detach_aux_heads:
            _h_aux = h.detach()
        else:
            _h_aux = h
        mdn = self.mdn_head(_h_aux)
        mu, log_sigma = mdn.chunk(2, dim=-1)
        lat = self.ae_enc(_h_aux)
        recon = self.ae_dec(lat)
        z1 = self.contrast_head(self.ae_enc(self._corrupt(_h_aux)))
        z2 = self.contrast_head(self.ae_enc(self._corrupt(_h_aux)))

        out = {
            "point": point,
            "mdn_mu": mu.squeeze(-1), "mdn_log_sigma": log_sigma.squeeze(-1),
            "recon": recon, "lat": lat, "h_aux": _h_aux, "z1": z1, "z2": z2,
            "sigma_h": h.std().detach().float(),
        }
        if self.fixed_aux_weight > 0:
            # the variance-shrinking aux head reads the FILTERED representation, so
            # STF (hard/soft) and full-detach (STF-0) can actually shield the encoder
            # from it -- making this a genuine test of spectral toxicity filtering.
            Wh = _h_aux @ self.W_fixed.t()
            out["fixed_aux"] = Wh.var(dim=0).sum()
        return point, out


def _nt_xent(z1, z2, tau: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = (z1 @ z2.T) / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


def rank_margin_loss(s, y, beta: float = 1e-3, kind: str = "quad"):
    """Pairwise rank loss (from the toy testbed). Zero at BOTH a perfect ranking
    AND a constant output -- the prediction-level degenerate trap. beta adds a
    small scale-pinning term so a perfect ranking is preferred over a constant.
    kind=quad  -> quadratic margin  [y_hat_j - y_hat_i]_+^2
    kind=sqrt  -> sqrt-margin        sqrt([y_hat_j - y_hat_i]_+)
    """
    s = s.float().squeeze(-1)
    D = s.unsqueeze(1) - s.unsqueeze(0)
    P = (y.unsqueeze(1) > y.unsqueeze(0)).float()
    margin = torch.relu(-D)
    npos = P.detach().sum().clamp_min(1.0)
    if kind == "sqrt":
        l_rank = (P * torch.sqrt(margin + 1e-8)).sum() / npos
    else:
        l_rank = (P * margin * margin).sum() / npos
    base = 1.0 if kind == "sqrt" else (margin * margin).clamp_max(1.0).mean()
    return l_rank + beta * base


# ---------------------------------------------------------------- training
def compute_rank_ic(pred, y, d):
    """Per-day Spearman mean (pooled rank IC across dates)."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    d = np.asarray(d, dtype=np.int64).ravel()
    ics = []
    for t in np.unique(d):
        m = d == t
        if m.sum() < 10:
            continue
        p, yy = pred[m], y[m]
        if np.std(p) < 1e-12:
            continue
        rp = np.argsort(np.argsort(p)).astype(np.float64)
        ry = np.argsort(np.argsort(yy)).astype(np.float64)
        rp -= rp.mean(); ry -= ry.mean()
        denom = np.sqrt((rp ** 2).sum() * (ry ** 2).sum())
        if denom < 1e-12:
            continue
        ics.append((rp * ry).sum() / denom)
    return float(np.mean(ics)) if ics else 0.0


def _predict(model, X, ms, device, batch=2048):
    model.eval()
    if not torch.is_tensor(X):
        X = torch.as_tensor(X, dtype=torch.float32, device=device)
        ms = torch.as_tensor(ms, dtype=torch.float32, device=device)
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            p, _ = model(X[i:i + batch], ms[i:i + batch])
            outs.append(p.detach().float().cpu().numpy())
    return np.concatenate(outs)


def train_fold(fold: int, detach: bool, epochs: int, batch: int, device: str,
               lr: float, patience: int, seed: int, aux_scale: float = 1.0,
               fixed_aux_weight: float = 0.0, fixed_aux_k: int = 2,
               rank_loss: bool = False, rank_beta: float = 1e-3,
               stf: Optional[str] = None, stf_beta: float = 8.0,
               stf_rho_c: float = 0.02):
    torch.manual_seed(seed)
    np.random.seed(seed)
    path = FOLD_PATTERN.format(n=fold)
    print(f"[fold {fold}] detach={detach} stf={stf} load {path}", flush=True)
    d = np.load(path, mmap_mode="r")
    # GPU-resident preload: a whole fold fits trivially in A100 HBM; this removes
    # the per-batch mmap gather + H2D copy (the finance hot path).
    def _g(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    Xtr, ytr, MStr = _g(d["train_X"]), _g(d["train_y"]), _g(d["train_ms"])
    Xva, yva, MSva = _g(d["val_X"]), _g(d["val_y"]), _g(d["val_ms"])
    Xte, yte, MSte = _g(d["test_X"]), _g(d["test_y"]), _g(d["test_ms"])
    yva_np, yte_np = np.asarray(d["val_y"]), np.asarray(d["test_y"])
    dtr, dva, dte = d["train_d"], d["val_d"], d["test_d"]

    model = StudentModel(detach_aux_heads=detach,
                         fixed_aux_weight=fixed_aux_weight,
                         fixed_aux_k=fixed_aux_k,
                         stf_mode=stf, stf_beta=stf_beta, stf_rho_c=stf_rho_c).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xtr)

    best_ic, best_sd, bad = -999.0, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_sum = torch.zeros((), device=device, dtype=torch.float64)
        cnt = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb, msb = Xtr[idx], ytr[idx], MStr[idx]
            opt.zero_grad()
            p, o = model(xb, msb)
            if rank_loss:
                loss_p = rank_margin_loss(p, yb, beta=rank_beta)
            else:
                loss_p = F.mse_loss(p, yb)
            loss_mdn = (o["mdn_log_sigma"]
                        + (yb - o["mdn_mu"]) ** 2 / (2 * torch.exp(2 * o["mdn_log_sigma"]) + 1e-8)).mean()
            loss_ae = F.mse_loss(o["recon"], o["h_aux"])
            loss_cont = _nt_xent(o["z1"], o["z2"], model.contrast_tau)
            loss = loss_p + aux_scale * (0.3 * loss_mdn + 0.2 * loss_ae + 0.1 * loss_cont)
            if fixed_aux_weight > 0:
                loss = loss + fixed_aux_weight * o["fixed_aux"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_sum += loss.detach().double() * len(idx)
            cnt += len(idx)
        val_ic = compute_rank_ic(_predict(model, Xva, MSva, device), yva_np, dva)
        if val_ic > best_ic + 1e-5:
            best_ic, bad = val_ic, 0
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"    ep {ep + 1:2d}/{epochs} loss={float(ep_sum) / max(cnt, 1):.5f} "
              f"val_ic={val_ic:+.5f} sigma_h={float(o['sigma_h']):.4f}", flush=True)
        if bad >= patience:
            print(f"    [early-stop] {patience} no-improve", flush=True)
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    te_pred = _predict(model, Xte, MSte, device)
    oos_ic = compute_rank_ic(te_pred, yte_np, dte)
    # collapse indicators on test. NOTE: the encoder's final LayerNorm forces
    # per-sample feature std to 1, so the raw h.std() stays ~1 even when samples
    # converge to one vector. The correct order parameter is the CROSS-SAMPLE
    # scatter (std of the batch mean of h), not the intra-sample spread.
    model.eval()
    with torch.no_grad():
        xb, msb = Xte[:8192], MSte[:8192]
        hb = model.encoder(xb)                      # (B, d)
        hb = hb - hb.mean(0, keepdim=True)          # center over samples
        cross_std = float(hb.std(dim=0).mean().cpu().numpy())  # mean over feats
        sigma_h = float(hb.norm().cpu().numpy())    # scatter magnitude (order param)
        if model.fixed_aux_weight > 0:
            Wh = xb.new_zeros(0)
            Wb = model.W_fixed
            hfull = model.encoder(xb)
            fixed_var = float((hfull @ Wb.t()).var(dim=0).mean().cpu().numpy())
        else:
            fixed_var = float("nan")
    # aux-task quality + STF filter stats on val (best ckpt): the tradeoff that
    # separates a surgical filter (soft/hard keep aux fit mostly intact) from the
    # blunt full-detach (STF-0 sacrifices aux to save the encoder).
    model.eval()
    with torch.no_grad():
        xb, yb, msb = Xva[:4096], yva[:4096], MSva[:4096]
        _, o = model(xb, msb)
        loss_mdn = float((o["mdn_log_sigma"]
                          + (yb - o["mdn_mu"]) ** 2 / (2 * torch.exp(2 * o["mdn_log_sigma"]) + 1e-8)).mean().item())
        loss_ae = float(F.mse_loss(o["recon"], o["h_aux"]).item())
        aux_val = loss_mdn + 0.2 * loss_ae   # representative aux block (excl. O(B^2) contrast)
    stf_rho = float(getattr(model, "last_rho", float("nan")))
    stf_w = float(getattr(model, "last_w", float("nan")))
    print(f"[fold {fold}] detach={detach} stf={stf} aux_scale={aux_scale} fixed_aux={fixed_aux_weight} "
          f"val_ic={best_ic:+.5f} oos_ic={oos_ic:+.5f} sigma_h={sigma_h:.4f} "
          f"collapse={cross_std:.4f} fixed_var={fixed_var:.4f} "
          f"aux_val={aux_val:.4f} stf_rho={stf_rho:.4f} stf_w={stf_w:.4f}", flush=True)
    return {"fold": fold, "detach": bool(detach), "aux_scale": aux_scale,
            "stf": stf, "fixed_aux": fixed_aux_weight, "val_ic": best_ic,
            "oos_ic": oos_ic, "sigma_h": sigma_h, "collapse": cross_std,
            "fixed_var": fixed_var, "aux_val": aux_val, "stf_rho": stf_rho,
            "stf_w": stf_w}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "train"], default="train")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--detach", type=int, default=-1, help="-1=both, 0=off, 1=on")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux-scale", type=float, default=1.0,
                    help="multiplier on the auxiliary loss block (toxicity sweep)")
    ap.add_argument("--fixed-aux", type=float, default=0.0,
                    help="weight of the fixed random-projection variance head (toy mechanism)")
    ap.add_argument("--fixed-k", type=int, default=2,
                    help="number of fixed random-projection rows")
    ap.add_argument("--rank-loss", action="store_true",
                    help="use pairwise quad-margin rank loss (prediction-level trap)")
    ap.add_argument("--stf", default=None, choices=[None, "off", "hard", "soft"],
                    help="spectral toxicity filtering mode (overrides detach)")
    ap.add_argument("--stf-beta", type=float, default=8.0)
    ap.add_argument("--stf-rho-c", type=float, default=0.02)
    ap.add_argument("--max-folds", type=int, default=10)
    ap.add_argument("--n-stocks", type=int, default=0, help="0=all")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="wf_repro.json")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    if args.stage == "data":
        t0 = time.time()
        print(f"[data] load {PKL_PATH}", flush=True)
        with open(PKL_PATH, "rb") as f:
            sd = pickle.load(f)
        if args.n_stocks:
            sd = dict(list(sd.items())[:args.n_stocks])
        print(f"[data] {len(sd)} stocks, build cs-rank cache...", flush=True)
        cs = build_cs_rank_cache(sd)
        print(f"[data] cs-rank done in {time.time() - t0:.0f}s, precompute features...", flush=True)
        precomp = _precompute_all(sd, cs, LABEL_HORIZON)
        print(f"[data] precompute done in {time.time() - t0:.0f}s", flush=True)
        for f in folds:
            build_fold_npz(sd, cs, f, max_folds=args.max_folds, precomp=precomp)
        print(f"[data] done in {time.time() - t0:.0f}s", flush=True)
        return

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[train] device={device}", flush=True)
    detaches = [args.detach] if args.detach >= 0 else [0, 1]
    results = []
    for f in folds:
        for det in detaches:
            r = train_fold(f, bool(det), args.epochs, args.batch, device,
                           args.lr, args.patience, args.seed, args.aux_scale,
                           args.fixed_aux, args.fixed_k, args.rank_loss,
                           stf=args.stf, stf_beta=args.stf_beta,
                           stf_rho_c=args.stf_rho_c)
            results.append(r)
    with open(args.out, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
