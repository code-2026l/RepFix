"""RepFix backend data loader (t3).

Self-contained, DCU-portable (numpy-only, no numba/torch hard dependency).

Verified data contract (probed against the 296MB cache + baseline_fold*.npz):

  stock_data_cache.pkl
      dict[ts_code] -> {
        "dates":    list[str]           # trade dates, ascending
        "date_idx": dict[str, int]      # date -> row index
        "open","high","low","close","vol","amount","pct_chg","turnover":
                    np.ndarray float32 (T,)
      }

  baseline_fold{N}.npz   (N = 0..9)
      train_X (n,1680) f32   train_y (n,) f32
      val_X   (v,1680) f32   val_y   (v,) f32
      test_X  (t,1680) f32   test_y  (t,) f32

  model input
      x_seq (B, 60, 28) with 28 = 20 base features + 8 CS-rank features
      (flattened (B, 1680) on disk; seq_len=60, feat_per_day=28)

Provided API:
    load_all_stocks_data()                    -> cached pkl dict
    build_cs_rank_cache(stock_data)           -> {ts_code: (T, 8) float32}
    generate_walk_forward_splits(all_dates,*) -> list[fold dict]
    load_fold_from_npz(fold)                  -> (trX,trY,vaX,vaY,teX,teY)
    build_fold_data(...)                      -> same tuple, regenerated from pkl
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------- paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Raw/fold data caches are not distributed due to third-party licensing;
# set REPFIX_DATA_DIR to point at your prepared data (defaults to repo data/).
DATA_DIR = os.environ.get("REPFIX_DATA_DIR") or os.path.join(THIS_DIR, "..", "data")

PKL_PATH = os.path.join(DATA_DIR, "stock_data_cache.pkl")
CS_CACHE_PATH = os.path.join(DATA_DIR, "cs_rank_cache.npz")
FOLD_PATTERN = os.path.join(DATA_DIR, "baseline_fold{n}.npz")

SEQ_LEN = 60
FEAT_PER_DAY = 28          # 20 base + 8 CS-rank
N_BASE = 20
N_CS = 8
INPUT_DIM = SEQ_LEN * FEAT_PER_DAY  # 1680


# ---------------------------------------------------------------- pkl
def load_all_stocks_data(path: str = PKL_PATH) -> Dict:
    """Return the raw per-stock time series dict (see header schema)."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- CS-rank
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
    """Cross-sectional rank -> pct -> normal qf -> N(0,1)."""
    from scipy.stats import norm
    r = np.argsort(np.argsort(values, kind="mergesort")).astype(np.float64) + 1.0
    pct = np.clip(r / (len(values) + 1.0), 1e-8, 1 - 1e-8)
    return norm.ppf(pct).astype(np.float32)


def _stock_raw_8(stock: dict) -> np.ndarray:
    """8 raw cross-sectional factors per stock-day (lagged rank / momentum terms)."""
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


def build_cs_rank_cache(stock_data: Dict, save: bool = True,
                        path: str = CS_CACHE_PATH) -> Dict[str, np.ndarray]:
    """{ts_code: (T, 8)} CSRankNorm features, cross-sectionally ranked per date."""
    # stocks aligned to the global date axis
    all_dates = sorted({d for s in stock_data.values() for d in s["dates"]})
    date_pos = {d: i for i, d in enumerate(all_dates)}
    n_dates = len(all_dates)

    codes = list(stock_data.keys())
    raw = {c: _stock_raw_8(stock_data[c]) for c in codes}

    # pooled ranking per date across stocks (vectorised)
    # panel[t, code_idx, :] -> for each factor column, rank across stocks present that day
    panel = np.full((n_dates, len(codes), 8), np.nan, dtype=np.float32)
    for ci, c in enumerate(codes):
        s = stock_data[c]
        idx = np.array([date_pos[d] for d in s["dates"]], dtype=np.int64)
        panel[idx, ci] = raw[c]

    # cross-sectional rank per date, then write back DATE-ALIGNED per stock so each
    # {ts_code: (T, 8)} lines up with stock_data[c]["dates"] (matching the positional
    # contract of train_v6.build_features, which zero-pads / right-truncates cs ranks).
    ranked = np.full_like(panel, np.nan)
    for t in range(n_dates):
        slab = panel[t]                      # (n_codes, 8)
        valid = np.isfinite(slab).all(1)
        if valid.sum() < 5:
            continue
        for f in range(8):
            ranked[t, valid, f] = _cs_rank_norm(slab[valid, f])

    out: Dict[str, np.ndarray] = {}
    for ci, c in enumerate(codes):
        s = stock_data[c]
        idx = np.array([date_pos[d] for d in s["dates"]], dtype=np.int64)
        v = ranked[idx, ci]                                   # (T, 8)
        out[c] = np.nan_to_num(v, nan=0.0).astype(np.float32)

    if save:
        np.savez(path, **{c: v for c, v in out.items()})
    return out


# ---------------------------------------------------------------- walk-forward
def generate_walk_forward_splits(
    all_dates: List[str], train_days: int = 252, step_days: int = 21,
    max_folds: Optional[int] = None,
) -> List[dict]:
    """Expand-by-k walk-forward with val/test gaps (as used by train.py)."""
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


# ---------------------------------------------------------------- fold npz (baseline reproduction)
def load_fold_from_npz(fold: int = 0, base: str = DATA_DIR) -> Tuple[np.ndarray, ...]:
    """Read a pre-built baseline fold. Fully correct baseline-reproduction path."""
    d = np.load(FOLD_PATTERN.format(n=fold).replace("{n}", str(fold)))
    return (d["train_X"], d["train_y"], d["val_X"], d["val_y"],
            d["test_X"], d["test_y"])


# ---------------------------------------------------------------- fold regeneration
# 20 base features: order + normalisation recovered VERBATIM from the
# production feature builder `build_features` (feat20), the reference the original
# walk-forward loader aligns with (28 = 20 base + 8 CS-rank; see the header contract).
_BASE_20_NAMES: Tuple[str, ...] = (
    "close_ma20", "open_close", "high_close", "low_close",
    "log_vol", "log_amount", "pct_chg",
    "ret1", "ret5", "ret10", "mom20",
    "ma5_diff", "ma20_diff", "vol20", "vol_ratio",
    "amount_ratio", "hl_range", "body", "rsi", "cz20",
)

# Label definition for fold regeneration, EMPIRICALLY PINNED by probing the shipped
# baseline_fold0.npz: train_y is a RAW h-day forward return (decimal fraction, not
# percent, not z-scored).  fold0 train_y std = 0.0683; pooled cross-sectional std of
# h-day forward returns across all 4,993 stocks = [h:std] 1:0.0335 3:0.0598 5:0.0767
# 10:0.1087 20:0.1762 -> h=5 is the match (fold0's earlier, lower-vol train window
# accounts for 0.0683 < 0.0767).  (train_baselines.py on the old cluster is no longer
# reachable, so this empirical pinning is the ground truth for fold regeneration.)
LABEL_HORIZON = 5


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
        close / (ma20 + eps),
        open_ / (close + eps),
        high / (close + eps),
        low / (close + eps),
        np.log1p(np.clip(vol, 0.0, None)) / 20.0,
        np.log1p(np.clip(amount, 0.0, None)) / 25.0,
        pct_norm,
        ret1,
        ret5,
        ret10,
        mom20,
        ma5_diff,
        ma20_diff,
        vol20,
        vol_ratio,
        amount_ratio,
        hl_range,
        body,
        rsi / 100.0,
        cz20,
    ])
    feat20 = np.nan_to_num(feat20, nan=0.0, posinf=0.0, neginf=0.0)
    return feat20.astype(np.float32)


def _forward_return(close: np.ndarray, horizon: int) -> np.ndarray:
    """(T,) forward h-day return; NaN in the trailing h days (no lookahead)."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n > horizon:
        out[: n - horizon] = (close[horizon:] - close[:-horizon]) / (np.abs(close[:-horizon]) + 1e-8)
    return out


def _assemble_samples(stock_data, cs_cache, horizon: int):
    """Build the (stock_code, date, x1680, raw_label) sample pool.

    A sample for stock s at date-index t uses the 60-day window [t-59..t] (inclusive)
    flattened to 1680, and labels the forward horizon-day return.  Returns aligned
    arrays + the sorted global date axis so splits can bucket samples by date.
    """
    all_dates = sorted({d for s in stock_data.values() for d in s["dates"]})
    date_idx = {d: i for i, d in enumerate(all_dates)}

    xs, ys, dts, codes = [], [], [], []
    for code, s in stock_data.items():
        T = len(s["close"])
        if T <= SEQ_LEN + horizon:
            continue
        feat20 = _base_20_features(s)                       # (T, 20)
        cs = cs_cache.get(code)
        if cs is None:
            cs_r = np.zeros((T, N_CS), dtype=np.float32)
        elif cs.shape[0] >= T:
            cs_r = cs[:T]
        else:  # shorter (older cache): right-align, zero-pad the front
            cs_r = np.zeros((T, N_CS), dtype=np.float32)
            cs_r[-cs.shape[0]:] = cs
        feat = np.concatenate([feat20, cs_r], axis=1)       # (T, 28)
        fwd = _forward_return(s["close"].astype(np.float64), horizon)

        for t in range(SEQ_LEN - 1, T - horizon):
            x = feat[t - SEQ_LEN + 1: t + 1].reshape(-1)     # (1680,)
            y = fwd[t]
            if not np.isfinite(y):
                continue
            xs.append(x)
            ys.append(np.float32(y))
            dts.append(date_idx[s["dates"][t]])
            codes.append(code)

    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    d = np.asarray(dts, dtype=np.int64)
    return X, y, d, all_dates


def _zscore_by_date(y: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Per-day cross-sectional z-score (mean 0 / unit var across stocks per date)."""
    out = np.zeros_like(y, dtype=np.float32)
    for t in np.unique(d):
        m = d == t
        seg = y[m]
        mu, sd = seg.mean(), seg.std()
        out[m] = (seg - mu) / (sd + 1e-8) if sd > 1e-8 else np.zeros_like(seg)
    return out


def build_fold_data(stock_data, cs_cache, all_dates,
                    tr_s, tr_e, va_s, va_e, te_s, te_e,
                    horizon: int = LABEL_HORIZON, zscore: bool = False):
    """Assemble (X,y) for one fold: X=(n,1680) flattened (60,28), y=forward return.

    Splits are date-string ranges (as produced by generate_walk_forward_splits).
    Labels are RAW `horizon`-day forward returns (decimal) by default — matching the
    shipped baseline_fold*.npz contract (train_y std ~0.068, NOT z-scored).  Set
    `zscore=True` to additionally apply per-day cross-sectional z-scoring.
    Output layout matches load_fold_from_npz() exactly.
    """
    X, y_raw, d, _ = _assemble_samples(stock_data, cs_cache, horizon)
    y = _zscore_by_date(y_raw, d) if zscore else y_raw
    date_set = {k: all_dates.index(v) for k, v in [
        ("tr_s", tr_s), ("tr_e", tr_e), ("va_s", va_s), ("va_e", va_e),
        ("te_s", te_s), ("te_e", te_e)]}

    seg = {
        "train": (d >= date_set["tr_s"]) & (d <= date_set["tr_e"]),
        "val":   (d >= date_set["va_s"]) & (d <= date_set["va_e"]),
        "test":  (d >= date_set["te_s"]) & (d <= date_set["te_e"]),
    }
    pick = lambda k: (X[seg[k]], y[seg[k]])
    trX, trY = pick("train")
    vaX, vaY = pick("val")
    teX, teY = pick("test")
    return trX, trY, vaX, vaY, teX, teY


if __name__ == "__main__":
    sd = load_all_stocks_data()
    print(f"[pkl] {len(sd)} stocks; sample keys: {sorted(next(iter(sd.values())))[:12]}")
    folds = generate_walk_forward_splits(
        sorted({d for s in sd.values() for d in s["dates"]}), max_folds=10)
    print(f"[splits] {len(folds)} folds; first: {folds[0]}")
    trX, trY, vaX, vaY, teX, teY = load_fold_from_npz(0)
    print(f"[fold0] train {trX.shape}/{trY.dtype} val {vaX.shape} test {teX.shape}")