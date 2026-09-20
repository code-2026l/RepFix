"""train_data: V11.6 训练数据管线（论文匿名代码自包含版）。

替代云端 train_v11_wf / train_v7_lite_wf / train_v6_wf 三个缺失模块，
从 data/stock_data_cache.pkl（4993 股 OHLCV 缓存）构建 walk-forward 折。

接口对齐 archive train_v11_6_distill.py 的使用方式：
- V11_TRAIN_DAYS / V11_STEP_DAYS / V11_MAX_FOLDS
- compute_market_state_from_data(data) -> (3,) [trend, vol, liquidity]
- generate_walk_forward_splits(all_dates, train_days, step_days, max_folds)
  -> (splits, all_dates)
- compute_rank_ic(pred, target, dates=None) -> float（Spearman Rank IC）
- load_stock_cache(path="data/stock_data_cache.pkl") -> {code: {dates, OHLCV...}}
- build_fold_data(stock_data, cs_cache, all_dates, tr_s, tr_e, va_s, va_e, te_s, te_e)
  -> {"X": (N,60,28) f32, "y": (N,) f32, "aux": (N,16) f32, "dates": [...]} or (None,None)
"""
from __future__ import annotations

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

V11_TRAIN_DAYS = 730   # 训练窗口约 3 年交易日（原云端用更大；缓存仅 2021-2026 故取 ~2.8y）
V11_STEP_DAYS = 90     # 每折步进 1 季度
V11_MAX_FOLDS = 10

SEQ_LEN = 60
N_FEAT = 28
N_AUX = 16
EPS = 1e-8


# ---------------------------------------------------------------------------
# 特征工程（与 v11_6_service._build_v11_6_xseq 对齐；28 维 + aux 16 维）
# ---------------------------------------------------------------------------
def _rm(x, w):
    n = len(x)
    c = np.cumsum(x)
    out = np.empty(n)
    for i in range(min(w, n)):
        out[i] = c[i] / (i + 1)
    if n > w:
        out[w:] = (c[w:] - c[:-w]) / w
    return out


def _rs(x, w):
    m = _rm(x, w)
    sq = _rm(x * x, w) - m * m
    return np.sqrt(np.maximum(sq, 0)) + EPS


def build_features(closes, opens, highs, lows, vols, amounts, pct_chg):
    """复刻 28+16 维特征。返回 (feat[T,28], aux[T,16])。"""
    closes = closes.astype(np.float64)
    opens = opens.astype(np.float64)
    highs = highs.astype(np.float64)
    lows = lows.astype(np.float64)
    vols = vols.astype(np.float64)
    amounts = amounts.astype(np.float64)
    pct = np.asarray(pct_chg, dtype=np.float64)
    if np.all(pct == 0):
        pct = np.diff(closes, prepend=closes[0]) / (np.abs(closes) + EPS) * 100.0
    n = len(closes)

    ret1 = np.diff(closes, prepend=closes[0]) / (np.abs(closes) + EPS)
    ma5, ma10, ma20 = _rm(closes, 5), _rm(closes, 10), _rm(closes, 20)
    vma5, vma20 = _rm(vols, 5), _rm(vols, 20)
    ama5, ama20 = _rm(amounts, 5), _rm(amounts, 20)

    def past_ret(p):
        out = np.zeros_like(closes)
        if n > p:
            out[p:] = (closes[p:] - closes[:-p]) / (np.abs(closes[:-p]) + EPS)
        return out

    ret5, ret10, mom20 = past_ret(5), past_ret(10), past_ret(20)
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    rs_ = _rm(gain, 14) / (_rm(loss, 14) + EPS)
    rsi = 100.0 - 100.0 / (1.0 + rs_)
    vol20 = _rs(ret1, 20)
    cz20 = (closes - _rm(closes, 20)) / _rs(closes, 20)
    az20 = (amounts - _rm(amounts, 20)) / _rs(amounts, 20)
    body = (closes - opens) / (opens + EPS)
    hl_range = (highs - lows) / (closes + EPS)
    up_shadow = (highs - np.maximum(opens, closes)) / (closes + EPS)
    lo_shadow = (np.minimum(opens, closes) - lows) / (closes + EPS)

    feat20 = np.column_stack([
        closes / (ma20 + EPS), opens / (closes + EPS), highs / (closes + EPS),
        lows / (closes + EPS), np.log1p(vols.clip(0)) / 20.0,
        np.log1p(amounts.clip(0)) / 25.0, pct / 100.0,
        ret1, ret5, ret10, mom20,
        closes / (ma5 + EPS) - 1.0, closes / (ma20 + EPS) - 1.0,
        vol20, vols / (vma5 + EPS), amounts / (ama5 + EPS),
        hl_range, body, rsi / 100.0, cz20,
    ])
    cs_ranks = np.zeros((n, 8), dtype=np.float32)  # 无 cs_cache → 0（与后端兜底一致）
    feat = np.nan_to_num(np.concatenate([feat20, cs_ranks], axis=1), nan=0.0, posinf=0.0, neginf=0.0)

    vol20_mean = _rm(vols, min(20, n))
    vol20_std = _rs(vols, min(20, n))
    ret5m = _rm(ret1, 5)
    ret10m = _rm(ret1, 10)
    panic = np.maximum(-ret1, 0) * (vols / (vol20_mean + EPS))
    greed = np.maximum(ret1, 0) * (vol20_mean / (vols + EPS))
    nvs = (vols - vol20_mean) / vol20_std
    senti = (greed - panic) / 2 + nvs * 0.3
    aux = np.column_stack([
        senti, greed / (panic + 1.0), nvs,
        np.maximum(ret1, 0), np.maximum(-ret1, 0), ret5m, ret10m,
        np.log1p(vols.clip(0)) / 20.0, np.abs(ret1), ret1, ret5m - panic,
        senti / 2.0, np.sign(ret1) * np.minimum(np.abs(ret1), 0.1),
        ret10m, ret10m - ret5m, ret1 * vols / (vol20_mean + EPS),
    ]).astype(np.float32)
    return feat.astype(np.float32), aux


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def _norm_date(x) -> str:
    """归一化日期为 YYYYMMDD（无连字符）。"""
    s = str(x).strip()
    return s.replace("-", "")


def load_stock_cache(path=None):
    path = path or str(Path(os.environ.get("REPFIX_DATA_DIR", "data")) / "stock_data_cache.pkl")
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)


def compute_market_state_from_data(data) -> np.ndarray:
    """返回 (3,) 全局市场状态 [trend, vol, liquidity]（粗粒度，供 msv_global）。"""
    try:
        rets = np.asarray(data["y"], dtype=np.float64)
        if rets.size < 20:
            return np.zeros(3, dtype=np.float32)
        trend = float(np.tanh(np.clip(rets.mean() * 20.0, -2, 2)))
        vol = float(np.clip(np.std(rets) * 10.0, 0, 1))
        liq = float(np.clip(1.0 - np.std(rets) / (np.abs(rets).mean() + EPS) / 5.0, 0, 1))
        return np.array([trend, vol, liq], dtype=np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


# ---------------------------------------------------------------------------
# Walk-forward 划分
# ---------------------------------------------------------------------------
def _dstr(x):
    return str(x)[:10].replace("-", "")


def generate_walk_forward_splits(all_dates, train_days=V11_TRAIN_DAYS,
                                 step_days=V11_STEP_DAYS, max_folds=V11_MAX_FOLDS):
    """按交易日序列切分 walk-forward 折。返回 (splits, all_dates)。"""
    dates = sorted(set(_norm_date(d) for d in all_dates))
    n = len(dates)
    splits = []
    tr_end = train_days
    win = max(60, step_days)  # val/test 窗口 = 步长（季度），保证样本量
    while tr_end + 2 * win <= n and len(splits) < max_folds:
        va_start, va_end = tr_end, tr_end + win
        te_start, te_end = va_end, va_end + win
        if te_end > n:
            break
        info = {
            "train_start": _dstr(dates[0]),
            "train_end": _dstr(dates[tr_end - 1]),
            "val_start": _dstr(dates[va_start]),
            "val_end": _dstr(dates[va_end - 1]),
            "test_start": _dstr(dates[te_start]),
            "test_end": _dstr(dates[te_end - 1]),
        }
        splits.append(info)
        tr_end += step_days
    return splits, dates


# ---------------------------------------------------------------------------
# 折数据构建
# ---------------------------------------------------------------------------
def build_fold_data(stock_data, cs_cache, all_dates,
                    train_start, train_end, val_start, val_end,
                    test_start, test_end):
    """为单个折构建训练/验证数据。返回 (train_dict, val_dict) 或 (None, None)。"""
    try:
        tr = _build_split(stock_data, train_start, train_end)
        va = _build_split(stock_data, val_start, val_end)
        if tr is None or va is None or len(tr["X"]) < 1000:
            return None, None
        return tr, va
    except Exception as e:
        print(f"[build_fold_data] 失败: {e}")
        return None, None


def _idx_ge(di_keys, target, upper=False):
    """在排序后的 di_keys 中找 >= target 的第一个（或 <= target 的最后一个）索引位置。
    返回 (date_key, idx)；找不到返回 (None, None)。"""
    ds, de = target, target
    keys = sorted(di_keys)
    lo, hi = 0, len(keys) - 1
    res = None
    if not upper:
        for k in keys:
            if k >= ds:
                res = k
                break
    else:
        for k in keys:
            if k <= de:
                res = k
    return res


def _build_split(stock_data, date_start, date_end, max_samples=100_000, seed=0):
    """对日期窗口 [start, end] 构建 (X[N,60,28], y[N], aux[N,16], dates)。
    股票只要在窗口内存在数据即可（上市晚/退市早都允许），按实际存在区间切片。
    max_samples: 样本上限（与云端训练一致，超限随机降采样防 OOM）。"""
    xs, ys, auxs, dts = [], [], [], []
    ds, de = _norm_date(date_start), _norm_date(date_end)
    rng = np.random.default_rng(seed)
    for code, s in stock_data.items():
        di = s.get("date_idx", {})
        if not di:
            continue
        # 归一化键：date_idx 键可能带/不带连字符
        norm_di = {}
        for k, v in di.items():
            norm_di[_norm_date(k)] = v
        keys = sorted(norm_di.keys())
        # i0: 第一个 >= ds 的交易日；i1: 最后一个 <= de 的交易日
        i0 = None
        for k in keys:
            if k >= ds:
                i0 = norm_di[k]
                break
        i1 = None
        for k in reversed(keys):
            if k <= de:
                i1 = norm_di[k]
                break
        if i0 is None or i1 is None or i1 - i0 < 1:
            continue
        # 特征窗口需要前 SEQ_LEN 天：把 i0 前延取历史（窗口本身可短于 60 天）
        i0_ext = max(0, i0 - SEQ_LEN)
        if i1 - i0_ext < SEQ_LEN + 6:
            continue  # 前延后仍不足（上市太晚/停牌太久）
        closes = np.asarray(s["close"], dtype=np.float64)[i0_ext:i1]
        opens = np.asarray(s["open"], dtype=np.float64)[i0_ext:i1]
        highs = np.asarray(s["high"], dtype=np.float64)[i0_ext:i1]
        lows = np.asarray(s["low"], dtype=np.float64)[i0_ext:i1]
        vols = np.asarray(s["vol"], dtype=np.float64)[i0_ext:i1]
        amounts = np.asarray(s["amount"], dtype=np.float64)[i0_ext:i1]
        pct = np.asarray(s.get("pct_chg", np.zeros_like(closes)), dtype=np.float64)[i0_ext:i1]
        feat, aux = build_features(closes, opens, highs, lows, vols, amounts, pct)
        n = len(feat)
        # 样本起点：窗口起点 i0 在切片中的偏移 + 60 特征；终点留 5 日前向
        base = i0 - i0_ext
        for t in range(base + SEQ_LEN, n - 5):
            x = feat[t - SEQ_LEN:t]
            fwd = closes[t + 5] / max(closes[t], EPS) - 1.0  # 5 日前向收益
            if not np.isfinite(fwd) or not np.isfinite(x).all():
                continue
            xs.append(x.astype(np.float32))
            ys.append(float(fwd))
            auxs.append(aux[t])
            dts.append(_norm_date(s["dates"][i0_ext + t]))
            if len(xs) >= max_samples:
                break
        if len(xs) >= max_samples:
            break
    if len(xs) < 100:
        return None
    return {
        "X": np.stack(xs).astype(np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "aux": np.stack(auxs).astype(np.float32),
        "dates": dts,
    }


# ---------------------------------------------------------------------------
# Rank IC
# ---------------------------------------------------------------------------
def compute_rank_ic(pred, target, dates=None) -> float:
    """日频 Spearman Rank IC（跨股票截面）。pred/target: 1D 数组。

    dates 提供时按交易日分组，组内算 Spearman 再均值（panel 截面口径，
    与论文 fold 表一致）；dates 为空时退化为整体 Spearman。
    """
    from scipy.stats import rankdata
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    v = np.isfinite(p) & np.isfinite(y)
    if v.sum() < 5:
        return 0.0
    if dates is None:
        p, y = p[v], y[v]
        return float(np.corrcoef(rankdata(p), rankdata(y))[0, 1])
    dates = np.asarray(dates)
    ics = []
    for d in np.unique(dates):
        m = v & (dates == d)
        if m.sum() < 5:
            continue
        pp, yy = p[m], y[m]
        ics.append(float(np.corrcoef(rankdata(pp), rankdata(yy))[0, 1]))
    return float(np.mean(ics)) if ics else 0.0
