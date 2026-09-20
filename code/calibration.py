"""Temporal Conformal Calibration
===============================================
Based on arXiv:2507.05470 (TCP) + ACI (Adaptive Conformal Inference,
Gibbs & Candes, NeurIPS 2021):

Maintains a rolling calibration window (recent N validation samples MDN quantiles
vs actual returns), computes adaptive calibration quantile alpha_t for finite-sample
validity (marginal coverage ≈ target), mitigating interval distortion under IC-DA
divergence.

- Rolling-window split-conformal: empirical coverage from history, then widen/narrow.
- ACI online correction of miscoverage gamma_t with envelope bounds:
    gamma_{t+1} = clip(gamma_t + eta*(target - coverage), -alpha_base, gamma_cap)
    alpha_t     = clip(alpha_base + gamma_t, alpha_min, alpha_max)
- No trainable parameters; called each val epoch during inference.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class TemporalConformalCalibrator:
    """Rolling-window temporal conformal calibrator.

    update(point_true, pred_quantiles): add calibration batch
    calibrate(mdn_out) -> (lower, upper, coverage): reliable prediction intervals
    coverage_report() -> dict: window statistics
    """

    def __init__(
        self,
        target_coverage: float = 0.90,
        window_size: int = 500,
        q_levels: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
        alpha_min: float = 0.02,
        alpha_max: float = 0.50,
        adapt_rate: float = 0.30,
    ):
        assert 0.0 < target_coverage < 1.0
        self.target = float(target_coverage)
        self.window_size = int(window_size)
        self.q_levels = np.asarray(q_levels, dtype=np.float64)
        self.alpha_base = 1.0 - self.target
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.adapt_rate = float(adapt_rate)
        # ACI 状态
        self.gamma = 0.0
        self.gamma_cap = float(alpha_max)
        # 滚动窗口
        self._yt: list = []
        self._q: list = []

    # ── 窗口维护 ──────────────────────────────────────────────
    def update(self, point_true, pred_quantiles) -> None:
        """添加一批校准样本（true returns + MDN quantiles）。"""
        if isinstance(point_true, torch.Tensor):
            point_true = point_true.detach().cpu().numpy()
        if isinstance(pred_quantiles, torch.Tensor):
            pred_quantiles = pred_quantiles.detach().cpu().numpy()
        yt = np.asarray(point_true, dtype=np.float64).reshape(-1)
        q = np.asarray(pred_quantiles, dtype=np.float64)
        if q.ndim == 1:
            q = q[None, :]
        self._yt.append(yt)
        self._q.append(q)
        # 限制窗口样本数
        total = sum(len(a) for a in self._yt)
        while total > self.window_size and len(self._yt) > 1:
            removed = self._yt.pop(0)
            self._q.pop(0)
            total -= len(removed)

    def _window_arrays(self):
        if not self._yt:
            return None, None
        return np.concatenate(self._yt), np.concatenate(self._q, axis=0)

    def _empirical_coverage(self, lo_col: int, hi_col: int) -> float:
        yt, q = self._window_arrays()
        if yt is None or len(yt) == 0:
            return self.target
        lo = q[:, lo_col]
        hi = q[:, hi_col]
        inside = (yt >= lo) & (yt <= hi)
        return float(inside.mean())

    # ── 校准 ──────────────────────────────────────────────────
    def calibrate(self, mdn_out: dict) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """返回 (lower, upper, coverage)。

        lower/upper: 预测区间 [B]；coverage: 滚动窗口经验覆盖率。
        """
        q = mdn_out.get("quantiles")
        if isinstance(q, torch.Tensor):
            q_np = q.detach().cpu().numpy()
            device, dtype = q.device, q.dtype
        else:
            q_np = np.asarray(q, dtype=np.float64)
            device, dtype = None, torch.float32
        if q_np.ndim == 1:
            q_np = q_np[None, :]
        n_cols = q_np.shape[-1]
        # 1) ACI 在线校正 miscoverage gamma_t
        cov_nom = self._empirical_coverage(0, n_cols - 1)
        self.gamma = float(np.clip(
            self.gamma + self.adapt_rate * (self.target - cov_nom),
            -self.alpha_base, self.gamma_cap))
        alpha_t = float(np.clip(self.alpha_base + self.gamma, self.alpha_min, self.alpha_max))
        # 2) alpha_t → 最近 MDN 分位列
        lo_level = alpha_t / 2.0
        hi_level = 1.0 - alpha_t / 2.0
        lo_col = int(np.argmin(np.abs(self.q_levels - lo_level))) if len(self.q_levels) else 0
        hi_col = int(np.argmin(np.abs(self.q_levels - hi_level))) if len(self.q_levels) else n_cols - 1
        lo_col = min(max(lo_col, 0), n_cols - 1)
        hi_col = min(max(hi_col, 0), n_cols - 1)
        lower = q_np[:, lo_col]
        upper = q_np[:, hi_col]
        # 3) 报告校准区间经验覆盖率
        cov = self._empirical_coverage(lo_col, hi_col)

        def _to_t(x):
            return torch.as_tensor(x, dtype=dtype) if device is None else torch.as_tensor(x, device=device, dtype=dtype)

        return _to_t(lower), _to_t(upper), cov

    def coverage_report(self) -> dict:
        yt, _ = self._window_arrays()
        n = 0 if yt is None else len(yt)
        cov = self._empirical_coverage(0, len(self.q_levels) - 1)
        return {
            "window_samples": int(n),
            "target_coverage": self.target,
            "empirical_coverage": cov,
            "aci_gamma": float(self.gamma),
        }
