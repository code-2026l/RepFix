"""v11.6 Self-contained Training Loss (Real-Task Multi-Objective Loss)

V11.6 student (StudentModel) is **self-contained**: no teacher loading, no fincast_v11.pth dependency.
All 8 differentiable losses are real financial tasks (FAMO dynamic weighting, NeurIPS 2023):

1. reg Point regression MSE(student_point, y_true) -- magnitude alignment
2. sign Direction consistency sign(pred)==sign(y) -- direction (binary)
3. quant Quantile pinball/CRPS (pred_quantiles vs y_true) -- calibration
4. true True NLL (MDN.nll_loss, anti-collapse / PIT uniformity) -- distribution fitting
5. rank_y ListNet listwise true returns ranking (ICML'07) -- listwise ranking
6. dir Direction contrastive: InfoNCE-style direction alignment (CPC) -- direction robustness
7. ic Direct IC loss (=1-Pearson corr(pred, y)) -- rank correlation
8. focal_rank Focal pairwise ranking (Focal + RankNet + PR-Loss) -- hard pair focus"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _component_responsibilities(
    mu_s: torch.Tensor,
    log_sigma_s: torch.Tensor,
    mu_t: torch.Tensor,
    log_sigma_t: torch.Tensor,
    temperature: float = 0.5,
    ) -> torch.Tensor:
    """compute student -> teacher r[k, t] ( batch )

    Args:
    mu_s: [B, K_s]
    log_sigma_s: [B, K_s]
    mu_t: [B, K_t]
    log_sigma_t: [B, K_t]
    temperature:
    Returns:
    r: [B, K_s, K_t] ( t normalization)
    """
# : +
# [B, K_s, 1] - [B, 1, K_t] -> [B, K_s, K_t]
    d_mu = (mu_s.unsqueeze(2) - mu_t.unsqueeze(1)) ** 2
    d_sigma = (log_sigma_s.unsqueeze(2) - log_sigma_t.unsqueeze(1)) ** 2
    cost = d_mu + d_sigma # [B, K_s, K_t]
    r = torch.softmax(-cost / max(temperature, 1e-3), dim=-1)
    return r


def _soft_mdn_loss(
    s: dict,
    t: dict,
    temperature: float = 0.5,
    ) -> torch.Tensor:
    """loss (Parameters)"""
    mu_s, ls_s, pi_s, al_s = s["mu"], s["log_sigma"], s["pi"], s["alpha"]
    mu_t, ls_t, pi_t, al_t = t["mu"], t["log_sigma"], t["pi"], t["alpha"]
    r = _component_responsibilities(mu_s, ls_s, mu_t, ls_t, temperature) # [B, Ks, Kt]
# student k Parameters ( r )
    mu_t_w = (r * mu_t.unsqueeze(1)).sum(-1) # [B, Ks]
    ls_t_w = (r * ls_t.unsqueeze(1)).sum(-1)
    al_t_w = (r * al_t.unsqueeze(1)).sum(-1)
    pi_t_w = (r * pi_t.unsqueeze(1)).sum(-1) # [B, Ks]
    l_mu = F.mse_loss(mu_s, mu_t_w)
    l_ls = F.mse_loss(ls_s, ls_t_w)
    l_al = F.mse_loss(al_s, al_t_w)
    l_pi = F.kl_div(torch.log(pi_s + 1e-8), pi_t_w, reduction="batchmean")
    return l_mu + l_ls + l_al + l_pi


# ---- loss (v11.6, v11.5 ) ----
# v11.6 (JS/FinStack/GBDT-LSTM/FinalMLP/SCARF/RCMoE/
# TSMixer/iTransformer+RevIN), training:
# Point regression (MSE) + direction consistency + quantile pinball (CRPS) + true NLL
# + ListNet true returns + MADL loss


def _pinball_loss(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Quantile pinball loss (CRPS equivalent), student quantiles vs true returns."""
    Q = q.shape[-1]
    taus = torch.linspace(0.05, 0.95, Q, device=q.device, dtype=q.dtype)
    yy = y.unsqueeze(-1).expand_as(q)
    e = yy - q
    return torch.max(taus * e, (taus - 1) * e).mean()


def listnet_loss(scores_s: torch.Tensor, scores_t: torch.Tensor,
    temp: float = 1.0) -> torch.Tensor:
    """ListNet (Cao et al., ICML'07 WS)

    /() top-1, KL(student||target),
    optimization(IC) ——,
    true returns( IC-DA )

    Args:
    scores_s: [B] ()
    scores_t: [B] ( true returns y_true)
    temp: softmax ; std
    """
    t_std = scores_t.std().clamp(min=1e-4)
    scale = (temp * t_std + 1e-8)
    p_s = F.softmax(scores_s / scale, dim=0)
    p_t = F.softmax(scores_t / scale, dim=0).detach()
    return F.kl_div(torch.log(p_s + 1e-8), p_t, reduction="batchmean")


def mad_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
    gamma: float = 1.0) -> torch.Tensor:
    """MADL loss (Chaturvedi et al., arXiv:2309.10546, 2023)

    MADL = mean( |e| * (1 + gamma * I[sign(ŷ) != sign(y)]) )
    (gamma=1 -> 2x), (DA), IC-DA
, training, ONNX
"""
    e = (y_true - y_pred).abs()
    wrong = (torch.sign(y_true) * torch.sign(y_pred)) < 0
    return (e * (1.0 + gamma * wrong.float())).mean()


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """ IC loss: 1 - (, Parameters)

    IC(Pearson) ; ->,
    optimizationevaluation IC:
    Blondel et al., Fast Differentiable Sorting and Ranking (ICML 2020)
    """
    p = pred - pred.mean()
    t = target - target.mean()
    denom = (p.norm() * t.norm()).clamp(min=eps)
    corr = (p * t).sum() / denom
    return (1.0 - corr).clamp(min=0.0, max=2.0)


def focal_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    n_pairs: int = 4096,
    ) -> torch.Tensor:
    """Focal loss (/, IC↔DA )

    (Burges et al., RankNet, NeurIPS 2005) + Focal
    (Lin et al., Focal Loss, ICCV 2017) +
    (PR-Loss, 2024): /true returns focal,
    IC

    weight = (1 - 0.5*sign_match) * (|y_i-y_j|/mean|y|)^gamma
    pair = softplus(-sign(y_i-y_j)*(p_i-p_j)) # RankNet ()
    loss = mean(weight * pair)

    B<=100 O(B^2) (), random n_pairs (O(n_pairs), )
    """
    B = pred.shape[0]
    if B < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if B <= 100:
        ii, jj = torch.triu_indices(B, B, offset=1)
    else:
        idx = torch.randint(0, B, (2 * n_pairs,), device=pred.device)
        ii, jj = idx[:n_pairs], idx[n_pairs:]
    yi, yj = target[ii], target[jj]
    pi, pj = pred[ii], pred[jj]
    dy = yi - yj
    dp = pi - pj
    s_true = torch.sign(dy)
    valid = s_true != 0
    if valid.sum() == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
    s_true = s_true[valid]; dy = dy[valid]; dp = dp[valid]
    s_pred = torch.sign(dp)
    sign_match = (s_true == s_pred).float()
    pair = F.softplus(-s_true * dp) # = log(1+exp(-x)),
    mag = dy.abs() / (dy.abs().mean() + 1e-6)
    weight = (1.0 - 0.5 * sign_match) * mag
    weight = weight.clamp(min=0.0) ** gamma
    return (weight * pair).mean()


def ic_da_contrastive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: float = 0.1,
    ) -> torch.Tensor:
    """IC↔DA optimization: InfoNCE loss (Parameters, 9 )

    IC↔DA (/""): normalization
    batch, ****, InfoNCE
    (Oord et al., Representation Learning with Contrastive Predictive Coding,
    NeurIPS 2018) "" —— IC(
    ) DA(, )

    Implementation: z = normalize(pred - pred.mean()); sim = z·zᵀ/τ;
    = {j : sign(y_j)==sign(y_i), j≠i}; InfoNCE(log_softmax)
    Parameters (/),, matmul/softmax/cross_entropy, ONNX
    B<2 (), NaN/
    """
    B = pred.shape[0]
    if B < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    pred = pred.reshape(-1)
    target = target.reshape(-1)
# normalization (Parameters);
    z = pred - pred.mean()
    z = z / (z.norm() + 1e-8)
    sim = (z.unsqueeze(1) @ z.unsqueeze(0)) / tau # (B, B) ()
    sy = torch.sign(target) # (B,)
# :
    pos = (sy.unsqueeze(0) == sy.unsqueeze(1)).float()
    pos = pos - torch.eye(B, device=pred.device, dtype=torch.float32)
    n_pos = pos.sum(dim=1) # (B,)
    safe_pos = pos.clone()
    no_pos = n_pos == 0 # -> :
    safe_pos[no_pos, no_pos] = 1.0
    logp = F.log_softmax(sim, dim=1) # (B, B)
# loss = -Σ_j safe_pos[i,j]·logp[i,j] / Σ_j safe_pos[i,j]
    denom = safe_pos.sum(dim=1).clamp(min=1.0)
    loss = -((safe_pos * logp).sum(dim=1) / denom)
    return loss.mean()


def _sign_loss(pred, target, weight: float) -> torch.Tensor:
    if weight <= 0:
        return torch.zeros_like(pred.sum())
    ps = torch.sign(pred)
    ts = torch.sign(target)
    mask = (ps != 0) & (ts != 0)
    if mask.sum() == 0:
        return torch.zeros_like(pred.sum())
    disagree = (ps[mask] != ts[mask]).float().mean()
    return weight * disagree


def v11_6_distill_loss(
    student_out: dict,
    y_true: torch.Tensor,
    mdn_head,
    w_reg: float = 1.0,
    w_sign: float = 0.5,
    w_quantile: float = 0.5,
    w_true: float = 0.2,
    w_rank: float = 0.5,
    w_dir: float = 1.0,
    w_ic: float = 0.3,
    w_focal_rank: float = 0.3,
    w_ic_da: float = 0.2,
    temperature: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """组合 8 路真实任务损失（v11.6 自包含训练）。"""
    l_reg = F.mse_loss(student_out["point"], y_true)
    l_sign = _sign_loss(student_out["point"], y_true, w_sign)
    l_quant = _pinball_loss(student_out["quantiles"], y_true)
    preds = {
        "point": student_out["point"],
        "quantiles": student_out["quantiles"],
        "pi": student_out["pi"],
        "mu": student_out["mu"],
        "sigma": student_out["sigma"],
        "alpha": student_out["alpha"],
        "log_sigma": student_out["log_sigma"],
    }
    l_true = mdn_head.nll_loss(preds, y_true)
    l_rank_y = listnet_loss(student_out["point"], y_true)
    l_dir = mad_loss(y_true, student_out["point"])
    l_ic = pearson_loss(student_out["point"], y_true)
    l_focal_rank = focal_rank_loss(student_out["point"], y_true)
    l_ic_da = ic_da_contrastive_loss(student_out["point"], y_true)
    total = (w_reg * l_reg + w_sign * l_sign + w_quantile * l_quant
             + w_true * l_true + w_rank * l_rank_y + w_dir * l_dir
             + w_ic * l_ic + w_focal_rank * l_focal_rank + w_ic_da * l_ic_da)
    comp = {
        "l_reg": l_reg.item(),
        "l_sign": l_sign.item() if torch.is_tensor(l_sign) else float(l_sign),
        "l_quant": l_quant.item(),
        "l_true": l_true.item(),
        "l_rank_y": l_rank_y.item(),
        "l_dir": l_dir.item() if torch.is_tensor(l_dir) else float(l_dir),
        "l_ic": l_ic.item(),
        "l_focal_rank": l_focal_rank.item(),
        "l_ic_da": l_ic_da.item(),
    }
    return total, comp

def v11_6_distill_components(
    student_out: dict,
    y_true: torch.Tensor,
    mdn_head,
    temperature: float = 0.5,
    use_dir_calib: bool = False,
) -> tuple[list[torch.Tensor], dict]:
    """返回 8 路(或 9 路)独立损失供 FAMO 动态加权。"""
    l_reg = F.mse_loss(student_out["point"], y_true)
    l_sign = _sign_loss(student_out["point"], y_true, 1.0)
    l_quant = _pinball_loss(student_out["quantiles"], y_true)
    preds = {
        "point": student_out["point"],
        "quantiles": student_out["quantiles"],
        "pi": student_out["pi"],
        "mu": student_out["mu"],
        "sigma": student_out["sigma"],
        "alpha": student_out["alpha"],
        "log_sigma": student_out["log_sigma"],
    }
    l_true = mdn_head.nll_loss(preds, y_true)
    l_rank_y = listnet_loss(student_out["point"], y_true)
    l_dir = mad_loss(y_true, student_out["point"])
    l_ic = pearson_loss(student_out["point"], y_true)
    l_focal_rank = focal_rank_loss(student_out["point"], y_true)
    l_ic_da = ic_da_contrastive_loss(student_out["point"], y_true)
    # P1: 方向校准 Brier 损失（可选第 9 路）
    if use_dir_calib and student_out.get("p_up") is not None:
        _target = (y_true > 0).float()
        l_dir_calib = ((student_out["p_up"] - _target) ** 2).mean()
    else:
        l_dir_calib = None
    comp = {
        "l_reg": l_reg.item(),
        "l_sign": l_sign.item() if torch.is_tensor(l_sign) else float(l_sign),
        "l_quant": l_quant.item(),
        "l_true": l_true.item(),
        "l_rank_y": l_rank_y.item(),
        "l_dir": l_dir.item() if torch.is_tensor(l_dir) else float(l_dir),
        "l_ic": l_ic.item(),
        "l_focal_rank": l_focal_rank.item(),
        "l_ic_da": l_ic_da.item(),
        "l_dir_calib": l_dir_calib.item() if l_dir_calib is not None else 0.0,
    }
    _losses = [l_reg, l_quant, l_true, l_rank_y, l_dir, l_ic, l_focal_rank, l_ic_da]
    if l_dir_calib is not None:
        _losses.append(l_dir_calib)
    return _losses, comp

class FAMO:
    """Fast Adaptive Multitask Optimization (NeurIPS 2023)

    loss running EMA, weight:
    w_i = softmax(-η * (L_i / EMA_i))
    normalization, ( 4905 eiil),
    loss

    :
    famo = FAMO(n_tasks=6, eta=0.5)
    losses, comp = v11_6_distill_components(...)
    w = famo.update(losses) # detached update EMA/weight
    total = sum(w[i] * losses[i]) # loss ()
    """

    def __init__(self, n_tasks: int, eta: float = 0.5, ema_alpha: float = 0.9, eps: float = 1e-8):
        self.n = n_tasks
        self.eta = eta
        self.ema_alpha = ema_alpha
        self.eps = eps
        self.ema = None # (n,) loss EMA
        self.weights = None # (n,) weight
        self.step = 0

    def update(self, losses) -> torch.Tensor:
        """losses: list[Tensor] N loss Returns detached weight (n,)
        NaN-safe: NaN/inf 时 (1) update EMA; (2) weight 0, renormalize
        (EMA 慢 -> boosting)"""
        vals = torch.stack([l.detach() for l in losses])  # (n,)
        finite = torch.isfinite(vals)
        if finite.all():
            if self.ema is None:
                self.ema = vals.clone()
            else:
                self.ema = self.ema_alpha * self.ema + (1.0 - self.ema_alpha) * vals
        elif self.ema is None:
            self.ema = torch.ones_like(vals)  # 首个 NaN
        self.step += 1
        safe_ema = torch.where(finite, self.ema, torch.ones_like(self.ema))
        norm = vals / (safe_ema + self.eps)
        norm = torch.clamp(norm, -10.0, 10.0)
        logw = -self.eta * norm
        w = torch.softmax(logw, dim=0)
        # NaN weight 0, renormalize (weighted_loss nan-safe)
        w = torch.where(finite, w, torch.zeros_like(w))
        w = w / (w.sum() + 1e-8)
        self.weights = w
        return w
    def weighted_loss(self, losses, weights=None) -> torch.Tensor:
                """ (detached) weightloss,
                ⚠️ NaN-safe (2026-07-10): loss 0,,
                NaN total=NaN training"""
                w = weights if weights is not None else self.weights
                safe = [torch.where(torch.isfinite(l), l, torch.zeros_like(l)) for l in losses]
                return sum(w[i] * safe[i] for i in range(self.n))

    def weight_dict(self, keys=None) -> dict:
        if keys is None:
            keys = ["reg", "quant", "true", "rank_y", "dir", "ic", "focal_rank", "ic_da"]
            if self.n == 9:
                keys.append("dir_calib")
        return {keys[i]: float(self.weights[i]) for i in range(self.n)}
