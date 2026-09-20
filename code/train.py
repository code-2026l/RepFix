"""v11.6 自包含训练 (集成顶级量化架构, 不蒸馏 v11.5 教师)
=========================================================
v11.6 已集成顶级量化架构范式 (JS-AE / FinStack 交叉 / GBDT-LSTM / FinalMLP
/ SCARF / RCMoE / TSMixer / iTransformer+RevIN), 并自包含地从原始多变量时序
(B,60,28) 学习 (RawTemporalEncoder 替代教师 backbone 提 fused)。

训练目标完全面向真实金融任务:
  点回归(MSE) + 方向一致性 + 分位数 pinball + 真实 NLL
  + ListNet 真实收益排名 + MADL 方向损失 + AE/SCARF 自监督 + SAM 平坦极小。

安全约束:
  - 完全不加载 v11.5 教师, 不依赖 fincast_v11.pth。
  - 学生很小 (~90K 参数) -> 训练可在 GPU 空闲后快速完成, 或降级 CPU。
  - 保留 --wait-gpu: 等 v11.5 训练进程释放显存后再上 GPU 快训, 零干扰教师。

用法:
  python train.py            # 全部折 (需 GPU 空闲)
  python train.py --fold 1   # 单折
  python train.py --wait-gpu # 轮询 GPU 空闲后启动
"""
from __future__ import annotations

import os
import sys
import argparse
import time
import math
from loguru import logger
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# 自包含数据管线（替代云端 train_v11_wf / train_v7_lite_wf / train_v6_wf）
from train_data import (  # noqa: E402
    V11_TRAIN_DAYS, V11_STEP_DAYS, V11_MAX_FOLDS,
    compute_market_state_from_data, compute_rank_ic,
    generate_walk_forward_splits, load_stock_cache, build_fold_data,
)
from model import StudentModel  # noqa: E402  本地 StudentModel
V11_6Student = StudentModel  # 兼容 archive 命名
from losses import (  # noqa: E402
    v11_6_distill_components,
    FAMO,
)
from calibration import TemporalConformalCalibrator  # noqa: E402
# 可选 NaN 诊断 (默认关闭; 自包含实现, 无内部依赖)
def diagnose_first_nan(s_out, losses, yb, i, ep):
    bad = [j for j, l in enumerate(losses) if not torch.isfinite(l).all()]
    print(f"[DIAG-NaN] ep={ep} batch@{i} non-finite losses: {bad}")
_DIAG_NAN_AVAILABLE = True
DIAGNOSE_NAN = False

DIAG_NAN_FLAG = [False]  # 全局可变 flag, 首次 NaN 后锁死


# ---- SAM (Sharpness-Aware Minimization, Forem et al., ICLR'21, arXiv:2010.01412) ----
# 训练期优化器, 寻找平坦极小 -> 更好 OOS 泛化 (Rank IC / DA)。纯训练侧, 不影响导出权重, ONNX 安全。
class SAM(torch.optim.Optimizer):
    """SAM: 在权重空间爬到局部最大(loss 尖锐方向)再回退更新, 等价于最小化邻域内最差损失。"""

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def _grad_norm(self):
        # 跨所有参数组的梯度全局 L2 范数 (各参数 grad 展平后拼接再求范数)
        grads = [p.grad.detach().view(-1) for group in self.param_groups
                 for p in group["params"] if p.grad is not None]
        if len(grads) == 0:
            return torch.zeros((), device=self.param_groups[0]["params"][0].device)
        return torch.norm(torch.cat(grads), 2)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        scale = self.param_groups[0]["rho"] / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state.setdefault(p, {})["old_p"] = p.detach().clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # 爬到局部最大
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # 还原到原始权重
        self.base_optimizer.step()               # 用扰动处的梯度做真实更新
        if zero_grad:
            self.base_optimizer.zero_grad()


V11_6_SAVE_PATH = Path(_BACKEND_DIR) / "models" / "fincast_v11_6.pth"
# [加速] 训练/验证集降采样上限, 防止单折训练过久 (RTX3060 6GB 实测 val 29万拖垮单折)
TRAIN_CAP = 100_000   # 训练样本上限 (与主循环内部 _cap 一致, 可 --train-cap 覆盖)
VAL_CAP = 40_000      # 验证样本上限 (解决 val 29万每2epoch拖慢单折, 可 --val-cap 覆盖)

# 蒸馏超参
D_MODEL = 64                  # 控参数量 (<100K, 适配 6GB GPU)
N_COMPONENTS = 3
USE_LINEAR_RETENTION = False
USE_GATED_DELTANET = True     # 用 GatedDeltaNet 替换 SSM/LinearRetention
GDN_EXPAND = 1                # 控参数量(=1)
USE_AE = True                 # 范式1: JS 自编码器表征
USE_CROSS_NET = True          # 范式2: FinStack 可微交叉
USE_STATIC_BRANCH = True      # 范式3: 第三个静态 MLP 基模型
AE_W = 0.3                    # AE 自监督重建损失权重
DISTILL_EPOCHS = 40          # 余弦退火 + warmup, 配早停
DISTILL_BATCH = 256          # 学生很小, 大 batch 反更快
USE_DIR_CALIB = False         # P1: 方向校准概率头 (默认 OFF; 开启则 FAMO 增为 10 路)
USE_AMP = False              # AMP: _assoc_scan 自定义算子不兼容 autocast. DCU 无 tensor core, 收益有限.
DISTILL_LR = 5e-4              # [opt] base 模型小(2.4M), LR 过高致震荡, 降至 5e-4 更稳
GRAD_CLIP_NORM = 10.0         # 梯度裁剪 (SSM 标准训练技术, 防 GatedDeltaNet 状态爆炸)
WARMUP_EPOCHS = 5             # [opt] 线性 warmup, 配合低 LR 确保平稳启动
LR_MIN_RATIO = 0.01           # 余弦退火到 LR_MIN_RATIO * DISTILL_LR
EARLY_STOP_PATIENCE = 2       # val_IC 早停耐心 (用户要求加速, 连续2轮未改善即停, 约省1-2epoch/折)
SAM_START_EPOCH = 10          # [SAM-late] SAM 仅在最后 40/50 轮启用 (Zhou+ ICLR'25 Spotlight)
MIXUP_ALPHA = 0.4            # [v4.2 OFF] Mixup 混样(跨股票混合60×28)产生分布外样本
USE_MIXUP = False             # 导致 BN+Swish 放大异常值 → NaN; 关掉从源头防 NaN
FAMO_ETA = 0.5               # FAMO 自适应速率
W_CONTRAST = 0.2             # SCARF 对比自监督损失权重
USE_SAM = True                # SAM (ICLR'21) 平坦极小 -> 更好 OOS 泛化 (实测 val_IC 从 -0.03 拉到 +0.03)
SAM_RHO = 0.01                # SAM 扰动半径 (base 模型 + GatedDeltaNet: 0.05 致 NaN, 降 5×)
# [COLLAPSE-FIX 2026-07-10 v2.5.2] v2.5 ep2 pstd=0.0012 仍偏低 -> 提高方差正则力度
#   ep1 pstd=0.0081 (从 0.0001 破开) -> ep2 pstd=0.0012 (又萎缩).
#   诊断: var-reg(W=0.05) 仅占总 loss 0.3%, 不足以对抗 reg MSE 的常数吸引子.
#   修复: W=0.05->0.30(6x), MARGIN=0.10->0.15(给更多空间).
#         也防止 point_head 权重衰减到 0 (AdamW weight_decay=0.01 长期也会拖低).
#   现象回顾:
#     v1(z-score y + point 方差正则)        -> epoch1 即 pstd=0 塌死: 因编码器同时被 aux 头拖塌,
#                                               方差正则无信号可放大, 常数处梯度消失, 无解。
#     v2(仅 aux 头 detach)                  -> f_hstd=0.34 编码器健康, 但塌缩下移到 point 路径
#                                               (h_hstd=0.04, pstd=0.0001): 即"常数不动点"——
#                                               常数 point 下 ic/sign/focal_rank/ic_da 梯度全=0,
#                                               仅 reg MSE 生效, 把 point 拉向 mean(y)≈0。
#   根因精炼: 塌缩有两层 —— ① 编码器被 aux 常数吸引子拖塌(已用 v2 detach 解决);
#                           ② point 目标函数的常数不动点(reg 独占梯度, 排序损失在常数处梯度消失)。
#   v2.5 修复(组合, 非试错):
#     1) DETACH_AUX_HEADS=True  —— 保住编码器方差(f_hstd 健康)。
#     2) ZSCORE_Y_PER_DAY=True  —— 目标逐交易日横截面 z-score(std≈1)。使 reg MSE 的目标本身
#                                   有方差, reg 梯度把 point 拉向"有方差的 y"而非常数 mean。
#     3) POINT_VAR_W=0.05, MARGIN=0.10 —— 温和方差正则(仅当 std<0.10 激活), 作为不动点安全网:
#                                   常数 point 处该 loss 梯度巨大(∝1/std), 直接把 point 推开,
#                                   且编码器现已能提供放大所需的信号。
# [COLLAPSE-FIX 2026-07-10 v3] 结构级修复: LayerNorm 前 point_head + 温和正则
#   诊断总结(5轮迭代):
#     v1(v2.5.3及之前): 所有基于正则/梯度的修复(weight norm/var-reg)在epoch 2后失败,
#       因为常数不动点处总损失 < 正常训练9路损失总和, 优化器选常数。
#   根因: point_head 输入 h_h 方差塌缩 → point_head 权重 decay 归零 → 常数锁死。
#   v3 结构修复: 在 point_head 前加 LayerNorm, 确保输入始终单位方差,
#       即使上游 head_hidden 塌缩, point_head 也有标准化输入, 权重永不归零。
#       正则保留但降回温和水平 (var-reg W=0.05, margin=0.10 作安全网)。
#   v3+BW 验证: 预期 epoch1-12 pstd 持续健康(>0.01), 不再 epoch 2 塌缩。
ZSCORE_Y_PER_DAY = True      
POINT_VAR_MARGIN = 0.0       # [opt] 归一化解耦后 pstd=1.0 构造保证, 方差正则不需要
POINT_VAR_W = 0.0            # [opt] 移除死代码, 损失可减 ~0.3% 无影响
DETACH_AUX_HEADS = True      
# 固定权重已废弃 -> 改用 FAMO 动态加权 (保留作参考/回退)
W_REG, W_SOFT, W_HINT, W_SIGN, W_QUANT, W_TRUE = 1.0, 1.0, 0.5, 0.5, 0.5, 0.2


# 单独控制 head 参数 (point_head + point_scale): 跳过梯度裁剪 + 更高 LR
HEAD_LR_MULT = 1.0            # [v5] 排序/幅度解耦后不再需要 head 高 LR

# (原 _norm_teacher_info / load_teacher / teacher_targets 已删)


def _clip_grads(student, norm, exclude_head=True):
    """梯度裁剪: 默认跳过 point_head/point_scale (1153 参数, 占 43.9M 的 0.0026%).
    被全局梯度裁剪严重稀释 → 每步有效 LR 仅大模块 1/100 → ep2 权重归零.
    跳过裁剪让 head 用原始梯度更新, 保持输出幅度正常。"""
    if not exclude_head:
        return torch.nn.utils.clip_grad_norm_(student.parameters(), norm)
    head_names = {'point_head.weight', 'point_head.bias'}
    saved = {}
    for n, p in student.named_parameters():
        if n in head_names and p.grad is not None:
            saved[n] = p.grad.clone()
    ret = torch.nn.utils.clip_grad_norm_(student.parameters(), norm)
    for n, p in student.named_parameters():
        if n in saved:
            p.grad = saved[n]
    return ret


def _grad_ok(student) -> bool:
    """检查所有参数梯度是否不含 NaN/inf. 反向传播后调用, 防 MDN σ→0 梯度爆炸传染."""
    for p in student.parameters():
        if p.grad is not None and not p.grad.isfinite().all():
            return False
    return True


def _to_seq(X: torch.Tensor, seq_len: int, feat: int) -> torch.Tensor:
    """(B, D) 或 (B, seq_len, feat) -> (B, seq_len, feat); 截断或零填充到 seq_len*feat。"""
    if X.dim() == 3 and X.shape[1:] == (seq_len, feat):
        return X
    B = X.shape[0]
    if X.dim() == 3:
        X = X.reshape(B, -1)
    need = seq_len * feat
    if X.shape[-1] >= need:
        x = X[..., :need]
    else:
        pad = torch.zeros(B, need - X.shape[-1], device=X.device, dtype=X.dtype)
        x = torch.cat([X, pad], dim=-1)
    return x.reshape(B, seq_len, feat)


def _student_predict_chunks(student, Xv, auxv, msv, batch: int = 128):
    """分块前向, 避免大验证集 (Nv 可能达数十万) 一次性上 GPU 触发 OOM。
    ⚠️ batch 必须与训练一致(128): GatedDeltaNet 的 chunk_delta_rule 在 L222 一次性物化
    转移矩阵 A/kk, 形状 (b,c,M,h,p,p) 随 b 线性膨胀; b=2048 时单张 A≈27GB 直接 OOM。
    验证仅前向(无梯度/无SAM), b=128 显存与训练同档(~10GB), 分块循环开销可忽略。
    返回 (point[B], quantiles[B, Q]) 均在 CPU。"""
    student.eval()
    pts, quants = [], []
    n = Xv.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch):
            xb = Xv[i:i + batch]
            ab = auxv[i:i + batch] if auxv is not None else None
            mb = msv[i:i + batch]
            Xv_seq = _to_seq(xb, student.seq_len, student.feat_per_day)
            p, o = student(Xv_seq, ab, mb)
            pts.append(p.detach().cpu())
            quants.append(o["quantiles"].detach().cpu())
    return torch.cat(pts, dim=0), torch.cat(quants, dim=0)


def _zscore_y_per_day(y: "np.ndarray", dates) -> "np.ndarray":
    """逐交易日横截面 z-score (量化 ML 标准): 同日多股票收益归一化为秩信号。
    无跨日/跨样本泄漏 (仅用当日截面)。返回与 y 同形 float32。"""
    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    from collections import defaultdict
    groups = defaultdict(list)
    for i, d in enumerate(dates):
        groups[d].append(i)
    for d, idxs in groups.items():
        yi = y[idxs]
        m = yi.mean()
        s = yi.std()
        if s < 1e-6:
            out[idxs] = 0.0
        else:
            out[idxs] = (yi - m) / s
    return out.astype(np.float32)


def train_v11_6_fold(student, train_data, val_data, device, fold_idx, return_student=False):
    # [v7.2 data-clean] 训练数据可能含NaN/inf (来自CS缓存缺失值/技术指标除零等),
    # 在喂入GPU前过滤: 防止单个脏样本经GatedDeltaNet链式传染全折训练。
    # ⚠️ 必须同步过滤所有 list 字段 (dates 等), 否则后续 z-score/grouping 索引越界。
    _X_raw = np.asarray(train_data["X"], dtype=np.float32)
    _y_raw = np.asarray(train_data["y"], dtype=np.float32)
    _valid = np.isfinite(_X_raw).reshape(len(_X_raw), -1).all(axis=1) & np.isfinite(_y_raw)
    if not _valid.all():
        n_bad = (~_valid).sum()
        print(f"    [data-clean] fold {fold_idx} 移除 {n_bad}/{len(_valid)} 脏样本, 保留 {_valid.sum()}")
        train_data["X"] = _X_raw[_valid]; train_data["y"] = _y_raw[_valid]
        if train_data.get("aux") is not None:
            train_data["aux"] = np.asarray(train_data["aux"], dtype=np.float32)[_valid]
        # 同步过滤所有字段 (dates 可能是 list 或 ndarray)
        for _k in ["dates"]:
            if _k in train_data:
                _arr = np.asarray(train_data[_k])
                train_data[_k] = _arr[_valid] if isinstance(train_data[_k], np.ndarray) else [_arr[i] for i in range(len(_valid)) if _valid[i]]
    # [v7.4 subsample] 超大折随机降采样, 防止单折 18h 的训练瓶颈
    _n_tr = len(train_data["X"])
    _cap = TRAIN_CAP  # [加速] 可配 --train-cap (默认100K)
    if _n_tr > _cap:
        _rng = np.random.default_rng(42 + fold_idx)
        _idx = _rng.choice(_n_tr, _cap, replace=False)
        train_data["X"] = train_data["X"][_idx]
        train_data["y"] = train_data["y"][_idx]
        if train_data.get("aux") is not None:
            train_data["aux"] = train_data["aux"][_idx]
        # 同步子采样 list/array 字段
        for _k in ["dates"]:
            if _k in train_data:
                _arr = np.asarray(train_data[_k])
                train_data[_k] = _arr[_idx] if isinstance(train_data[_k], np.ndarray) else [_arr[i] for i in _idx]
        print(f"    [subsample] fold {fold_idx} {_n_tr} -> {_cap} 样本 (超限降采样)")
    X = torch.tensor(train_data["X"], dtype=torch.float32, device=device)
    y = torch.tensor(train_data["y"], dtype=torch.float32, device=device)
    aux = torch.tensor(train_data["aux"], dtype=torch.float32, device=device) if train_data.get("aux") is not None else None
    ms_global = torch.tensor(compute_market_state_from_data(train_data), dtype=torch.float32, device=device)  # (3,) 全局市场状态
    ms = ms_global.unsqueeze(0).expand(X.shape[0], -1).contiguous()  # (N,3) 每样本同享全局市场状态
    _Xv_raw = np.asarray(val_data["X"], dtype=np.float32)
    _yv_raw = np.asarray(val_data["y"], dtype=np.float32)
    _val_valid = np.isfinite(_Xv_raw).reshape(len(_Xv_raw), -1).all(axis=1) & np.isfinite(_yv_raw)
    if not _val_valid.all():
        print(f"    [data-clean] fold {fold_idx} 验证集移除 {(~_val_valid).sum()}/{len(_val_valid)} 脏样本")
        val_data["X"] = _Xv_raw[_val_valid]; val_data["y"] = _yv_raw[_val_valid]
        if val_data.get("aux") is not None:
            val_data["aux"] = np.asarray(val_data["aux"], dtype=np.float32)[_val_valid]
        # 同步过滤 val dates
        for _k in ["dates"]:
            if _k in val_data:
                _arr = np.asarray(val_data[_k])
                val_data[_k] = _arr[_val_valid] if isinstance(val_data[_k], np.ndarray) else [_arr[i] for i in range(len(_val_valid)) if _val_valid[i]]
    # [加速] 验证集降采样 (与 train 对称, 防止 29万 val 拖垮单折)
    _n_v = len(val_data["X"])
    if _n_v > VAL_CAP:
        _rngv = np.random.default_rng(99 + fold_idx)
        _idxv = _rngv.choice(_n_v, VAL_CAP, replace=False)
        val_data["X"] = np.asarray(val_data["X"], dtype=np.float32)[_idxv]
        val_data["y"] = np.asarray(val_data["y"], dtype=np.float32)[_idxv]
        if val_data.get("aux") is not None:
            val_data["aux"] = np.asarray(val_data["aux"], dtype=np.float32)[_idxv]
        for _k in ["dates"]:
            if _k in val_data:
                _arr = np.asarray(val_data[_k])
                val_data[_k] = _arr[_idxv] if isinstance(val_data[_k], np.ndarray) else [_arr[i] for i in _idxv]
        print(f"    [subsample-val] fold {fold_idx} {_n_v} -> {VAL_CAP} 样本")
    Xv = torch.tensor(val_data["X"], dtype=torch.float32, device=device)
    yv = torch.tensor(val_data["y"], dtype=torch.float32, device=device)
    auxv = torch.tensor(val_data["aux"], dtype=torch.float32, device=device) if val_data.get("aux") is not None else None
    msv_global = torch.tensor(compute_market_state_from_data(val_data), dtype=torch.float32, device=device)  # (3,)
    print(f"[mem] fold {fold_idx} 训练集 n={len(X)} 已分配显存={torch.cuda.memory_allocated()/1024**3:.1f}GB DISTILL_BATCH={DISTILL_BATCH}")
    msv = msv_global.unsqueeze(0).expand(Xv.shape[0], -1).contiguous()  # (Nv,3)

    # [COLLAPSE-FIX] 目标 y 逐交易日横截面 z-score (与 IC 度量一致, 消除"预测绝对均值"退化解)
    if ZSCORE_Y_PER_DAY:
        y = torch.tensor(_zscore_y_per_day(train_data["y"], train_data["dates"]),
                         dtype=torch.float32, device=device)
        yv = torch.tensor(_zscore_y_per_day(val_data["y"], val_data["dates"]),
                          dtype=torch.float32, device=device)
        print(f"[zscore-y] 目标已按交易日横截面 z-score (train n={len(y)} val n={len(yv)})")

    # 动态多目标加权 (8 路真实任务损失, FAMO, 去掉 zero-grad sign)
    famo = FAMO(n_tasks=9 if USE_DIR_CALIB else 8, eta=FAMO_ETA)
    # 时序共形校准器 (val 即校准集)
    calibrator = TemporalConformalCalibrator(target_coverage=0.90, window_size=500)

    # [pstd-fix v4] 分离 head 参数组: 零权重衰减 + 跳过梯度裁剪
    head_names = {'point_head.weight', 'point_head.bias'}
    _h = [p for n, p in student.named_parameters() if n in head_names]
    _b = [p for n, p in student.named_parameters() if n not in head_names]
    opt = torch.optim.AdamW([
        {'params': _b, 'lr': DISTILL_LR, 'weight_decay': 1e-4},
        {'params': _h, 'lr': DISTILL_LR * HEAD_LR_MULT, 'weight_decay': 0.0},
    ])
    if USE_SAM:
        opt = SAM([
            {'params': _b, 'lr': DISTILL_LR, 'weight_decay': 1e-4},
            {'params': _h, 'lr': DISTILL_LR * HEAD_LR_MULT, 'weight_decay': 0.0},
        ], torch.optim.AdamW, rho=SAM_RHO)
    # AMP 混合精度缩放器 (BWX 支持, 约 1.6× 提速; 默认 OFF 不影响当前训练)
    _scaler = torch.cuda.amp.GradScaler(init_scale=128) if USE_AMP else None  # 调低 init_scale 适配本模型 loss 量级
    # 余弦退火 + 线性 warmup 学习率调度
    def _lr_lambda(ep: int) -> float:
        if ep < WARMUP_EPOCHS:
            return (ep + 1) / max(WARMUP_EPOCHS, 1)
        prog = (ep - WARMUP_EPOCHS) / max(DISTILL_EPOCHS - WARMUP_EPOCHS, 1)
        prog = min(max(prog, 0.0), 1.0)
        return LR_MIN_RATIO + (1.0 - LR_MIN_RATIO) * 0.5 * (1.0 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    swa_model = AveragedModel(student)   # SWA: 权值 EMA 平均, 提升泛化与稳定性
    n = len(X)
    best_ic = -999.0
    best_state = None
    best_ep = 0
    best_path = f"data/v116_fold{fold_idx}_best.pth"   # 磁盘 best 检查点, 防 kill 丢成果
    epochs_since_best = 0
    cov_history = []
    step_no = 0
    step_win = []
    _last_f_hstd = float("nan")     # [COLLAPSE-FIX v2] 共享特征 final_h 批内 std (监控塌缩)
    _last_h_hstd = float("nan")     # head_hidden 输出批内 std
    for ep in range(1, DISTILL_EPOCHS + 1):
        # [SAM-late] ICLR'25: 仅最后 ~80% 轮启用 SAM, 前半程用单 pass AdamW 加速
        sam_active = USE_SAM and (ep >= SAM_START_EPOCH)
        if USE_SAM:
            # rho=0 → first_step 无扰动, 等效纯 AdamW; rho>0 → SAM 双 pass
            for pg in opt.param_groups:
                pg["rho"] = SAM_RHO if sam_active else 0.0
        student.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, DISTILL_BATCH):
            idx = perm[i:i + DISTILL_BATCH]
            ab = aux[idx] if aux is not None else None
            mb = ms[idx]
            _t0 = time.time()
            # ---- Mixup 数据增强 (ICLR'18): 混合样本特征与标签 ----
            if USE_MIXUP and X.size(0) > 1:
                lam = float(torch.distributions.Beta(MIXUP_ALPHA, MIXUP_ALPHA)
                            .sample().clamp(0.1, 0.9))
                perm2 = torch.randperm(X.size(0), device=device)[:idx.size(0)]
                xb = lam * X[idx] + (1.0 - lam) * X[perm2]
                yb = lam * y[idx] + (1.0 - lam) * y[perm2]
                if ab is not None:
                    ab = lam * ab + (1.0 - lam) * aux[perm2]
                mb = lam * mb + (1.0 - lam) * ms[perm2]
            else:
                xb, yb = X[idx], y[idx]
            # v11.6 自包含: 学生直接吃原始时序 X_seq, 不加载教师
            X_seq = _to_seq(xb, student.seq_len, student.feat_per_day)  # 必须用批次 xb, 非全量 X
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    sp, s_out = student(X_seq, ab, mb)
            else:
                sp, s_out = student(X_seq, ab, mb)
            # [COLLAPSE-FIX v2] 记录共享特征方差 (早于损失计算, 任何分支都抓得到)
            if "f_hstd" in s_out:
                _last_f_hstd = float(s_out["f_hstd"])
                _last_h_hstd = float(s_out["h_hstd"])
            # [DIAG-NaN] 精准定位 NaN 源头: 前向输出 vs 损失分量
            if _DIAG_NAN_AVAILABLE and DIAGNOSE_NAN and not DIAG_NAN_FLAG[0] and not torch.isfinite(s_out.get("point", torch.tensor(0.0))).all():
                print(f"[DIAG-NaN] ep={ep} batch@{i} FORWARD point has NaN, skipping loss-level diag")
                DIAG_NAN_FLAG[0] = True
            # FAMO 动态加权: 取 9 路真实任务损失 -> 自适应权重 -> 加权总损失
            losses, comp = v11_6_distill_components(s_out, yb, student.mdn_head, use_dir_calib=USE_DIR_CALIB)
            if _DIAG_NAN_AVAILABLE and DIAGNOSE_NAN and not DIAG_NAN_FLAG[0]:
                if not all(torch.isfinite(l).all() for l in losses):
                    diagnose_first_nan(s_out, losses, yb, i, ep)
                    DIAG_NAN_FLAG[0] = True
            w = famo.update(losses)
            total = famo.weighted_loss(losses, w)
            # 范式1: AE 自监督重建损失 (从学生 forward 输出取出)
            total = total + AE_W * s_out["ae_loss"]
            # SCARF 对比自监督损失 (训练时由 _corrupt 双视图产生, 评测时为 0)
            cl = s_out.get("contrast_loss", None)
            if cl is not None:
                total = total + W_CONTRAST * cl
            # [COLLAPSE-FIX v2] point 头方差正则 (POINT_VAR_W>0 才启用)
            if POINT_VAR_W > 0:
                _p = s_out["point"]
                _p_std = _p.std()
                l_var = torch.relu(torch.tensor(POINT_VAR_MARGIN, device=_p.device, dtype=_p.dtype) - _p_std)
                total = total + POINT_VAR_W * l_var
            # ⚠️ NaN/inf 防护 (2026-07-10): 任一步总损失非有限则跳过该步更新,
            # 避免污染模型权重导致全盘发散 (实测 epoch2 因 MDN σ→0 / ic 分母→0 触发首例 NaN)。
            if not torch.isfinite(total):
                print(f"    [nan-guard] batch@{i} total 非有限, 跳过更新")
                opt.zero_grad()
                continue
            if sam_active:
                # ---- SAM (ICLR'21 + ICLR'25 SAM-late): 双 pass 平坦极小 ----
                opt.zero_grad()
                if _scaler is not None:
                    _scaler.scale(total).backward()
                    _scaler.unscale_(opt)
                else:
                    total.backward()
                _clip_grads(student, GRAD_CLIP_NORM)
                if not _grad_ok(student): opt.zero_grad(); continue  # [v7.3]
                opt.first_step(zero_grad=True)
                # 在扰动权重上重算 (教师冻结, 仅重跑学生前向); FAMO 权重 w 保持第一次结果
                if USE_AMP:
                    with torch.cuda.amp.autocast():
                        _, s_out2 = student(X_seq, ab, mb)
                else:
                    _, s_out2 = student(X_seq, ab, mb)
                losses2, _ = v11_6_distill_components(s_out2, yb, student.mdn_head, use_dir_calib=USE_DIR_CALIB)
                total2 = famo.weighted_loss(losses2, w) + AE_W * s_out2["ae_loss"]
                cl2 = s_out2.get("contrast_loss", None)
                if cl2 is not None:
                    total2 = total2 + W_CONTRAST * cl2
                # [COLLAPSE-FIX v2.5] SAM 第二跳同样加方差正则 (保持一致)
                if POINT_VAR_W > 0:
                    _p2 = s_out2["point"]
                    _p_std2 = _p2.std()
                    l_var2 = torch.relu(torch.tensor(POINT_VAR_MARGIN, device=_p2.device, dtype=_p2.dtype) - _p_std2)
                    total2 = total2 + POINT_VAR_W * l_var2
                if not torch.isfinite(total2):
                    # 扰动前向 NaN: 还原权重且不更新 (防中毒), 不污染
                    opt.zero_grad()
                    opt.second_step(zero_grad=True)
                    continue
                opt.zero_grad()
                if _scaler is not None:
                    _scaler.scale(total2).backward()
                    _scaler.unscale_(opt)
                else:
                    total2.backward()
                _clip_grads(student, GRAD_CLIP_NORM)
                if not _grad_ok(student): opt.zero_grad(); opt.second_step(zero_grad=True); continue  # [v7.3] 清NaN梯度+还原权重,不更新
                opt.second_step(zero_grad=True)   # 还原权重并用扰动处梯度做真实更新
                if _scaler is not None:
                    _scaler.update()
            elif USE_SAM:
                # [SAM-late] pre-SAM 轮: SAM wrapper rho=0, 单 pass = 普通 AdamW step
                opt.zero_grad()
                if _scaler is not None:
                    _scaler.scale(total).backward()
                    _scaler.unscale_(opt)
                else:
                    total.backward()
                _clip_grads(student, GRAD_CLIP_NORM)
                if not _grad_ok(student): opt.zero_grad(); continue  # [v7.3] 梯度爆炸拦截
                opt.first_step(zero_grad=False)  # rho=0 无扰动 + 保梯度
                opt.second_step(zero_grad=True)  # 用保存梯度执行 base AdamW step
                if _scaler is not None:
                    _scaler.update()
            else:
                opt.zero_grad()
                if _scaler is not None:
                    _scaler.scale(total).backward()
                    _scaler.unscale_(opt)
                else:
                    total.backward()
                _clip_grads(student, GRAD_CLIP_NORM)
                if not _grad_ok(student): opt.zero_grad(); continue  # [v7.3]
                if _scaler is not None:
                    _scaler.step(opt)
                    _scaler.update()
                else:
                    opt.step()
            _dt = time.time() - _t0
            step_no += 1
            step_win.append(_dt)
            if step_no % 20 == 0:
                _avg = sum(step_win[-20:]) / 20.0
                print(f"    [step] #{step_no} avg={_avg:.2f}s last={_dt:.2f}s")
        scheduler.step()                       # 余弦退火 + warmup
        swa_model.update_parameters(student)   # 每个 epoch 更新 SWA EMA
        # 验证: 每 2 epoch 一次 (加速: 验证集 29.4万样本, 耗时与训练相当)
        if ep % 2 == 1:
            sv_point, sv_quant = _student_predict_chunks(student, Xv, auxv, msv)
            # [第九轮审计修复] 原始: compute_rank_ic(sv_point.numpy(), yv.cpu().numpy())
            #   修改: 传入 val_data["dates"] 按交易日分组算截面 IC 再均值。
            #   原实现退化为整体 Spearman, 对 panel 数据 (多日多股) IC 估计有偏 (跨日混叠)。
            val_ic = compute_rank_ic(sv_point.numpy(), yv.cpu().numpy(), dates=val_data.get("dates"))
            calibrator.update(yv.cpu(), sv_quant)
            _, _, cov = calibrator.calibrate({"quantiles": sv_quant})
            cov_history.append(cov)
            # 早停: val_IC 连续若干轮无提升则终止 (保留 best_state)
            if val_ic > best_ic + 1e-5:
                epochs_since_best = 0
            else:
                epochs_since_best += 1
                if epochs_since_best >= EARLY_STOP_PATIENCE:
                    print(f"    [early-stop] ep {ep} val_IC 连续 {EARLY_STOP_PATIENCE} 轮未提升, 终止训练")
                    break
            if val_ic > best_ic:
                best_ic = val_ic
                best_state = {k: v.clone() for k, v in student.state_dict().items()}
                best_ep = ep
                torch.save(best_state, best_path)   # 磁盘存盘, 防进程被杀丢成果
            wd = famo.weight_dict()
            _pstd = float(sv_point.std())
            print(f"    ep {ep:2d} loss={total.item():.4f} val_IC={val_ic:+.5f} pstd={_pstd:.4f} "
                  f"cov={cov:.3f} f_hstd={_last_f_hstd:.4f} h_hstd={_last_h_hstd:.4f} "
                  f"W={wd} (best={best_ic:+.4f})")
        else:
            wd = famo.weight_dict()
            print(f"    ep {ep:2d} loss={total.item():.4f} (skip val)")
    # 最终模型: 优先 SWA (EMA 平均, 泛化更稳), 但其含 NaN 则回退 best_state
    swa_sd = swa_model.module.state_dict()
    if best_state is not None:
        _swa_finite = all(torch.isfinite(v).all() for v in swa_sd.values())
        if _swa_finite:
            student.load_state_dict(swa_sd)
            print(f"    [ckpt] 最终用 SWA 权重 (best ep{best_ep} val_IC={best_ic:+.4f})")
        else:
            student.load_state_dict(best_state)
            print(f"    [ckpt] SWA 含 NaN, 回退 best_state (ep{best_ep} val_IC={best_ic:+.4f})")
    else:
        student.load_state_dict(swa_sd)
    student.eval()
    svp, svq = _student_predict_chunks(student, Xv, auxv, msv)
    # [第九轮审计修复] 同上: 传入 dates 算截面 IC
    swa_ic = compute_rank_ic(svp.numpy(), yv.cpu().numpy(), dates=val_data.get("dates"))
    mean_cov = float(np.mean(cov_history)) if cov_history else float("nan")
    if return_student:
        return best_ic, mean_cov, famo.weight_dict(), swa_ic, student
    return best_ic, mean_cov, famo.weight_dict(), swa_ic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--size", type=str, default="tiny", choices=["tiny", "base", "large", "xl", "xxl"],
                    help="容量档: tiny(94K,RTX3060)/base(~2M,A800)/large(~8M,A800)/xl(~22M,BW)/xxl(~60M,BW顶配)")
    ap.add_argument("--wait-gpu", action="store_true", help="轮询直到 v11.5 训练释放 GPU")
    ap.add_argument("--wait-ckpt", action="store_true", help="(已废弃) v11.6 自包含, 不再等待 v11.5 教师")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU 训练 (学生很小)")
    ap.add_argument("--batch", type=int, default=None,
                    help="覆盖 DISTILL_BATCH (xxl 在 BWX 64GB 下建议 128 以留出 delta-rule 并行 scan 显存)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="覆盖 DISTILL_EPOCHS (冒烟测试用, 默认 40)")
    # Ablation flags
    ap.add_argument("--no-detach", action="store_true", help="消融: 关 Detach-Aux-Heads")
    ap.add_argument("--no-zscore", action="store_true", help="消融: 关逐日 Z-score y")
    ap.add_argument("--no-ae", action="store_true", help="消融: 关 AutoEncoder")
    ap.add_argument("--no-cross", action="store_true", help="消融: 关 CrossNet")
    ap.add_argument("--no-sam", action="store_true", help="消融: 关 SAM")
    ap.add_argument("--no-contrast", action="store_true", help="消融: 关 SCARF 对比学习")
    ap.add_argument("--max-folds", type=int, default=V11_MAX_FOLDS,
                    help=f"最多折数 (默认={V11_MAX_FOLDS}, 加速用5)")
    ap.add_argument("--n-stocks", type=int, default=5000,
                    help="加载股票数 (默认5000; 内存受限时调小如 800 防 build_fold_data OOM)")
    ap.add_argument("--train-cap", type=int, default=100_000,
                    help="训练样本上限 (默认100K; 加速可设小如 50000, 质量优先可设大)")
    ap.add_argument("--val-cap", type=int, default=40_000,
                    help="验证样本上限 (默认40K; 解决 val 29万拖慢单折, 加速可设小)")
    ap.add_argument("--chunk-size", type=int, default=64,
                    help="GatedDeltaNet scan chunk size (默认64, 加速用32可翻倍batch)")
    args = ap.parse_args()

    # 覆盖训练 batch: xxl 基座大, b=256 时 delta-rule 并行 scan 物化 A/C 会 OOM, 降到 128
    if args.batch is not None:
        global DISTILL_BATCH
        DISTILL_BATCH = max(1, args.batch)
    if args.epochs is not None:
        global DISTILL_EPOCHS
        DISTILL_EPOCHS = max(1, args.epochs)
    print(f"[batch] DISTILL_BATCH={DISTILL_BATCH} (args.batch={args.batch}) epochs={DISTILL_EPOCHS}")
    global TRAIN_CAP, VAL_CAP
    TRAIN_CAP = args.train_cap
    VAL_CAP = args.val_cap
    print(f"[cap] TRAIN_CAP={TRAIN_CAP} VAL_CAP={VAL_CAP}")
    # Apply ablation overrides
    _abl = []
    if args.no_detach: global DETACH_AUX_HEADS; DETACH_AUX_HEADS = False; _abl.append("no_detach")
    if args.no_zscore: global ZSCORE_Y_PER_DAY; ZSCORE_Y_PER_DAY = False; _abl.append("no_zscore")
    if args.no_ae: global USE_AE, AE_W; USE_AE = False; AE_W = 0.0; _abl.append("no_ae")
    if args.no_cross: global USE_CROSS_NET; USE_CROSS_NET = False; _abl.append("no_cross")
    if args.no_sam: global USE_SAM; USE_SAM = False; _abl.append("no_sam")
    if args.no_contrast: global W_CONTRAST; W_CONTRAST = 0.0; _abl.append("no_contrast")
    if _abl: print(f"[ablation] active: {_abl}")

    # 单实例锁: 防止重复启动(自动化/并发/双进程)导致双 GPU 训练
    import atexit
    _LOCK = Path(_BACKEND_DIR) / "data" / "v116_train.lock"
    try:
        _lfd = os.open(str(_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"[guard] 检测到 v11.6 训练锁 {_LOCK}, 已有实例在运行, 退出以避免重复训练")
        return
    def _release_lock():
        try: os.close(_lfd)
        except Exception as e: logger.warning(f"[V11.6] 释放锁fd失败: {e}")
        try: os.unlink(str(_LOCK))
        except Exception as e: logger.warning(f"[V11.6] 删除锁文件失败: {e}")
    atexit.register(_release_lock)

    # v11.6 自包含: 不等待/不加载 v11.5 教师检查点, 直接面向真实任务训练
    if args.wait_gpu:
        import psutil
        TEACHER_MARKER = "train_v11_wf.py"   # v11.5 教师脚本(含 .py), 精确匹配避免误伤学生
        def _teacher_alive():
            for p in psutil.process_iter(["pid", "cmdline"]):
                try:
                    cmd = p.info.get("cmdline") or []
                    if any(TEACHER_MARKER in (c or "") for c in cmd):
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            return False
        print(f"[guard] 等待 v11.5 教师进程({TEACHER_MARKER})完全退出以释放 GPU...")
        # 关键: 必须等【教师进程整体退出】(顺序训练多折, 折间有瞬时空闲窗口会误判),
        #       不能只看瞬时空闲显存, 否则会抢 GPU 与教师并发 -> OOM/争用。
        while True:
            if _teacher_alive():
                time.sleep(30)
                continue
            # 教师进程已退; 若 CUDA 可用再确认设备级空闲显存充足(防其他进程残留占用)
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info()   # 字节; 设备级(全局)含所有进程
                if free <= 4.0 * 1024 ** 3:
                    time.sleep(30)
                    continue
            break
        print("[guard] v11.5 教师已退出且 GPU 空闲, 启动 v11.6 自包含训练")
    device = "cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda"

    V11_6_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # v11.6 自包含: 不加载 v11.5 教师

    # CS-Rank 跨截面排名缓存 (8 维/天): 可选，缺失时特征置 0（与后端兜底一致）
    cs_cache = {}
    try:
        _cs_path = Path(os.environ.get("REPFIX_DATA_DIR", "data")) / "cs_rank_cache.npz"
        if _cs_path.exists():
            _cs = np.load(str(_cs_path))
            cs_cache = {k: _cs[k] for k in _cs.files}
            print(f"[CS-Rank] 加载预计算缓存: {len(cs_cache)} 股 (X 维度=1680)")
    except Exception as e:
        print(f"[CS-Rank] 缓存不可用(失败开放): {e}")

    _DATA_CACHE = Path(os.environ.get("REPFIX_DATA_DIR", "data")) / "stock_data_cache.pkl"
    if _DATA_CACHE.exists():
        print(f"[cache] 加载缓存股票数据: {_DATA_CACHE}")
        stock_data = pickle.loads(_DATA_CACHE.read_bytes())
        if not isinstance(stock_data, dict):
            print(f"[cache] 缓存格式异常，走缓存子集")
            stock_data = load_stock_cache(str(_DATA_CACHE))
    else:
        stock_data = load_stock_cache(str(_DATA_CACHE))
        if not stock_data:
            print("[cache] 无数据缓存，无法训练（需云端数据管线）")
            return 1
        _DATA_CACHE.write_bytes(pickle.dumps(stock_data))
        print(f"[cache] 缓存股票数据至 {_DATA_CACHE}")
    all_dates = sorted(set(d for s in stock_data.values() for d in s["dates"]))
    splits, _ = generate_walk_forward_splits(
        all_dates, train_days=V11_TRAIN_DAYS, step_days=V11_STEP_DAYS, max_folds=args.max_folds)

    fold_ics = []
    for fi, info in enumerate(splits):
        if args.fold > 0 and (fi + 1) != args.fold:
            continue
        tr, va, te = build_fold_data(
            stock_data, cs_cache, all_dates,
            info["train_start"], info["train_end"],
            info["val_start"], info["val_end"],
            info["test_start"], info["test_end"])
        if tr is None or va is None:
            continue
        # [断点续训] 已完成折跳过, 防进程被回收后重训已完成的折
        _fold_done = V11_6_SAVE_PATH.parent / f"fincast_v11_6_fold{fi + 1}.pth"
        if _fold_done.exists() and _fold_done.stat().st_size > 5_000_000:
            print(f"  Fold {fi+1} 已完成 ({_fold_done.name}, {_fold_done.stat().st_size // 1024 // 1024}MB), 跳过")
            continue
        # tiny 保留旧显式容量(=94,135, 向后兼容); base/large 由 size 预设驱动
        _cap = (dict(d_model=D_MODEL, n_components=N_COMPONENTS, gdn_expand=GDN_EXPAND)
                if args.size == "tiny" else {})
        student = V11_6Student(size=args.size, chunk_size=args.chunk_size, **_cap,
                               use_linear_retention=USE_LINEAR_RETENTION,
                               use_gated_deltanet=USE_GATED_DELTANET,
                               use_ae=USE_AE, use_cross_net=USE_CROSS_NET,
                               use_static_branch=USE_STATIC_BRANCH,
                               use_bilinear=True, use_rcmoe=True, use_contrast=True,
                               use_dir_calib=USE_DIR_CALIB,
                               detach_aux_heads=DETACH_AUX_HEADS).to(device)
        ic, cov, famo_w, swa_ic = train_v11_6_fold(student, tr, va, device, fi)
        fold_ics.append(ic)
        fold_path = V11_6_SAVE_PATH.parent / f"fincast_v11_6_fold{fi + 1}.pth"
        torch.save({"fold": fi + 1, "size": args.size,
                    "model_state_dict": student.state_dict(),
                    "result": {"val_ic": ic,
                               "swa_val_ic": swa_ic,
                               "conformal_coverage": cov,
                               "famo_weights": famo_w}}, str(fold_path))
        print(f"  Fold {fi+1} val_IC={ic:+.5f} SWA_val_IC={swa_ic:+.5f} "
              f"conformal_cov={cov:.3f} -> {fold_path.name}")
    print(f"\n  v11.6 蒸馏完成, 均值 val_IC={(np.mean(fold_ics) if fold_ics else 0.0):+.4f}")

    # 结果聚合: 写入 JSON 供后续论文/分析用
    import json
    _abl_key = "_".join(_abl) if _abl else args.size
    _res_path = Path("data") / f"v116_results_{_abl_key}.json"
    _existing = {}
    if _res_path.exists():
        try:
            _existing = json.loads(_res_path.read_text())
        except Exception as e: logger.warning(f"读取已有结果文件失败 {_res_path}: {e}")
    _existing[str(args.fold) if args.fold > 0 else "all"] = {
        "size": args.size, "fold": args.fold, "ablations": _abl,
        "fold_ics": [round(float(x), 6) for x in fold_ics],
        "mean_ic": round(float(np.mean(fold_ics)), 6),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _res_path.write_text(json.dumps(_existing, indent=2))
    print(f"  [results] 保存至 {_res_path}")


if __name__ == "__main__":
    main()
