"""v11.6 蒸馏学生模型 (StudentModel)
====================================
目标: 比 v11.5 teacher (V11LiteModel) 更小、更快、显存友好，同时保留
      MDN 分布预测与方向能力，便于 ONNX(int8) 部署到模拟盘实时决策。

设计 (对照 teacher + 蒸馏顶级量化范式):
  teacher 重模块: MIMFlow(~大) + LightINFlow + 2x LightSSMv2 + MASTER Lite
                  + EIIL+HRM(对抗) + AERegime AE + AdaptiveRegimeHead(双路)
                  + SkewedTMDNHead(K=4) + GOLD(推理)

  student (本文件) 蒸馏三大顶级量化真实范式:
    范式1 Jane Street Market Prediction 冠军 = AutoEncoder+MLP:
        BN 预处理 + 高斯噪声增强 + Swish; 轻量 AE 生成新表征, 与原始 fused 拼接。
    范式2 FinStack-Net (ACM2025) = 层次特征交叉(HCFM) + 堆叠集成 + 元融合:
        DCN-v3 升级交叉网络(arXiv:2407.13349, KDD'24: Self-Mask 减半参 + 指数阶 Deep Crossing)+门控筛选(替代不可微互信息/Lasso);
        多异构 NN 基分支(A: GatedDeltaNet 时序 / B: 交叉MLP / C: 静态MLP)并行;
        可学习 softmax 元融合头(等价 LR 元学习器)。
    范式3 GBDT+LSTM hybrid: 时序(GatedDeltaNet) 与 表格交叉/静态(MLP) 分工互补。

  二次增强 (顶刊顶会精读, 2026-07-09 第三轮):
    - DCN-V3 (arXiv:2407.13349, KDD'24) 交叉升级: D/2 交叉向量 + Self-Mask 正则 + 指数阶反馈 → 已落地 CrossNet
    - TinyTimeMixer/TSMixer (arXiv:2401.03955, NeurIPS'24) 无注意力 patch MLP 混合 → 已落地 TSMixerBlock(时序分支前处理)
    - (已落地) FT-Transformer + iTransformer 反转注意力 (NeurIPS'21 / ICLR'24): branchC 将每个因子视为 token(feature-as-token), 手写多头注意力(纯 matmul/softmax) 建模跨因子依赖, 100% ONNX 可导出; 配 RevIN(ICLR'22) 校正分布漂移 -> 替换 branchC 静态 MLP
    - (候选未落地) DCN-V3 Tri-BCE 浅深双交叉辅助监督: 需训练侧浅/深交叉双分支改造

  保留: 自包含 RawTemporalEncoder(TSMixer 风格, 原始 X[1680]=60步×28因子 → fused[B,192], 不加载教师);
        SkewedTMDNHead(K=3); point_head; GatedDeltaNet 时序层; 种子集成推理接口。
  约束: 参数量 <100K, 适配 6GB GPU, CPU 可训可验证; 完全不依赖 v11.5 教师 ckpt。
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict
import math
import os
import torch
from torch import nn
import torch.nn.functional as F


def _scan_budget_gb() -> float:
    """delta-rule 并行 scan 允许的峰值内存预算(GB)。
    - 显式 env V11_6_SCAN_BUDGET_GB 优先;
    - GPU 可用 → 15GB (原值, 选大 M 提速);
    - CPU → 按可用系统内存的 35% 动态限制(psutil 可用时), 上限 1.5GB, 下限 0.3GB,
      使 M 自适应缩小(数值等价, 仅稍慢), 防止低内存机器 scan 峰值 OOM 段错误。
    """
    env = os.environ.get("V11_6_SCAN_BUDGET_GB")
    if env:
        try:
            return max(0.1, float(env))
        except ValueError:
            pass
    try:
        if torch.cuda.is_available():
            return 15.0
    except Exception:
        pass
    avail_gb = None
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        avail_gb = None
    if avail_gb is not None:
        # [PERF-v116] 上限死守 1.5GB -> 当前 watchlist(~83只) scan 峰值 @M=8 约 2.0GB,
        # 故默认恒选 M=4。注意: M=4<->M=8 **并非数值等价**——chunk_delta_rule 内
        # nan_to_num(A/C, posinf=1e4) 与关联扫描深度(log2 M)耦合, 不同 M 链式矩阵乘次数不同,
        # 钳制值被不同倍数放大(T=5 差~1e4, T=60 差~1e31)。故**严禁在修复该不稳定前自动升 M=8**,
        # 否则会改变相对排序预测。保持 M=4 与现行部署一致、安全。
        return float(min(1.5, max(0.3, avail_gb * 0.35)))
    return 1.0

from light_ssm_v2 import LightSSMv2
from skewed_t_mdn import SkewedTMDNHead

_DELTA_FIRST = False  # 一次性诊断打印开关


# ===== 新增通用组件: DropPath(随机深度) + RoPE(旋转位置编码) + CrossAttn =====

class DropPath(nn.Module):
    """Stochastic Depth (ICCV'17 / Deep Networks with Stochastic Depth).
    训练时以概率 drop_prob 随机丢弃整条路径; 推理恒等映射。
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        mask = x.new_empty(x.shape[0], 1, *([1]*(x.dim()-2))).bernoulli_(keep_prob)
        return x * mask / keep_prob


class RotaryEmbedding(nn.Module):
    """旋转位置编码 RoPE (NeurIPS'21 / LLaMA / GPT-NeoX).
    对每对 (d, d+1) 维度应用旋转矩阵, 无需可学习参数。
    """
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos = freqs.cos()   # (T, d/2)
        sin = freqs.sin()   # (T, d/2)
        x_rot = x[..., ::2], x[..., 1::2]  # (..., T, d/2) each
        x_out = torch.stack([x_rot[0]*cos - x_rot[1]*sin,
                              x_rot[0]*sin + x_rot[1]*cos], dim=-1)
        return x_out.flatten(-2)


class CrossAttention(nn.Module):
    """轻量多头交叉注意力: query 来自时序 fused, key/value 来自分支特征。
    Q/K/V 统一投影到 n_heads*d_head 同维度空间, 保证 QK^T 维度匹配。
    """
    def __init__(self, q_dim: int, kv_dim: int, n_heads: int = 4, d_head: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        d_model = n_heads * d_head
        self.wq = nn.Linear(q_dim, d_model, bias=False)
        self.wk = nn.Linear(kv_dim, d_model, bias=False)
        self.wv = nn.Linear(kv_dim, d_model, bias=False)
        self.out = nn.Linear(d_model, q_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, q, k, v):
        B = q.shape[0]
        Q = self.wq(q).reshape(B, -1, self.n_heads, self.d_head).transpose(1,2)  # (B,h,T,d)
        K = self.wk(k).reshape(B, -1, self.n_heads, self.d_head).transpose(1,2)
        V = self.wv(v).reshape(B, -1, self.n_heads, self.d_head).transpose(1,2)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2,-1)) / (self.d_head**0.5), dim=-1)
        out = torch.matmul(attn, V).transpose(1,2).reshape(B, -1, self.n_heads*self.d_head)
        out = self.out(out)  # (B, 1, q_dim)
        return self.drop(out[:, 0])


# ===== Existing Utilities (unchanged) =====

def _assoc_scan(A: torch.Tensor, C: torch.Tensor):
    """结合律并行前缀扫描 (G优先于 A 的半环): (A_t, C_t) 组合为
        (A_2, C_2) ∘ (A_1, C_1) = (A_2 @ A_1,  A_2 @ C_1 + C_2)
    使 state_l = (A_l ... A_1) S_0 + Σ_{j<=l} (A_l ... A_{j+1}) C_j。

    A, C: (b, L, h, p, p)   —— 块内位置维在 dim=1。返回 (cumA, cumC) 同形状。
    O(log L) 次大矩阵乘替代逐位置循环, 大幅减少细碎 kernel launch (壁仞 BWX 友好)。
    """
    L = A.shape[1]
    outA, outC = A.clone(), C.clone()
    step = 1
    while step < L:
        A_prev = torch.cat([outA[:, :step], outA[:, :-step]], dim=1)
        C_prev = torch.cat([outC[:, :step], outC[:, :-step]], dim=1)
        newA = torch.einsum("blhpq,blhqr->blhpr", outA, A_prev)
        newC = torch.einsum("blhpq,blhqr->blhpr", outA, C_prev) + outC
        outA = torch.cat([outA[:, :step], newA[:, step:]], dim=1)
        outC = torch.cat([outC[:, :step], newC[:, step:]], dim=1)
        step *= 2
    return outA, outC


def chunk_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    S0: torch.Tensor | None,
    chunk_size: int,
    mem_budget_gb: float = 8.0,
):
    """Chunkwise Gated Delta Rule — 并行 scan 版 (纯 PyTorch, 无 triton/cuda 扩展)。

    块内因果递推 (参考 arXiv:2412.06464 Gated DeltaNet + light_ssm_v2.ssd 块分解):
        S_t = α_t ⊙ S_{t-1} + β_t (v_t − S_{t-1} k_t) k_tᵀ
        o_t = S_t q_t
    其中 α_t ∈ (0,1) 元素门控(状态遗忘), β_t ∈ (0,1) delta 学习率, 状态 S∈ℝ^{P×P}。

    性能优化 (针对壁仞 BWX 等 launch-overhead 较高的设备):
    原实现为 `for ci ... for i ...` 双层 Python 循环, T=60 时单 forward 产生 ~256 次
    细碎 kernel launch, 在 BWX 上累积成灾难。这里改为:
      - 自适应内部 scan 块大小 M (把 A/C 物化显存压到 mem_budget_gb 内, 避免 OOM);
      - 块内用结合律并行前缀扫描 (associative scan) 以 O(log M) 次大矩阵乘替代逐位置循环;
      - 块间仅用少量 Python 迭代传递状态 S_prev。
    数值等价于原递推, autograd 通畅。

    Args:
        q,k,v: (B, T, H, P)
        beta : (B, T, H)  已 sigmoid
        alpha: (B, T, H)  已 sigmoid (逐头状态门控)
        S0   : (B, H, P, P) | None
        chunk_size: 内部 scan 块大小上界 (实际 M ≤ 此值)
        mem_budget_gb: A/C 物化显存预算 (GB), 用于自适应 M
    Returns:
        Y: (B, T, H, P)   o_t 序列
        final_state: (B, H, P, P)
    """
    b, T, h, p = q.shape
    dev, dt = q.device, q.dtype

    # ---- 自适应内部 scan 块 M: 受【并行 scan 峰值显存】约束 (防 OOM) ----
    # 单块 A/C 张量 ≈ b*M*h*p*p*4 字节; 但 _assoc_scan 在扫描过程中峰值 ≈
    # (A,C:2 + clone:2 + 每步~4) × 单块, 即 ~ (4 + 4·log2(M)) × 单块, 远大于"空闲显存"
    # 直觉(空闲大时会误选过大 M 反而 OOM)。故直接按 scan 峰值预算选最大可行 M。
    SCAN_PEAK_BUDGET_GB = _scan_budget_gb()   # 自适应: GPU=15GB / CPU 按可用内存动态(防低内存机段错误)
    def _scan_peak_gb(Mc):
        Tm = b * Mc * h * p * p * 4 / (1024 ** 3)        # 单块 A 或 C (GB)
        steps = max(1.0, math.log2(Mc))
        return (4.0 + 4.0 * steps) * Tm                  # 粗略上界
    M = 4
    for cand in (4, 8, 16, 32, 64):
        if cand > chunk_size:
            break
        if _scan_peak_gb(cand) <= SCAN_PEAK_BUDGET_GB:
            M = cand
        else:
            break
    M = int(min(chunk_size, max(4, M)))
    if M < 1:
        M = 1

    # ---- 一次性诊断: 确认 b/M/A-C 尺寸 ----
    global _DELTA_FIRST
    if not _DELTA_FIRST:
        _DELTA_FIRST = True
        _alloc = torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else -1.0
        print(f"[delta-debug] b={b} T={T} h={h} p={p} chunk_size={chunk_size} M={M} "
              f"scan_peak~{_scan_peak_gb(M):.1f}GB alloc_before={_alloc:.1f}GB "
              f"AC_per_block_GB={b*M*h*p*p*2*4/1024**3:.2f}", flush=True)

    # ---- padding 到 M 的整数倍 ----
    if T % M != 0:
        pad = M - (T % M)
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
        alpha = F.pad(alpha, (0, 0, 0, pad))
        T2 = T + pad
    else:
        T2 = T
        pad = 0
    c = T2 // M

    qc = q.reshape(b, c, M, h, p)
    kc = k.reshape(b, c, M, h, p)
    vc = v.reshape(b, c, M, h, p)
    betac = beta.reshape(b, c, M, h)       # (b,c,M,h)
    alphac = alpha.reshape(b, c, M, h)     # (b,c,M,h)

    I = torch.eye(p, device=dev, dtype=dt)
    # A_t = α ⊙ I − β k kᵀ ,  C_t = β v kᵀ   (b,c,M,h,p,p)
    kt = kc.unsqueeze(-1)              # (b,c,M,h,p,1)
    kk = kt @ kt.transpose(-1, -2)     # (b,c,M,h,p,p)
    A = alphac[..., None, None] * I - betac[..., None, None] * kk
    vt = vc.unsqueeze(-1)              # (b,c,M,h,p,1)
    vk = vt @ kc.unsqueeze(-2)         # (b,c,M,h,p,p)
    C = betac[..., None, None] * vk
    # [v7 NaN-source-guard] _assoc_scan 内的 cat/einsum 不防 NaN,
    # 极值数据点使 kk/vk 矩阵含 inf → A/C 含 inf → scan 链式 NaN → 三路分支全零.
    A = torch.nan_to_num(A, nan=0.0, posinf=1e4, neginf=-1e4)
    C = torch.nan_to_num(C, nan=0.0, posinf=1e4, neginf=-1e4)

    Y = torch.zeros(b, c, M, h, p, device=dev, dtype=dt)
    S_prev = S0 if S0 is not None else torch.zeros(b, h, p, p, device=dev, dtype=dt)

    # 块间仅少量 Python 迭代; 块内并行扫描消除逐位置 launch
    for ci in range(c):
        cumA, cumC = _assoc_scan(A[:, ci], C[:, ci])   # (b,M,h,p,p)
        # 注入块初态 S_prev:  state[l] = cumA[l] @ S_prev + cumC[l]
        S_t = torch.einsum("bhpq,blhqr->blhpr", S_prev, cumA) + cumC  # (b,M,h,p,p)
        Y[:, ci] = torch.einsum("blhpq,blhq->blhp", S_t, qc[:, ci])   # (b,M,h,p)
        S_prev = S_t[:, -1]                              # 跨块传递末态

    Y = Y.reshape(b, c * M, h, p)[:, :T]
    return Y, S_prev



class GatedDeltaNet(nn.Module):
    """Gated DeltaNet 时序层 (替换 LightSSMv2, 更强更快)。

    参考 arXiv:2412.06464 (Gated DeltaNet, ICLR2025) 的 gated delta rule。
    纯 PyTorch chunkwise 实现 (无 triton/cuda 扩展), 可训练, CPU 可跑通。

    设计:
      - q,k,v 投影 d_model → d_inner (=d_model*expand), 多头 (n_heads)
      - beta gate (sigmoid): delta 学习率 ∈ (0,1)
      - alpha gate (sigmoid): 逐头状态门控 ∈ (0,1) —— "Gated" 来源
      - 输出 GLU 门控 + 残差聚合
    状态 S_t ∈ ℝ^{P×P} (P=d_head), 故 d_state 参数仅用于接口兼容。
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        chunk_size: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        seq_len: int = 5,   # 2D 输入扩展为短序列以启用递推 (对齐 LightSSMv2)
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(d_model * expand)
        assert self.d_inner % n_heads == 0, "d_inner 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.d_head = self.d_inner // n_heads
        self.chunk_size = chunk_size
        self.seq_len = seq_len

        self.qkv = nn.Linear(d_model, self.d_inner * 3, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=False)
        self.alpha_proj = nn.Linear(d_model, n_heads, bias=False)
        self.out_gate = nn.Linear(d_model, self.d_inner, bias=False)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        self.reset_parameters()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[GatedDeltaNet] d_model={d_model} d_inner={self.d_inner} "
              f"heads={n_heads} d_head={self.d_head} params={n_params:,}")

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight, gain=0.5)
        nn.init.xavier_uniform_(self.beta_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.alpha_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_gate.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, C) 或 (B, T, C) → (B, d_model) 聚合后的单帧特征。"""
        if u.dim() == 2:
            B, C = u.shape
            x = u.unsqueeze(1).expand(-1, self.seq_len, -1).contiguous()  # (B, seq_len, C)
        else:
            x = u
        B, T, C = x.shape
        h, p = self.n_heads, self.d_head

        qkv = self.qkv(x)  # (B, T, 3*d_inner)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, h, p) * (p ** -0.5)   # [v7 opt] Attention 风格缩放: 防 k@kᵀ 外积 overflow → inf
        k = k.view(B, T, h, p) * (p ** -0.5)
        v = v.view(B, T, h, p)
        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, h)   # (B,T,h)
        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, T, h) # (B,T,h)

        Y, _ = chunk_delta_rule(q, k, v, beta, alpha, None, self.chunk_size)  # (B,T,h,p)
        Y = Y.reshape(B, T, self.d_inner)
        gate = torch.sigmoid(self.out_gate(x))                  # (B,T,d_inner)
        Y = Y * gate
        Y = self.out_norm(Y.mean(dim=1))                        # (B, d_inner) 聚合
        Y = self.drop(Y)
        out = self.out_proj(Y)                                  # (B, d_model)
        return out


class LinearRetention(nn.Module):
    """可选线性注意力/retention 块 (替代 SSM, O(N) 并行, 无状态递归, 更快)。"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] 或 [B, D] -> 扩为 [B,1,D]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # 各 [B,L,D]
        # 因果 retention: 使用标量衰减 (固定), 矩阵形式
        decay = 0.9
        # 简化: 标准线性注意力 (无因果) 用于单帧特征足够; 多帧则逐位衰减累加
        kv = torch.einsum("bld,blc->bdc", k, v)  # [B,D,D]
        out = torch.einsum("bld,bdc->blc", q, kv) / D
        out = self.out(out) + x
        out = self.norm(out)
        out = self.drop(out)
        return out[:, -1, :] if L > 1 else out.squeeze(1)


class GaussianNoise(nn.Module):
    """JS 范式: encoder 前的高斯噪声数据增强 (训练时注入, 评测时关闭)。"""
    def __init__(self, std: float = 0.05):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class Swish(nn.Module):
    """JS 范式: Swish = x·σ(x) 激活 (nn.SiLU 等价)。"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class AutoEncoder(nn.Module):
    """JS 范式: 轻量自编码器 —— 在 d_model 表征空间内 编码→latent→重建。

    自监督重建损失 (MSE) 作为辅助任务, latent 作为"编码器表征"与原始特征拼接。
    此处 AE 作用于 encoder 输出的 h(d_model), 重建 h (detached 目标), 避免污染主分支梯度。
    """
    def __init__(self, d_model: int, ae_latent: int = 32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_model, ae_latent), Swish())
        self.dec = nn.Linear(ae_latent, d_model)

    def forward(self, h: torch.Tensor):
        lat = self.enc(h)            # [B, ae_latent]
        recon = self.dec(lat)        # [B, d_model]
        return lat, recon


class CrossNet(nn.Module):
    """DCN-V3 (arXiv:2407.13349, KDD'24) 升级版交叉网络 (替换 DCN-v2):
      - 交叉向量维度 D/2: 参数量减半 (v3 Self-Mask 的核心收益)
      - Self-Mask: Mask(c)=c⊙ReLU(LayerNorm(c)), 约 50% 元素置零, 噪声正则化
      - 指数阶 Deep Crossing: 外层乘积反馈累计 x (而非 x0), 交互阶 2^l 指数增长
      - 逐特征门控 g=σ(gate) 软筛选 (替代不可微 Lasso 硬剪枝)
    全程可微, 端到端可训练; 比 DCN-v2 交叉更强且更省参。
    """
    def __init__(self, d_model: int, n_layers: int = 2):
        super().__init__()
        self.n_layers = n_layers
        self.half = d_model // 2
        self.W = nn.Linear(d_model, self.half, bias=True)   # D/2 交叉向量 → 参数减半
        self.ln = nn.LayerNorm(self.half)
        self.gate = nn.Parameter(torch.zeros(d_model))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for _ in range(self.n_layers):
            c = self.W(x)                       # [B, D/2] 交叉向量
            m = c * torch.relu(self.ln(c))      # Self-Mask (~50% 置零)
            c_full = torch.cat([c, m], dim=-1)  # [B, D] 拼接还原维度
            x = x * c_full + x                  # 指数阶交叉 (反馈累计 x)
        g = torch.sigmoid(self.gate)            # (d,) 逐特征门控
        return g * x + (1.0 - g) * x0


class TSMixerBlock(nn.Module):
    """TinyTimeMixer/TSMixer (arXiv:2401.03955, NeurIPS'24) 轻量无注意力特征混合。

    将 d_model 特征视为 1D 信号, 非重叠分块 (patch_len), 依次做:
      - intra-patch MLP 混合 (patch 内特征交互, 共享权重)
      - inter-patch MLP 混合 (patch 间交互, 共享权重)
    无自注意力, 纯 MLP, 参数量极小; 作为 GatedDeltaNet 时序分支前处理,
    增强局部/结构化特征交互 (替代对重复帧的退化递推)。
    """
    def __init__(self, d_model: int, patch_len: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % patch_len == 0, "d_model 必须能被 patch_len 整除"
        self.n_patch = d_model // patch_len
        self.patch_len = patch_len
        self.intra = nn.Sequential(
            nn.Linear(patch_len, patch_len), Swish(), nn.Dropout(dropout))
        self.inter = nn.Sequential(
            nn.Linear(self.n_patch, self.n_patch), Swish(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        xp = x.reshape(B, self.n_patch, self.patch_len)   # [B, np, pl]
        xp = self.intra(xp)                                # intra-patch 混合
        xp = self.inter(xp.transpose(1, 2)).transpose(1, 2)  # inter-patch 混合
        return x + xp.reshape(B, D)                        # 残差

class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR'22) —— 处理金融特征分布漂移。

    对每个样本的特征维做标准化 + 可学习仿射 (γ,β), 输出再反标准化。
    极轻量 (2*d 参数), ONNX 安全 (纯 matmul/均值/标准差), 增强非平稳数据鲁棒性,
    是现代时序栈 (iTransformer 等) 的标准组件。这里用于稳定送入注意力分支的特征分布。
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True) + self.eps
        return (x - mean) / std * self.gamma + self.beta


class FTTabularTransformer(nn.Module):
    """FT-Transformer (Gorishniy et al., NeurIPS'21) 轻量表分支蒸馏 — 手写注意力(ONNX 友好)。

    采用 iTransformer (Liu et al., ICLR'24) 的**反转注意力**思想: 将每个特征 (factor)
    视为一个 token (feature-as-token), 自注意力在特征维度上建模**跨因子依赖**
    (金融因子间交互), FFN 逐 token 学习因子内表示。手写多头注意力(纯 matmul/softmax,
    非 nn.Transformer), 100% ONNX 可导出。配合 RevIN 归一化增强非平稳鲁棒性。
    替代静态 MLP 分支(branchC), 引入真实注意力表建模(顶会 SOTA 范式)。

    将 base(d_model) 切为 n_feat 个 d_tok 维 token -> 共享 Linear 特征 tokenizer ->
    L 个 Transformer block(预 LN + 多头缩放点积注意力, 纯 matmul/softmax 实现, 不用 nn.Transformer) ->
    平均池化 -> Linear(d_tok -> d_model)。
    替代静态 MLP 分支(branchC), 引入真实注意力表建模(顶会 SOTA 范式), 且 100% ONNX 可导出
    (全部为 matmul/softmax/reshape/permute/relu/LN 等标准算子, 无控制流/自定义算子)。
    """
    def __init__(self, d_model: int, n_feat: int = 8, n_layers: int = 2,
                 n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_feat == 0, "d_model 必须能被 n_feat 整除"
        self.n_feat = n_feat
        self.d_tok = d_model // n_feat
        assert self.d_tok % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = self.d_tok // n_heads
        self.tokenizer = nn.Linear(self.d_tok, self.d_tok)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(self.d_tok),
                "qkv": nn.Linear(self.d_tok, self.d_tok * 3, bias=False),
                "out_proj": nn.Linear(self.d_tok, self.d_tok),
                "ln2": nn.LayerNorm(self.d_tok),
                "fc1": nn.Linear(self.d_tok, self.d_tok * 2),
                "fc2": nn.Linear(self.d_tok * 2, self.d_tok),
            }))
        self.out_proj = nn.Linear(self.d_tok, d_model)
        self.drop = nn.Dropout(dropout)

    def _mha(self, x: torch.Tensor, blk) -> torch.Tensor:
        B, n, d = x.shape
        h, hd = self.n_heads, self.head_dim
        qkv = blk["qkv"](x).reshape(B, n, 3, h, hd).permute(2, 0, 3, 1, 4)  # [3,B,h,n,hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.einsum("bhnd,bhmd->bhnm", q, k) / (hd ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bhnm,bhmd->bhnd", attn, v)        # [B,h,n,hd]
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, n, d)        # [B,n,d]
        return self.drop(blk["out_proj"](ctx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        tok = self.tokenizer(x.reshape(B, self.n_feat, self.d_tok))   # [B,n,d_tok]
        for blk in self.blocks:
            tok = tok + self._mha(blk["ln1"](tok), blk)
            ff = blk["fc2"](torch.relu(blk["fc1"](blk["ln2"](tok))))
            tok = tok + self.drop(ff)
        pooled = tok.mean(dim=1)                                      # [B, d_tok]
        return self.out_proj(pooled)                                  # [B, d_model]

class BilinearInteraction(nn.Module):
    """FinalMLP (AAAI'23) 风格低秩双线性特征交互。

    显式二阶交叉: out_i = Σ_j (UVᵀ)_{ij} · x_i · x_j + b_i,
    其中 UV = U Vᵀ 为低秩重构 (参数仅 2·d·rank, 紧凑)。
    与 CrossNet(元素级高阶交叉)互补, 提供成对二阶交互信号, 强化表格特征交叉。
    """

    def __init__(self, d_model: int, rank: int = 4):
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        UV = self.U @ self.V.t()                              # [d, d] 低秩
        out = torch.einsum('bd,di,bi->bd', x, UV, x) + self.b  # [B, d] 二阶交互
        return out


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """SCARF (ICML'21) NT-Xent 对比损失: 两视图表征互信息最大化。"""
    B = z1.size(0)
    if B < 2:
        return torch.zeros((), device=z1.device, dtype=z1.dtype)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)               # [2B, d]
    sim = z @ z.t() / tau                         # [2B, 2B]
    sim = sim.masked_fill(torch.eye(2 * B, device=z.device, dtype=torch.bool), -torch.finfo(sim.dtype).max)
    labels = torch.cat([torch.arange(B, device=z.device) + B,
                        torch.arange(B, device=z.device)])   # 正样本对索引
    return F.cross_entropy(sim, labels)


class MLPBranch(nn.Module):
    """通用 MLP 基分支 (FinStack 堆叠基学习器的 NN 等价)。"""
    def __init__(self, d_model: int, hidden: int, dropout: float, act=Swish):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), act(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RawTemporalEncoder(nn.Module):
    """多层 TSMixer 时序编码器 (多块残差堆叠, 支持 Stochastic Depth)。

    顶层入口: (B,60,28) 原始多变量时序 → (B, feature_dim)。
    每层: step_proj → TimeMix(跨T) → FeatMix(跨d) → 残差 + DropPath。
    最终层: TimeXer 全局 token + mean-pool 双通道聚合。
    """
    def __init__(self, seq_len: int = 60, feat_per_day: int = 28,
                 d_model: int = 64, feature_dim: int = 192,
                 n_blocks: int = 1, drop_path: float = 0.0, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.step = nn.Linear(feat_per_day, d_model)

        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            sd = drop_path * i / max(1, n_blocks - 1) if n_blocks > 1 else drop_path
            self.blocks.append(nn.ModuleDict({
                "time_mix": nn.Sequential(nn.Linear(seq_len, seq_len), Swish(), nn.Dropout(dropout)),
                "feat_mix": nn.Sequential(nn.Linear(d_model, d_model), Swish(), nn.Dropout(dropout),
                                           nn.Linear(d_model, d_model), Swish(), nn.Dropout(dropout)),
                "drop_path": DropPath(sd),
            }))

        self.global_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.global_token, mean=0.0, std=0.02)
        self.out = nn.Linear(d_model, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.step(x)
        for blk in self.blocks:
            shortcut = h
            h = h.permute(0, 2, 1)
            h = blk["time_mix"](h)
            h = h.permute(0, 2, 1)
            h = blk["feat_mix"](h)
            h = shortcut + blk["drop_path"](h)

        mean_pool = h.mean(dim=1)
        B, _ = mean_pool.shape
        g = self.global_token.unsqueeze(0).expand(B, -1)
        scores = torch.einsum("bd,btd->bt", g, h) / (self.d_model ** 0.5)
        attn = torch.softmax(scores, dim=1)
        ctx = torch.einsum("bt,btd->bd", attn, h)
        fused = mean_pool + ctx
        return self.out(fused)


# ===== 可配置容量三档预设 (一套自包含架构, size 参数生成不同规模) =====
# 论文双贡献轴: (1) tiny=RTX3060 6GB 实时推理紧凑可部署变体;
#              (2) base/large=A800 性能变体, 支撑 scaling 消融。
# 显式传入对应超参 (非 None) 会覆盖 size 预设, 便于消融实验。
V11_6_SIZE_PRESETS = {
    # tiny: 向后兼容 (params=94,135), RTX3060 6GB 部署变体
    "tiny": dict(
        feature_dim=192, d_model=64,  n_components=3, ssm_d_state=8,  gdn_expand=1,
        ae_latent=24, branch_hidden=24, ft_layers=2, ft_heads=2,
        gdn_heads=4, bilinear_rank=4, mdn_hidden=32, head_layers=1,
        te_blocks=1, cross_attn_heads=0,
    ),
    # base: A800 性能变体 (~3M 参数), 更宽 d_model/feature_dim + 更深注意力 + 更多 MDN 分量
    "base": dict(
        feature_dim=320, d_model=320, n_components=4, ssm_d_state=16, gdn_expand=2,
        ae_latent=64, branch_hidden=96, ft_layers=4, ft_heads=8,
        gdn_heads=8, bilinear_rank=8, mdn_hidden=96, head_layers=2,
        te_blocks=2, cross_attn_heads=4,
    ),
    # large: A800 上限变体 (~10M 参数), 冲顶会 scaling 上界
    "large": dict(
        feature_dim=512, d_model=640, n_components=5, ssm_d_state=32, gdn_expand=2,
        ae_latent=128, branch_hidden=192, ft_layers=6, ft_heads=8,
        gdn_heads=8, bilinear_rank=16, mdn_hidden=128, head_layers=3,
        te_blocks=3, cross_attn_heads=4,
    ),
    # xl: BW 68.7GB 超大变体 (~21.8M 参数), 多层 TSMixer + 交叉注意力
    "xl": dict(
        feature_dim=768, d_model=896, n_components=5, ssm_d_state=32, gdn_expand=2,
        ae_latent=128, branch_hidden=512, ft_layers=6, ft_heads=8,
        gdn_heads=8, bilinear_rank=16, mdn_hidden=128, head_layers=3,
        te_blocks=4, cross_attn_heads=4,
    ),
    # xxl: BW 68.7GB 顶配变体 (~60M 参数), 冲顶会 SOTA 上界
    #   宽 d_model=1152 + 6×TSMixer + 12头 FT-Transformer + 更深 MDN/交叉网络
    "xxl": dict(
        feature_dim=1024, d_model=1152, n_components=6, ssm_d_state=32, gdn_expand=2,
        ae_latent=256, branch_hidden=768, ft_layers=8, ft_heads=12,
        gdn_heads=12, bilinear_rank=32, mdn_hidden=256, head_layers=4,
        te_blocks=6, cross_attn_heads=6,
    ),
}


class StudentModel(nn.Module):
    def __init__(
        self,
        size: str = "tiny",          # 容量档: tiny(94K,RTX3060部署)/base(~3M,A800)/large(~11M,A800上限)
        feature_dim: Optional[int] = None,   # 学生 fused 输出维度 (None=按 size 预设; 内部维度, 不改 ONNX I/O)
        input_dim: int = 1680,       # 原始特征 X 维度 (60步 × 28因子)
        seq_len: int = 60,           # 时间步数 W
        feat_per_day: int = 28,      # 每步因子数
        d_model: Optional[int] = None,       # 学生隐藏 (None=按 size 预设)
        aux_dim: int = 16,
        market_dim: int = 3,
        n_components: Optional[int] = None,  # MDN 分量 K (None=按 size 预设)
        ssm_d_state: Optional[int] = None,
        use_linear_retention: bool = False,  # False=1x SSM, True=线性注意力
        use_gated_deltanet: bool = True,     # True=用 GatedDeltaNet 替换 SSM/LinearRetention
        gdn_expand: Optional[int] = None,    # GatedDeltaNet 扩展系数 (None=按 size 预设)
        dropout: float = 0.15,
        df: float = 4.0,
        # --- 蒸馏新开关 (默认值保证旧调用兼容) ---
        use_ae: bool = True,                 # 范式1: JS 自编码器表征分支
        use_cross_net: bool = True,          # 范式2: FinStack HCFM 可微交叉
        use_static_branch: bool = True,      # 范式3: 第三个静态 MLP 基模型(可选)
        ae_latent: Optional[int] = None,     # AE 隐维度 (None=按 size 预设)
        noise_std: float = 0.05,             # 范式1: encoder 前高斯噪声
        branch_hidden: Optional[int] = None, # 基 MLP 分支隐维度 (None=按 size 预设)
        use_bilinear: bool = True,           # FinalMLP 低秩双线性二阶特征交互
        use_rcmoe: bool = True,              # Regime-Conditioned MoE 门控
        use_contrast: bool = True,           # SCARF 对比自监督
        use_ft_transformer: bool = True,     # FT-Transformer(NeurIPS'21) 替代静态 MLP 分支
        use_revin: bool = True,              # RevIN(ICLR'22) 特征分布漂移校正
        use_dir_calib: bool = False,         # P1: 方向校准概率头 (默认 OFF, 不破坏旧行为)
        detach_aux_heads: bool = False,       # [COLLAPSE-FIX v2] MDN/AE/SCARF 辅助头从共享编码器梯度 detach
        # --- 可配置容量细项 (None=按 size 预设, 显式传参用于消融) ---
        ft_layers: Optional[int] = None,     # FT-Transformer 层数
        ft_heads: Optional[int] = None,      # FT-Transformer 注意力头数
        gdn_heads: Optional[int] = None,     # GatedDeltaNet 头数
        bilinear_rank: Optional[int] = None, # 双线性交互低秩
        mdn_hidden: Optional[int] = None,    # MDN 头隐维度
        head_layers: Optional[int] = None,   # 输出头前 MLP 层数
        n_feat: int = 8,                     # FT-Transformer feature-as-token 分组数
        chunk_size: int = 64,                # GatedDeltaNet scan chunk size (加速用32可翻倍batch)
    ):
        super().__init__()
        self.chunk_size = chunk_size
        # ===== 容量档预设解析: 显式传参 (非 None) 覆盖 size 预设 =====
        if size not in V11_6_SIZE_PRESETS:
            raise ValueError(f"size 必须是 {list(V11_6_SIZE_PRESETS)}, 收到 {size!r}")
        _p = V11_6_SIZE_PRESETS[size]
        feature_dim   = _p["feature_dim"]   if feature_dim   is None else feature_dim
        d_model       = _p["d_model"]       if d_model       is None else d_model
        n_components  = _p["n_components"]   if n_components  is None else n_components
        ssm_d_state   = _p["ssm_d_state"]    if ssm_d_state   is None else ssm_d_state
        gdn_expand    = _p["gdn_expand"]     if gdn_expand    is None else gdn_expand
        ae_latent     = _p["ae_latent"]      if ae_latent     is None else ae_latent
        branch_hidden = _p["branch_hidden"]  if branch_hidden is None else branch_hidden
        ft_layers     = _p["ft_layers"]      if ft_layers     is None else ft_layers
        ft_heads      = _p["ft_heads"]       if ft_heads      is None else ft_heads
        gdn_heads     = _p["gdn_heads"]      if gdn_heads     is None else gdn_heads
        bilinear_rank = _p["bilinear_rank"]  if bilinear_rank is None else bilinear_rank
        mdn_hidden    = _p["mdn_hidden"]     if mdn_hidden    is None else mdn_hidden
        head_layers   = _p["head_layers"]    if head_layers   is None else head_layers
        te_blocks     = _p.get("te_blocks", 1)
        cross_attn_heads = _p.get("cross_attn_heads", 0)
        self.size = size
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.feat_per_day = feat_per_day
        self.d_model = d_model
        self.K = n_components
        self.ae_latent = ae_latent
        self.noise_std = noise_std
        self.market_dim = market_dim
        self.use_dir_calib = use_dir_calib

        in_dim = feature_dim + aux_dim

        # ===== 范式1: JS 预处理 (BN + Swish + 高斯噪声) =====
        self.noise = GaussianNoise(noise_std)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.BatchNorm1d(d_model),
            Swish(),
            nn.Dropout(dropout),
        )

        # ===== 范式1: 轻量自编码器 (d_model 空间内 编码→latent→重建) =====
        self.ae = AutoEncoder(d_model, ae_latent) if use_ae else None

        # 主分支拼接 [原始 fused(192), AE 表征(ae_latent)] -> fuser -> d_model
        fuse_in = feature_dim + (ae_latent if use_ae else 0)
        self.fuser = nn.Sequential(
            nn.Linear(fuse_in, d_model),
            nn.BatchNorm1d(d_model),
            Swish(),
            nn.Dropout(dropout),
        )

        # ===== 范式2: 可微交叉网络 (FinStack HCFM 等价) =====
        self.cross_net = CrossNet(d_model) if use_cross_net else None

        # ===== FinalMLP (AAAI'23): 低秩双线性二阶特征交互 (补 CrossNet 元素级交叉) =====
        self.bilinear = BilinearInteraction(d_model, rank=bilinear_rank) if use_bilinear else None

        # ===== 范式3: 多基模型并行 (模拟 FinStack 堆叠基学习器) =====
        # 基模型 A: GatedDeltaNet 时序分支 (保留现有实现)
        if use_gated_deltanet:
            self.branchA = nn.Sequential(
                TSMixerBlock(d_model, patch_len=8, dropout=dropout),  # TSMixer 无注意力特征混合
                GatedDeltaNet(d_model=d_model, d_state=ssm_d_state,
                              expand=gdn_expand, n_heads=gdn_heads,
                              chunk_size=self.chunk_size, dropout=dropout),
            )
            self.temporal_kind = "TSMixer+GatedDeltaNet"
        elif use_linear_retention:
            self.branchA = LinearRetention(d_model, dropout)
            self.temporal_kind = "LinRet"
        else:
            self.branchA = LightSSMv2(d_model=d_model, d_state=ssm_d_state,
                                      n_layers=1, dropout=dropout)
            self.temporal_kind = "SSM1"

        # 基模型 B: 特征交叉 MLP 分支
        self.branchB = MLPBranch(d_model, branch_hidden, dropout)
        # 基模型 C: FT-Transformer 表分支(NeurIPS'21 蒸馏) 或 静态 MLP 分支(可选)
        if use_static_branch:
            self.branchC = (FTTabularTransformer(d_model, n_feat=n_feat, n_layers=ft_layers,
                                                 n_heads=ft_heads, dropout=dropout)
                            if use_ft_transformer else MLPBranch(d_model, branch_hidden, dropout))
        else:
            self.branchC = None
        self.n_base = 3 if use_static_branch else 2

        # ===== Regime-Conditioned MoE 门控 (金融 regime switching 思想) =====
        # 由 market_state 条件化元融合门控: 不同市场状态路由到不同基模型权重
        self.regime_gate = nn.Sequential(
            nn.Linear(market_dim, 8), Swish(), nn.Linear(8, self.n_base)
        ) if use_rcmoe else None

        # ===== SCARF (ICML'21) 对比自监督: AE latent 投影头 =====
        self.contrast_head = nn.Linear(ae_latent, ae_latent) if (use_contrast and use_ae) else None
        self.contrast_tau = 0.1

        self.use_bilinear = use_bilinear
        self.use_rcmoe = use_rcmoe
        self.use_contrast = use_contrast
        self.use_ft_transformer = use_ft_transformer
        self.use_revin = use_revin
        self.detach_aux_heads = detach_aux_heads
        # RevIN(ICLR'22) 特征分布漂移校正 (用于稳定 branchC 注意力输入)
        self.revin = RevIN(d_model) if use_revin else None

        # ===== 自包含时序编码器 (多层 TSMixer 堆叠 + DropPath) =====
        drop_path_rate = 0.05 * (d_model / 64.0)  # 随 d_model 缩放
        self.temporal_encoder = RawTemporalEncoder(
            seq_len=seq_len, feat_per_day=feat_per_day,
            d_model=d_model, feature_dim=feature_dim,
            n_blocks=te_blocks, drop_path=drop_path_rate, dropout=dropout)

        # ===== 时序-特征交叉注意力 (多头交叉, 增强 fused 与辅助特征的语义交互) =====
        # d_head=32 固定, 使 QK^T 维度独立于 feature_dim/aux_dim
        self.cross_attn = (CrossAttention(q_dim=feature_dim, kv_dim=aux_dim,
                                          n_heads=cross_attn_heads, d_head=32, dropout=dropout)
                           if cross_attn_heads > 0 else None)

        # ===== 范式3: 元融合头 (FinStack LR 元学习器的 softmax 等价) =====
        # 可学习 W·[out_A,out_B,out_C] -> softmax 权重 -> 加权融合
        self.meta_head = nn.Linear(self.n_base * d_model, self.n_base)

        _head_stack = []
        for _ in range(max(1, head_layers)):
            _head_stack += [nn.Linear(d_model, d_model), Swish(), nn.Dropout(dropout)]
        self.head_hidden = nn.Sequential(*_head_stack)

        # 点预测头 — [v6] 仅归一化输出(std=1), 对齐 z-scored 目标, 无需学习 scale
        self.point_head = nn.Linear(d_model, 1)

        # P1: 方向校准概率头 (USE_DIR_CALIB, 默认 OFF) — 输出 P(y>0) 经温度缩放校准,
        # 显式建模"涨的概率"可信度, 攻击论文自曝的 IC-DA 背离 (IC 高但 DA<50%)。
        self.dir_head = nn.Linear(d_model, 1) if use_dir_calib else None
        self.dir_temp = nn.Parameter(torch.ones(1)) if use_dir_calib else None

        # MDN 头 (复用 teacher 的 skewed-t 实现, K 按 size 预设)
        self.mdn_head = SkewedTMDNHead(
            feature_dim=d_model, n_components=n_components,
            hidden_dim=mdn_hidden, df=df, dropout=dropout,
        )
        # 特征提示投影已移除: v11.6 自包含, 不再对齐教师 phi (去 v11.5 教师依赖)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[StudentModel] size={size} params={n_params:,} d_model={d_model} "
              f"K={n_components} enc=TSMixer(自包含) temporal={self.temporal_kind} "
              f"AE={use_ae} cross={use_cross_net} nbases={self.n_base} "
              f"bilinear={use_bilinear} rcmoe={use_rcmoe} contrast={use_contrast}")

    def _base_forward(self, base: torch.Tensor,
                      market_state: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """单帧前向: 多基并行 + 元融合 + 输出头。返回 (point, out_dict, final_h)。"""
        # [v4 NaN-guard] 入口处防极端值扩散: ~0.8% 样本在 fuser/bilinear 后爆炸,
        # 替换 NaN 为 0 阻止 meta_head 全维度污染; 仅主路径, 不影响编码器梯度。
        # ONNX 兼容: 无条件执行 nan_to_num (去掉 data-dependent if)
        base = torch.nan_to_num(base, nan=0.0, posinf=1e4, neginf=-1e4)
        # 基模型 A: 时序分支
        out_A = self.branchA(base)                       # [B, d]
        out_A = torch.nan_to_num(out_A, nan=0.0, posinf=1e4, neginf=-1e4)
        # 基模型 B: 交叉 MLP 分支
        b_in = self.cross_net(base) if self.cross_net is not None else base
        out_B = self.branchB(b_in)                       # [B, d]
        out_B = torch.nan_to_num(out_B, nan=0.0, posinf=1e4, neginf=-1e4)
        # 基模型 C: 静态 MLP 分支 (可选)
        outs = [out_A, out_B]
        if self.branchC is not None:
            c_in = self.revin(base) if self.revin is not None else base
            out_C = self.branchC(c_in)
            out_C = torch.nan_to_num(out_C, nan=0.0, posinf=1e4, neginf=-1e4)
            outs.append(out_C)
        stacked = torch.stack(outs, dim=1)               # [B, nb, d]
        meta_logits = self.meta_head(stacked.reshape(stacked.size(0), -1))  # [B, nb]
        # Regime-Conditioned MoE: 由 market_state 条件化门控 (金融 regime switching)
        if self.regime_gate is not None and market_state is not None:
            ms = market_state[..., -self.market_dim:] if market_state.shape[-1] >= self.market_dim else market_state
            meta_logits = meta_logits + self.regime_gate(ms)
        # [v7.1 guard] meta_logits 若含 inf 组合 → softmax 产 NaN → final_h 全链崩塌
        meta_logits = torch.nan_to_num(meta_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        w = torch.softmax(meta_logits, dim=-1)           # [B, nb] 和为1
        final_h = (w.unsqueeze(-1) * stacked).sum(dim=1) # [B, d]

        # [v3.1] LayerNorm 编码器输出: 防止 f_hstd 爆炸到 1.5B (v3 过夜实测).
        # v3 的 layer_norm 仅加在 point_head 前, 但 final_h 无约束, 8轮后爆炸.
        # 加在 head_hidden 前归一化, 既防爆炸又保留head_hidden学习能力.
        final_h = F.layer_norm(final_h, (final_h.size(-1),))
        h_h = self.head_hidden(final_h)
        # [COLLAPSE-FIX v3] LayerNorm 前 point_head: 即使上游 head_hidden 输出塌缩,
        # point_head 仍收到单位方差输入, 防止权重归零导致常数不动点锁死。
        h_h = F.layer_norm(h_h, (h_h.size(-1),))
        # [v5 RANKING-MAGNITUDE DECOUPLING] 排序与幅度完全解耦:
        # 归一化后 std=1, 与 z-scored 目标(std≈1)直接对齐。MSE 梯度无正反馈循环。
        raw = self.point_head(h_h).squeeze(-1)           # [B] — 纯排序信号(未标准化回归值)
        if self.training:
            # 训练时按 batch 标准化(std=1)，对齐 z-scored 目标，ranking loss 需要
            point = (raw - raw.mean()) / (raw.std() + 1e-8)  # [B] — pstd=1.0 由归一化保证
        else:
            # [INFER-FIX] 推理时 batch 常为1，batch 标准化会坍缩为0(raw-raw.mean()==0)；
            # 直接用 raw 回归值作为点预测。标准化是单调变换，raw 完整保留排序与符号，
            # 且携带幅度信息，比恒0更可用。ONNX 导出在 eval 模式追踪此分支。
            point = raw
        # [v7 NaN catch-all] 归一化前 raw 可能含 NaN (上游传播), 末端截断
        point = torch.nan_to_num(point, nan=0.0, posinf=1e4, neginf=-1e4)
        # [COLLAPSE-FIX v2] DETACH_AUX_HEADS: MDN 头吃 detach 的 h_h, 不把分布"常数吸引子"
        # (nll_loss 只用 mu/sigma/pi/alpha) 经 head_hidden 反向传回共享 final_h/编码器。
        # MDN 自身参数仍经 nll_loss 正常训练, 但不再拖拽共享特征 -> 根治 val_IC 塌 0。
        _h_h_for_mdn = h_h.detach() if self.detach_aux_heads else h_h
        mdn = self.mdn_head(_h_h_for_mdn)                # dict

        # P1: 方向校准概率头 -> P(y>0) (温度缩放校准)
        if self.dir_head is not None:
            _t = self.dir_temp.clamp(min=1e-3) if self.dir_temp is not None else torch.ones(1, device=h_h.device)
            _p_up = torch.sigmoid(self.dir_head(h_h).squeeze(-1) / _t)
        else:
            _p_up = None

        out = {
            "point": point,
            "raw": raw,                                  # [INFER-FIX] 未标准化回归值, 供排查/校准
            "quantiles": mdn["quantiles"],
            "pi": mdn["pi"],
            "mu": mdn["mu"],
            "sigma": mdn["sigma"],
            "alpha": mdn["alpha"],
            "log_sigma": mdn["log_sigma"],
            "meta_weights": w.detach(),                  # 元融合权重(验证和为1)
            "base_outs": stacked.detach(),               # 各基模型输出
            "f_hstd": final_h.std().detach(),            # [COLLAPSE-FIX v2] 共享特征方差监控
            "h_hstd": h_h.std().detach(),                # head_hidden 输出方差监控
            "p_up": _p_up,                               # P1: 方向校准概率 (None if 关闭)
        }
        return point, out, final_h

    def forward(
        self,
        x_seq: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
        market_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # [v7.1 input-clean] 输入特征可能含 NaN/inf (CS缓存缺失值、指标除零),
        # 由源头截断可防 ep8+ 累积 NaN 全模型崩塌。
        # ONNX 兼容: 无条件执行 nan_to_num (去掉 data-dependent if 条件)
        x_seq = torch.nan_to_num(x_seq, nan=0.0, posinf=1e4, neginf=-1e4)
        if aux is not None:
            aux = torch.nan_to_num(aux, nan=0.0, posinf=1e4, neginf=-1e4)
        if market_state is not None:
            market_state = torch.nan_to_num(market_state, nan=0.0, posinf=1e4, neginf=-1e4)
        # ===== 自包含时序编码器 (多层 TSMixer 堆叠 + DropPath) =====
        z = self.temporal_encoder(x_seq)                 # (B, feature_dim), 原始时序 fused
    
        # ===== V11.6 增强: 多头交叉注意力 (temporal fused ↔ branch 特征) =====
        if self.cross_attn is not None and aux is not None:
            # 用时序 fused 作为 query, aux 作为 key/value, 做时序→特征交叉注意力
            ca_out = self.cross_attn(z.unsqueeze(1), aux.unsqueeze(1), aux.unsqueeze(1))
            z = z + 0.1 * ca_out  # 残差 + 缩放注入
        
        x = z
        if aux is not None:
            fused_raw = z                            # [B, feature_dim], 原 fused
            x = torch.cat([z, aux], dim=-1)
        else:
            fused_raw = z[..., :self.feature_dim]
            if x.shape[-1] != self.encoder[0].in_features:
                pad = torch.zeros(*x.shape[:-1],
                                  self.encoder[0].in_features - x.shape[-1],
                                  device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=-1)

        x = self.noise(x)                               # 范式1: 高斯噪声 (仅训练)
        h = self.encoder(x)                             # [B, d_model]

        # [COLLAPSE-FIX v2] DETACH_AUX_HEADS: 辅助头(AE/SCARF)从共享编码器梯度 detach,
        # 编码器只由 point 任务驱动 -> 防 MDN/AE "常数吸引子" 把 final_h 拖向常数塌缩。
        # (aux 头自身参数仍经 recon/NT-Xent 正常训练, 只是不再反向污染主编码器)
        _h_for_aux = h.detach() if self.detach_aux_heads else h

        # SCARF 对比自监督: 两视图随机腐蚀 -> AE latent -> NT-Xent
        contrast_loss = torch.zeros((), device=h.device, dtype=h.dtype)
        if self.contrast_head is not None and self.ae is not None and self.training:
            h1 = self._corrupt(_h_for_aux); h2 = self._corrupt(_h_for_aux)
            z1 = self.contrast_head(self.ae.enc(h1))
            z2 = self.contrast_head(self.ae.enc(h2))
            contrast_loss = _nt_xent(z1, z2, self.contrast_tau)

        # 范式1: AE 自监督重建 (latent 作为表征)
        if self.ae is not None:
            lat, recon = self.ae(_h_for_aux)
            ae_loss = F.mse_loss(recon, h.detach())
            main = torch.cat([fused_raw, lat], dim=-1)  # [B, 192 + ae_latent]
        else:
            ae_loss = torch.zeros((), device=h.device, dtype=h.dtype)
            main = fused_raw

        base = self.fuser(main)                         # [B, d_model]
        # FinalMLP 低秩双线性二阶交互 (残差增强)
        if self.bilinear is not None:
            base = base + self.bilinear(base)
        # [v4 NaN-guard] 在进入 _base_forward 前截断极端值: 0.8% 样本的 NaN 在 bilinear 二次型处产生
        base = torch.nan_to_num(base, nan=0.0, posinf=1e4, neginf=-1e4)

        point, out, final_h = self._base_forward(base, market_state)
        out["ae_loss"] = ae_loss
        out["contrast_loss"] = contrast_loss
        self._last_hint = final_h
        return point, out

    def forward_ensemble(
        self,
        x_seq: torch.Tensor,
        aux: Optional[torch.Tensor] = None,
        market_state: Optional[torch.Tensor] = None,
        n_ensemble: int = 3,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """种子集成 / MC-dropout 平均推理 (蒸馏 JS 的 3-种子集成)。

        开启 dropout 多次前向取平均, 等价于多个子网络平均预测。
        注: 真正的 model-soup 是训练 2-3 个随机种子后平均 state_dict,
        可在此 forward 之外对多个加载的模型做 weight averaging 推理。
        """
        self.train()  # 启用 dropout mask 多样性
        points, mus, sigmas, pis, alphas, quants = [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(n_ensemble):
                p, o = self.forward(x_seq, aux, market_state)
                points.append(p)
                mus.append(o["mu"]); sigmas.append(o["sigma"])
                pis.append(o["pi"]); alphas.append(o["alpha"])
                quants.append(o["quantiles"])
        self.eval()
        avg = lambda t: torch.stack(t, 0).mean(0)
        out = {
            "point": avg(points),
            "quantiles": avg(quants),
            "pi": avg(pis), "mu": avg(mus),
            "sigma": avg(sigmas), "alpha": avg(alphas),
            "log_sigma": torch.log(avg(sigmas) + 1e-6),
        }
        return out["point"], out

    def _corrupt(self, x: torch.Tensor, ratio: float = 0.3) -> torch.Tensor:
        """SCARF 风格随机特征腐蚀 (训练时): 以 ratio 概率将特征置零。"""
        if not self.training:
            return x
        mask = torch.rand_like(x) < ratio
        return x.masked_fill(mask, 0.0)

    def hint_feature(self) -> torch.Tensor:
        return self._last_hint
