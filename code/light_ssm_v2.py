"""LightSSM-V2: 基于 Mamba-2/SSD 的轻量状态空间模型
=====================================================
V10 核心模块：用 State Space Duality (SSD) 算子替换 V9 的简化 selective scan

升级要点（vs V9 LightSSM）:
1. SSD 块分解算法: 2-8x 训练加速，支持更大状态容量
2. 真实时序建模: 修复 V9 单帧复制 5 次的设计缺陷
3. 多头结构: n_heads=4, d_head=48 (d_model=192)
4. 门控融合: 保留 V9 的 GLU 融合机制
5. 纯 PyTorch 实现: 无需 CUDA 编译，Windows 兼容

参考文献:
- Gu & Dao, "Transformers are SSMs: Generalized Models and Efficient Algorithms
  Through Structured State Space Duality," arXiv:2408.10705, 2024.
- 源码: https://github.com/state-spaces/mamba/blob/mamba-2/ssd_minimal.py

参数量: ~15K (vs V9 LightSSM ~5K)，但时序建模能力显著增强
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from loguru import logger

try:
    from einops import rearrange
    _HAS_EINOPS = True
except ImportError:
    _HAS_EINOPS = False
    rearrange = None


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """计算 segment sum（SSD 块分解核心算子）

    L[i,j] = sum(x[i:j]) for i>=j, else -inf

    Args:
        x: (B, ..., T) 累积衰减率
    Returns:
        (B, ..., T, T) 下三角 segment sum 矩阵
    """
    T = x.size(-1)
    # 扩展为 (..., T, T)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    # 对角线及以下的累积和
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    x = x.masked_fill(~mask, 0)
    return torch.cumsum(x, dim=-2).masked_fill(~mask, -float('inf'))


def ssd(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
        chunk_size: int = 64, initial_states: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Structured State Space Duality (SSD) 算子

    块分解矩阵形式计算 Y = M·X，其中 M 是 1-半可分矩阵

    Args:
        x: (b, T, h, p) 输入
        A: (b, T, h) 状态转移衰减率（负值）
        B: (b, T, h, n) 输入矩阵
        C: (b, T, h, n) 输出矩阵
        chunk_size: 块大小（必须整除 T）
        initial_states: (b, h, n, p) 初始状态

    Returns:
        Y: (b, T, h, p) 输出
        final_state: (b, h, n, p) 最终状态
    """
    b, T, h, p = x.shape
    n = B.shape[-1]

    # 确保 T 能被 chunk_size 整除
    if T % chunk_size != 0:
        # padding 到最近的倍数
        pad_len = chunk_size - (T % chunk_size)
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        A = F.pad(A, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
        T_padded = T + pad_len
    else:
        T_padded = T
        pad_len = 0

    c = T_padded // chunk_size

    # 重排为块结构
    x_b = rearrange(x, 'b (c l) h p -> b c l h p', l=chunk_size)
    A_b = rearrange(A, 'b (c l) h -> b h c l', l=chunk_size)
    B_b = rearrange(B, 'b (c l) h n -> b c l h n', l=chunk_size)
    C_b = rearrange(C, 'b (c l) h n -> b c l h n', l=chunk_size)

    # A 的累积和
    A_cumsum = torch.cumsum(A_b, dim=-1)  # (b, h, c, l)

    # ── 1. 块内对角块计算 ──
    L = torch.exp(_segsum(A_b))  # (b, h, c, l, l)
    # Y_diag = sum over l: C * B * L * x
    Y_diag = torch.einsum('bclhn,bcshn,bhcls,bcshp->bclhp',
                          C_b, B_b, L, x_b)

    # ── 2. 块内最终状态 ──
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # (b, h, c, l)
    states = torch.einsum('bclhn,bhcl,bclhp->bchpn',
                          B_b, decay_states, x_b)  # (b, c, h, p, n)

    # ── 3. 块间递推 ──
    if initial_states is None:
        initial_states = torch.zeros(b, 1, h, p, n, device=x.device, dtype=x.dtype)
    states = torch.cat([initial_states, states], dim=1)  # (b, c+1, h, p, n)

    # 块间衰减
    A_last = A_cumsum[:, :, :, -1]  # (b, h, c)
    A_last_padded = F.pad(A_last, (1, 0))  # (b, h, c+1)
    decay_chunk = torch.exp(_segsum(A_last_padded))  # (b, h, c+1, c+1)
    new_states = torch.einsum('bhzc,bchpn->bzhpn', decay_chunk, states)
    states = new_states[:, :-1]  # (b, c, h, p, n)
    final_state = new_states[:, -1]  # (b, h, p, n)

    # ── 4. 块间 → 输出贡献 ──
    state_decay_out = torch.exp(A_cumsum)  # (b, h, c, l)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp',
                         C_b, states, state_decay_out)

    # 合并对角块和块间贡献
    Y = Y_diag + Y_off
    Y = rearrange(Y, 'b c l h p -> b (c l) h p')

    # 去除 padding
    if pad_len > 0:
        Y = Y[:, :T]

    return Y, final_state


class Mamba2Block(nn.Module):
    """Mamba-2 SSD 块（轻量金融时序版）

    结构:
        x → in_proj (D → 2D: Δ 和 BC)
          → conv1d (short conv, kernel=3)
          → SSD 算子
          → norm
          → out_proj
          → GLU 门控融合

    参数量（d_model=192, n_heads=4, d_head=48, d_state=16）:
        in_proj: 192*384 = 73,728
        conv1d: 192*3 = 576
        A_log: 4
        D: 192
        out_proj: 192*192 = 36,864
        总计: ~111K (可调)
    """

    def __init__(self, d_model: int = 192, n_heads: int = 4, d_state: int = 16,
                 chunk_size: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_state = d_state
        self.chunk_size = chunk_size

        assert d_model % n_heads == 0, f"d_model {d_model} 必须能被 n_heads {n_heads} 整除"

        # 输入投影: 生成 Δ (B,C) 和 x'
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        # 短时卷积（Mamba 标准组件）
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

        # 状态参数（可学习）
        # A_log: 每头一个，负指数保证稳定性
        self.A_log = nn.Parameter(torch.zeros(n_heads))
        self.D = nn.Parameter(torch.ones(d_model) * 0.1)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # 门控融合（GLU）
        self.gate = nn.Linear(d_model, d_model)
        # LayerNorm
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.5)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"[LightSSM-V2] Mamba2Block: d_model={d_model}, n_heads={n_heads}, "
                    f"d_state={d_state}, chunk={chunk_size}, params={n_params}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 输入序列
        Returns:
            y: [B, L, D] 输出序列
        """
        B, L, D = x.shape
        h, p = self.n_heads, self.d_head
        n = self.d_state

        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2D)
        x_proj, z = xz.chunk(2, dim=-1)  # 各 (B, L, D)

        # 短时卷积
        x_proj = x_proj.transpose(1, 2)  # (B, D, L)
        x_proj = self.conv1d(x_proj)
        x_proj = x_proj.transpose(1, 2)  # (B, L, D)

        # 选择性参数
        delta = F.softplus(x_proj)  # (B, L, D) 确保 > 0
        # A: 负值保证稳定
        A = -torch.exp(self.A_log) * delta[..., :h]  # 广播: (B, L, h)

        # B, C: 从输入投影（共享以节省参数）
        BC = x_proj  # (B, L, D)
        # 重排为多头形式
        x_mh = x_proj.view(B, L, h, p)  # (B, L, h, p)
        B_mat = BC.view(B, L, h, p)[..., :n]  # (B, L, h, n) 取前 n 维
        C_mat = BC.view(B, L, h, p)[..., :n]  # (B, L, h, n)

        # SSD 算子
        Y, _ = ssd(x_mh, A, B_mat, C_mat, chunk_size=min(self.chunk_size, L))

        # 重排回 (B, L, D)
        Y = Y.reshape(B, L, D)
        # 残差连接 + D 跳跃
        Y = Y + self.D * x_proj

        # GLU 门控融合
        gate = torch.sigmoid(self.gate(z))
        Y = Y * gate

        # 输出投影 + Norm + Dropout
        Y = self.out_proj(Y)
        Y = self.norm(Y + x)  # 残差
        Y = self.dropout(Y)
        return Y


class LightSSMv2(nn.Module):
    """LightSSM-V2: Mamba-2 SSD 增强的时序特征模块

    用于 FinCast V10：替换 V9 的 LightSSM

    特性:
    1. 真实时序建模（修复 V9 单帧复制缺陷）
    2. SSD 块分解算法（2-8x 训练加速）
    3. 多头结构（n_heads=4）
    4. 残差连接 + GLU 门控
    5. 支持变长输入（自动 padding）
    """

    def __init__(self, d_model: int = 192, n_heads: int = 4, d_state: int = 16,
                 n_layers: int = 2, chunk_size: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # 堆叠多个 Mamba-2 块
        self.layers = nn.ModuleList([
            Mamba2Block(d_model, n_heads, d_state, chunk_size, dropout)
            for _ in range(n_layers)
        ])

        # 最终的聚合层：序列 → 单帧
        self.aggregator = nn.Linear(d_model * 2, d_model)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"[LightSSM-V2] 总参数量: {n_params} ({n_layers} 层 Mamba-2)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 时序输入（支持真实多帧序列）
               或 [B, D] 单帧输入（向后兼容 V9，会自动扩展为 L=5 序列）
        Returns:
            h: [B, D] 增强后的单帧特征
        """
        if x.dim() == 2:
            # 向后兼容 V9：单帧输入 → 扩展为序列
            B, D = x.shape
            x = x.unsqueeze(1).expand(-1, 5, -1).contiguous()  # (B, 5, D)
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(f"输入维度 {x.dim()} 不支持，期望 2 或 3")

        # 通过 Mamba-2 层
        for layer in self.layers:
            x = layer(x)

        # 聚合：最后一帧 + 平均池化
        last = x[:, -1, :]  # (B, D)
        mean = x.mean(dim=1)  # (B, D)
        h = self.aggregator(torch.cat([last, mean], dim=-1))  # (B, D)
        return h
