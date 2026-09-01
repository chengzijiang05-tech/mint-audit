"""MINT 模型组件：边际歼灭预处理、编码器、分数头、联合损失。

设计依据：整体研究方案 §3.2 / §4.2。
  - Gaussianize：rank → (r+0.5)/n → 标准正态分位。边际分布在输入端
    被歼灭，编码器无法靠边际作弊，必须使用依赖结构（相位/时序）。
    真实窗、增强正样本、轨道负样本、推断候选全部施以同一变换；
  - Encoder：轻量 1D 卷积栈（stride 4 ×4 层 → 感受野覆盖全窗），
    全局平均池化后投影到 d 维并 L2 归一化（InfoNCE 余弦相似度）；
  - ScoreHead：小 MLP 二分类头（真窗口 vs 轨道成员），与编码器
    联合训练，输出非负分数（e 值构造要求 s ≥ 0）；
  - mint_loss：InfoNCE（anchor=增强正样本对，负样本=本窗轨道 +
    批内跨窗轨道池）+ BCE（分数头）。

CPU 训练规模：~100k 参数，31 窗 × K=16 轨道全批，分钟级。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


def gaussianize(x: np.ndarray) -> np.ndarray:
    """rank → (r+0.5)/n → Φ⁻¹。单序列自归一化（训练/推断一致）。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    ranks = np.argsort(np.argsort(x))
    return stats.norm.ppf((ranks + 0.5) / n)


def gaussianize_batch(X: np.ndarray) -> np.ndarray:
    """按行 Gaussianize，输入 (n, T) → 输出 (n, T)。"""
    return np.vstack([gaussianize(row) for row in X])


class Encoder(nn.Module):
    """1D 卷积编码器：R^T → R^d（L2 归一化）。

    stride (2,2,2,4)：1000 天窗保留 30 个时间步（G1 诊断发现
    stride 4×4 仅剩 2 步会把时间不对称结构在池化中抹平，杠杆
    滞后 1–5 天的模式必须在一层卷积核内可见）。变长输入安全
    （Conv+GroupNorm+mean-pool 均与 T 无关）。
    """

    def __init__(self, d: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv1d(128, 128, 5, stride=4), nn.GELU(),
        )
        self.proj = nn.Linear(128, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.unsqueeze(1)).mean(dim=2)
        return F.normalize(self.proj(z), dim=1)


class ScoreHead(nn.Module):
    """真窗口 vs 轨道成员的二分类分数头。

    forward 返回 logit（保序、不饱和，sigmoid 饱和会压缩测试
    分数序，G1 诊断发现这是 AUC 衰减来源之一）。e 值构造处
    自行 sigmoid 得到非负概率。
    """

    def __init__(self, d: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z).squeeze(-1)


def info_nce(z: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor,
             tau: float = 0.1) -> torch.Tensor:
    """InfoNCE：anchor=真实窗，正样本=自身增强，负样本=轨道池。

    z (B,d) 真实窗嵌入；z_pos (B,d) 增强窗嵌入；z_neg (M,d) 批内
    全部轨道成员嵌入（含本窗轨道硬负样本与跨窗轨道易负样本）。
    """
    logits_pos = torch.sum(z * z_pos, dim=1) / tau          # (B,)
    logits_neg = z @ z_neg.T / tau                           # (B,M)
    logits = torch.cat([logits_pos.unsqueeze(1), logits_neg],
                       dim=1)                                # (B,1+M)
    return F.cross_entropy(logits, torch.zeros(len(z), dtype=torch.long,
                                               device=z.device))


def bce_loss(s_pos: torch.Tensor, s_neg: torch.Tensor,
             pos_weight: float = 1.0) -> torch.Tensor:
    """分数头 BCE（logit 域）：真窗口 label 1，轨道成员 label 0。

    pos_weight 为正类损失权重，用于平衡 1:K 的正负比例。"""
    logits = torch.cat([s_pos, s_neg])
    y = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)])
    return F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=torch.as_tensor(float(pos_weight)))


def orbit_e_value(s_x: float, s_orbit: np.ndarray) -> float:
    """轨道 e 值 W(x) = (K+1)s(x) / Σ_j s(y_j)，y_0 = x。

    可交换性下 E[W]=1；W ≫ 1 表示 x 在其保指纹零假设族中极端，
    证实存在零假设族无法复制的真实结构；W ≈ 1 表示与族不可分（可疑）。
    """
    denom = s_x + float(np.sum(s_orbit))
    return float((len(s_orbit) + 1) * s_x / max(denom, 1e-12))
