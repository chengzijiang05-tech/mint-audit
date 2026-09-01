"""统一分形特征提取器：将收益序列映射为 8/15 维分形指纹向量。

默认（full=False）返回 8 维经典分形指纹（兼容 E1-E4 全部历史结论）；
full=True 追加 7 维相位结构特征（E5 N5 盲区攻克：|r| ACF 多尺度/
|r| DFA Hurst/IAAFT surrogate z-score 系，见 features/spectral.py）。
"""
from __future__ import annotations

import numpy as np

from .garch_feat import garch_persistence
from .hill import hill_tail_index
from .mfdfa import dfa_hurst, mfdfa_hq, multifractal_spectrum
from .rs import hurst_rs
from .phase_ext import PHASE_EXT_FEATURE_NAMES, extract_phase_ext_features
from .spectral import SPECTRAL_FEATURE_NAMES, extract_spectral_features

FEATURE_NAMES = [
    "H_DFA",      # 去趋势波动标度指数（长记忆）
    "H_RS",       # R/S 重标极差 Hurst（长记忆对照）
    "H_qneg3",    # 广义 Hurst 谱 q=-3（大波动结构）
    "H_qpos3",    # 广义 Hurst 谱 q=3（小波动结构）
    "dH",         # H(q) 谱宽度（多重分形强度）
    "dalpha",     # 多重分形谱宽度 f(alpha)
    "beta_tail",  # Hill 尾部 Pareto 指数（厚尾）
    "garch_pers", # GARCH(1,1) 持续性（波动聚集）
]

FULL_FEATURE_NAMES = FEATURE_NAMES + SPECTRAL_FEATURE_NAMES
ALL_22_FEATURE_NAMES = FEATURE_NAMES + SPECTRAL_FEATURE_NAMES + PHASE_EXT_FEATURE_NAMES


def _core_features(x: np.ndarray) -> np.ndarray:
    """8 维经典分形指纹（幅度谱/边际分布决定）。"""
    Hq = mfdfa_hq(x, qs=np.array([-3.0, 2.0, 3.0]))
    spec = multifractal_spectrum(x)

    return np.array(
        [
            float(Hq[1]) if np.isfinite(Hq[1]) else np.nan,   # H_DFA (q=2)
            hurst_rs(x),                                       # H_RS
            float(Hq[0]) if np.isfinite(Hq[0]) else np.nan,   # H(q=-3)
            float(Hq[2]) if np.isfinite(Hq[2]) else np.nan,   # H(q=3)
            spec["dH"],                                        # dH
            spec["dalpha"],                                    # dalpha
            hill_tail_index(x),                                # beta_tail
            garch_persistence(x),                              # garch_pers
        ]
    )


def extract_features(x: np.ndarray, full: bool = False,
                     phase_ext: bool = False) -> np.ndarray:
    """输入：收益序列（一维 float）。

    输出：
      full=False, phase_ext=False → 8 维经典分形指纹（默认，兼容 E1-E4）；
      full=True,  phase_ext=False → 15 维：8 经典 + 7 相位结构特征；
      full=True,  phase_ext=True  → 22 维：8 经典 + 7 相位 + 7 深层扩展。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 512:
        raise ValueError(f"序列过短：{len(x)} < 512")

    core = _core_features(x)
    result = core
    if full:
        result = np.concatenate([result, extract_spectral_features(x)])
    if phase_ext:
        result = np.concatenate([result, extract_phase_ext_features(x)])
    return result
