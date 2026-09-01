"""短窗相位结构特征（E6 真实 LLM 输出适配）。

背景：extract_features（8/15 维分形指纹）要求 ≥512 点（DFA/mfdfa
标度拟合需要），而真实 LLM 输出数字序列通常只有几十~上百个 token
（如复述 60~120 个日收益率）。因此为"LLM 输出→序列→检验"链路设计
短窗特征：全部来自短窗下数值稳定的相位/时序结构特征（N5 命中的
维度），不依赖长标度拟合。

短窗特征清单（10 维，120 点窗口实证稳定）：
  abs_acf1/3/5    |r| 自相关（波动聚集）
  abs_dfa         |r| DFA Hurst（短窗版，min_box=4）
  surr_z          IAAFT surrogate z（max(abs_acf1, abs_acf5)）
  surr_z_dfa      IAAFT surrogate z（abs_dfa 统计量）
  sign_acf1       符号序列一阶自相关（phase_ext）
  lev_asym        杠杆不对称 corr(|r_{t+1}|, r_t)（phase_ext）
  samp_ent        样本熵（m=2, r=0.2σ）
  vol_res_acf1    GARCH 残差波动 ACF(1)

设计：单一入口 extract_short_features(x)；N5Discriminator(short=True)
使用该特征。FSCT 仍要求长序列（≥512），短窗下仅 N5 判别 + 断言层。
"""
from __future__ import annotations

import numpy as np

from .phase_ext import (garch_resid_vol_acf1, leverage_asym,
                        sample_entropy, sign_acf1)
from .spectral import _dfa_hurst, abs_autocorr, surrogate_z

SHORT_FEATURE_NAMES = [
    "abs_acf1", "abs_acf3", "abs_acf5", "abs_dfa",
    "surr_z", "surr_z_dfa",
    "sign_acf1", "lev_asym", "samp_ent", "vol_res_acf1",
]

MIN_SHORT_LEN = 32


def _abs_dfa_short(x: np.ndarray) -> float:
    """|r| DFA（短窗：min_box=4, 4 尺度）。"""
    a = np.abs(np.asarray(x, dtype=float))
    if len(a) < 24:
        return np.nan
    return _dfa_hurst(a, min_box=4, n_box=4)


def extract_short_features(x: np.ndarray, n_surr: int = 12) -> np.ndarray:
    """提取 10 维短窗相位结构特征（顺序与 SHORT_FEATURE_NAMES 一致）。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_SHORT_LEN:
        raise ValueError(f"短窗序列过短：{len(x)} < {MIN_SHORT_LEN}")
    feats = np.array([
        abs_autocorr(x, lag=1),
        abs_autocorr(x, lag=3),
        abs_autocorr(x, lag=5),
        _abs_dfa_short(x),
        surrogate_z(x, stat="max", n_surr=n_surr, seed=1),
        surrogate_z(x, stat="abs_dfa", n_surr=n_surr, seed=2),
        sign_acf1(x),
        leverage_asym(x),
        sample_entropy(x, m=2),
        garch_resid_vol_acf1(x),
    ], dtype=float)
    return feats
