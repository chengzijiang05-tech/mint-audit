"""R/S (Rescaled Range) 重标极差法 Hurst 指数。"""
from __future__ import annotations

import numpy as np


def hurst_rs(x: np.ndarray, min_scale: int = 8, n_scales: int = 16) -> float:
    """经典 R/S 分析估计 Hurst 指数。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    max_s = max(n // 4, min_scale + 1)
    scales = np.unique(
        np.floor(np.logspace(np.log10(min_scale), np.log10(max_s), n_scales))
    ).astype(int)
    scales = scales[(scales >= min_scale) & (scales <= n // 4)]
    if len(scales) < 3:
        return np.nan

    rs = np.zeros(len(scales))
    for i, s in enumerate(scales):
        n_seg = n // s
        vals = []
        for v in range(n_seg):
            seg = x[v * s : (v + 1) * s]
            mean = seg.mean()
            dev = np.cumsum(seg - mean)
            R = dev.max() - dev.min()
            S = seg.std(ddof=1)
            if S > 0:
                vals.append(R / S)
        if vals:
            rs[i] = np.mean(vals)
        else:
            rs[i] = np.nan

    mask = np.isfinite(rs) & (rs > 0)
    if mask.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(scales[mask]), np.log(rs[mask]), 1)[0])
