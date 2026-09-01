"""Hill 尾部指数估计（厚尾 Pareto 指数 beta）。"""
from __future__ import annotations

import numpy as np


def hill_tail_index(
    x: np.ndarray, tail_fraction: float = 0.1, min_k: int = 10
) -> float:
    """Hill 估计。

    对对称厚尾序列取绝对值后，对尾部 k 个顺序统计量估计
    xi = (1/k) * sum_{i=1..k} log(x_(i) / x_(k+1))
    返回 Pareto 尾部指数 beta = 1 / xi。
    """
    x = np.asarray(x, dtype=float)
    x = np.abs(x)
    n = len(x)
    k = max(int(np.floor(tail_fraction * n)), min_k)
    k = min(k, n - 1)
    x_sorted = np.sort(x)[::-1]  # 降序
    if x_sorted[k] <= 0 or x_sorted[k - 1] <= 0:
        return np.nan
    log_vals = np.log(x_sorted[:k])
    xi = log_vals.mean() - np.log(x_sorted[k])
    if xi <= 1e-9:
        return np.nan
    return float(1.0 / xi)
