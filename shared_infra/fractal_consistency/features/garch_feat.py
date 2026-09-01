"""GARCH(1,1) 波动聚集持续性 (alpha+beta) 估计（scipy MLE）。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def garch_persistence(r: np.ndarray) -> float:
    """对收益序列 r 拟合 GARCH(1,1): sigma2_t = w + a*r_{t-1}^2 + b*sigma2_{t-1}
    返回 a + b（持续性系数，接近1表示强波动聚集）。
    """
    r = np.asarray(r, dtype=float)
    r = r - r.mean()
    n = len(r)
    if n < 100:
        return np.nan
    init_var = np.var(r)

    def nll(params: np.ndarray) -> float:
        w, a, b = params
        if w <= 0 or a < 0 or b < 0 or (a + b) >= 1.0 or a + b <= 0:
            return 1e12
        sigma2 = np.empty(n)
        sigma2[0] = init_var
        for t in range(1, n):
            sigma2[t] = w + a * r[t - 1] ** 2 + b * sigma2[t - 1]
        eps = 1e-12
        return float(
            np.sum(np.log(np.maximum(sigma2[1:], eps)) + r[1:] ** 2 / np.maximum(sigma2[1:], eps))
        )

    best = None
    best_val = np.inf
    # 多起点避免局部最优
    for a0, b0 in [(0.05, 0.90), (0.08, 0.85), (0.10, 0.80)]:
        try:
            res = minimize(
                nll,
                x0=np.array([init_var * 0.05, a0, b0]),
                method="L-BFGS-B",
                bounds=[(1e-12, None), (0.0, 1.0), (0.0, 1.0)],
                options={"maxiter": 300},
            )
            if res.fun < best_val:
                best_val = res.fun
                best = res.x
        except Exception:
            continue
    if best is None:
        return np.nan
    w, a, b = best
    return float(a + b)
