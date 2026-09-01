"""MF-DFA (Multifractal Detrended Fluctuation Analysis) 实现。

Kantelhardt et al. (2002) 标准算法：
- DFA 是其 q=2 特例
- 输出广义 Hurst 谱 H(q)、质量指数 tau(q)、多重分形谱 f(alpha)
"""
from __future__ import annotations

import numpy as np


def _profile(x: np.ndarray) -> np.ndarray:
    """累积偏离序列 Y(i) = sum_{k=1..i} (x_k - mean(x))"""
    x = np.asarray(x, dtype=float)
    return np.cumsum(x - x.mean())


def _detrended_variances(y: np.ndarray, s: int, order: int = 1) -> np.ndarray:
    """对尺度 s 分段去趋势，返回各段方差 F^2(s, nu)。"""
    n = len(y)
    n_seg = n // s
    if n_seg < 2:
        return np.array([])
    f2 = np.zeros(2 * n_seg)
    idx = 0
    xs = np.arange(s, dtype=float)
    for v in range(n_seg):
        seg = y[v * s : (v + 1) * s]
        coef = np.polyfit(xs, seg, order)
        trend = np.polyval(coef, xs)
        f2[idx] = np.mean((seg - trend) ** 2)
        idx += 1
    y_rev = y[::-1]
    for v in range(n_seg):
        seg = y_rev[v * s : (v + 1) * s]
        coef = np.polyfit(xs, seg, order)
        trend = np.polyval(coef, xs)
        f2[idx] = np.mean((seg - trend) ** 2)
        idx += 1
    return f2


def _fq(f2: np.ndarray, q: float) -> float:
    """q 阶波动函数（q=0 用对数极限）。"""
    if q == 0.0:
        return float(np.exp(0.5 * np.mean(np.log(np.maximum(f2, 1e-300)))))
    return float(np.mean(f2 ** (q / 2.0)) ** (1.0 / q))


def _loglog_slope(scales: np.ndarray, yvals: np.ndarray) -> float:
    mask = np.isfinite(yvals) & (yvals > 0)
    if mask.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(scales[mask]), np.log(yvals[mask]), 1)[0])


def mfdfa_hq(
    x: np.ndarray,
    qs: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    order: int = 1,
) -> np.ndarray:
    """广义 Hurst 谱 H(q)。qs 默认 [-5,-3,-1,0,1,3,5]。"""
    if qs is None:
        qs = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    qs = np.asarray(qs, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if scales is None:
        # 标准尺度网格：16 ~ N/4，等比
        max_s = max(n // 4, 4)
        scales = np.unique(np.floor(np.logspace(np.log10(4), np.log10(max_s), 18))).astype(int)
    scales = scales[(scales >= 4) & (scales <= n // 4)]
    if len(scales) < 3:
        return np.full(len(qs), np.nan)

    y = _profile(x)
    f2_by_s = [_detrended_variances(y, int(s), order) for s in scales]
    hq = np.zeros(len(qs))
    for i, q in enumerate(qs):
        Fq = np.array([_fq(f2, q) if len(f2) else np.nan for f2 in f2_by_s])
        hq[i] = _loglog_slope(scales, Fq)
    return hq


def dfa_hurst(x: np.ndarray, scales: np.ndarray | None = None) -> float:
    """DFA (q=2) Hurst 指数。"""
    h = mfdfa_hq(x, qs=np.array([2.0]), scales=scales)
    return float(h[0])


def multifractal_spectrum(
    x: np.ndarray,
    qs: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> dict:
    """多重分形谱：返回 {Hq, tau, alpha, f_alpha, dH, dalpha, qs}。"""
    if qs is None:
        qs = np.linspace(-5, 5, 21)
    qs = np.asarray(qs, dtype=float)
    Hq = mfdfa_hq(x, qs, scales)
    tau = qs * Hq - 1.0
    # alpha = d(tau)/dq 数值微分
    dq = np.gradient(qs)
    alpha = np.gradient(tau, qs)
    f_alpha = qs * alpha - tau
    valid = np.isfinite(alpha)
    dalpha = float(np.nanmax(alpha[valid]) - np.nanmin(alpha[valid]))
    valid_h = np.isfinite(Hq)
    dH = float(np.nanmax(Hq[valid_h]) - np.nanmin(Hq[valid_h]))
    return {
        "Hq": Hq,
        "tau": tau,
        "alpha": alpha,
        "f_alpha": f_alpha,
        "dalpha": dalpha,
        "dH": dH,
        "qs": qs,
    }
