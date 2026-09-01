"""深层相位/时序结构扩展特征（E6，N5 严格预算提升核心）。

E5 已证明：N5 保谱伪造（保留幅度谱+随机相位+边际校准）使经典 8 维
特征全部失明，7 维浅层相位特征（|r| ACF/DFA + IAAFT z）可将判别
AUC 推到 ≈0.885，但 FPR≤0.10 工作点检出率受分离度上限限制
（0.675）。E6 增加"更深"的相位时序结构特征，捕捉 N5 随机相位
破坏的额外可分离信号：

  1. sign_acf1    符号序列一阶自相关（真实收益符号的持续/反持续模式）
  2. sign_run_ent 符号游程长度分布熵（正负符号聚集结构）
  3. lev_asym     杠杆不对称 corr(|r_{t+1}|, r_t)（真实金融负值更强；
                  N5 随机相位打散杠杆效应 → 接近 0）
  4. svd_eff      延迟嵌入轨道矩阵奇异谱有效维数（真实低维结构 →
                  有效维小；随机相位 → 有效维接近嵌入维 m）
  5. knn_pred_err 延迟嵌入 kNN 一步预测归一化误差（真实序列短程可预测；
                  N5 打散相位结构 → 误差更大）
  6. samp_ent     样本熵（m=2, r=0.2σ；真实金融序列复杂度低于 N5）
  7. vol_res_acf1 GARCH(1,1) 拟合后残差波动 |a_t| 的一阶自相关
                  （真实残差近白噪声而 N5 残差带结构？实证检验）

设计原则：
  - 全部特征仅依赖相位/时序结构（幅度谱固定下 N5 无法伪装）；
  - 计算复杂度 O(n)~O(n·m)（窗口 n=1000 秒级以内）；
  - 返回有限标量，非法输入返回 np.nan。
"""
from __future__ import annotations

import numpy as np

from .spectral import abs_autocorr


# ---------------------------------------------------------------------------
# 1. 符号序列结构
# ---------------------------------------------------------------------------
def sign_acf1(x: np.ndarray) -> float:
    """sign(r) 一阶自相关：正负收益交替的短程依赖。"""
    x = np.asarray(x, dtype=float)
    s = np.sign(x)
    n = len(s)
    if n < 6 or np.ptp(s) == 0:
        return np.nan
    r = np.corrcoef(s[:-1], s[1:])
    return float(r[0, 1])


def sign_run_entropy(x: np.ndarray) -> float:
    """符号游程长度分布熵（归一化 [0,1]，log2 单位）。

    真实序列常出现"下跌聚集/上涨聚集"（游程偏长、分布不均 → 熵低）；
    随机相位打散符号结构 → 游程接近几何分布 → 熵接近 1。
    """
    x = np.asarray(x, dtype=float)
    s = np.sign(x)
    if len(s) == 0:
        return np.nan
    runs: list[int] = []
    cur = s[0]
    cnt = 1
    for v in s[1:]:
        if v == cur:
            cnt += 1
        else:
            runs.append(cnt)
            cur, cnt = v, 1
    runs.append(cnt)
    arr = np.asarray(runs, dtype=float)
    p = arr / arr.sum()
    if len(p) < 2:
        return 0.0
    h = -np.sum(p * np.log2(p))
    return float(h / np.log2(len(arr))) if len(arr) > 1 else 0.0


# ---------------------------------------------------------------------------
# 2. 杠杆不对称（波动对滞后收益的非对称响应）
# ---------------------------------------------------------------------------
def leverage_asym(x: np.ndarray, lag: int = 1) -> float:
    """corr(|r_{t+lag}|, r_t)：真实金融存在负杠杆效应（下跌 → 波动放大），
    该相关性显著为负；N5 随机相位打散时序 → 相关性趋 0。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < lag + 6:
        return np.nan
    a_f = np.abs(x[lag:])
    r_lag = x[:-lag]
    if np.std(a_f) < 1e-12 or np.std(r_lag) < 1e-12:
        return np.nan
    return float(np.corrcoef(a_f, r_lag)[0, 1])


# ---------------------------------------------------------------------------
# 3. 延迟嵌入奇异谱（Takens 重构的低维结构）
# ---------------------------------------------------------------------------
def _embed_matrix(x: np.ndarray, m: int = 10, tau: int = 1,
                  n_eff: int | None = None) -> np.ndarray:
    """延迟嵌入轨道矩阵。

    M[i, :] = [x[i], x[i+tau], ..., x[i+(m-1)*tau]]，
    i ∈ [0, n_eff-1]，n_eff 默认 n - (m-1)*tau（尾部 tau 信息冗余但
    无越界风险）。kNN 预测等需要"可预测目标"的场景传入更小的
    n_eff = n - m*tau。
    """
    n = len(x)
    if n_eff is None:
        n_eff = n - (m - 1) * tau
    if n_eff < m + 1:
        return np.empty((0, m))
    idx = np.arange(n_eff)[:, None] + np.arange(m)[None, :] * tau
    return x[idx]


def svd_effective_dim(x: np.ndarray, m: int = 10, tau: int = 1) -> float:
    """延迟嵌入轨道矩阵奇异谱的有效维数（指数化香农熵）。

    真实金融序列是低维确定性+噪声混合 → 奇异值快速衰减 → 有效维小；
    N5 随机相位使序列更接近白噪声化 → 奇异值平缓 → 有效维接近 m。
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if np.std(x) < 1e-12:
        return np.nan
    M = _embed_matrix(x, m=m, tau=tau)
    if M.shape[0] < 2:
        return np.nan
    s = np.linalg.svd(M, compute_uv=False)
    s = s[:m]
    p = s ** 2 / (np.sum(s ** 2) + 1e-12)
    p = p[p > 1e-12]
    if len(p) < 2:
        return float(len(p))
    h = -np.sum(p * np.log(p))
    return float(np.exp(h))


# ---------------------------------------------------------------------------
# 4. 延迟嵌入 kNN 一步预测误差
# ---------------------------------------------------------------------------
def knn_pred_error(x: np.ndarray, m: int = 10, tau: int = 1,
                   k: int = 5, max_points: int = 400) -> float:
    """延迟嵌入空间中 kNN 一步预测的归一化误差。

    真实金融序列存在短程相位结构（动量/均值回归）→ 局部可预测；
    N5 随机相位破坏结构 → 预测误差趋近无条件波动。
    返回 rmse / std(x)。

    为控制 O(n²) 距离矩阵开销，用大步长采样（≤ max_points 个嵌入
    向量）估算，Chebyshev 邻域搜索不需要全精度。
    """
    x = np.asarray(x, dtype=float)
    std = float(np.std(x))
    if std < 1e-12:
        return np.nan
    n = len(x)
    n_eff = n - m * tau  # 可预测点：i=0..n_eff-1，目标 x[i+m*tau]
    if n_eff < k + 2:
        return np.nan
    M = _embed_matrix(x, m=m, tau=tau, n_eff=n_eff)  # (n_eff, m)
    y = x[np.arange(n_eff) + m * tau]  # (n_eff,)

    step = max(1, n_eff // max_points)  # 大步长采样
    idx = np.arange(0, n_eff, step)
    Mq = M[idx]
    # 成对 Chebyshev 距离（批量向量化）
    d = np.abs(Mq[:, None, :] - M[None, :, :]).max(axis=2)  # (q, n_eff)
    # 排除"同一原始时刻"（Mq 第 i 行来自原序列 idx[i] 时刻）
    d[np.arange(len(idx)), idx] = np.inf
    nn = np.argpartition(d, k, axis=1)[:, :k]  # (q, k)
    pred = np.take_along_axis(np.broadcast_to(y[None, :], d.shape),
                              nn, axis=1).mean(axis=1)
    errs = (y[idx] - pred) ** 2
    return float(np.sqrt(np.mean(errs)) / std)


# ---------------------------------------------------------------------------
# 5. 样本熵（Richman & Moorman 2000）
# ---------------------------------------------------------------------------
def sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2,
                   max_window: int = 5000) -> float:
    """样本熵 SampEn(m, r=0.2σ)：序列复杂度/规律性。

    真实金融序列（含噪声+结构）复杂度低于纯随机相位化版本，
    N5 破坏结构 → 更接近随机 → SampEn 更高。
    """
    x = np.asarray(x, dtype=float)
    std = float(np.std(x))
    if std < 1e-12:
        return np.nan
    r = r_frac * std
    n = len(x)
    if n < m + 2:
        return np.nan

    def _count_matches(l: int) -> int:
        # 向量化：构造 l 维模式窗口，两两 Chebyshev 距离 ≤ r 的对数
        n_w = n - l + 1
        if n_w < 2:
            return 0
        windows = np.lib.stride_tricks.sliding_window_view(x, l)
        # 限制比较规模（内存 O(n_w²·l)），过长序列用大步长采样
        step = max(1, n_w // max_window)
        idx = np.arange(0, n_w, step)
        w = windows[idx]
        d = np.abs(w[:, None, :] - windows[None, :, :]).max(axis=2)
        d[np.arange(len(idx)), idx] = np.inf  # 排除自匹配
        return int(np.sum(d <= r))

    b = _count_matches(m)
    a = _count_matches(m + 1)
    if b == 0:
        return 0.0
    return float(-np.log(a / b))


# ---------------------------------------------------------------------------
# 6. GARCH(1,1) 残差波动结构
# ---------------------------------------------------------------------------
def _garch_fit_gaussian(x: np.ndarray) -> tuple[float, float, float, float]:
    """用条件最大似然的简化矩估计拟合 GARCH(1,1)：返回 (w, a, b, sigma2_0)。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sigma2 = np.var(x) + 1e-12
    # 从样本自相关粗估：b ≈ min(0.97, max(0.0, corr(|x|_t, |x|_{t-1})))
    r1 = abs_autocorr(x, lag=1)
    if not np.isfinite(r1):
        r1 = 0.0
    b = min(0.95, max(0.0, r1)) * 0.8
    a = 0.10
    w = (1.0 - a - b) * np.var(x)
    w = max(w, 1e-8)
    return float(w), float(a), float(b), float(sigma2)


def garch_resid_vol_acf1(x: np.ndarray) -> float:
    """GARCH(1,1) 拟合后标准化残差的 |a_t| 一阶自相关。

    真实 GARCH 序列：模型吸收波动聚集 → 残差接近白噪声（ACF≈0）；
    N5 边际校准保留了 GARCH 式的肥尾但时序被打散 → 拟合后残差带
    残余结构（ACF 偏离 0）。实证检验该特征方向。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 50:
        return np.nan
    mu = float(np.mean(x))
    xc = x - mu
    w, a, b, s0 = _garch_fit_gaussian(xc)
    sigma2 = np.empty(n)
    sigma2[0] = s0
    for t in range(1, n):
        sigma2[t] = w + a * xc[t - 1] ** 2 + b * sigma2[t - 1]
    h = np.sqrt(sigma2 + 1e-12)
    resid = xc / h
    return abs_autocorr(resid, lag=1)


# ---------------------------------------------------------------------------
# 特征清单与提取
# ---------------------------------------------------------------------------
PHASE_EXT_FEATURE_NAMES = [
    "sign_acf1",       # 符号序列一阶自相关
    "sign_run_ent",    # 符号游程熵
    "lev_asym",        # 杠杆不对称
    "svd_eff",         # 延迟嵌入奇异谱有效维
    "knn_pred_err",    # kNN 一步预测归一化误差
    "samp_ent",        # 样本熵
    "vol_res_acf1",    # GARCH 残差波动 ACF(1)
]


def extract_phase_ext_features(
    x: np.ndarray,
    m_embed: int = 10,
    k_knn: int = 5,
    m_samp: int = 2,
) -> np.ndarray:
    """提取 7 维深层相位结构特征（顺序与 PHASE_EXT_FEATURE_NAMES 一致）。"""
    feats = np.array([
        sign_acf1(x),
        sign_run_entropy(x),
        leverage_asym(x),
        svd_effective_dim(x, m=m_embed),
        knn_pred_error(x, m=m_embed, k=k_knn),
        sample_entropy(x, m=m_samp),
        garch_resid_vol_acf1(x),
    ], dtype=float)
    return feats
