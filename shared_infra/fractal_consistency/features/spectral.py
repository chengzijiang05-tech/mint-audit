"""相位结构特征模块（N5 盲区攻克核心，E5）。

N5 相位随机化保谱伪造：FFT → 保留幅度谱 + 随机化相位 → IFFT →
边际分布校准。因此：
  - 一切仅依赖幅度谱/功率谱/边际分布的特征（H_DFA、H_RS、beta_tail、
    garch_pers、Welch PSD 斜率……）对 N5 完全失明（被逐一保留）；
  - 唯有时序/相位结构特征可检：波动聚集的时序持续性（|r| ACF、
    |r| DFA Hurst）、以及"与自身保谱随机相位代理族的一致性"
    （IAAFT surrogate z-score 系）。

E5 实证（最终 7 特征，方向归一化单特征 AUC on N5(1.0) vs 真实股票）：
  - surr_z_acf_sum（IAAFT z on Σ|r| ACF(1..20)） ≈0.91  ← 最强
  - abs_dfa / surr_z_dfa（|r| DFA 原始与自参照 z） ≈0.85
  - abs_acf3（|r| 3阶自相关）                      ≈0.83
  - surr_z（IAAFT z on max(abs_acf1, abs_acf5)）   ≈0.76
  - abs_acf1 / abs_acf5（|r| 短/中期自相关）        ≈0.70~0.74

理论依据（非线性时间序列分析经典工具）：
  - IAAFT/AAFT surrogate 检验（Theiler et al. 1992; Schreiber &
    Schmitz 2000）：候选序列若与"其自身保谱随机相位代理族（且做
    同一边际校准）"统计一致（z≈0），则它极可能是保谱随机相位伪造
    （N5 正是如此）；真实序列因非线性相位结构显著偏离代理族。
    关键修正（E5 探索发现）：naive surrogate（代理不做边际校准）对
    N5 无效，N5 的边际校准本身引入非线性，代理必须同样校准（IAAFT）。
  - 波动聚集持续性 |r| ACF / DFA：GARCH 聚集需要相位一致排列；
    随机相位打散大波动聚集的时序 → ACF 显著下降 → 自参照 z 高。
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# 内部工具：rank-based 边际校准（IAAFT 代理用；与 bench 独立避免循环依赖）
# ---------------------------------------------------------------------------
def _rank_calibrate(modified: np.ndarray, target: np.ndarray) -> np.ndarray:
    """把 modified 的秩顺序映射到 target 的分位数（保秩单调变换）。"""
    m = np.asarray(modified, dtype=float)
    t = np.sort(np.asarray(target, dtype=float))
    n = len(m)
    idx = np.argsort(np.argsort(m))
    # 分位数插值到 t
    ranks = (np.arange(n) + 0.5) / n
    quant = np.interp(ranks, np.linspace(0, 1, n), t)
    return quant[idx]


# ---------------------------------------------------------------------------
# 1. 波动率时序结构特征（N5 最强命中维度）
# ---------------------------------------------------------------------------
def abs_autocorr(x: np.ndarray, lag: int = 1) -> float:
    """|x| 的滞后自相关（波动率聚集持续性）。

    真实 GARCH 序列大|r|连续出现 → 正 ACF；N5 随机相位打散聚集 →
    ACF 显著下降。lag=1 短期、lag=3/5 中期。
    """
    x = np.asarray(x, dtype=float)
    a = np.abs(x)
    n = len(a)
    if n < lag + 4:
        return np.nan
    r = np.corrcoef(a[:-lag], a[lag:])
    return float(r[0, 1])


def _dfa_hurst(series: np.ndarray, min_box: int = 8, max_box: int | None = None,
               n_box: int = 6) -> float:
    """单序列 DFA H（供波动率长记忆标度使用）。"""
    s = np.asarray(series, dtype=float)
    s = s - s.mean()
    y = np.cumsum(s)
    m = len(y)
    max_box = max_box or m // 4
    scales = []
    for k in range(n_box):
        box = int(round(min_box * (max_box / min_box) ** (k / (n_box - 1))))
        if box < 4 or box > m // 2:
            continue
        n_seg = m // box
        if n_seg < 2:
            continue
        t = np.arange(box)
        f2 = 0.0
        for v in range(n_seg):
            seg = y[v * box:(v + 1) * box]
            p = np.polyfit(t, seg, 1)
            f2 += float(np.mean((seg - np.polyval(p, t)) ** 2))
        scales.append((box, np.sqrt(f2 / n_seg)))
    if len(scales) < 3:
        return np.nan
    log_s = np.log(np.array([sc[0] for sc in scales]))
    log_f = np.log(np.array([sc[1] for sc in scales]))
    return float(np.polyfit(log_s, log_f, 1)[0])


def abs_dfa_hurst(x: np.ndarray) -> float:
    """|r| 的 DFA Hurst（波动率长记忆标度指数）。

    E5 探索最强特征（Cohen's d ≈ +1.02）：GARCH 波动聚集是长记忆
    过程（|r| 强持久），N5 随机相位打散聚集 → |r| 标度指数显著下降。
    """
    x = np.asarray(x, dtype=float)
    a = np.abs(x)
    return _dfa_hurst(a)


# ---------------------------------------------------------------------------
# 2. 双相干（非线性相位耦合强度）
# ---------------------------------------------------------------------------
def mean_bicoherence(x: np.ndarray, nfft: int | None = None,
                     lo_frac: float = 0.25, n_seg: int = 8) -> float:
    """平均归一化双相干 bic^2（Kim & Powers 1979 分段估计）。

    相位耦合（非线性）时各段三谱相干累积 → bic 大；
    相位独立（随机相位）时平均后 B → 0 → bic → 0。
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    nfft = nfft or min(512, max(128, n // n_seg))
    if n < nfft * 2:
        nfft = max(32, n // 4)
    n_seg = max(4, min(n_seg, n // nfft))
    step = max(1, (n - nfft) // max(1, n_seg - 1))
    Xk = np.stack([np.fft.rfft(x[i:i + nfft]) for i in range(0, n - nfft + 1, step)])
    nf = Xk.shape[1]
    n_lo = max(8, min(int(nf * lo_frac), nf // 3))
    f1, f2 = np.meshgrid(np.arange(1, n_lo), np.arange(1, n_lo))
    f3 = f1 + f2
    mask = f3 < nf
    f1, f2, f3 = f1[mask], f2[mask], f3[mask]
    if len(f1) == 0:
        return np.nan
    B = (Xk[:, f1] * Xk[:, f2] * np.conj(Xk[:, f3])).mean(axis=0)
    P1 = (np.abs(Xk[:, f1] * Xk[:, f2]) ** 2).mean(axis=0)
    P2 = (np.abs(Xk[:, f3]) ** 2).mean(axis=0)
    bic2 = np.abs(B) ** 2 / (P1 * P2 + 1e-12)
    return float(np.mean(bic2))


# ---------------------------------------------------------------------------
# 3. 排列熵（|x| 序模式复杂度；对 rank-preserving 校准稳健）
# ---------------------------------------------------------------------------
def permutation_entropy(x: np.ndarray, m: int = 4, tau: int = 1,
                        abs_mode: bool = True) -> float:
    """Bandt-Pompe 排列熵（归一化 [0,1]）。

    对 |x| 计算：GARCH 波动聚集使 |r| 序模式有结构（低熵）；N5 随机
    相位破坏聚集 → 序模式趋随机 → 熵升高。完全随机序 → 1.0。
    """
    x = np.asarray(x, dtype=float)
    if abs_mode:
        x = np.abs(x)
    n = len(x)
    if n < m * tau + 1:
        return np.nan
    counts: dict[tuple, int] = {}
    for i in range(n - m * tau + 1):
        win = x[i: i + m * tau: tau]
        key = tuple(np.argsort(win, kind="stable"))
        counts[key] = counts.get(key, 0) + 1
    arr = np.array(list(counts.values()), dtype=float)
    p = arr / arr.sum()
    h = -np.sum(p * np.log(p))
    return float(h / math.log(math.factorial(m)))


# ---------------------------------------------------------------------------
# 4. IAAFT surrogate 一致性 z-score（N5 直接检验）
# ---------------------------------------------------------------------------
def surrogate_z(x: np.ndarray, stat: str = "abs_acf5",
                n_surr: int = 16, seed: int = 0) -> float:
    """候选序列 vs 其保谱随机相位代理族（IAAFT）的标准化偏离 z。

    IAAFT 代理 = 保留幅度谱 + 随机相位 + 同一边际校准（_rank_calibrate）。
    检验统计量默认 |r| 5阶自相关（实证最强）。

    解读：
      - 真实序列（非线性相位结构）：T(x) 显著偏离代理族 → z 大；
      - N5 伪造（本身即 IAAFT 代理产物）：与代理族一致 → z ≈ 0。
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x - x.mean()

    def _stat(y: np.ndarray, kind: str) -> float:
        if kind == "abs_acf1":
            return abs_autocorr(y, lag=1)
        if kind == "abs_acf5":
            return abs_autocorr(y, lag=5)
        if kind == "abs_acf_sum":
            return float(sum(abs_autocorr(y, lag=k) for k in range(1, 21)))
        if kind == "abs_dfa":
            return abs_dfa_hurst(y)
        if kind == "bicoh":
            return mean_bicoherence(y)
        if kind == "perm_ent":
            return permutation_entropy(y, abs_mode=True)
        raise ValueError(f"未知检验统计量: {kind}")

    def _surr(y: np.ndarray) -> np.ndarray:
        X = np.fft.rfft(y)
        amp = np.abs(X)
        ph = rng.uniform(0.0, 2.0 * np.pi, size=len(amp))
        z = np.fft.irfft(amp * np.exp(1j * ph), n=len(y))
        z = z - z.mean()
        return _rank_calibrate(z, y)   # IAAFT：代理也校准到同一边际

    def _z(kind: str) -> float:
        t0 = _stat(x, kind)
        ts = np.empty(n_surr)
        for i in range(n_surr):
            ts[i] = _stat(_surr(x), kind)
        std = float(np.std(ts))
        if not np.isfinite(std) or std < 1e-12:
            return 0.0
        return float((t0 - float(np.mean(ts))) / std)

    if stat == "max":
        return float(max(abs(_z("abs_acf1")), abs(_z("abs_acf5"))))
    return _z(stat)


# ---------------------------------------------------------------------------
# 特征清单
# ---------------------------------------------------------------------------
SPECTRAL_FEATURE_NAMES = [
    "abs_acf1",          # |r| 一阶自相关（短期波动聚集；N5 命中，d≈0.69）
    "abs_acf3",          # |r| 3阶自相关（波动聚集；N5 命中，d≈0.98）
    "abs_acf5",          # |r| 5阶自相关（中期波动聚集；N5 命中，d≈0.77）
    "abs_dfa",           # |r| DFA Hurst（波动率长记忆；N5 命中，d≈1.02）
    "surr_z",            # IAAFT surrogate z（max(abs_acf1, abs_acf5)）
    "surr_z_dfa",        # IAAFT surrogate z（abs_dfa 统计量；自参照长记忆）
    "surr_z_acf_sum",    # IAAFT surrogate z（Σ|r| ACF(1..20)；总波动持续性）
]


def extract_spectral_features(x: np.ndarray) -> np.ndarray:
    """提取 7 维相位结构特征向量（顺序与 SPECTRAL_FEATURE_NAMES 一致）。"""
    feats = np.array([
        abs_autocorr(x, lag=1),
        abs_autocorr(x, lag=3),
        abs_autocorr(x, lag=5),
        abs_dfa_hurst(x),
        surrogate_z(x, stat="max"),
        surrogate_z(x, stat="abs_dfa"),
        surrogate_z(x, stat="abs_acf_sum"),
    ], dtype=float)
    return feats
