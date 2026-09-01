"""Fractal-Hallu-Bench 幻觉样本构造器（技术路线 §4.2 幻觉构造四法）。

N1 数值置换  ★★  真实数值替换为"相邻量级"错误值（±10%~30%偏差，语法完美）
N2 时间倒置  ★★  将未来行情序列嫁接入过去时间线（跨期拼接/反转）
N3 标度破坏  ★★★ 把日频序列统计特征写入分钟频描述（标度指数错配）
N4 跨域嫁接  ★★★★ 将异市场分形特征写入本市场情景（参数域错配）
"""
from __future__ import annotations

import numpy as np


def n1_numerical_perturb(s: np.ndarray, frac: float = 0.25,
                         mag: float = 0.3, seed: int = 0) -> np.ndarray:
    """N1 数值置换：随机选 frac 比例的点做 ±(1~mag) 倍乘法扰动。

    保持时序结构与统计形状，仅在数值精度层面产生"相邻量级"错误
    ，仿真 LLM 数值问答中的精细数值幻觉（语法完美、量级相近）。
    """
    rng = np.random.default_rng(seed)
    out = s.copy()
    idx = rng.choice(len(s), size=max(1, int(len(s) * frac)), replace=False)
    factors = 1.0 + rng.uniform(-mag, mag, size=len(idx))
    out[idx] = out[idx] * factors
    return out


def n2_time_reversal(s: np.ndarray, seed: int = 0) -> np.ndarray:
    """N2 时间倒置（反转）：未来走势嫁接入过去时间线。

    反转保持单变量边际分布与大多数频谱特征，但破坏因果方向与
    趋势结构，仿真 LLM 将"事后走势"写成"事前预测"的因果幻觉。
    """
    rng = np.random.default_rng(seed)
    return s[::-1].copy()


def n2_time_graft(s: np.ndarray, other: np.ndarray, seed: int = 0) -> np.ndarray:
    """N2 时间嫁接：将另一段序列的前半段嫁接到本序列后半段（跨期拼接）。"""
    rng = np.random.default_rng(seed)
    half = len(s) // 2
    graft = rng.choice(len(other) - half) if len(other) > half else 0
    out = s.copy()
    out[half:] = other[graft:graft + (len(s) - half)]
    return out


def n3_scale_mismatch(s: np.ndarray, factor: int = 5, seed: int = 0) -> np.ndarray:
    """N3 标度破坏：日频序列线性插值放大 factor 倍 → "分钟频"描述。

    插值将相邻点连成直线，高频起伏被抹平 → H_DFA 显著抬升、
    多重分形谱收窄，正是"把日频统计特征写进分钟频描述"造成的
    标度指数错配。
    """
    rng = np.random.default_rng(seed)
    t_old = np.arange(len(s))
    t_new = np.linspace(0, len(s) - 1, len(s) * factor)
    ups = np.interp(t_new, t_old, s)
    # 叠加原始日频噪声的缩微副本，保留部分真实高频（仿真"看起来合理"）
    noise = s[rng.integers(0, len(s), size=len(ups))] * 0.02
    ups = ups + noise
    # 高仿真：边际分布校准至与原始序列一致（仅保留标度结构错位）
    return calibrate_marginal(ups, s)


def calibrate_marginal(modified: np.ndarray, target: np.ndarray) -> np.ndarray:
    """边际分布校准（高仿真幻觉核心步骤）：分位数映射。

    将 modified 的秩结构保留，但把其边际分布完全替换为 target 的
    经验分布（排序统计量映射），使幻觉序列的均值/方差/偏度/峰度
    及全部高阶矩与真实序列一致，仅保留结构（标度/长记忆）错位。
    统计量基线在此类样本上完全失效，唯有分形结构特征可检。
    """
    t_sorted = np.sort(target)
    n_t = len(t_sorted)
    # 分位数映射（支持不同长度）：modified 每元素按其排名分位取 target 经验分位数
    ranks = np.argsort(np.argsort(modified))
    fracs = (ranks + 0.5) / len(modified)
    idx = np.clip((fracs * n_t).astype(int), 0, n_t - 1)
    return t_sorted[idx]


def n4_cross_domain(base: np.ndarray, seed: int = 0) -> np.ndarray:
    """N4 跨域嫁接：以本市场序列的"反向"参数域生成嫁接序列。

    仿真：把美股（高H、弱多重分形）特征写入 A 股情景描述。
    具体实现：对 base 做离散小波域系数衰减 + 重排，改变其
    长记忆结构而不改变总波动率，产生"另一种市场"的统计面貌。
    """
    from scipy.signal import medfilt
    rng = np.random.default_rng(seed)
    med = medfilt(base, kernel_size=11)          # 提取趋势骨架
    resid = base - med                            # 高频残差
    # 破坏长记忆：残差重排（去自相关）+ 骨架随机游走替换
    permuted = rng.permutation(resid)
    rw = np.cumsum(rng.normal(0, np.std(permuted), size=len(base)))
    out = med + 0.6 * permuted + 0.4 * rw
    out = out - out.mean()
    out = out * (np.std(base) / max(np.std(out), 1e-12))
    # 高仿真：边际分布校准至与原始序列一致（仅保留长记忆/参数域错位）
    return calibrate_marginal(out, base)


def n5_phase_forge(s: np.ndarray, seed: int = 0) -> np.ndarray:
    """N5 高仿真幻觉：相位随机化保谱伪造（E4 对抗攻击核心）。

    对真实序列做 FFT → 保留幅度谱、随机化相位 → IFFT 重构：
    - 功率谱形状（PSD 斜率 beta_tail）、DFA H、长记忆结构完全保留
      （因为这些由幅度谱决定）；
    - 具体相位/时序形态全新 → 仿真 LLM "逆向工程真实统计特征后
      伪造内容"（R6 攻击模型中攻击者已知检验器特征、刻意拟合）。
    - 最后做边际分布校准，使均值/方差/高阶矩与真实一致。
    ， 统计层完全失效，分形指纹层接近不可检（H_DFA 不变），
       是比 N4 更难的对抗样本：攻击者"过拟合"了检验器的频域特征。
    """
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(s)
    amp = np.abs(X)
    phase = rng.uniform(0, 2 * np.pi, size=len(amp))
    X2 = amp * np.exp(1j * phase)
    out = np.fft.irfft(X2, n=len(s))
    out = out - out.mean()
    out = out * (np.std(s) / max(np.std(out), 1e-12))
    return calibrate_marginal(out, s)


def n5_blend(s: np.ndarray, eps: float = 0.5, seed: int = 0) -> np.ndarray:
    """N5 对抗强度混合：eps∈[0,1] 控制伪造比例（攻击强度扫描）。

    out = (1-eps)·真实 + eps·相位伪造 → eps=0 为真实、eps=1 为纯伪造。
    用于绘制"检出率 vs 攻击强度"曲线，量化检验器的鲁棒性边界。
    """
    forged = n5_phase_forge(s, seed=seed)
    out = (1.0 - eps) * s + eps * forged
    return calibrate_marginal(out, s)


HALLUCINATION_BUILDERS = {
    "N1数值置换": n1_numerical_perturb,
    "N2时间倒置": n2_time_reversal,
    "N3标度破坏": n3_scale_mismatch,
    "N4跨域嫁接": n4_cross_domain,
}
