"""MINT 五替代算子库：制造零假设（manufactured nulls）的核心组件。

轨道族设计（整体研究方案 §3.1）：五个算子各对应一个零假设，
并集构成"保住审计可见低阶指纹前提下攻击者能做的一切"：

    permute       iid 零假设：保边际（精确）、毁全部时间依赖；精确可交换
    time_reverse  可逆零假设：保边际/振幅谱/ACF（精确）、毁时间箭头；
                  N2 攻击的原型；对可逆过程类精确
    aaft          线性高斯零假设：近似保边际与谱、毁相位依赖
    iaaft         保谱线性零假设：精确保边际、近似保振幅谱（迭代收敛）、
                  毁相位依赖；N5 攻击的原型；可交换性近似（ε_G 披露）
    block_perm    短记忆零假设：保块内短程依赖（精确）、毁长程依赖；
                  块可交换精确

统一接口：fn(x, rng, **kw) -> np.ndarray（与 x 等长）。
generate_orbit 对窗口做算子混合抽样，得到训练用原则性负样本族。

诊断工具：
    surrogate_diagnostics   事后量化"算子保住了什么"（边际/谱/ACF）
    null_uniformity_diag    近似算子的可交换性偏差 ε_G 代理（秩均匀性
                            KS 检验；T3 有效性披露用）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# 精确算子
# ---------------------------------------------------------------------------
def permute(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """iid 零假设：整体随机置换。边际分布逐点保持，时间依赖全部销毁。"""
    return rng.permutation(np.asarray(x, dtype=float))


def time_reverse(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """可逆零假设：时间反转。边际、振幅谱、ACF 与全部对称统计量精确
    保持；时间箭头（不可逆结构，如杠杆不对称）销毁。N2 攻击原型。"""
    return np.asarray(x, dtype=float)[::-1].copy()


def block_perm(x: np.ndarray, rng: np.random.Generator,
               block: int = 20) -> np.ndarray:
    """短记忆零假设：分块置换。块内短程依赖（lag < block）保持，
    长程依赖销毁；对块可交换类精确。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_blk = n // block
    if n_blk < 2:
        return permute(x, rng)
    head = x[: n_blk * block].reshape(n_blk, block)
    return head[rng.permutation(n_blk)].reshape(-1)


# ---------------------------------------------------------------------------
# 傅里叶族近似算子
# ---------------------------------------------------------------------------
def _rank_map(target_sorted: np.ndarray, y: np.ndarray) -> np.ndarray:
    """把 y 的秩顺序映射到 target 的分位数（保秩单调变换）。"""
    order = np.argsort(np.argsort(y))
    return target_sorted[order]


def aaft(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """AAFT（Theiler et al. 1992）：高斯化 → 相位随机化 → 逆高斯化。

    线性高斯零假设代理：边际精确保持（秩映射），振幅谱近似保持。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    xs = np.sort(x)
    # 高斯化：经验分位 → 标准正态分位
    ranks = np.argsort(np.argsort(x))
    z = stats.norm.ppf((ranks + 0.5) / n)
    # 相位随机化
    Z = np.fft.rfft(z)
    ph = rng.uniform(0.0, 2.0 * np.pi, size=len(Z))
    z2 = np.fft.irfft(np.abs(Z) * np.exp(1j * ph), n=n)
    # 逆映射回原边际
    return _rank_map(xs, z2)


def iaaft(x: np.ndarray, rng: np.random.Generator, max_iter: int = 200,
          tol: float = 1e-6) -> np.ndarray:
    """IAAFT（Schreiber & Schmitz 1996）：谱步与边际步交替迭代至不动点。

    保谱线性零假设：边际精确保持（秩映射），振幅谱至迭代容差保持。
    N5 相位伪造攻击的原型；收敛诊断见 iaaft_diag。"""
    return _iaaft(x, rng, max_iter, tol)[0]


def iaaft_diag(x: np.ndarray, rng: np.random.Generator, max_iter: int = 200,
               tol: float = 1e-6) -> tuple[np.ndarray, dict]:
    """IAAFT 带收敛诊断：返回（代理, {n_iter, rel_spec_err}）。"""
    return _iaaft(x, rng, max_iter, tol)


def _iaaft(x: np.ndarray, rng: np.random.Generator, max_iter: int,
           tol: float) -> tuple[np.ndarray, dict]:
    x = np.asarray(x, dtype=float)
    n = len(x)
    X = np.fft.rfft(x)
    amp = np.abs(X)
    amp_scale = float(np.max(amp)) + 1e-12
    xs = np.sort(x)

    ph = rng.uniform(0.0, 2.0 * np.pi, size=len(amp))
    s = np.fft.irfft(amp * np.exp(1j * ph), n=n)

    prev_err = np.inf
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        # 谱步：替换振幅谱，保留当前相位
        S = np.fft.rfft(s)
        s = np.fft.irfft(amp * np.exp(1j * np.angle(S)), n=n)
        # 边际步：按当前秩把原值放回
        s = _rank_map(xs, s)
        err = float(np.max(np.abs(np.abs(np.fft.rfft(s)) - amp)) / amp_scale)
        if err < tol or abs(prev_err - err) < 1e-12:
            break
        prev_err = err
    return s, {"n_iter": n_iter, "rel_spec_err": err}


# ---------------------------------------------------------------------------
# 算子注册表
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperatorSpec:
    name: str
    fn: Callable[..., np.ndarray]
    null: str            # 对应零假设
    preserves: str       # 保什么
    destroys: str        # 毁什么
    exchangeability: str  # "exact" | "near"


OPERATOR_SPECS: dict[str, OperatorSpec] = {
    "permute": OperatorSpec(
        permute.__name__, permute, "iid（可交换）",
        "边际分布（精确）", "全部时间依赖", "exact"),
    "time_reverse": OperatorSpec(
        time_reverse.__name__, time_reverse, "时间可逆过程",
        "边际/振幅谱/ACF（精确）", "时间箭头（不可逆结构）", "exact"),
    "aaft": OperatorSpec(
        aaft.__name__, aaft, "线性高斯过程",
        "边际（精确）、谱（近似）", "相位依赖", "near"),
    "iaaft": OperatorSpec(
        iaaft.__name__, iaaft, "保谱线性过程（任意边际）",
        "边际（精确）、振幅谱（至容差）", "相位依赖", "near"),
    "block_perm": OperatorSpec(
        block_perm.__name__, block_perm, f"短记忆（块可交换）",
        f"块内短程依赖（lag<block）", "长程依赖", "exact"),
}

ORBIT_NAMES: list[str] = ["permute", "time_reverse", "aaft", "iaaft",
                          "block_perm"]


def generate_orbit(w: np.ndarray, n: int,
                   names: Sequence[str] = ORBIT_NAMES,
                   rng: np.random.Generator | None = None,
                   **op_kwargs) -> tuple[np.ndarray, list[str]]:
    """对窗口 w 生成 n 条混合轨道样本。

    算子轮转分配（均匀混合）；返回 (n, len(w)) 数组与逐样本算子名。
    训练时的原则性负样本即由此产生（方案 §3.2）。"""
    if rng is None:
        rng = np.random.default_rng(0)
    names = list(names)
    surrs = np.empty((n, len(w)), dtype=float)
    used: list[str] = []
    for i in range(n):
        op = names[i % len(names)]
        surrs[i] = OPERATOR_SPECS[op].fn(w, rng, **op_kwargs)
        used.append(op)
    return surrs, used


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------
def surrogate_diagnostics(x: np.ndarray, s: np.ndarray,
                          max_lag: int = 20) -> dict:
    """量化代理 s 相对原序列 x 保住了什么：边际 KS、振幅谱相对误差、
    ACF 偏差、ACF1 保持率。"""
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    ks = stats.ks_2samp(x, s)
    ax = np.abs(np.fft.rfft(x - x.mean()))
    as_ = np.abs(np.fft.rfft(s - s.mean()))
    spec_err = float(np.max(np.abs(as_ - ax)) / (np.max(ax) + 1e-12))

    def acf(z: np.ndarray, lag: int) -> float:
        z = z - z.mean()
        return float(np.corrcoef(z[:-lag], z[lag:])[0, 1])

    acf_dev = float(np.mean([abs(acf(s, k) - acf(x, k))
                             for k in range(1, max_lag + 1)]))
    return {
        "marginal_ks_stat": float(ks.statistic),
        "marginal_ks_p": float(ks.pvalue),
        "rel_spec_err": spec_err,
        "mean_abs_acf_dev_lag1_20": acf_dev,
        "acf1_x": acf(x, 1), "acf1_s": acf(s, 1),
    }


def null_uniformity_diag(null_factory: Callable[[np.random.Generator],
                                                 np.ndarray],
                         operator: str, n_trials: int = 200,
                         orbit_k: int = 19,
                         rng: np.random.Generator | None = None) -> dict:
    """可交换性偏差 ε_G 的代理诊断（T3 披露用）。

    每轮从真零假设工厂抽 x，生成其 k 条轨道，记录 x 在 {x}∪轨道 中的
    秩。可交换性成立 ⇔ 秩在 {0..k} 上均匀。秩为离散值，用卡方检验
    （连续 KS 对离散秩反保守）；直方图最大偏差即 ε_G 的经验量化。"""
    if rng is None:
        rng = np.random.default_rng(0)
    spec = OPERATOR_SPECS[operator]
    ranks = np.empty(n_trials, dtype=int)
    for t in range(n_trials):
        x = null_factory(rng)
        surrs, _ = generate_orbit(x, orbit_k, names=[spec.name], rng=rng)
        # 秩按任意对称分数计算；可交换性结论与分数选择无关
        ranks[t] = int(np.sum(_score(surrs) >= _score(x[np.newaxis, :])[0]))
    exp_p = 1.0 / (orbit_k + 1)
    counts = np.bincount(ranks, minlength=orbit_k + 1)
    hist = counts / n_trials
    chi = stats.chisquare(counts)
    return {
        "operator": operator, "n_trials": n_trials, "orbit_k": orbit_k,
        "rank_mean": float(ranks.mean()),
        "expected_rank_mean": (orbit_k + 1) / 2.0,
        "rank_hist_max_dev": float(np.max(np.abs(hist - exp_p))),
        "chi2_p": float(chi.pvalue),
    }


def _score(Z: np.ndarray) -> np.ndarray:
    """占位对称分数：|ACF1| + |x| 均值。仅用于秩均匀性诊断。"""
    out = np.empty(len(Z))
    for i, z in enumerate(Z):
        zc = z - z.mean()
        acf1 = float(np.corrcoef(zc[:-1], zc[1:])[0, 1]) if len(z) > 2 else 0.0
        out[i] = abs(acf1) + float(np.mean(np.abs(z)))
    return out


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    """AR(1) 高斯工厂（IAAFT 零假设成员）。"""
    e = rng.standard_normal(n + 50)
    y = np.empty(n + 50)
    y[0] = 0.0
    for t in range(1, n + 50):
        y[t] = phi * y[t - 1] + e[t]
    return y[50:]


# ---------------------------------------------------------------------------
# 自检：python operators.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1000
    # AR(1) + GARCH(1,1) 风格合成序列（金融日频替代）
    eps = rng.standard_normal(n + 200)
    r = np.empty(n + 200)
    sig2 = 1e-4
    for t in range(1, n + 200):
        sig2 = 1e-5 + 0.08 * r[t - 1] ** 2 + 0.90 * sig2
        r[t] = np.sqrt(sig2) * eps[t]
    x = r[200:]

    print(f"自检序列: n={len(x)} std={x.std():.5f}")
    print(f"{'算子':<14}{'边际KS':>9}{'谱误差':>9}{'ACF偏差':>9}  交换性")
    print("-" * 60)
    for name in ORBIT_NAMES:
        s = OPERATOR_SPECS[name].fn(x, rng)
        d = surrogate_diagnostics(x, s)
        print(f"{name:<14}{d['marginal_ks_stat']:>9.4f}"
              f"{d['rel_spec_err']:>9.4f}{d['mean_abs_acf_dev_lag1_20']:>9.4f}"
              f"  {OPERATOR_SPECS[name].exchangeability}")

    s2, diag = iaaft_diag(x, rng)
    print(f"\nIAAFT 收敛: {diag['n_iter']} 轮, 谱误差 {diag['rel_spec_err']:.2e}")

    orbit, used = generate_orbit(x, 10, rng=rng)
    print(f"轨道生成: 10 条（{set(used)}），shape={orbit.shape}")

    # 精确算子的可交换性应在真零假设下成立：置换算子 × iid 工厂
    d_perm = null_uniformity_diag(
        lambda g: g.standard_normal(200), "permute", n_trials=200,
        orbit_k=19, rng=rng)
    print(f"置换×iid 秩均匀性: chi2 p={d_perm['chi2_p']:.3f} "
          f"(应不拒绝均匀)")
    # 近似算子偏差量化：IAAFT × 线性 AR(1) 高斯工厂
    d_iaaft = null_uniformity_diag(
        lambda g: _ar1(g, 200, 0.3), "iaaft", n_trials=200,
        orbit_k=19, rng=rng)
    print(f"IAAFT×AR(1) 秩均匀性: chi2 p={d_iaaft['chi2_p']:.3f} "
          f"直方图最大偏差={d_iaaft['rank_hist_max_dev']:.4f}")
