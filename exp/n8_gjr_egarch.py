"""N8：带杠杆的并集外参数化敌手（GJR-GARCH / EGARCH，ML 拟合）。

实验目的：N6 用对称 t 创新拟合 GARCH(1,1)，
恰好缺失编码器最易读的签名（杠杆不对称），其 AUC 0.895 与 N2 共享
同一检测坐标，"编码器只是放大的杠杆读取器"这一替代解释未被排除。
N8 与 N6 同在算子并集之外，但显式携带杠杆不对称（GJR 的 γ 项、
EGARCH 的 log-|σ| 非对称响应），构成签名正交的压力测试。

两型敌手 × 两型创新（四配置）：
  GJR-GARCH(1,1)   h_t = w + (a + γ·1[r_{t-1}<0]) r²_{t-1} + b·h_{t-1}
  EGARCH(1,1)      log h_t = w + a(|e_{t-1}| - E|e|) + θ e_{t-1} + b log h
  创新：偏态 t（skew=−0.2，负偏）与拟合层经验残差重采样（保真最高）

若 MINT 检出 N8 显著弱于 N6：如报告并收缩 C4 主张为"指纹保持类
伪造的泛化"；若仍强：替代解释被排除。认证行为一并报告（领域 W6：
模仿型敌手认证率）。

产出：results/n8_gjr_egarch.npz + results/n8_gjr_egarch.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import skewnorm, t as tdist

MINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSCT_ROOT = os.path.join(MINT_ROOT, "shared_infra", "fractal_consistency")
for p in (MINT_ROOT, FSCT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")

WINDOW, STEP = 1000, 50
SPLIT_SPEC = {
    "equity":  {"n_fit": 15, "cal": (34, 43), "test": (63, 99), "cap": 8},
    "bond":    {"n_fit": 15, "cal": (34, 43), "test": (63, 94), "cap": 8},
    "fx":      {"n_fit": 11, "cal": (31, 36), "test": (37, 42), "cap": 6},
    "gold":    {"n_fit": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
    "copper":  {"n_fit": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
}
PER_ASSET = 4          # 每资产每配置条数（对齐测试窗数下限）
CONFIGS = ["gjr_skewt", "gjr_boot", "egarch_skewt", "egarch_boot"]
SEED = 20260821


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


# ---------------------------------------------------------------------------
# GJR-GARCH(1,1) MLE
# ---------------------------------------------------------------------------
def fit_gjr(r):
    r = np.asarray(r, float) - np.mean(r)
    n = len(r)
    iv = np.var(r)

    def nll(p):
        w, a, g, b = p
        if w <= 0 or a < 0 or b < 0 or (a + g) < 0 or (a + b + max(g, 0)) >= 0.999:
            return 1e12
        h = np.empty(n)
        h[0] = iv
        for t in range(1, n):
            ind = 1.0 if r[t - 1] < 0 else 0.0
            h[t] = w + (a + g * ind) * r[t - 1] ** 2 + b * h[t - 1]
        eps = 1e-12
        h = np.maximum(h, eps)
        return float(np.sum(np.log(h[1:]) + r[1:] ** 2 / h[1:]))

    best, bv = None, np.inf
    for a0, g0, b0 in [(0.05, 0.06, 0.88), (0.08, 0.08, 0.82),
                       (0.03, 0.10, 0.85)]:
        try:
            res = minimize(nll, x0=np.array([iv * 0.02, a0, g0, b0]),
                           method="L-BFGS-B",
                           bounds=[(1e-12, None)] * 4,
                           options={"maxiter": 400})
            if res.fun < bv:
                bv, best = res.fun, res.x
        except Exception:
            continue
    return tuple(map(float, best)) if best is not None else (None,) * 4


def gjr_simulate(w, a, g, b, n, h0, innov_fn, rng):
    r = np.empty(n)
    h = np.empty(n)
    h[0] = h0
    r[0] = np.sqrt(h0) * innov_fn(rng)
    for t in range(1, n):
        ind = 1.0 if r[t - 1] < 0 else 0.0
        h[t] = max(w + (a + g * ind) * r[t - 1] ** 2 + b * h[t - 1], 1e-14)
        r[t] = np.sqrt(h[t]) * innov_fn(rng)
    return r


# ---------------------------------------------------------------------------
# EGARCH(1,1) MLE（高斯 QML 简化拟合，模拟时换目标创新）
# ---------------------------------------------------------------------------
def fit_egarch(r):
    r = np.asarray(r, float) - np.mean(r)
    n = len(r)
    iv = np.var(r)

    def nll(p):
        w, a, th, b = p
        if abs(b) >= 0.999 or a < 0:
            return 1e12
        lh = np.empty(n)
        lh[0] = np.log(iv)
        for t in range(1, n):
            e = r[t - 1] / np.sqrt(np.exp(lh[t - 1]))
            lh[t] = w + a * (abs(e) - np.sqrt(2 / np.pi)) + th * e + b * lh[t - 1]
        h = np.exp(np.clip(lh, -20, 20))
        eps = 1e-12
        return float(np.sum(0.5 * lh[1:] + 0.5 * r[1:] ** 2 / np.maximum(h[1:], eps)))

    best, bv = None, np.inf
    for a0, t0, b0 in [(0.10, -0.10, 0.90), (0.15, -0.15, 0.85),
                       (0.08, -0.05, 0.92)]:
        try:
            res = minimize(nll, x0=np.array([np.log(iv) * 0.05, a0, t0, b0]),
                           method="L-BFGS-B", options={"maxiter": 400})
            if res.fun < bv:
                bv, best = res.fun, res.x
        except Exception:
            continue
    return tuple(map(float, best)) if best is not None else (None,) * 4


def egarch_simulate(w, a, th, b, n, h0, innov_fn, rng):
    r = np.empty(n)
    lh = np.empty(n)
    lh[0] = np.log(h0)
    r[0] = np.sqrt(h0) * innov_fn(rng)
    for t in range(1, n):
        e = r[t - 1] / np.sqrt(np.exp(lh[t - 1]))
        lh[t] = w + a * (abs(e) - np.sqrt(2 / np.pi)) + th * e + b * lh[t - 1]
        lh[t] = np.clip(lh[t], -20, 20)
        r[t] = np.sqrt(np.exp(lh[t])) * innov_fn(rng)
    return r


# ---------------------------------------------------------------------------
# 创新分布
# ---------------------------------------------------------------------------
def skewt_innov(rng):
    """偏态 t(5)，skew=−0.2：单位方差、负偏（金融收益典型）。"""
    x = skewnorm.rvs(-0.2, random_state=rng) * 0.98 + rng.standard_t(5) * 0.2
    return float(x / 1.045)   # 近似单位方差


def boot_innov_factory(fit_resid):
    resid = np.asarray(fit_resid, float)
    resid = resid / max(np.std(resid), 1e-12)

    def innov(rng):
        return float(resid[rng.integers(0, len(resid))])
    return innov


def resid_from_gjr(r, w, a, g, b):
    r = np.asarray(r, float)
    n = len(r)
    h = np.empty(n)
    h[0] = np.var(r)
    for t in range(1, n):
        ind = 1.0 if r[t - 1] < 0 else 0.0
        h[t] = max(w + (a + g * ind) * r[t - 1] ** 2 + b * h[t - 1], 1e-14)
    return r[1:] / np.sqrt(h[1:])


def resid_from_egarch(r, w, a, th, b):
    r = np.asarray(r, float)
    n = len(r)
    lh = np.empty(n)
    lh[0] = np.log(np.var(r))
    for t in range(1, n):
        e = r[t - 1] / np.sqrt(np.exp(lh[t - 1]))
        lh[t] = np.clip(
            w + a * (abs(e) - np.sqrt(2 / np.pi)) + th * e + b * lh[t - 1],
            -20, 20)
    return r[1:] / np.sqrt(np.exp(lh[1:]))


def main():
    print("=" * 66)
    print("N8 带杠杆并集外敌手：GJR/EGARCH × 偏态t/经验重采样")
    print("=" * 66, flush=True)
    t0 = time.time()
    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    out = {}
    params_log = {}
    for ai, (asset, r) in enumerate(returns.items()):
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        fit_wins = wins[:sp["n_fit"]]
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        target_std = float(np.mean([np.std(w) for w in test_wins]))
        allfit = np.concatenate(fit_wins)
        allfit = allfit - allfit.mean()
        rng = np.random.default_rng(SEED + ai * 100)

        gjr = fit_gjr(allfit)
        eg = fit_egarch(allfit)
        resid_gjr = resid_from_gjr(allfit, *gjr)
        resid_eg = resid_from_egarch(allfit, *eg)
        params_log[asset] = {
            "gjr_wagr_b": [round(v, 5) for v in gjr],
            "egarch_wath_b": [round(v, 5) for v in eg],
            "gjr_persistence": round(gjr[1] + gjr[3] + max(gjr[2], 0) / 2, 4),
            "egarch_persistence": round(abs(eg[3]), 4),
            "n_test": len(test_wins)}

        iv = np.var(allfit)
        sims = {
            "gjr_skewt": lambda: gjr_simulate(*gjr, WINDOW, iv, skewt_innov, rng),
            "gjr_boot": lambda: gjr_simulate(
                *gjr, WINDOW, iv, boot_innov_factory(resid_gjr), rng),
            "egarch_skewt": lambda: egarch_simulate(
                *eg, WINDOW, iv, skewt_innov, rng),
            "egarch_boot": lambda: egarch_simulate(
                *eg, WINDOW, iv, boot_innov_factory(resid_eg), rng),
        }
        for cfg in CONFIGS:
            for j in range(PER_ASSET):
                x = sims[cfg]()
                x = x - x.mean()
                x = x * (target_std / max(np.std(x), 1e-12))
                out[f"{asset}/{cfg}/{j}"] = x
        print(f"[{asset}] GJR(w,a,g,b)={np.round(gjr,4).tolist()} | "
              f"EGARCH(w,a,th,b)={np.round(eg,4).tolist()}", flush=True)

    np.savez_compressed(
        os.path.join(RESULTS_DIR, "n8_gjr_egarch.npz"), **out)
    with open(os.path.join(RESULTS_DIR, "n8_gjr_egarch.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "protocol": {
                "models": "GJR(1,1) & EGARCH(1,1), MLE on fit layer, "
                          "per-asset params",
                "innovations": {"skewt": "skewnorm(-0.2)+t(5) mix, unit var",
                                "boot": "fit-layer residual bootstrap"},
                "per_asset_per_config": PER_ASSET,
                "scale_align": "asset test-layer mean std",
                "leverage_by_construction": True,
                "configs": CONFIGS},
            "params": params_log,
            "n_series": len(out),
            "elapsed_s": round(time.time() - t0, 1),
        }, fh, ensure_ascii=False, indent=2)
    print(f"完成：{len(out)} 条（{len(CONFIGS)} 配置 × {PER_ASSET} 条"
          f" × 5 资产）已存 npz+json  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
