"""E15：ε_G 条件偏差扩测（编辑决定书 Round 4 R5 / SC-3）。

Round 4 R5 要求两件 e14 未做的事：
  1. null family 覆盖从 5 选 2（iid / AR(1)φ=0.3）扩到强相依族，
     新增 AR(1) φ=0.9（强短记忆，仍在线性零族内）与 ARFIMA(0,d,0)
     d=0.25（长记忆，块算子的短记忆零族边界），报每族认证方向
     worst-cell E[W_cert]；
  2. 部署监控流是 95% 重叠滑窗，逐窗独立抽样测的是边际均值；新增
     滑窗依赖流条件偏差实验：单一长流滑窗，逐窗新鲜轨道，直接量
     (a) 边缘均值、(b) 漂移校正财富轨迹上确界、(c) 前窗高值条件下
     的下一窗条件均值，检验 E[W_t | F_{t-1}] ≤ 1+ε_G 是否在重叠流
     下仍成立。

协议与 e14 逐位一致：candidate x=h(y)，轨道五算子混合，读出=部署
五资产编码器 s_θ（A0，与 e2_main 同种子）。W_cert=(K+1)softmax(+s)_x，
W_flag=(K+1)softmax(−s)_x。

产出：results/e15_eg_conditional.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

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

from e2_main import (  # noqa: E402
    CONFIGS, SPLIT_SPEC, SEED, WINDOW, STEP, train_model, logits_var,
)
from mint.operators import OPERATOR_SPECS, ORBIT_NAMES, generate_orbit  # noqa: E402

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")

K = 32
N_TRIALS = 150          # 强相依族新增格
SEED_DIAG = 20260828
EPS_FLAG = 0.009        # 现稿部署用指控方向 ε_G（与 e14 一致，正文 0.009）
H_STREAM = 115          # 滑窗流窗数（监控实验同口径）
T_BASE = WINDOW + (H_STREAM - 1) * STEP + 500  # 长流长度（含 burn-in）


def factory_ar1_strong(rng, phi=0.9):
    buf = 2000
    eps = rng.standard_normal(WINDOW + buf)
    x = np.empty_like(eps)
    x[0] = eps[0] / np.sqrt(1 - phi * phi)
    for t in range(1, len(eps)):
        x[t] = phi * x[t - 1] + eps[t]
    return x[buf:] / max(x[buf:].std(), 1e-12)


def factory_arfima(rng, d=0.25):
    buf = 4000
    n = WINDOW + buf
    eps = rng.standard_normal(n)
    psi = np.empty(n)
    psi[0] = 1.0
    for j in range(1, n):
        psi[j] = psi[j - 1] * (j - 1 + d) / j
    x = np.convolve(eps, psi, mode="full")[:n]
    x = x[buf:]
    return x / max(x.std(), 1e-12)


STRONG_FACTORIES = {"ar1_strong": factory_ar1_strong, "arfima": factory_arfima}

# 滑窗流也纳入全部源族（含 e14 已测的三种，作同口无缝对照）
STREAM_FACTORIES = {
    "iid": lambda rng: rng.standard_normal(T_BASE),
    "ar1": lambda rng: _ar1_stream(rng, 0.3),
    "ar1_strong": lambda rng: _ar1_stream(rng, 0.9),
    "arfima": lambda rng: _arfima_stream(rng, 0.25),
    "gjr": lambda rng: _gjr_stream(rng),
}


def _ar1_stream(rng, phi):
    eps = rng.standard_normal(T_BASE + 2000)
    x = np.empty_like(eps)
    x[0] = eps[0] / np.sqrt(1 - phi * phi)
    for t in range(1, len(eps)):
        x[t] = phi * x[t - 1] + eps[t]
    return x[2000:]


def _arfima_stream(rng, d=0.25):
    n = T_BASE + 4000
    eps = rng.standard_normal(n)
    psi = np.empty(n)
    psi[0] = 1.0
    for j in range(1, n):
        psi[j] = psi[j - 1] * (j - 1 + d) / j
    x = np.convolve(eps, psi, mode="full")[:n]
    return x[4000:]


def _gjr_stream(rng):
    omega, alpha, beta, gamma = 1e-5, 0.02, 0.92, 0.10
    n = T_BASE + 3000
    eps = rng.standard_normal(n)
    x = np.empty_like(eps)
    s2 = omega / max(1e-3, 1 - alpha - beta - gamma / 2)
    for t in range(n):
        x[t] = np.sqrt(s2) * eps[t]
        s2 = omega + alpha * x[t] ** 2 + beta * s2 + gamma * x[t] ** 2 * (x[t] < 0)
        s2 = max(s2, 1e-12)
    x = x[3000:]
    return x / x.std()


def softmax_pair(s0, s):
    allv = np.concatenate([[s0], s])
    m = float(np.max(allv))
    lse = m + float(np.log(np.sum(np.exp(allv - m))))
    w_cert = float(len(allv) * np.exp(float(s0) - lse))
    alln = -allv
    mn = float(np.max(alln))
    lsn = mn + float(np.log(np.sum(np.exp(alln - mn))))
    w_flag = float(len(alln) * np.exp(-float(s0) - lsn))
    return w_cert, w_flag


def main():
    t0 = time.time()
    print("=" * 66)
    print("E15 ε_G 条件偏差扩测（Round 4 R5）")
    print("=" * 66, flush=True)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    encoders = {}
    for ai, (asset, r) in enumerate(returns.items()):
        sp = SPLIT_SPEC[asset]
        train_end = sp["cal"][0] * STEP
        cfg = CONFIGS["A0"]
        enc, head, last, _t = train_model(
            r, train_end, cfg, SEED + ai * 1000, 250, 150, 20)
        encoders[asset] = (enc, head)
        print(f"[{asset}] A0 trained loss={last['loss']:.4f} ({_t:.0f}s)",
              flush=True)

    assets = list(returns.keys())
    rng = np.random.default_rng(SEED_DIAG)
    out = {"n": WINDOW, "K": K, "n_trials": N_TRIALS, "seed": SEED_DIAG,
           "h_stream": H_STREAM, "eps_flag": EPS_FLAG,
           "protocol": "candidate x=h(y); orbit=five-operator mixture; "
                       "reader=deployed per-asset encoder s_theta",
           "strong_cells": {}, "sliding_stream": {}}

    # --- 强相依族逐族认证方向 worst-cell ---------------------------------
    for fname, fac in STRONG_FACTORIES.items():
        for null in ORBIT_NAMES:
            for asset in assets:
                enc, head = encoders[asset]
                w_c = np.empty(N_TRIALS)
                w_f = np.empty(N_TRIALS)
                for t in range(N_TRIALS):
                    y = fac(rng)
                    x = OPERATOR_SPECS[null].fn(y, rng)
                    orbit, _ = generate_orbit(x, K, names=ORBIT_NAMES, rng=rng)
                    s = logits_var(enc, head, [x] + list(orbit), True)
                    w_c[t], w_f[t] = softmax_pair(float(s[0]),
                                                  np.asarray(s[1:]))
                out["strong_cells"][f"{fname}|{null}|{asset}"] = {
                    "e_cert_mean": float(w_c.mean()),
                    "e_flag_mean": float(w_f.mean()),
                    "p_cert_ge_20": float(np.mean(w_c >= 20)),
                    "p_flag_ge_20": float(np.mean(w_f >= 20)),
                }
            cs = [out["strong_cells"][f"{fname}|{null}|{a}"] for a in assets]
            print(f"[cells] {fname:<12}{null:<13} "
                  f"E[Wcert] mean={np.mean([c['e_cert_mean'] for c in cs]):.3f} "
                  f"max={np.max([c['e_cert_mean'] for c in cs]):.3f} | "
                  f"E[Wflag] max={np.max([c['e_flag_mean'] for c in cs]):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # --- 滑窗依赖流条件偏差 -------------------------------------------------
    for fname, fac in STREAM_FACTORIES.items():
        stream = fac(rng)
        for asset in assets:
            enc, head = encoders[asset]
            wc = np.empty(H_STREAM)
            wf = np.empty(H_STREAM)
            for t in range(H_STREAM):
                w = stream[t * STEP: t * STEP + WINDOW]
                orbit, _ = generate_orbit(w, K, names=ORBIT_NAMES, rng=rng)
                s = logits_var(enc, head, [w] + list(orbit), True)
                wc[t], wf[t] = softmax_pair(float(s[0]), np.asarray(s[1:]))
            # 边缘 + 漂移校正财富轨迹 + 前窗高值条件均值
            m_flag = np.cumprod(wf / (1 + EPS_FLAG))
            m_cert = np.cumprod(wc / (1 + EPS_FLAG))
            hi = wf[:-1] >= np.median(wf[:-1])
            cond_flag_hi = float(wf[1:][hi].mean()) if hi.any() else float("nan")
            cond_flag_lo = float(wf[1:][~hi].mean()) if (~hi).any() else float("nan")
            key = f"{fname}|{asset}"
            out["sliding_stream"][key] = {
                "mean_W_cert": float(wc.mean()),
                "mean_W_flag": float(wf.mean()),
                "max_M_flag_corrected": float(m_flag.max()),
                "max_M_cert_corrected": float(m_cert.max()),
                "p_flag_ge_20": float(np.mean(wf >= 20)),
                "p_cert_ge_20": float(np.mean(wc >= 20)),
                "cond_flag_given_prev_hi": cond_flag_hi,
                "cond_flag_given_prev_lo": cond_flag_lo,
            }
        print(f"[stream] {fname:<12} "
              f"mean Wflag="
              f"{np.mean([out['sliding_stream'][f'{fname}|{a}']['mean_W_flag'] for a in assets]):.3f} "
              f"max Mflag="
              f"{np.max([out['sliding_stream'][f'{fname}|{a}']['max_M_flag_corrected'] for a in assets]):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # --- 汇总：强相依族认证方向 worst-cell（正偏差，跨五编码器取最坏） ----
    cert_hi = 0.0
    cert_hi_key = None
    flag_hi = 0.0
    flag_hi_key = None
    for k, c in out["strong_cells"].items():
        if c["e_cert_mean"] - 1.0 > cert_hi:
            cert_hi, cert_hi_key = c["e_cert_mean"] - 1.0, k
        if c["e_flag_mean"] - 1.0 > flag_hi:
            flag_hi, flag_hi_key = c["e_flag_mean"] - 1.0, k
    # 按族（source）聚合认证方向 worst-cell：每族内跨 5 算子 × 5 编码器最坏
    per_family = {}
    for fname in STRONG_FACTORIES:
        vals = [out["strong_cells"][k] for k in out["strong_cells"]
                if k.startswith(fname + "|")]
        per_family[fname] = {
            "max_E_cert": round(max(c["e_cert_mean"] for c in vals), 4),
            "max_E_flag": round(max(c["e_flag_mean"] for c in vals), 4),
            "p_cert_ge_20_max": round(max(c["p_cert_ge_20"] for c in vals), 4),
            "p_flag_ge_20_max": round(max(c["p_flag_ge_20"] for c in vals), 4),
        }
    # 滑窗流条件均值跨源族最坏
    stream_flag_cond_hi_max = max(
        (out["sliding_stream"][k]["cond_flag_given_prev_hi"]
         for k in out["sliding_stream"]
         if not np.isnan(out["sliding_stream"][k]["cond_flag_given_prev_hi"])),
        default=0.0)
    stream_flag_mean_max = max(
        out["sliding_stream"][k]["mean_W_flag"] for k in out["sliding_stream"])
    stream_M_flag_max = max(
        out["sliding_stream"][k]["max_M_flag_corrected"] for k in out["sliding_stream"])

    out["summary"] = {
        "strong_cert_worst_cell": {"eps": round(cert_hi, 4), "cell": cert_hi_key},
        "strong_flag_worst_cell": {"eps": round(flag_hi, 4), "cell": flag_hi_key},
        "strong_per_family": per_family,
        "stream_flag_mean_max": round(stream_flag_mean_max, 4),
        "stream_flag_cond_given_prev_hi_max": round(stream_flag_cond_hi_max, 4),
        "stream_M_flag_corrected_max": round(stream_M_flag_max, 4),
        "tau": H_STREAM,
        "note": "strong_cert eps is max positive deviation of E[W_cert] from 1 "
                "over the two strong-dependence families (5 operators x 5 "
                "encoders x N_TRIALS); stream quantities are over the "
                "95%-overlap sliding stream with fresh orbits per window",
    }

    print("\nsummary:", json.dumps(out["summary"], indent=2), flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    path = os.path.join(RESULTS_DIR, "e15_eg_conditional.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\nsaved: e15_eg_conditional.json ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()