"""E14：部署编码器读出下的 ε_G 测量（R3）。

编辑决定书 R3：定理 3 的 anytime 界携带 (1+ε_G)^τ 因子，而 ε_G 此前
只用两个手工代理读出（classical/arrow）测量，从未用部署编码器本身；
"编码器不占据任何盲读出格"是论证不是测量。要求：以编码器分数 s_θ
为读出重跑附录 A 诊断全部 30 格；报告编码器专属 ε_G；声明 115 窗
监控实验运行的缩放水平与有限视界有效界。

协议（与 _diag_eG_deployed.py v2 相同的部署语义）：
  零假设候选 x = h_0(y)（y 来自工厂 iid/AR(1)/GJR，h_0 为被检算子）
  轨道 O(x) = generate_orbit(x, K=32, 五算子混合)
  读出 = 部署编码器 s_θ（五资产各训一个 A0，与 e2_main 同协议）
  报告两个方向的 e 值（都是部署在用的货币）：
    W_cert(x) = (K+1)·softmax(+s)_x   认证方向（定理 3 的 W）
    W_flag(x) = (K+1)·softmax(−s)_x   判伪方向（V 通道监控货币）
  ε_G^cert / ε_G^flag = E[W] 相对 1 的最大正偏差（负偏差保守不计）

产出：results/e14_eg_encoder.json
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
    CONFIGS, SPLIT_SPEC, SEED, WINDOW, STEP, span_windows, train_model,
    logits_var,
)
from mint.operators import OPERATOR_SPECS, ORBIT_NAMES, generate_orbit  # noqa: E402

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")

K = 32
N_TRIALS = 300
SEED_DIAG = 20260820


def factory_iid(rng):
    return rng.standard_normal(WINDOW)


def factory_ar1(rng, phi=0.3):
    eps = rng.standard_normal(WINDOW + 200)
    x = np.empty_like(eps)
    x[0] = eps[0] / np.sqrt(1 - phi * phi)
    for t in range(1, len(eps)):
        x[t] = phi * x[t - 1] + eps[t]
    return x[200:]


def factory_gjr(rng):
    omega, alpha, beta, gamma = 1e-5, 0.02, 0.92, 0.10
    eps = rng.standard_normal(WINDOW + 500)
    x = np.empty_like(eps)
    s2 = omega / max(1e-3, 1 - alpha - beta - gamma / 2)
    for t in range(len(eps)):
        x[t] = np.sqrt(s2) * eps[t]
        s2 = omega + alpha * x[t] ** 2 + beta * s2 + gamma * x[t] ** 2 * (x[t] < 0)
        s2 = max(s2, 1e-12)
    x = x[500:]
    return x / x.std()


FACTORIES = {"iid": factory_iid, "ar1": factory_ar1, "gjr": factory_gjr}


def softmax_pair(s0, s):
    """单窗双向 e 值：W_cert = (K+1)softmax(+s)_x, W_flag = (K+1)softmax(−s)_x。"""
    allv = np.concatenate([[s0], s])
    m = float(np.max(allv))
    lse = m + float(np.log(np.sum(np.exp(allv - m))))
    w_cert = float((len(allv)) * np.exp(float(s0) - lse))
    alln = -allv
    mn = float(np.max(alln))
    lsn = mn + float(np.log(np.sum(np.exp(alln - mn))))
    w_flag = float((len(alln)) * np.exp(-float(s0) - lsn))
    rank = int(np.sum(s >= s0))
    return w_cert, w_flag, rank


def main():
    t0 = time.time()
    print("=" * 66)
    print(f"E14 部署编码器 ε_G 诊断（R3）：{N_TRIALS} trials × 30 cells × 5 encoders")
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
        print(f"[{asset}] A0 trained loss={last['loss']:.4f} "
              f"({_t:.0f}s)", flush=True)

    rng = np.random.default_rng(SEED_DIAG)
    out = {"n": WINDOW, "K": K, "n_trials": N_TRIALS, "seed": SEED_DIAG,
           "protocol": "candidate x = h(y), orbit = five-operator mixture, "
                       "reader = deployed per-asset encoder s_theta; "
                       "W_cert=(K+1)softmax(+s)_x, W_flag=(K+1)softmax(-s)_x",
           "cells": {}}

    assets = list(returns.keys())
    for fname, fac in FACTORIES.items():
        for null in ORBIT_NAMES:
            for asset in assets:
                enc, head = encoders[asset]
                w_c = np.empty(N_TRIALS)
                w_f = np.empty(N_TRIALS)
                ranks = np.empty(N_TRIALS)
                for t in range(N_TRIALS):
                    y = fac(rng)
                    x = OPERATOR_SPECS[null].fn(y, rng)
                    orbit, _ = generate_orbit(x, K, names=ORBIT_NAMES, rng=rng)
                    s = logits_var(enc, head, [x] + list(orbit), True)
                    w_c[t], w_f[t], ranks[t] = softmax_pair(float(s[0]),
                                                            np.asarray(s[1:]))
                cell = {
                    "mean_rank": float(ranks.mean()),
                    "uniform_rank": (K + 1) / 2.0,
                    "e_cert_mean": float(w_c.mean()),
                    "e_flag_mean": float(w_f.mean()),
                    "p_cert_ge_20": float(np.mean(w_c >= 20)),
                    "p_flag_ge_20": float(np.mean(w_f >= 20)),
                }
                out["cells"][f"{fname}|{null}|{asset}"] = cell
            # 每格打印五编码器聚合
            cs = [out["cells"][f"{fname}|{null}|{a}"] for a in assets]
            print(f"{fname:<5}{null:<13} rank="
                  f"{np.mean([c['mean_rank'] for c in cs]):6.2f} "
                  f"E[Wcert]={np.mean([c['e_cert_mean'] for c in cs]):.3f}"
                  f"(max {np.max([c['e_cert_mean'] for c in cs]):.3f}) "
                  f"E[Wflag]={np.mean([c['e_flag_mean'] for c in cs]):.3f}"
                  f"(max {np.max([c['e_flag_mean'] for c in cs]):.3f}) "
                  f"({time.time()-t0:.0f}s)", flush=True)

    eps_cert, eps_flag = 0.0, 0.0
    arg_cert = arg_flag = None
    for key, c in out["cells"].items():
        if c["e_cert_mean"] - 1.0 > eps_cert:
            eps_cert, arg_cert = c["e_cert_mean"] - 1.0, key
        if c["e_flag_mean"] - 1.0 > eps_flag:
            eps_flag, arg_flag = c["e_flag_mean"] - 1.0, key
    out["summary"] = {
        "eps_G_cert": round(eps_cert, 4), "argmax_cert": arg_cert,
        "eps_G_flag": round(eps_flag, 4), "argmax_flag": arg_flag,
        "p_any_cert_ge_20": float(np.mean(
            [c["p_cert_ge_20"] for c in out["cells"].values()])),
        "p_any_flag_ge_20": float(np.mean(
            [c["p_flag_ge_20"] for c in out["cells"].values()])),
        "note": "negative deviations are conservative and not counted; "
                "eps_G is the maximal positive deviation of E[W] from 1 "
                "across all 150 cells (30 protocol cells x 5 encoders)",
    }
    # 缩放水平：115 窗监控实验（V 通道 Ville 界）
    tau = 115
    for tag, eps in (("cert", eps_cert), ("flag", eps_flag)):
        out["summary"][f"scaled_alpha_{tag}_tau{tau}"] = round(
            0.05 / (1 + eps) ** tau, 5)
        out["summary"][f"finite_horizon_bound_{tag}_tau{tau}"] = round(
            0.05 * (1 + eps) ** tau, 5)

    print("\nsummary:", json.dumps(out["summary"], indent=2), flush=True)

    out["elapsed_s"] = round(time.time() - t0, 1)
    path = os.path.join(RESULTS_DIR, "e14_eg_encoder.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\nsaved: e14_eg_encoder.json ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
