"""E13：洗钱家族实验，真实段 + 伪造延续的混合比检测衰减（R10）。

编辑决定书 R10：全部已评估伪造者均为静态；论文公开了针对自身
算子并集的训练配方，MINT 感知的伪造者是显然的攻击；无混合真实段
+ 伪造延续（检索增强式）家族被测试。要求：至少一个洗钱家族
（真实窗 + 伪造延续，多混合比，报告检测衰减），理想情况加一个
针对已公开分数优化的伪造者探针。

攻击协议（敌手视角：拿一段真实序列作前缀，把伪造续在其后）：
  混合比 ρ = 伪造延续占比 ∈ {0.25, 0.50, 0.75}，参考点 ρ=0（纯真实）
  与 ρ=1（纯伪造）。窗口 = 真实前缀 1000·(1-ρ) 点 + 伪造延续。

  伪造延续源（两个，覆盖两类敌手强度）：
    iaaft_match  检索增强式：对完整真实窗做 IAAFT（保边际保谱），
                 取其尾部作延续，伪造者知道并模仿被嵌入真实窗自身
                 的分布（最强信息条件）
    gjr_boot     参数式：GJR-GARCH(1,1) 仅拟合于真实前缀（MLE），
                 拟合层残差 bootstrap 创新，伪造者只见前缀的现实条件

  对抗探针（score-aware best-of-n）：伪造者针对已公开的认证分数
  做 20 选 1，生成 20 条 gjr_boot 延续，保留 e 值 W 最大（最像
  真实）的一条入窗。这是"对已公开分数优化"的有界近似。

检测读数：A0 e 值（与 e2_main 相同训练种子与协议）。
  检测率 = -W 超过逐资产 cal q95 阈值的比例（阈值与主表同源）
  误认证率 = W ≥ 1/α 的比例（洗钱成功 = 被认证为真实）
  检测衰减 = 上述两率随 ρ 下降/上升的曲线

产出：results/e13_laundering.json
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
    CONFIGS, SPLIT_SPEC, SEED, ALPHA, K_EVAL, WINDOW, STEP,
    span_windows, train_model, e_values,
)
from mint.operators import iaaft  # noqa: E402
from n8_gjr_egarch import (  # noqa: E402
    fit_gjr, gjr_simulate, resid_from_gjr, boot_innov_factory,
)

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")

RATIOS = [0.25, 0.50, 0.75]
SOURCES = ["iaaft_match", "gjr_boot"]
N_ADAPT = 20
GEN_SEED = 40404040


def cut_len(rho):
    return int(round(WINDOW * (1 - rho)))


def continuation_iaaft(w, rng):
    s = iaaft(w, rng)
    return s


def gjr_fit_prefix(prefix):
    w_par, a, g, b = fit_gjr(prefix)
    if w_par is None:
        return None
    resid = resid_from_gjr(prefix, w_par, a, g, b)
    innov = boot_innov_factory(resid)
    h0 = float(np.var(prefix[-250:]))
    return (w_par, a, g, b, h0, innov)


def gjr_continue(fit, n, rng):
    if fit is None:
        return rng.standard_normal(n) * 0.01
    w_par, a, g, b, h0, innov = fit
    return gjr_simulate(w_par, a, g, b, n, h0, innov, rng)


def continuation_gjr(prefix, n, rng):
    return gjr_continue(gjr_fit_prefix(prefix), n, rng)


def main():
    t0 = time.time()
    print("=" * 66)
    print(f"E13 洗钱家族：真实段+伪造延续 混合比检测衰减（R10）")
    print(f"混合比 ρ={RATIOS} | 延续源 {SOURCES} | 对抗 best-of-{N_ADAPT}")
    print("=" * 66, flush=True)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    out = {"protocol": {
        "ratios_fabricated": RATIOS, "sources": SOURCES,
        "adaptive_probe": f"best-of-{N_ADAPT} gjr_boot vs published score",
        "K_eval": K_EVAL, "alpha": ALPHA, "seed": SEED,
        "detection": "-W > per-asset cal q95 (threshold same as main table)",
        "false_certification": "W >= 1/alpha",
    }, "per_asset": {}, "decay": {}, "adaptive": {}}

    W_store = {"real": [], "hybrid": {(s, r): [] for s in SOURCES
                                      for r in RATIOS + [1.0]},
               "adaptive": {r: [] for r in RATIOS}}
    flags_store = {"real": []}

    for ai, (asset, r) in enumerate(returns.items()):
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        cal_wins = span_windows(wins, *sp["cal"])
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        train_end = sp["cal"][0] * STEP
        print(f"\n[{asset}] cal{len(cal_wins)} test{len(test_wins)}",
              flush=True)

        cfg = CONFIGS["A0"]
        enc, head, last, _t = train_model(
            r, train_end, cfg, SEED + ai * 1000 + 0, 250,
            150, 20)
        print(f"    A0 trained loss={last['loss']}", flush=True)

        base = 960000 + ai * 10000
        w_cal = e_values(enc, head, cal_wins, cfg["rank"], K_EVAL, base)
        thr = float(np.quantile(-w_cal[np.isfinite(-w_cal)], 1 - ALPHA))
        w_real = e_values(enc, head, test_wins, cfg["rank"], K_EVAL,
                          base + 100)
        flags_store["real"].append((-w_real) > thr)
        W_store["real"].append(w_real)

        pa = {"threshold": thr,
              "real": {"certify": float(np.mean(w_real >= 1 / ALPHA)),
                       "detect": float(np.mean((-w_real) > thr))}}

        for rho in RATIOS:
            cl = cut_len(rho)
            for si, src in enumerate(SOURCES):
                Ws = []
                for wi, w in enumerate(test_wins):
                    prefix = np.asarray(w[:cl], float)
                    gseed = GEN_SEED + ai * 10000 + wi * 100 + si * 10
                    if src == "iaaft_match":
                        cont = continuation_iaaft(
                            w, np.random.default_rng(gseed))[cl:]
                    else:
                        cont = continuation_gjr(
                            prefix, WINDOW - cl,
                            np.random.default_rng(gseed))
                    hybrid = np.concatenate([prefix, cont])[:WINDOW]
                    wv = e_values(enc, head, [hybrid], cfg["rank"],
                                  K_EVAL, base + 300 + si * 100 + wi)
                    Ws.append(float(wv[0]))
                Ws = np.array(Ws)
                W_store["hybrid"][(src, rho)].append(Ws)
                pa[f"{src}/rho{rho}"] = {
                    "certify": float(np.mean(Ws >= 1 / ALPHA)),
                    "detect": float(np.mean((-Ws) > thr)),
                    "mean_W": float(np.mean(Ws))}
                print(f"    ρ={rho} {src}: detect="
                      f"{pa[f'{src}/rho{rho}']['detect']:.3f} "
                      f"certify={pa[f'{src}/rho{rho}']['certify']:.3f}",
                      flush=True)

        # ρ=1 参考点（纯伪造）
        for si, src in enumerate(SOURCES):
            Ws = []
            for wi, w in enumerate(test_wins):
                gseed = GEN_SEED + 707000 + ai * 10000 + wi * 100 + si * 10
                rng = np.random.default_rng(gseed)
                if src == "iaaft_match":
                    full = iaaft(w, rng)
                else:
                    full = continuation_gjr(
                        np.asarray(w, float), WINDOW, rng)
                wv = e_values(enc, head, [np.asarray(full)], cfg["rank"],
                              K_EVAL, base + 500 + si * 100 + wi)
                Ws.append(float(wv[0]))
            Ws = np.array(Ws)
            W_store["hybrid"][(src, 1.0)].append(Ws)
            pa[f"{src}/rho1.0"] = {
                "certify": float(np.mean(Ws >= 1 / ALPHA)),
                "detect": float(np.mean((-Ws) > thr)),
                "mean_W": float(np.mean(Ws))}

        # 对抗探针：best-of-20 gjr_boot
        for rho in RATIOS:
            cl = cut_len(rho)
            Ws = []
            for wi, w in enumerate(test_wins):
                prefix = np.asarray(w[:cl], float)
                best = -1.0
                for c in range(N_ADAPT):
                    gseed = (GEN_SEED + 808000 + ai * 10000 + wi * 1000
                             + c)
                    if c == 0:
                        fit_cache = gjr_fit_prefix(prefix)
                    cont = gjr_continue(
                        fit_cache, WINDOW - cl,
                        np.random.default_rng(gseed))
                    hybrid = np.concatenate([prefix, cont])[:WINDOW]
                    wv = e_values(enc, head, [hybrid], cfg["rank"],
                                  K_EVAL, base + 700 + wi * 10 + c)
                    best = max(best, float(wv[0]))
                Ws.append(best)
            Ws = np.array(Ws)
            W_store["adaptive"][rho].append(Ws)
            pa[f"adaptive/rho{rho}"] = {
                "certify": float(np.mean(Ws >= 1 / ALPHA)),
                "detect": float(np.mean((-Ws) > thr)),
                "mean_W": float(np.mean(Ws))}
            print(f"    ρ={rho} adaptive(best-of-{N_ADAPT}): detect="
                  f"{pa[f'adaptive/rho{rho}']['detect']:.3f} "
                  f"certify={pa[f'adaptive/rho{rho}']['certify']:.3f}",
                  flush=True)

        out["per_asset"][asset] = pa

    # ---- 池化衰减曲线 ----
    real_w = np.concatenate(W_store["real"])
    out["decay"]["real"] = {
        "certify": float(np.mean(real_w >= 1 / ALPHA))}
    fr = np.concatenate(flags_store["real"])
    out["decay"]["real"]["detect"] = float(np.mean(fr))
    for src in SOURCES:
        for rho in RATIOS + [1.0]:
            Ws = np.concatenate(W_store["hybrid"][(src, rho)])
            out["decay"][f"{src}/rho{rho}"] = {
                "certify": float(np.mean(Ws >= 1 / ALPHA))}
    for rho in RATIOS:
        Ws = np.concatenate(W_store["adaptive"][rho])
        out["adaptive"][f"rho{rho}"] = {
            "certify": float(np.mean(Ws >= 1 / ALPHA))}

    # 池化检测率（阈值逐资产，池化标记向量拼接）
    for src in SOURCES:
        for rho in RATIOS + [1.0]:
            flags = []
            for seg, asset in enumerate(out["per_asset"]):
                thr = out["per_asset"][asset]["threshold"]
                flags.append((-W_store["hybrid"][(src, rho)][seg]) > thr)
            out["decay"][f"{src}/rho{rho}"]["detect"] = float(
                np.mean(np.concatenate(flags)))
    for rho in RATIOS:
        flags = []
        for seg, asset in enumerate(out["per_asset"]):
            thr = out["per_asset"][asset]["threshold"]
            flags.append((-W_store["adaptive"][rho][seg]) > thr)
        out["adaptive"][f"rho{rho}"]["detect"] = float(
            np.mean(np.concatenate(flags)))

    print("\n池化衰减（detect / certify）：")
    print(f"  real ρ=0: {out['decay']['real']['detect']:.3f} / "
          f"{out['decay']['real']['certify']:.3f}")
    for src in SOURCES:
        for rho in RATIOS + [1.0]:
            v = out["decay"][f"{src}/rho{rho}"]
            print(f"  {src} ρ={rho}: detect={v['detect']:.3f} "
                  f"certify={v['certify']:.3f}")
    for rho in RATIOS:
        v = out["adaptive"][f"rho{rho}"]
        print(f"  adaptive ρ={rho}: detect={v['detect']:.3f} "
              f"certify={v['certify']:.3f}")

    out["elapsed_s"] = round(time.time() - t0, 1)
    path = os.path.join(RESULTS_DIR, "e13_laundering.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved: e13_laundering.json ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
