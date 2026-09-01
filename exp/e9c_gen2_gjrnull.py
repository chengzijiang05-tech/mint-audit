"""E9c：第二代基线的另一条腿，fitted-GJR model-based null 校准（S4）。

实验目的：论文的 Gen2 基线只用 surrogate-data
程序的一条校准腿（IAAFT constrained realization）。计量文献中该程序
还有另一条标准腿：model-based null，对被检窗口拟合一个参数模型
（此处 GJR-GARCH(1,1)，带杠杆的金融计量标准件），从拟合模型模拟
B 条路径作为零假设分布，p 值读法与 surrogate 版一致。

本脚本把四个 TR 统计量（L / Ldir / CCK / bispec）换成这条腿重跑：
  每窗口 x：fit GJR-GARCH(1,1)（MLE，同 N8 的拟合器）→ 残差 bootstrap
  创新 → 模拟 B=99 条同长度路径 → 双侧 p = min(p_hi, p_lo)。
判读方向与 E9 一致：p 大 = 统计量落在自身零假设分布内 = 判伪造。

对比面：真实测试窗 + N1–N5（bench 构造，同种子）+ N7a/N7b/N7c/N8
（npz 读入）。协议、SPLIT_SPEC、统计量定义与 e9_gen2_surrogate.py
逐位一致，仅零假设不同，两行直接可比。

产出：results/e9c_gen2_gjrnull.json
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

from e9_gen2_surrogate import (  # noqa: E402
    ALPHA, B_SURR, SPLIT_SPEC, STATS, WINDOW, STEP, SEED,
    auc_pair, span_windows,
)
from n8_gjr_egarch import (  # noqa: E402
    boot_innov_factory, fit_gjr, gjr_simulate, resid_from_gjr,
)

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
FAM_FILES = {
    "N7a": "n7a_llm_forge.npz",
    "N7b": "n7b_quantgan.npz",
    "N7c": "n7c_chronos_forge.npz",
    "N8": "n8_gjr_egarch.npz",
}
GJR_DEFAULT = None  # 拟合失败的回退参数（用窗口方差与保守起点重拟）


def gjr_null_pvalue(x, stat_fns, seed, b=B_SURR):
    """Model-based null：拟合 GJR → 模拟 b 条 → 各统计量双侧 p。"""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    params = fit_gjr(x)
    if params[0] is None:
        iv = max(np.var(x), 1e-12)
        params = (iv * 0.02, 0.05, 0.06, 0.88)
    resid = resid_from_gjr(x, *params)
    innov = boot_innov_factory(resid)
    t_obs = {name: fn(x) for name, fn in stat_fns.items()}
    t_sim = {name: np.empty(b) for name in stat_fns}
    iv = max(np.var(x), 1e-12)
    for i in range(b):
        sim = gjr_simulate(*params, len(x), iv, innov, rng)
        for name, fn in stat_fns.items():
            t_sim[name][i] = fn(sim)
    out = {}
    for name in stat_fns:
        p_hi = (1 + np.sum(t_sim[name] >= t_obs[name])) / (b + 1)
        p_lo = (1 + np.sum(t_sim[name] <= t_obs[name])) / (b + 1)
        out[name] = float(min(p_hi, p_lo))
    return out, params


def main():
    t0 = time.time()
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--families="):
            only = a.split("=", 1)[1].split(",")
    print("=" * 66)
    print(f"E9c 第二代基线 model-based null 腿：GJR-GARCH(1,1) 拟合 + "
          f"残差 bootstrap 模拟 B={B_SURR}")
    print("=" * 66, flush=True)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    from bench import HALLUCINATION_BUILDERS, n5_phase_forge

    fam_series = {}
    for fam_name in ["N1数值置换", "N2时间倒置", "N3标度破坏",
                     "N4跨域嫁接", "N5相位伪造"]:
        if only is not None and fam_name[:2] not in only:
            continue
        fam_series[fam_name] = {}
        for asset, r in returns.items():
            sp = SPLIT_SPEC[asset]
            wins = [r[i:i + WINDOW]
                    for i in range(0, len(r) - WINDOW + 1, STEP)]
            test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
            fi = ["N1数值置换", "N2时间倒置", "N3标度破坏",
                  "N4跨域嫁接", "N5相位伪造"].index(fam_name)
            if fam_name == "N5相位伪造":
                fam_series[fam_name][asset] = [
                    n5_phase_forge(w, seed=1000 + wi * 10 + fi)
                    for wi, w in enumerate(test_wins)]
            else:
                fam_series[fam_name][asset] = [
                    HALLUCINATION_BUILDERS[fam_name](w, seed=1000 + wi * 10 + fi)
                    for wi, w in enumerate(test_wins)]

    for tag, fn in FAM_FILES.items():
        if only is not None and tag not in only:
            continue
        path = os.path.join(RESULTS_DIR, fn)
        if not os.path.exists(path):
            print(f"!! 缺 {fn}，跳过 {tag}")
            continue
        with np.load(path, allow_pickle=True) as d:
            by_asset = {}
            for key in d.files:
                asset, cfg, j = key.split("/")
                by_asset.setdefault(asset, []).append(d[key])
            fam_series[tag] = by_asset
    print(f"敌手族就绪：{list(fam_series)}", flush=True)

    stat_fns = dict(STATS)
    pvalues = {name: {} for name in stat_fns}
    params_log = []
    n_win = 0
    for ai, (asset, r) in enumerate(returns.items()):
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW]
                for i in range(0, len(r) - WINDOW + 1, STEP)]
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        for wi, w in enumerate(test_wins):
            pv, params = gjr_null_pvalue(
                w, stat_fns, SEED + 700000 + ai * 100 + wi)
            for name in stat_fns:
                pvalues[name].setdefault(asset, {"real": []})
                pvalues[name][asset]["real"].append(pv[name])
            params_log.append({"asset": asset, "kind": "real", "idx": wi,
                               "gjr_wagb": [round(v, 5) for v in params]})
            n_win += 1
        for f_idx, (fam, per_asset) in enumerate(fam_series.items()):
            for wi, w in enumerate(per_asset.get(asset, [])):
                pv, params = gjr_null_pvalue(
                    w, stat_fns, SEED + 800000 + ai * 100 + f_idx * 97 + wi)
                for name in stat_fns:
                    pvalues[name][asset].setdefault(fam, []).append(pv[name])
                params_log.append({"asset": asset, "kind": fam, "idx": wi,
                                   "gjr_wagb": [round(v, 5) for v in params]})
                n_win += 1
        print(f"[{asset}] real {len(test_wins)} + families done "
              f"(cum {n_win} windows)", flush=True)

    summary = {}
    for stat_name in stat_fns:
        s = {}
        for fam in fam_series:
            pv_f, pv_r = [], []
            for asset in returns:
                blk = pvalues[stat_name][asset]
                pv_r.extend(blk["real"])
                pv_f.extend(blk.get(fam, []))
            if pv_f:
                a = auc_pair(pv_f, pv_r)
                fl_f = float(np.mean(np.array(pv_f) >= ALPHA))
                fl_r = float(np.mean(np.array(pv_r) >= ALPHA))
                s[fam] = {"auc": round(a, 3),
                          f"recall@p>={ALPHA}": round(fl_f, 3),
                          f"fpr@p>={ALPHA}": round(fl_r, 3),
                          "n_forge": len(pv_f), "n_real": len(pv_r)}
        summary[stat_name] = s
        print(f"\n[{stat_name}] 汇总（auc / recall / fpr）：")
        for fam, v in s.items():
            print(f"  {fam:8s} {v['auc']:.3f}  "
                  f"{v[f'recall@p>={ALPHA}']:.3f}  "
                  f"{v[f'fpr@p>={ALPHA}']:.3f}", flush=True)

    res = {
        "protocol": {
            "null": f"fitted GJR-GARCH(1,1) per window, residual-bootstrap "
                    f"innovations, B={B_SURR} simulated paths (model-based "
                    f"null; the surrogate program's other leg)",
            "fitter": "same MLE as N8 (L-BFGS-B, 3 starts)",
            "stats": list(stat_fns), "p_resolution": 1.0 / (B_SURR + 1),
            "seed": SEED,
            "reading": "p >= alpha -> judged forgery (same as E9)"},
        "summary": summary,
        "params_sample": params_log[:50],
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(RESULTS_DIR, "e9c_gen2_gjrnull.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved: e9c_gen2_gjrnull.json  ({res['elapsed_s']}s)")


if __name__ == "__main__":
    main()
