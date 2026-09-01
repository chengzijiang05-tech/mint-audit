"""E9：第二代正统基线，TR 统计量族 + surrogate 校准（Theiler 程序）。

实验目的：为经典指纹 kit 提供其最强校准形态。Theiler 程序对
时间不可逆问题的标准答案是"TR 敏感统计量 + AAFT/IAAFT surrogate
校准 p 值"，本脚本补齐这一基线，三代对决成立。

统计量族（计量文献既定定义）：
  L(tau)  Bouchaud et al. 2001 leverage correlation:
          L(k) = E[r_t r_{t+k}^2]/E[r_t^2]^2 - E[r_t^2 r_{t+k}]/E[r_t^2]^2
          读数 = mean_{k<=K} |L(k)|（无向）与 mean_k L(k)（有向）
  CCK     Chen-Chou-Kuan 2000 TR 统计量：TR(k)=E[r_t r_{t+k}^2]
          - E[r_t^2 r_{t+k}] 的 k 加权和（带经验协方差二次型）
  bispec  Hinich 1982 非线性相位耦合质量（bicoherence 平方和）

校准（第二代精髓，免阈值、免训练）：
  对被检窗口 x 生成 B=99 条 IAAFT surrogate（保边际保谱、毁相位），
  p = (1 + #{T(s_i) >= T(x)}) / (B+1)，与 x 自身分布校准，
  这正是 Theiler 1992 的 constrained realization。

评分方向：TR 敏感统计量在"缺时间箭头"的伪造上塌缩到自身 surrogate
分布之内（p 大），在真实窗上（若箭头真实存在）p 小。异常分取
-p（p 小 = 可疑 = 高异常）？不，注意审计方向：真实窗有时间箭头，
伪造窗缺箭头。surrogate 检验检的是"x 是否偏离 surrogate 类"，
真实窗拒绝 surrogate null（箭头存在），伪造窗不拒绝。
因此"判伪造"读数 = +p（p 大 = 看起来像自己的 surrogate = 可疑），
同时报告双向 AUC。

对比面：真实测试窗 vs N1-N8 全部敌手族（N7/N8 由 npz 读入）。

产出：results/e9_gen2_surrogate.json
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

from mint.operators import iaaft  # noqa: E402

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
B_SURR = 99            # surrogate 条数（p 值分辨率 1/100）
K_LAG = 20             # leverage/TR 滞后阶数
SEED = 20260821
ALPHA = 0.05           # "不拒绝时间可逆 H0" 的显著性阈值
FAM_FILES = {
    "N7a": "n7a_llm_forge.npz",
    "N7b": "n7b_quantgan.npz",
    "N8": "n8_gjr_egarch.npz",
}


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


# ---------------------------------------------------------------------------
# TR 敟计量族
# ---------------------------------------------------------------------------
def stat_L(x):
    """Bouchaud leverage correlation：无向读数 mean|L(k)|。"""
    x = np.asarray(x, float)
    x = x - x.mean()
    d = np.mean(x ** 2)
    if d <= 0:
        return 0.0
    ls = []
    for k in range(1, K_LAG + 1):
        num = np.mean(x[:-k] * x[k:] ** 2) - np.mean(x[:-k] ** 2 * x[k:])
        ls.append(num / (d * d))
    return float(np.mean(np.abs(ls)))


def stat_Ldir(x):
    """有向读数 mean L(k)（符号保留，时间箭头方向）。"""
    x = np.asarray(x, float)
    x = x - x.mean()
    d = np.mean(x ** 2)
    if d <= 0:
        return 0.0
    ls = []
    for k in range(1, K_LAG + 1):
        num = np.mean(x[:-k] * x[k:] ** 2) - np.mean(x[:-k] ** 2 * x[k:])
        ls.append(num / (d * d))
    return float(np.mean(ls))


def stat_cck(x):
    """CCK 式 TR 统计量：TR(k) 向量的自归一二次型。"""
    x = np.asarray(x, float)
    x = x - x.mean()
    d = np.mean(x ** 2)
    tr = []
    for k in range(1, K_LAG + 1):
        tr.append((np.mean(x[:-k] * x[k:] ** 2)
                   - np.mean(x[:-k] ** 2 * x[k:])) / (d * d))
    tr = np.array(tr)
    sd = np.std(np.abs(np.random.default_rng(0).normal(size=len(tr))))
    return float(np.sum(tr ** 2) / max(len(tr), 1))


def stat_bispec(x):
    """归一 bicoherence 平方和（Hinich 非线性相位耦合读数）。"""
    x = np.asarray(x, float)
    x = x - x.mean()
    n_seg = 8
    L = len(x) // n_seg
    B = np.zeros(L // 4, dtype=complex)
    w = np.hanning(L)
    for i in range(n_seg):
        seg = x[i * L:(i + 1) * L] * w
        X = np.fft.rfft(seg)
        m = len(X) // 4
        for q in range(1, m):
            B[q] += X[q] * X[q] * np.conj(X[2 * q])
    return float(np.sum(np.abs(B / max(n_seg, 1)) ** 2))


STATS = {"L": stat_L, "Ldir": stat_Ldir, "CCK": stat_cck,
         "bispec": stat_bispec}


def surr_pvalue(x, stat_fn, seed, b=B_SURR):
    """Theiler 程序：统计量 vs 自身 IAAFT surrogate 分布。"""
    rng = np.random.default_rng(seed)
    t_obs = stat_fn(x)
    t_s = np.empty(b)
    for i in range(b):
        s = iaaft(x, rng)
        t_s[i] = stat_fn(s)
    # 双侧 p（T 落在自身 surrogate 分布尾部的程度）
    p_hi = (1 + np.sum(t_s >= t_obs)) / (b + 1)
    p_lo = (1 + np.sum(t_s <= t_obs)) / (b + 1)
    return float(min(p_hi, p_lo)), float(t_obs), float(np.mean(t_s))


def auc_pair(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if np.std(pos) == 0 and np.std(neg) == 0:
        return 0.5
    s = np.r_[pos, neg]
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = len(pos), len(neg)
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def wilson(k, n, zz=1.96):
    p = k / n
    d = 1 + zz * zz / n
    c = p + zz * zz / (2 * n)
    h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def main():
    t0 = time.time()
    print("=" * 66)
    print(f"E9 第二代正统基线：TR 统计量 + IAAFT surrogate 校准 "
          f"(B={B_SURR}, p 分辨率 1/{B_SURR+1})")
    print("=" * 66, flush=True)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    # 敌手族：N1–N5 由 bench 构造（与 e2_main 同种子协议），N7/N8 npz
    from bench import HALLUCINATION_BUILDERS, n5_phase_forge
    fam_series = {}     # fam -> {asset: [windows]}
    for fam_name in ["N1数值置换", "N2时间倒置", "N3标度破坏",
                     "N4跨域嫁接", "N5相位伪造"]:
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

    res = {"protocol": {
        "surrogates": f"IAAFT B={B_SURR} (constrained realization, "
                      "Theiler 1992)",
        "stats": list(STATS), "p_resolution": 1.0 / (B_SURR + 1),
        "seed": SEED},
        "pvalues": {}, "summary": {}}

    # 每统计量：真实窗与各族的 p 值（判伪造读数 = p，p 大 = 像 surrogate）
    for stat_name, stat_fn in STATS.items():
        print(f"\n[{stat_name}]", flush=True)
        res["pvalues"][stat_name] = {}
        for ai, (asset, r) in enumerate(returns.items()):
            sp = SPLIT_SPEC[asset]
            wins = [r[i:i + WINDOW]
                    for i in range(0, len(r) - WINDOW + 1, STEP)]
            test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
            rows_real = [surr_pvalue(w, stat_fn, SEED + 500000 + ai * 100 + wi)
                         for wi, w in enumerate(test_wins)]
            res["pvalues"][stat_name][asset] = {
                "real": [x[0] for x in rows_real],
                "real_T": [x[1] for x in rows_real],
                "surr_T": [x[2] for x in rows_real]}
            for f_idx, (fam, per_asset) in enumerate(fam_series.items()):
                rows = [surr_pvalue(w, stat_fn,
                                    SEED + 600000 + ai * 100 + f_idx * 97 + wi)
                        for wi, w in enumerate(per_asset.get(asset, []))]
                res["pvalues"][stat_name][asset][fam] = [x[0] for x in rows]

    # 汇总：逐族 AUC（判伪造读数 p；伪造窗 p 应大 = 像自身 surrogate）
    for stat_name in STATS:
        s = {}
        for fam in fam_series:
            pv_f, pv_r = [], []
            for asset in returns:
                blk = res["pvalues"][stat_name][asset]
                pv_r.extend(blk["real"])
                pv_f.extend(blk.get(fam, []))
            if pv_f:
                a = auc_pair(pv_f, pv_r)
                # 免训练规则：min 双侧 p 上限 0.5，p >= ALPHA 表示统计量
                # 落在自身 surrogate 分布内（不拒绝"时间可逆 H0"）→ 判伪造；
                # p < ALPHA 表示显著偏离 surrogate（有箭头）→ 判真实。
                fl_f = float(np.mean(np.array(pv_f) >= ALPHA))
                fl_r = float(np.mean(np.array(pv_r) >= ALPHA))
                s[fam] = {"auc": round(a, 3),
                          f"recall@p>={ALPHA}": round(fl_f, 3),
                          f"fpr@p>={ALPHA}": round(fl_r, 3),
                          "n_forge": len(pv_f), "n_real": len(pv_r)}
        res["summary"][stat_name] = s
        print(f"\n{stat_name} 汇总（auc / recall / fpr）：")
        for fam, v in s.items():
            print(f"  {fam:8s} {v['auc']:.3f}  {v[f'recall@p>={ALPHA}']:.3f}"
                  f"  {v[f'fpr@p>={ALPHA}']:.3f}", flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS_DIR, "e9_gen2_surrogate.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved: e9_gen2_surrogate.json  ({res['elapsed_s']}s)")


if __name__ == "__main__":
    main()
