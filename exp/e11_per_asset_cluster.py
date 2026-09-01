"""E11：逐资产分解 + 杠杆强度相关性 + 聚类稳健推断重算（表 2 修正版）。

实验目的：
  1. 五资产池化掩盖杠杆效应资产异质性，需逐资产 AUC/FPR/认证表；
  2. 逐资产杠杆强度 L(τ) 与 N2/N7 检出的相关性，机制证据；
  3. 重叠窗口/同源伪造/种子合并被当独立样本，全部推断按源窗聚类重算。

数据源：results/e2_main_scores.npz（三种子冻结分数缓存，逐资产分层）
+ returns.npz（原始序列，算杠杆强度）。

聚类口径（方法论 W2 建议原样执行）：
  - 单元 = 源真实窗（28 个）；族与种子随窗联动重抽；
  - block bootstrap 2000 次，AUC 95% CI 与 AUC 差的 CI；
  - "nominal level" 类措辞改报：点估计 + 聚类稳健上界；
  - McNemar 换窗口级符号检验（同窗多族判定并列）。

产出：results/e11_per_asset_cluster.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

MINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSCT_ROOT = os.path.join(MINT_ROOT, "shared_infra", "fractal_consistency")
for p in (MINT_ROOT, FSCT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
SCORES_NPZ = os.path.join(MINT_ROOT, "results", "e2_main_scores.npz")
OUT_JSON = os.path.join(MINT_ROOT, "results", "e11_per_asset_cluster.json")

WINDOW, STEP = 1000, 50
SPLIT_SPEC = {
    "equity":  {"n_fit": 15, "cal": (34, 43), "test": (63, 99), "cap": 8},
    "bond":    {"n_fit": 15, "cal": (34, 43), "test": (63, 94), "cap": 8},
    "fx":      {"n_fit": 11, "cal": (31, 36), "test": (37, 42), "cap": 6},
    "gold":    {"n_fit": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
    "copper":  {"n_fit": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
}
FAMS = ["N1", "N2", "N3", "N4", "N5"]
METHODS = ["A0", "A1", "A2", "A3", "A4", "A5", "A5b"]
ALPHA = 0.05
K_LAG = 20
N_BOOT = 2000
SEED_BOOT = 20260821


def leverage_strength(x):
    """尺度不变杠杆读数：L̃(k) = E[r_t r²_{t+k}]/E[|r|³]。
    Bouchaud 原式除以 E[r²]² 含 1/c 量纲（跨资产单位污染），三阶矩
    标准化后对幅度缩放不变，跨资产可比。有向（时间箭头方向）。"""
    x = np.asarray(x, float)
    x = x - x.mean()
    den = np.mean(np.abs(x) ** 3)
    if den <= 0:
        return 0.0
    ls = []
    for k in range(1, K_LAG + 1):
        ls.append(np.mean(x[:-k] * x[k:] ** 2) / den)
    return float(np.mean(ls))


def auc_pair(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.r_[pos, neg]
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = len(pos), len(neg)
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def load_scores():
    with np.load(SCORES_NPZ, allow_pickle=True) as d:
        assets = [str(a) for a in d["assets"]]
        data = {}
        for m in METHODS:
            for layer in ["cal", "real"] + FAMS:
                for ai in range(len(assets)):
                    sk = f"anom/{m}/{layer}/{ai}"
                    if sk in d.files:
                        data[(m, layer, ai)] = np.asarray(d[sk], float)
        for ai in range(len(assets)):
            if f"W/A0/real/{ai}" in d.files:
                data[("W", "real", ai)] = np.asarray(
                    d[f"W/A0/real/{ai}"], float)
                for f in FAMS:
                    if f"W/A0/{f}/{ai}" in d.files:
                        data[("W", f, ai)] = np.asarray(
                            d[f"W/A0/{f}/{ai}"], float)
    return assets, data


def main():
    print("=" * 66)
    print("E11 逐资产分解 + 杠杆相关性 + 聚类稳健推断（表 2 修正）")
    print("=" * 66, flush=True)

    assets, data = load_scores()
    print(f"资产：{assets} | 分数缓存键：{len(data)}", flush=True)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    res = {"protocol": {
        "cluster_unit": "source real window (28 total)",
        "bootstrap": f"window-level, {N_BOOT} reps, families/seeds tied",
        "note": "point estimates unchanged; intervals/p-values now "
                "cluster-robust"},
        "per_asset": {}, "leverage": {}, "pooled_cluster": {}}

    # ---- 逐资产表 + 杠杆强度 ----
    lev_scores = {}
    for ai, asset in enumerate(assets):
        sp = SPLIT_SPEC[asset]
        r = returns[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        test_wins = [wins[i] for i in np.unique(np.linspace(
            sp["test"][0], sp["test"][1],
            sp["test"][1] - sp["test"][0] + 1).astype(int))[:9999]]
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        lev = float(np.mean([abs(leverage_strength(w)) for w in test_wins]))
        lev_scores[asset] = lev
        row = {"leverage_L": round(lev, 5),
               "n_test": len(test_wins)}
        for m in METHODS:
            if (m, "real", ai) not in data:
                continue
            real = data[(m, "real", ai)]
            cal = data[(m, "cal", ai)]
            thr = np.quantile(cal[np.isfinite(cal)], 1 - ALPHA)
            row[m] = {
                "fpr": round(float(np.mean(real > thr)), 3),
                "auc": {}}
            for f in FAMS:
                if (m, f, ai) in data:
                    fa = data[(m, f, ai)]
                    row[m]["auc"][f] = round(
                        float(auc_pair(fa, real)), 3)
        # e 值认证（若缓存）
        if ("W", "real", ai) in data:
            Wr = data[("W", "real", ai)]
            row["certify_real"] = round(float(np.mean(Wr >= 1 / ALPHA)), 3)
            for f in FAMS:
                if ("W", f, ai) in data:
                    Wf = data[("W", f, ai)]
                    row.setdefault("certify_fam", {})[f] = round(
                        float(np.mean(Wf >= 1 / ALPHA)), 3)
        res["per_asset"][asset] = row
        print(f"[{asset}] leverage|L|={lev:.5f} "
              f"A0/N2={row.get('A0',{}).get('auc',{}).get('N2')}",
              flush=True)

    res["leverage"]["per_asset"] = {k: round(v, 5)
                                    for k, v in lev_scores.items()}

    # ---- 杠杆强度 vs N2 检出的相关性（Spearman）----
    from scipy.stats import spearmanr
    xs, ys = [], []
    for ai, asset in enumerate(assets):
        if ("A0", "N2", ai) in data:
            xs.append(abs(lev_scores[asset]))
            ys.append(auc_pair(data[("A0", "N2", ai)],
                               data[("A0", "real", ai)]))
    if len(xs) >= 3:
        rho, p = spearmanr(xs, ys)
        res["leverage"]["spearman_vs_N2_auc"] = {
            "rho": round(float(rho), 3), "p": round(float(p), 4),
            "n": len(xs)}
        print(f"杠杆强度 vs N2 AUC：ρ={rho:.3f} (p={p:.4f}, n={len(xs)})",
              flush=True)

    # ---- 池化聚类 bootstrap：A0 vs A5b 每族 AUC 差 ----
    # 单元 = 源窗：该窗的真实分 + 五族伪造分（各 1 条）+ 各方法联动
    rng = np.random.default_rng(SEED_BOOT)
    units = []      # 每单元: {asset_idx, win_idx}
    for ai in range(len(assets)):
        n_w = len(data.get(("A0", "real", ai), []))
        for wi in range(n_w):
            units.append((ai, wi))
    n_u = len(units)
    print(f"\n聚类单元（源窗）：{n_u}", flush=True)

    def pooled_scores(m, layer):
        """返回 (asset_idx, win_idx) → score 的字典数组形式。"""
        out = {}
        for ai in range(len(assets)):
            v = data.get((m, layer, ai))
            if v is None:
                return None
            out[ai] = v
        return out

    for fam in FAMS:
        sa = pooled_scores("A0", fam)
        sr = pooled_scores("A0", "real")
        sb_a = pooled_scores("A5b", fam)
        sb_r = pooled_scores("A5b", "real")
        if None in (sa, sr, sb_a, sb_r):
            continue
        # 点估计
        pos_a = np.concatenate([sa[ai] for ai in range(len(assets))])
        neg_a = np.concatenate([sr[ai] for ai in range(len(assets))])
        pos_b = np.concatenate([sb_a[ai] for ai in range(len(assets))])
        neg_b = np.concatenate([sb_r[ai] for ai in range(len(assets))])
        d_point = auc_pair(pos_a, neg_a) - auc_pair(pos_b, neg_b)
        # 聚类 bootstrap
        diffs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = rng.integers(0, n_u, n_u)
            pa, na, pb, nb = [], [], [], []
            for u in pick:
                ai, wi = units[u]
                if wi < len(sa[ai]):
                    pa.append(sa[ai][wi])
                    na.append(sr[ai][wi])
                    pb.append(sb_a[ai][wi])
                    nb.append(sb_r[ai][wi])
            diffs[b] = (auc_pair(np.array(pa), np.array(na))
                        - auc_pair(np.array(pb), np.array(nb)))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_two = float(min(1.0, 2 * min(np.mean(diffs <= 0),
                                       np.mean(diffs >= 0))))
        res["pooled_cluster"][fam] = {
            "auc_A0": round(float(auc_pair(pos_a, neg_a)), 3),
            "auc_A5b": round(float(auc_pair(pos_b, neg_b)), 3),
            "diff": round(float(d_point), 3),
            "diff_cluster_ci": [round(float(lo), 3), round(float(hi), 3)],
            "cluster_p": max(round(p_two, 5), 1.0 / N_BOOT)}
        print(f"  {fam}: ΔAUC={d_point:.3f} "
              f"CI[{lo:.3f},{hi:.3f}] p={p_two:.4f}", flush=True)

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved: {OUT_JSON}")


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


if __name__ == "__main__":
    main()
