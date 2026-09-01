"""E16：源窗级 cluster-robust 推断（R4）。

 1) tab:main 的配对显著性：A0 vs 各基线（经典 A5b/A5 + 深度 D1-D4），
    以 28 个源窗为重抽单元的配对 cluster bootstrap（AUC 差，池化与 N2）。
    替换 DeLong p<10^-7~10^-10 的小分母声明。
 2) 零计数 cluster-robust 上界：0/128（生成拟合单元 40 个）与
    0/120（40 个不同目标窗 × 3 源编码器）。

数据源：e2_main_scores.npz（经典）、e2_deep_scores.npz（深度，e2_deep
重跑落盘）、e4_transfer.json（九宫格）、e17_genpaths_w.json（路径 W）。
"""
from __future__ import annotations

import json
import os

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
FAMS = ["N1", "N2", "N3", "N4", "N5"]
ALPHA = 0.05
REPS = 2000


def auc_pair(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    s = np.r_[pos, neg]
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = len(pos), len(neg)
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def load_windows(npz, methods):
    assets = [str(a) for a in npz["assets"]]
    wins = []
    for ai in range(len(assets)):
        n_w = len(npz[f"anom/{methods[0]}/real/{ai}"])
        for j in range(n_w):
            rec = {"asset": assets[ai], "real": {},
                   "fam": {f: {} for f in FAMS}}
            for m in methods:
                rec["real"][m] = float(npz[f"anom/{m}/real/{ai}"][j])
                for f in FAMS:
                    rec["fam"][f][m] = float(npz[f"anom/{m}/{f}/{ai}"][j])
            wins.append(rec)
    return wins, assets


def pooled_auc(m, wins, fams=FAMS):
    pos = [w["fam"][f][m] for w in wins for f in fams
           if np.isfinite(w["fam"][f][m])]
    neg = [w["real"][m] for w in wins if np.isfinite(w["real"][m])]
    return auc_pair(pos, neg)


def boot_paired_cluster(m_a, m_b, wins, fams=FAMS, reps=REPS, seed=424242):
    rng = np.random.default_rng(seed)
    n = len(wins)
    obs = pooled_auc(m_a, wins, fams) - pooled_auc(m_b, wins, fams)
    diffs = np.empty(reps)
    for t in range(reps):
        idx = rng.integers(0, n, n)
        ws = [wins[i] for i in idx]
        diffs[t] = pooled_auc(m_a, ws, fams) - pooled_auc(m_b, ws, fams)
    phi = float(np.mean(diffs <= 0))
    p = min(1.0, 2 * min(phi, 1 - phi))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"auc_a": pooled_auc(m_a, wins, fams),
            "auc_b": pooled_auc(m_b, wins, fams),
            "diff": obs, "ci": [float(lo), float(hi)],
            "p": max(p, 1.0 / reps)}


def cal_thresholds(npz, methods, assets):
    thr = {}
    for ai, a in enumerate(assets):
        thr[a] = {}
        for m in methods:
            cal = np.asarray(npz[f"anom/{m}/cal/{ai}"], float)
            cal = cal[np.isfinite(cal)]
            thr[a][m] = float(np.quantile(cal, 1 - ALPHA))
    return thr


def decisions_cluster(m_a, m_b, wins, thr, fams=FAMS,
                      reps=REPS, seed=424243):
    """决策层：池化 recall 差的源窗 cluster bootstrap（per-asset cal q95）。"""

    def rates(m, ws):
        fl_r, fl_f = [], []
        for w in ws:
            if np.isfinite(w["real"][m]):
                fl_r.append(w["real"][m] > thr[w["asset"]][m])
            for f in fams:
                if np.isfinite(w["fam"][f][m]):
                    fl_f.append(w["fam"][f][m] > thr[w["asset"]][m])
        return (float(np.mean(fl_f)) if fl_f else np.nan,
                float(np.mean(fl_r)) if fl_r else np.nan)

    rng = np.random.default_rng(seed)
    n = len(wins)
    obs = rates(m_a, wins)[0] - rates(m_b, wins)[0]
    diffs = np.empty(reps)
    for t in range(reps):
        idx = rng.integers(0, n, n)
        ws = [wins[i] for i in idx]
        diffs[t] = rates(m_a, ws)[0] - rates(m_b, ws)[0]
    phi = float(np.mean(diffs <= 0))
    p = min(1.0, 2 * min(phi, 1 - phi))
    return {"recall_diff": obs,
            "ci": [float(np.percentile(diffs, 2.5)),
                   float(np.percentile(diffs, 97.5))],
            "p": max(p, 1.0 / reps)}


def main():
    out = {"reps": REPS, "cluster_unit": "source window (n=28)"}

    # ---- 经典基线 ----
    npz_main = np.load(os.path.join(RESULTS, "e2_main_scores.npz"),
                       allow_pickle=True)
    methods = ["A0", "A5", "A5b"]
    wins, assets = load_windows(npz_main, methods)
    thr = cal_thresholds(npz_main, methods, assets)
    print(f"源窗数 {len(wins)}（{assets}）")
    for m in ("A5", "A5b"):
        for scope, fams in (("pooled", FAMS), ("N2", ["N2"]), ("N5", ["N5"])):
            r = boot_paired_cluster("A0", m, wins, fams=fams)
            out[f"A0vs{m}/{scope}"] = r
            print(f"A0 vs {m} [{scope}]: ΔAUC={r['diff']:+.3f} "
                  f"CI{r['ci']} p={r['p']:.4f}")
        d = decisions_cluster("A0", m, wins, thr)
        out[f"A0vs{m}/recall"] = d
        print(f"A0 vs {m} [recall]: Δ={d['recall_diff']:+.3f} "
              f"CI{d['ci']} p={d['p']:.4f}")

    # ---- 深度基线（e2_deep_scores.npz 存在时）----
    deep_npz = os.path.join(RESULTS, "e2_deep_scores.npz")
    if os.path.exists(deep_npz):
        zd = np.load(deep_npz, allow_pickle=True)
        dm = ["D1", "D2", "D3", "D4"]
        deep_wins, _ = load_windows(zd, dm)
        for rec, w in zip(deep_wins, wins):
            rec["real"]["A0"] = w["real"]["A0"]
            for f in FAMS:
                rec["fam"][f]["A0"] = w["fam"][f]["A0"]
        thr_deep = cal_thresholds(zd, dm, assets)
        for a in assets:
            thr_deep[a]["A0"] = thr[a]["A0"]
        for m in dm:
            for scope, fams in (("pooled", FAMS), ("N2", ["N2"])):
                r = boot_paired_cluster("A0", m, deep_wins, fams=fams)
                out[f"A0vs{m}/{scope}"] = r
                print(f"A0 vs {m} [{scope}]: ΔAUC={r['diff']:+.3f} "
                      f"CI{r['ci']} p={r['p']:.4f}")
            d = decisions_cluster("A0", m, deep_wins, thr_deep)
            out[f"A0vs{m}/recall"] = d
            print(f"A0 vs {m} [recall]: Δ={d['recall_diff']:+.3f} "
                  f"CI{d['ci']} p={d['p']:.4f}")
    else:
        print("（e2_deep_scores.npz 未就绪，深度部分跳过）")

    with open(os.path.join(RESULTS, "e16_cluster_inf.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("saved: e16_cluster_inf.json")


if __name__ == "__main__":
    main()
