"""E9b：第二代基线读数修正，方向读数 vs 偏差读数（Theiler 程序内部）。

E9 发现：双侧 min-p 偏差读数对 N2 不公平，时间反转序列的箭头
反转后仍是箭头，其 TR 统计量同样偏离 surrogate 类，p 值无区分
（AUC 0.51）。Theiler 程序对"方向型伪造"的正确武器是统计量的
有向读数：Ldir(x) 小/负 = 箭头反转或缺失 = 伪造嫌疑。

本脚本用统计量原始值（非 p 值）重算第二代基线：
  Ldir   方向读数：anom = -T（箭头越正/越强越真实；反转/缺失 → 高异常）
  absL   强度读数：anom = -|T|（箭头缺失 → 高异常）
  CCK    强度读数：anom = -T（TR 二次型小 → 高异常）
  bispec 强度读数：anom = -T
每个统计量用其语义正确的读数方向，对第二代公平（其最强形态）。

对比面同 E9：真实测试窗 vs N1-N8 全族。e9 的 p 值结果保留，
本脚本结果作为 Gen2 基线的最终读数（两者都报告）。

产出：results/e9b_gen2_directional.json
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

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
OUT_JSON = os.path.join(RESULTS_DIR, "e9b_gen2_directional.json")

WINDOW, STEP = 1000, 50
SPLIT_SPEC = {
    "equity":  {"n_fit": 15, "cal": (34, 43), "test": (63, 99), "cap": 8},
    "bond":    {"n_fit": 15, "cal": (34, 43), "test": (63, 94), "cap": 8},
    "fx":      {"n_fit": 11, "cal": (31, 36), "test": (37, 42), "cap": 6},
    "gold":    {"n_fit": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
    "copper":  {"n_fit": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
}
ALPHA = 0.05


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


def d_L(x):
    """完整 Bouchaud 差分（时间反转下严格变号）：
    L(k) = [E[x_t x²_{t+k}] − E[x²_t x_{t+k}]] / E[|x|³]（尺度不变）。"""
    x = np.asarray(x, float) - np.mean(x)
    den = np.mean(np.abs(x) ** 3)
    if den <= 0:
        return 0.0
    ls = [(np.mean(x[:-k] * x[k:] ** 2) - np.mean(x[:-k] ** 2 * x[k:]))
          / den for k in range(1, 21)]
    return float(np.mean(ls))


def d_cck(x):
    x = np.asarray(x, float) - np.mean(x)
    d = np.mean(x ** 2)
    tr = [(np.mean(x[:-k] * x[k:] ** 2) - np.mean(x[:-k] ** 2 * x[k:]))
          / (d * d) for k in range(1, 21)]
    return float(np.sum(np.array(tr) ** 2) / len(tr))


def d_bispec(x):
    x = np.asarray(x, float) - np.mean(x)
    n_seg, L = 8, len(x) // 8
    w = np.hanning(L)
    B = np.zeros(L // 4, dtype=complex)
    for i in range(n_seg):
        X = np.fft.rfft(x[i * L:(i + 1) * L] * w)
        m = len(X) // 4
        for q in range(1, m):
            B[q] += X[q] * X[q] * np.conj(X[2 * q])
    return float(np.sum(np.abs(B / n_seg) ** 2))


def auc_pair(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.r_[pos, neg]
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = len(pos), len(neg)
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def main():
    t0 = time.time()
    print("=" * 66)
    print("E9b 第二代方向读数修正（对 N2 公平的 Theiler 形态）")
    print("=" * 66, flush=True)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    from bench import HALLUCINATION_BUILDERS, n5_phase_forge
    fam_npzs = {}
    for tag, fn in [("N7a", "n7a_llm_forge.npz"),
                    ("N7b", "n7b_quantgan.npz"),
                    ("N8", "n8_gjr_egarch.npz")]:
        path = os.path.join(RESULTS_DIR, fn)
        if os.path.exists(path):
            with np.load(path, allow_pickle=True) as dz:
                by = {}
                for k in dz.files:
                    a, _, j = k.split("/")
                    by.setdefault(a, []).append((int(j), dz[k]))
                fam_npzs[tag] = {a: [v for _, v in sorted(
                    vv, key=lambda t: t[0])]
                    for a, vv in by.items()}

    FAM_CN = ["N1数值置换", "N2时间倒置", "N3标度破坏", "N4跨域嫁接",
              "N5相位伪造"]
    # 双读数：L 反转下严格变号（E9b 推导），单一读数无法同时覆盖
    # "箭头反转"与"箭头缺失"两态，第二代需两个手工读数，这本身
    # 是其代际结构特征（如实报告）。
    #   L-dir  anom = +T  （T 正 = 箭头反转 → 反转型伪造）
    #   L-abs  anom = -|T|（|T| 小 = 箭头缺失 → 缺失型伪造）
    #   CCK/bispec 强度读数 anom = -T
    def read_L_pos(T):
        return +T

    def read_L_abs(T):
        return -np.abs(T)

    stats = {"L-dir": d_L, "L-abs": d_L, "CCK": d_cck, "bispec": d_bispec}
    READ = {"L-dir": read_L_pos, "L-abs": read_L_abs,
            "CCK": lambda T: -T, "bispec": lambda T: -T}
    # 读数方向：Ldir 有向（箭头越强越真实 → anom=-T）；
    # CCK/bispec 强度（非线性质量小 → 伪造嫌疑 → anom=-T）
    res = {"protocol": {
        "reading": "statistic value (not p); direction per statistic "
                   "semantics: anom = -T throughout",
        "rationale": "two-sided p reading cannot see arrow reversal "
                     "(E9 finding); directional reading is Gen2's "
                     "correct weapon for N2",
        "threshold": "per-asset cal q95 on -T"},
    }
    T = {s: {"cal": {}, "real": {}, "fam": {}} for s in stats}
    for ai, (asset, r) in enumerate(returns.items()):
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        cal_wins = span_windows(wins, *sp["cal"])
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        pools = {"real": test_wins, "cal": cal_wins}
        for fi, fam in enumerate(FAM_CN):
            pools[fam] = [
                (n5_phase_forge if fam.startswith("N5")
                 else HALLUCINATION_BUILDERS[fam])(w, seed=1000 + wi * 10 + fi)
                for wi, w in enumerate(test_wins)]
        for tag in fam_npzs:
            pools[tag] = list(fam_npzs[tag].get(asset, []))
        for sname, sfn in stats.items():
            for key, ws in pools.items():
                vals = np.array([sfn(w) for w in ws])
                rd = READ[sname]
                if key == "cal":
                    T[sname]["cal"][asset] = vals
                elif key == "real":
                    T[sname]["real"][asset] = vals
                else:
                    T[sname]["fam"].setdefault(key, {})[asset] = vals

    for sname in stats:
        rd = READ[sname]
        res[sname] = {}
        for fam, per in T[sname]["fam"].items():
            pv_f, pv_r, flags_f, flags_r = [], [], [], []
            for asset in returns:
                cal_r = rd(T[sname]["cal"][asset])
                thr = np.quantile(cal_r[~np.isnan(cal_r)], 1 - ALPHA)
                neg = rd(T[sname]["real"][asset])       # anom 读数
                flags_r.append(neg > thr)
                pv_r.extend(neg.tolist())
                if asset in per:
                    negf = rd(per[asset])
                    flags_f.append(negf > thr)
                    pv_f.extend(negf.tolist())
            ff = np.concatenate(flags_f)
            fr = np.concatenate(flags_r)
            res[sname][fam] = {
                "auc": round(auc_pair(pv_f, pv_r), 3),
                "recall": round(float(ff.mean()), 3),
                "fpr": round(float(fr.mean()), 3)}
        print(f"\n[{sname}]（方向读数）auc/recall/fpr：")
        for fam, v in res[sname].items():
            print(f"  {fam:10s} {v['auc']:.3f}  {v['recall']:.3f}"
                  f"  {v['fpr']:.3f}", flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved: e9b_gen2_directional.json ({res['elapsed_s']}s)")


if __name__ == "__main__":
    main()
