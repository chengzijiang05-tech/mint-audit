"""E12：训练自由乘积 e 值基线 + Gen2 公平校准行（R12）。

编辑决定书 R12：池化 AUC 混合训练相邻族（N2 为轨道成员、N5 在
AAFT/IAAFT 射程内）与真正未见族；古典读数在 N3/N4 上胜过 MINT；
未运行"校准经典读数的乘积 e 值"基线 → 学习编码器的边际贡献
未被测量。要求：(i) in-null vs out-of-union AUC 分开报告；
(ii) 2-3 个 surrogate 校准读数的乘积 e 值基线行；(iii) Gen2
加校准层阈值（与古典引擎同等待遇）。

乘积 e 值构造（validity 由本文定理 1 的分数无关性直接继承）：
  读数 r_j（越高 = 结构越真实）：
    r1 杠杆不对称 leverage_asym（lag-1，真实金融窗 > 0，轨道成员 ≈ 0）
    r2 波动聚集 vol_clust = mean_{k<=20} AC(|x|, k)（真实窗高，
       轨道相位随机化/置换摧毁）
    r3 -Mahalanobis_8d（古典 A5b 引擎读数取负，真实窗贴近拟合层分布）
  自校准（保持 (K+1) 集合对称性 → E[W]=1 精确）：
    z_j(v) = (r_j(v) - mean_{all K+1} r_j) / std_{all K+1} r_j
    s(v) = Σ_j z_j(v) = log ∏_j e^{z_j(v)}（校准读数的乘积）
    W(x) = (K+1) · softmax(s)_x
  z 在全部 K+1 成员上估计 → s 是集合的对称函数 → 重标号下
  E[W] = 1 精确成立（与引理 1 同一论证），认证规则 W ≥ 1/α 有效。

Gen2 公平校准行：Ldir TR 统计量 + IAAFT surrogate p 值（Theiler
程序，B=99，与 e9 同种子协议），异常分 = p（p 大 = 像自身
surrogate = 可疑），阈值 = 逐资产 cal 层 q95（与 A0/A5b 同等待遇）。

覆盖族：N1–N5（bench 构造，e2 种子协议）+ N7a/N7b/N7c/N8（npz）。
in-null = {N2, N5}；out-of-union = {N3, N4, N7a, N7b, N7c, N8}；
N1 为数值层（类型化通道对象，单独报告）。

产出：results/e12_product_evalue.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import defaultdict

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

from bench import HALLUCINATION_BUILDERS, n5_phase_forge  # noqa: E402
from features import extract_features  # noqa: E402
from features.phase_ext import leverage_asym  # noqa: E402
from mint.operators import generate_orbit  # noqa: E402

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")

WINDOW, STEP = 1000, 50
SPLIT_SPEC = {
    "equity":  {"n_anchor": 15, "cal": (34, 43), "test": (63, 99), "cap": 8},
    "bond":    {"n_anchor": 15, "cal": (34, 43), "test": (63, 94), "cap": 8},
    "fx":      {"n_anchor": 11, "cal": (31, 36), "test": (37, 42), "cap": 6},
    "gold":    {"n_anchor": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
    "copper":  {"n_anchor": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
}
FAM_BUILD = ["N1数值置换", "N2时间倒置", "N3标度破坏",
             "N4跨域嫁接", "N5相位伪造"]
FAM_FILES = {
    "N7a": "n7a_llm_forge.npz",
    "N7b": "n7b_quantgan.npz",
    "N7c": "n7c_chronos_forge.npz",
    "N8": "n8_gjr_egarch.npz",
}
IN_NULL = ["N2", "N5"]
OUT_UNION = ["N3", "N4", "N7a", "N7b", "N7c", "N8"]
K_EVAL = 32
ALPHA = 0.05
SEED = 20260820
B_SURR = 99
K_LAG = 20


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


def forge(x, family, seed):
    if family == "N5相位伪造":
        return n5_phase_forge(x, seed=seed)
    return HALLUCINATION_BUILDERS[family](x, seed=seed)


def vol_clust(x):
    """波动聚集读数：|x| 的滞后自相关均值（lags 1..K_LAG）。"""
    x = np.asarray(x, float) - np.mean(x)
    a = np.abs(x)
    a = a - a.mean()
    d = float(np.mean(a * a))
    if d <= 0:
        return 0.0
    cs = []
    for k in range(1, K_LAG + 1):
        cs.append(float(np.mean(a[:-k] * a[k:])) / d)
    return float(np.mean(cs))


def readings(x, mu, inv):
    """三个经典读数（越高 = 结构越真实）。"""
    r1 = float(leverage_asym(x))
    r2 = float(vol_clust(x))
    f = extract_features(x, full=False)
    if not np.isfinite(f).all():
        r3 = -1e9
    else:
        dev = f - mu
        r3 = -float(np.sqrt(dev @ inv @ dev))
    return np.array([r1, r2, r3])


def prod_e_value(x, k, rng, mu, inv):
    """乘积 e 值：轨道自校准三读数之和 → softmax 权重。"""
    surrs, _ = generate_orbit(x, k, rng=rng)
    members = [x] + list(surrs)
    R = np.array([readings(m, mu, inv) for m in members])
    sd = R.std(axis=0)
    sd[sd < 1e-12] = 1e-12
    Z = (R - R.mean(axis=0)) / sd
    s = Z.sum(axis=1)
    m = float(np.max(s))
    lse = m + float(np.log(np.sum(np.exp(s - m))))
    W = float((k + 1) * np.exp(s[0] - lse))
    return W


def stat_Ldir(x):
    """有向 TR 读数（与 e9 逐字一致）。"""
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


def iaaft(x, rng, n_iter=10):
    """IAAFT surrogate（保边际保谱），与 mint.operators 同算法。"""
    from mint.operators import iaaft as _iaaft
    return _iaaft(x, rng)


def gen2_pvalue(x, seed, b=B_SURR):
    """Theiler 程序 p 值（与 e9 相同的双侧 min 协议）。"""
    rng = np.random.default_rng(seed)
    t_obs = stat_Ldir(x)
    t_s = np.empty(b)
    for i in range(b):
        t_s[i] = stat_Ldir(iaaft(x, rng))
    p_hi = (1 + np.sum(t_s >= t_obs)) / (b + 1)
    p_lo = (1 + np.sum(t_s <= t_obs)) / (b + 1)
    return float(min(p_hi, p_lo))


def auc_pair(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.r_[pos, neg]
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = len(pos), len(neg)
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def wilson(k, n, zz=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + zz * zz / n
    c = p + zz * zz / (2 * n)
    h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def main():
    t0 = time.time()
    print("=" * 66)
    print("E12 乘积 e 值基线（训练自由）+ Gen2 公平校准行（R12）")
    print("=" * 66, flush=True)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    fams_npz = {}
    for tag, fn in FAM_FILES.items():
        path = os.path.join(RESULTS_DIR, fn)
        if not os.path.exists(path):
            print(f"!! 缺 {fn}，跳过 {tag}", flush=True)
            continue
        with np.load(path, allow_pickle=True) as d:
            by_asset = defaultdict(list)
            for key in d.files:
                asset, rest = key.split("/", 1)
                by_asset[asset].append(d[key])
        fams_npz[tag] = by_asset
        n_tot = sum(len(v) for v in by_asset.values())
        print(f"{tag}: {n_tot} 条", flush=True)
    fam_tags = FAM_BUILD + [t for t in FAM_FILES if t in fams_npz]

    anom = {"ProdE": {"cal": [], "real": [], "fam": {f: [] for f in fam_tags}},
            "Gen2cal": {"cal": [], "real": [], "fam": {f: [] for f in fam_tags}}}
    Wraw = {"ProdE": {"real": [], "fam": {f: [] for f in fam_tags}}}
    per_asset = {}
    done_assets = set()

    partial_path = os.path.join(RESULTS_DIR, "e12_partial.pkl")
    if os.path.exists(partial_path):
        with open(partial_path, "rb") as fh:
            partial = pickle.load(fh)
        anom = partial["anom"]
        Wraw = partial["Wraw"]
        done_assets = set(partial["done"])
        print(f"续跑缓存：已完成 {sorted(done_assets)}", flush=True)

    for ai, (asset, r) in enumerate(returns.items()):
        if asset in done_assets:
            continue
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        anchors = wins[:sp["n_anchor"]]
        cal_wins = span_windows(wins, *sp["cal"])
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        forged = {fam: [forge(w, fam, seed=1000 + wi * 10 + fi)
                        for wi, w in enumerate(test_wins)]
                  for fi, fam in enumerate(FAM_BUILD)}
        npz_fam = {tag: list(fams_npz[tag].get(asset, []))
                   for tag in fams_npz}
        print(f"\n[{asset}] cal{len(cal_wins)} test{len(test_wins)} "
              f"| npz 族 {({t: len(v) for t, v in npz_fam.items()})}",
              flush=True)

        A = np.array([extract_features(w, full=False) for w in anchors])
        A = A[np.isfinite(A).all(axis=1)]
        mu = A.mean(axis=0)
        inv = np.linalg.pinv(np.cov(A, rowvar=False)
                             + 1e-6 * np.eye(A.shape[1]))

        base = 950000 + ai * 10000
        w_cal = [prod_e_value(w, K_EVAL, np.random.default_rng(base + i),
                              mu, inv)
                 for i, w in enumerate(cal_wins)]
        w_real = [prod_e_value(w, K_EVAL,
                               np.random.default_rng(base + 100 + i),
                               mu, inv)
                  for i, w in enumerate(test_wins)]
        anom["ProdE"]["cal"].append(-np.array(w_cal))
        anom["ProdE"]["real"].append(-np.array(w_real))
        Wraw["ProdE"]["real"].append(np.array(w_real))
        for fi, fam in enumerate(FAM_BUILD):
            w_f = [prod_e_value(w, K_EVAL,
                                np.random.default_rng(base + 200 + fi * 100 + i),
                                mu, inv)
                   for i, w in enumerate(forged[fam])]
            anom["ProdE"]["fam"][fam].append(-np.array(w_f))
            Wraw["ProdE"]["fam"][fam].append(np.array(w_f))
        for fj, tag in enumerate(fams_npz):
            w_f = [prod_e_value(w, K_EVAL,
                                np.random.default_rng(base + 700 + fj * 100 + i),
                                mu, inv)
                   for i, w in enumerate(npz_fam[tag])]
            anom["ProdE"]["fam"][tag].append(-np.array(w_f))
            Wraw["ProdE"]["fam"][tag].append(np.array(w_f))

        # Gen2 公平校准：cal p 值 + 逐资产 q95（与古典引擎同等待遇）
        p_cal = [gen2_pvalue(w, 450000 + ai * 100 + i)
                 for i, w in enumerate(cal_wins)]
        p_real = [gen2_pvalue(w, 500000 + ai * 100 + i)
                  for i, w in enumerate(test_wins)]
        anom["Gen2cal"]["cal"].append(np.array(p_cal))
        anom["Gen2cal"]["real"].append(np.array(p_real))
        for fi, fam in enumerate(FAM_BUILD):
            p_f = [gen2_pvalue(w, 600000 + ai * 100 + fi * 97 + i)
                   for i, w in enumerate(forged[fam])]
            anom["Gen2cal"]["fam"][fam].append(np.array(p_f))
        for fj, tag in enumerate(fams_npz):
            p_f = [gen2_pvalue(w, 650000 + ai * 100 + fj * 97 + i)
                   for i, w in enumerate(npz_fam[tag])]
            anom["Gen2cal"]["fam"][tag].append(np.array(p_f))
        print(f"    ProdE/Gen2cal [{asset}] 完成 "
              f"({time.time()-t0:.0f}s)", flush=True)
        done_assets.add(asset)
        with open(partial_path, "wb") as fh:
            pickle.dump({"anom": anom, "Wraw": Wraw,
                         "done": sorted(done_assets)}, fh)

    # ---- 池化（逐资产 cal q95 阈值，与 e2 同协议）----
    res = {"protocol": {
        "prod_e": "3 readings (leverage_asym, vol_clust, -mahal8d) "
                  "self-calibrated over K+1 orbit members; "
                  "W=(K+1)*softmax(sum z); exact E[W]=1 via Lemma 1",
        "gen2cal": "Ldir TR stat + IAAFT B=99 surrogate p-value "
                   "(Theiler), anomaly=p, per-asset cal q95 threshold",
        "K_eval": K_EVAL, "alpha": ALPHA, "seed": SEED,
        "in_null": IN_NULL, "out_union": OUT_UNION,
    }, "methods": {}}

    for tag_m in ("ProdE", "Gen2cal"):
        recalls, flags_real = {}, []
        fam_auc, fam_wil = {}, {}
        real_all, fam_all = [], {}
        for seg, _asset in enumerate(returns):
            cal = np.asarray(anom[tag_m]["cal"][seg])
            thr = float(np.quantile(cal, 1 - ALPHA))
            real = np.asarray(anom[tag_m]["real"][seg])
            flags_real.append(real > thr)
            real_all.append(real)
        for fam in fam_tags:
            hit, tot, f_scores = 0, 0, []
            for seg, _asset in enumerate(returns):
                cal = np.asarray(anom[tag_m]["cal"][seg])
                thr = float(np.quantile(cal, 1 - ALPHA))
                f = np.asarray(anom[tag_m]["fam"][fam][seg])
                fl = f > thr
                hit += int(np.sum(fl))
                tot += len(fl)
                f_scores.append(f)
            fam_all[fam] = np.concatenate(f_scores)
            recalls[fam[:2] if fam in FAM_BUILD else fam] = hit / max(tot, 1)
            lo, hi = wilson(hit, tot)
            fam_wil[fam[:2] if fam in FAM_BUILD else fam] = [lo, hi]
        real_flat = np.concatenate(flags_real)
        real_sc = np.concatenate(real_all)
        for fam in fam_tags:
            key = fam[:2] if fam in FAM_BUILD else fam
            fam_auc[key] = round(
                auc_pair(fam_all[fam], real_sc[real_sc == real_sc]), 3)
        all_f = np.concatenate([fam_all[f] for f in fam_tags])
        s = {"fpr": float(np.mean(real_flat)),
             "recall": recalls, "recall_wilson": fam_wil,
             "auc_family": fam_auc,
             "auc_all": round(auc_pair(all_f, real_sc), 3)}
        s["auc_in_null"] = round(float(np.mean(
            [v for k, v in fam_auc.items() if k in IN_NULL])), 3)
        out_v = [v for k, v in fam_auc.items() if k in OUT_UNION]
        s["auc_out_union"] = (round(float(np.mean(out_v)), 3)
                              if out_v else None)
        if "N1" in fam_auc:
            s["auc_value_level_N1"] = fam_auc["N1"]
        res["methods"][tag_m] = s
        print(f"\n[{tag_m}] FPR={s['fpr']:.3f} AUC_all={s['auc_all']} "
              f"in-null={s['auc_in_null']} "
              f"out-union={s['auc_out_union']}")
        for k, v in fam_auc.items():
            w = fam_wil[k]
            print(f"  {k}: AUC={v} recall={recalls[k]:.3f}"
                  f" Wilson[{w[0]:.2f},{w[1]:.2f}]")

    # e 规则认证（仅 ProdE）
    Wr = np.concatenate(Wraw["ProdE"]["real"])
    e_rule = {"real_certify": float(np.mean(Wr >= 1 / ALPHA))}
    for fam in fam_tags:
        Wf = np.concatenate(Wraw["ProdE"]["fam"][fam])
        key = fam[:2] if fam in FAM_BUILD else fam
        e_rule[f"{key}_certify"] = float(np.mean(Wf >= 1 / ALPHA))
    res["e_rule_ProdE"] = e_rule
    print("\nProdE e-rule:", json.dumps(e_rule, indent=1), flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    out = os.path.join(RESULTS_DIR, "e12_product_evalue.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved: e12_product_evalue.json ({res['elapsed_s']}s)")


if __name__ == "__main__":
    main()
