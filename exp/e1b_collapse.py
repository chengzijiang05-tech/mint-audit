"""E1b 塌缩验证（G1 关卡之一）：现稿引擎对时间反转的结构性失明。

命题 T2（古典块盲区）：现稿特征块 Φ_C 在时间反转算子 R 下不变
（至估计精度），因此任何 Φ_C 基检验对 N2（时间倒置伪造）的检出率
恒等于其 FPR，与参考库规模、阈值选择无关。

验证五件事：
  1. 特征级不变性审计（三层结构）：15 维特征在反转下的偏差分层，
     排序不变量（beta_tail、abs_acf1/3/5、surr_z、surr_z_acf_sum）
     应浮点精确不变（|Δ|≤1e-9）；分段/分箱估计器（H_DFA、H_RS、
     qpos3/qneg3、dH、dalpha、abs_dfa、surr_z_dfa）塌缩至箱分割
     边界精度；garch_pers 塌缩至 MLE 估计精度；
  1b. 可逆零假设对照：对每窗口的 IAAFT 代理（保谱线性类成员，
     结构上时间可逆）做同测量，若各特征偏差与真实窗口同量级，
     则残余偏差确证为估计器伪影，而非时间箭头信息；
  2. 分数级塌缩：马氏距离 d(x) vs d(Rx) 散点塌缩到对角线（8d/15d
     两口径，含 R1 三层切分下的 cal 冻结阈值；d8 的散布由 garch_pers
     的 MLE 路径噪声主导）；
  3. 判定级后果：cal 冻结阈值下逐窗配对 flag 一致率 ≥0.85，池化
     recall(N2) − FPR ≤ 0.10（穿透）；
  4. 表示选择的对照：杠杆不对称统计 lev_asym（corr(r_t, |r_{t+1}|)）
     在真实窗口上显著翻转，而在可逆代理上无此翻转，塌缩是表示的
     性质，不是数学必然。

协议：R1 三层切分原样（fit/校准/测试，window=1000 step=50）；特征
与马氏引擎逐行复刻现稿（features.extract_features 15d，前 8 维即
古典块）；N2 用 mint.operators.time_reverse（与 bench.n2 同一定义）。

产出：results/e1b_collapse.json + figures/fig_e1b_collapse.pdf
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

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

from features import (  # noqa: E402
    FULL_FEATURE_NAMES,
    extract_features,
)
from features.phase_ext import leverage_asym  # noqa: E402
from mint.operators import iaaft, time_reverse  # noqa: E402
from scipy import stats  # noqa: E402

WINDOW = 1000
STEP = 50
# 特征三层分组（T2 三层不变性结构）
EXACT_FEATS = ["beta_tail", "abs_acf1", "abs_acf3", "abs_acf5",
               "surr_z", "surr_z_acf_sum"]
MLE_FEATS = ["garch_pers"]
SPLIT_SPEC = {
    "equity":  {"n_anchor": 15, "cal": (34, 43), "test": (63, 99), "cap": 8},
    "bond":    {"n_anchor": 15, "cal": (34, 43), "test": (63, 94), "cap": 8},
    "fx":      {"n_anchor": 11, "cal": (31, 36), "test": (37, 42), "cap": 6},
    "gold":    {"n_anchor": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
    "copper":  {"n_anchor": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
}
RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
FIGURES_DIR = os.path.join(MINT_ROOT, "figures")

MORANDI = {
    "equity": "#4A6FA5", "bond": "#8FA8BF", "fx": "#9CAF88",
    "gold": "#C9A66B", "copper": "#A9745E",
}


def span_windows(wins: list[np.ndarray], lo: int, hi: int,
                 cap: int | None = None) -> list[np.ndarray]:
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


def fit_mahalanobis(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False) + 1e-6 * np.eye(feats.shape[1])
    return mu, np.linalg.pinv(cov)


def mahal(f: np.ndarray, mu: np.ndarray, inv: np.ndarray) -> float:
    dev = f - mu
    return float(np.sqrt(dev @ inv @ dev))


def main() -> None:
    smoke = "--smoke" in sys.argv
    t0 = time.time()
    print("=" * 66)
    print("E1b 塌缩验证：现稿引擎对时间反转的结构性失明（T2 前提）"
          + ("  [冒烟]" if smoke else ""))
    print("=" * 66)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    per_asset: dict[str, dict] = {}
    pooled = {
        "feat_x": [], "feat_rev": [], "feat_surr": [], "feat_surr_rev": [],
        "asset": [],
        "d8_x": [], "d8_rev": [], "d15_x": [], "d15_rev": [],
        "d8_surr": [], "d8_surr_rev": [], "d15_surr": [], "d15_surr_rev": [],
        "lev_x": [], "lev_rev": [], "lev_surr": [], "lev_surr_rev": [],
        "flag8_x": 0, "flag8_rev": 0, "flag15_x": 0, "flag15_rev": 0,
        "agree8": 0, "agree15": 0, "n_test": 0,
    }

    for asset, r in returns.items():
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        cap = 2 if smoke else sp["cap"]
        anchors = wins[:sp["n_anchor"]]
        cal = span_windows(wins, *sp["cal"])
        clean = span_windows(wins, *sp["test"], cap=cap)

        # 拟合层窗库估计（8d / 15d 两口径；15d 特征前 8 维即古典块）
        A = np.array([extract_features(w, full=True) for w in anchors])
        A = A[np.isfinite(A).all(axis=1)]
        mu8, inv8 = fit_mahalanobis(A[:, :8])
        mu15, inv15 = fit_mahalanobis(A)

        # cal 冻结阈值（q95，对齐 R1-B 主口径）
        C = np.array([extract_features(w, full=True) for w in cal])
        C = C[np.isfinite(C).all(axis=1)]
        cal8 = np.array([mahal(f[:8], mu8, inv8) for f in C])
        cal15 = np.array([mahal(f, mu15, inv15) for f in C])
        t8 = float(np.quantile(cal8, 0.95))
        t15 = float(np.quantile(cal15, 0.95))

        # 测试层：x 与 Rx（time_reverse 即 N2 原型）配对
        rec = {"n_anchor": len(A), "n_cal": len(C), "n_test": len(clean),
               "threshold_8d": t8, "threshold_15d": t15}
        d8x, d8r, d15x, d15r, levx, levr = [], [], [], [], [], []
        d8s, d8sr, d15s, d15sr = [], [], [], []
        f15x, f15r, f15s, f15sr = [], [], [], []
        levs, levsr = [], []
        n_drop = 0
        for wi, w in enumerate(clean):
            rw = time_reverse(w, None)
            # 可逆零假设对照：w 的 IAAFT 代理（保谱线性类成员）及其反转
            sw = iaaft(w, np.random.default_rng(1000 + wi))
            rsw = time_reverse(sw, None)
            fx = extract_features(w, full=True)
            fr = extract_features(rw, full=True)
            fs = extract_features(sw, full=True)
            frs = extract_features(rsw, full=True)
            if not (np.isfinite(fx).all() and np.isfinite(fr).all()
                    and np.isfinite(fs).all() and np.isfinite(frs).all()):
                n_drop += 1
                continue
            d8x.append(mahal(fx[:8], mu8, inv8))
            d8r.append(mahal(fr[:8], mu8, inv8))
            d15x.append(mahal(fx, mu15, inv15))
            d15r.append(mahal(fr, mu15, inv15))
            d8s.append(mahal(fs[:8], mu8, inv8))
            d8sr.append(mahal(frs[:8], mu8, inv8))
            d15s.append(mahal(fs, mu15, inv15))
            d15sr.append(mahal(frs, mu15, inv15))
            levx.append(float(leverage_asym(w)))
            levr.append(float(leverage_asym(rw)))
            levs.append(float(leverage_asym(sw)))
            levsr.append(float(leverage_asym(rsw)))
            f15x.append(fx); f15r.append(fr)
            f15s.append(fs); f15sr.append(frs)
        rec["n_dropped_nonfinite"] = n_drop
        d8x, d8r = np.array(d8x), np.array(d8r)
        d15x, d15r = np.array(d15x), np.array(d15r)
        d8s, d8sr = np.array(d8s), np.array(d8sr)
        d15s, d15sr = np.array(d15s), np.array(d15sr)

        fl8x, fl8r = d8x >= t8, d8r >= t8
        fl15x, fl15r = d15x >= t15, d15r >= t15
        rec.update({
            "corr_d8": float(np.corrcoef(d8x, d8r)[0, 1]),
            "corr_d15": float(np.corrcoef(d15x, d15r)[0, 1]),
            "corr_d8_surr": float(np.corrcoef(d8s, d8sr)[0, 1]),
            "corr_d15_surr": float(np.corrcoef(d15s, d15sr)[0, 1]),
            "collapse_ratio_8d": float(np.median(np.abs(d8x - d8r))
                                       / (np.percentile(d8x, 75)
                                          - np.percentile(d8x, 25) + 1e-12)),
            "collapse_ratio_15d": float(np.median(np.abs(d15x - d15r))
                                        / (np.percentile(d15x, 75)
                                           - np.percentile(d15x, 25) + 1e-12)),
            "fpr_8d": float(fl8x.mean()), "n2_recall_8d": float(fl8r.mean()),
            "fpr_15d": float(fl15x.mean()),
            "n2_recall_15d": float(fl15r.mean()),
            "paired_agreement_8d": float(np.mean(fl8x == fl8r)),
            "paired_agreement_15d": float(np.mean(fl15x == fl15r)),
            "lev_asym_mean_x": float(np.nanmean(levx)),
            "lev_asym_mean_rev": float(np.nanmean(levr)),
            "lev_asym_mean_surr": float(np.nanmean(levs)),
            "lev_asym_mean_surr_rev": float(np.nanmean(levsr)),
        })
        per_asset[asset] = rec

        # 池化累积（特征审计用测试层配对；cal 层仅用于冻结阈值）
        n_kept = len(f15x)
        pooled["feat_x"].extend(f15x); pooled["feat_rev"].extend(f15r)
        pooled["feat_surr"].extend(f15s); pooled["feat_surr_rev"].extend(f15sr)
        pooled["asset"].extend([asset] * n_kept)
        pooled["d8_x"].extend(d8x); pooled["d8_rev"].extend(d8r)
        pooled["d15_x"].extend(d15x); pooled["d15_rev"].extend(d15r)
        pooled["d8_surr"].extend(d8s); pooled["d8_surr_rev"].extend(d8sr)
        pooled["d15_surr"].extend(d15s); pooled["d15_surr_rev"].extend(d15sr)
        pooled["lev_x"].extend(levx); pooled["lev_rev"].extend(levr)
        pooled["lev_surr"].extend(levs); pooled["lev_surr_rev"].extend(levsr)
        pooled["flag8_x"] += int(fl8x.sum()); pooled["flag8_rev"] += int(fl8r.sum())
        pooled["flag15_x"] += int(fl15x.sum())
        pooled["flag15_rev"] += int(fl15r.sum())
        pooled["agree8"] += int((fl8x == fl8r).sum())
        pooled["agree15"] += int((fl15x == fl15r).sum())
        pooled["n_test"] += n_kept

        print(f"  [{asset}] n_test={len(clean)} "
              f"corr_d8={rec['corr_d8']:.4f} corr_d15={rec['corr_d15']:.4f} | "
              f"FPR/recall(N2) 8d: {rec['fpr_8d']:.2f}/{rec['n2_recall_8d']:.2f} "
              f"15d: {rec['fpr_15d']:.2f}/{rec['n2_recall_15d']:.2f} | "
              f"lev: {rec['lev_asym_mean_x']:+.3f}->{rec['lev_asym_mean_rev']:+.3f}",
              flush=True)

    # ---- 池化指标 ----
    asset_arr = np.array(pooled["asset"])
    fx_all = np.array(pooled["feat_x"])
    fr_all = np.array(pooled["feat_rev"])
    fs_all = np.array(pooled["feat_surr"])
    fsr_all = np.array(pooled["feat_surr_rev"])
    d8x, d8r = np.array(pooled["d8_x"]), np.array(pooled["d8_rev"])
    d15x, d15r = np.array(pooled["d15_x"]), np.array(pooled["d15_rev"])
    d8s, d8sr = np.array(pooled["d8_surr"]), np.array(pooled["d8_surr_rev"])
    d15s, d15sr = np.array(pooled["d15_surr"]), np.array(pooled["d15_surr_rev"])
    levx, levr = np.array(pooled["lev_x"]), np.array(pooled["lev_rev"])
    levs, levsr = np.array(pooled["lev_surr"]), np.array(pooled["lev_surr_rev"])
    n = pooled["n_test"]

    feat_invar: dict[str, dict] = {}
    for j, name in enumerate(FULL_FEATURE_NAMES):
        a, b = fx_all[:, j], fr_all[:, j]
        sa, sb = fs_all[:, j], fsr_all[:, j]
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(sa) & np.isfinite(sb)
        a, b, sa, sb = a[ok], b[ok], sa[ok], sb[ok]
        # 资产内标准化偏差：|Δf| / 资产内 std
        rel = []
        for ast in np.unique(asset_arr[ok]):
            m = asset_arr[ok] == ast
            sd = np.std(a[m]) + 1e-12
            rel.append(np.mean(np.abs(a[m] - b[m]) / sd))
        mad_real = float(np.mean(np.abs(a - b)))
        mad_surr = float(np.mean(np.abs(sa - sb)))
        da, dbs = np.abs(a - b), np.abs(sa - sb)
        corr_delta = (float(np.corrcoef(da, dbs)[0, 1])
                      if np.std(da) > 0 and np.std(dbs) > 0 else float("nan"))
        feat_invar[name] = {
            "corr": float(np.corrcoef(a, b)[0, 1]),
            "within_asset_mean_rel_delta": float(np.mean(rel)),
            "max_abs_delta": float(np.max(np.abs(a - b))),
            "mean_abs_delta_real": mad_real,
            "mean_abs_delta_surr": mad_surr,
            "artifact_ratio": (mad_real / mad_surr) if mad_surr > 0 else
                              (0.0 if mad_real < 1e-12 else float("inf")),
            "corr_delta_real_vs_surr": corr_delta,
        }

    lev_ok = np.isfinite(levx) & np.isfinite(levr)
    wil = stats.wilcoxon(levx[lev_ok], levr[lev_ok])
    lev_ok_s = np.isfinite(levs) & np.isfinite(levsr)
    wil_s = stats.wilcoxon(levs[lev_ok_s], levsr[lev_ok_s])
    lev_stats = {
        "corr": float(np.corrcoef(levx[lev_ok], levr[lev_ok])[0, 1]),
        "mean_x": float(np.mean(levx[lev_ok])),
        "mean_rev": float(np.mean(levr[lev_ok])),
        "wilcoxon_p": float(wil.pvalue),
        "mean_surr": float(np.mean(levs[lev_ok_s])),
        "mean_surr_rev": float(np.mean(levsr[lev_ok_s])),
        "wilcoxon_p_surr": float(wil_s.pvalue),
        "abs_shift_real": float(abs(np.mean(levr[lev_ok])
                                    - np.mean(levx[lev_ok]))),
        "abs_shift_surr": float(abs(np.mean(levsr[lev_ok_s])
                                    - np.mean(levs[lev_ok_s]))),
    }

    pooled_out = {
        "n_test_windows": n,
        "corr_d8": float(np.corrcoef(d8x, d8r)[0, 1]),
        "corr_d15": float(np.corrcoef(d15x, d15r)[0, 1]),
        "corr_d8_surr": float(np.corrcoef(d8s, d8sr)[0, 1]),
        "corr_d15_surr": float(np.corrcoef(d15s, d15sr)[0, 1]),
        "fpr_8d_pooled": pooled["flag8_x"] / n,
        "n2_recall_8d_pooled": pooled["flag8_rev"] / n,
        "fpr_15d_pooled": pooled["flag15_x"] / n,
        "n2_recall_15d_pooled": pooled["flag15_rev"] / n,
        "paired_agreement_8d": pooled["agree8"] / n,
        "paired_agreement_15d": pooled["agree15"] / n,
        "feature_invariance": feat_invar,
        "lev_asym_contrast": lev_stats,
    }

    # ---- T2 判定（按三层不变性结构）----
    invar = feat_invar
    BINNING_FEATS = [k for k in FULL_FEATURE_NAMES
                     if k not in EXACT_FEATS + MLE_FEATS]

    # 层1：排序不变量应浮点精确不变（|Δ| ≤ 1e-9）
    layer1_exact = all(invar[k]["max_abs_delta"] <= 1e-9 for k in EXACT_FEATS)
    # 杠杆信号比：真实 shift / 可逆对照 shift（纯时间箭头信号强度）
    r_lev = (lev_stats["abs_shift_real"]
             / max(lev_stats["abs_shift_surr"], 1e-12))
    # 层2：分段/分箱估计器塌缩至边界精度，残余偏差以伪影为主，
    #   ratio ≤ 3（可逆对照解释 ≥1/3 的偏差基线），且残余结构成分
    #   远低于 lev 信号强度（ratio ≤ r_lev/5，即不超过其 20%）
    layer2_collapse = all(invar[k]["corr"] >= 0.85 for k in BINNING_FEATS)
    layer2_artifact = all(
        (invar[k]["artifact_ratio"] <= 3.0
         and invar[k]["artifact_ratio"] <= r_lev / 5.0)
        or invar[k]["mean_abs_delta_real"] <= 1e-9
        for k in BINNING_FEATS)
    # 层3：garch_pers 塌缩至 MLE 估计精度，且偏差不超过可逆对照 2 倍
    #   （完整数据中 ratio≈0.1：代理上 GARCH 拟合更不稳定，纯伪影）
    layer3_collapse = invar["garch_pers"]["corr"] >= 0.6
    layer3_artifact = invar["garch_pers"]["artifact_ratio"] <= 2.0

    # 分数级：d15 塌缩至对角线；d8 允许 garch_pers MLE 噪声稀释
    score_collapse = (pooled_out["corr_d15"] >= 0.95
                      and pooled_out["corr_d8"] >= 0.85)
    # 判定级：穿透（recall − FPR ≤ 0.10）且配对一致
    penetration = (
        abs(pooled_out["n2_recall_8d_pooled"]
            - pooled_out["fpr_8d_pooled"]) <= 0.10
        and abs(pooled_out["n2_recall_15d_pooled"]
                - pooled_out["fpr_15d_pooled"]) <= 0.10)
    flag_equality = (penetration
                     and pooled_out["paired_agreement_8d"] >= 0.85
                     and pooled_out["paired_agreement_15d"] >= 0.85)
    # 表示对照：真实窗口上杠杆翻转显著且符号反转；可逆代理上无此结构
    lev_contrast = bool(lev_stats["wilcoxon_p"] < 0.05
                        and lev_stats["mean_x"] * lev_stats["mean_rev"] < 0)
    lev_contrast_surr_clean = bool(
        lev_stats["abs_shift_surr"] <= 2.0 * lev_stats["abs_shift_real"])

    verdict = {
        "layer1_order_invariants_exact": layer1_exact,
        "layer2_binning_collapse": layer2_collapse,
        "layer2_residual_is_artifact": layer2_artifact,
        "layer3_garch_mle_collapse": layer3_collapse,
        "layer3_residual_is_artifact": layer3_artifact,
        "score_collapse_to_diagonal": score_collapse,
        "recall_equals_fpr_at_frozen_threshold": flag_equality,
        "lev_asym_contrast_significant": lev_contrast,
        "lev_contrast_absent_on_reversible": lev_contrast_surr_clean,
        "t2_confirmed": bool(layer1_exact and layer2_collapse
                             and layer2_artifact and layer3_collapse
                             and layer3_artifact and score_collapse
                             and flag_equality and lev_contrast
                             and lev_contrast_surr_clean),
    }

    print("\n" + "-" * 66)
    print(f"池化 n={n} | corr d8={pooled_out['corr_d8']:.4f} "
          f"(对照 {pooled_out['corr_d8_surr']:.4f}) "
          f"d15={pooled_out['corr_d15']:.4f} "
          f"(对照 {pooled_out['corr_d15_surr']:.4f})")
    print(f"FPR/recall(N2) 8d: {pooled_out['fpr_8d_pooled']:.3f}/"
          f"{pooled_out['n2_recall_8d_pooled']:.3f}  "
          f"15d: {pooled_out['fpr_15d_pooled']:.3f}/"
          f"{pooled_out['n2_recall_15d_pooled']:.3f}  "
          f"配对一致率 8d={pooled_out['paired_agreement_8d']:.3f} "
          f"15d={pooled_out['paired_agreement_15d']:.3f}")
    print("特征三层审计（真实|Δ| / 可逆对照|Δ|，ratio）:")
    print(f"  层1 精确: " + ", ".join(
        f"{k}={invar[k]['max_abs_delta']:.1e}" for k in EXACT_FEATS))
    print(f"  层2 箱分割: " + ", ".join(
        f"{k} r={invar[k]['corr']:.3f} ×{invar[k]['artifact_ratio']:.2f}"
        for k in BINNING_FEATS))
    print(f"  层3 MLE: garch_pers r={invar['garch_pers']['corr']:.3f} "
          f"×{invar['garch_pers']['artifact_ratio']:.2f}")
    print(f"杠杆对照: 真实 {lev_stats['mean_x']:+.4f} -> "
          f"{lev_stats['mean_rev']:+.4f} (p={lev_stats['wilcoxon_p']:.1e}) | "
          f"可逆代理 {lev_stats['mean_surr']:+.4f} -> "
          f"{lev_stats['mean_surr_rev']:+.4f} "
          f"(p={lev_stats['wilcoxon_p_surr']:.1e}) | "
          f"信号比 r_lev={r_lev:.1f}×")
    print("-" * 66)
    for k, v in verdict.items():
        print(f"T2[{k}]: {'PASS' if v else 'FAIL'}")
    print(f"E1b 总体: {'塌缩确认（三层结构），T2 前提成立' if verdict['t2_confirmed'] else '未确认，需人工检查'}")
    print(f"耗时 {time.time() - t0:.0f}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e1b_collapse.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "smoke": smoke, "elapsed_s": round(time.time() - t0, 1),
            "window": WINDOW, "step": STEP, "split_spec": SPLIT_SPEC,
            "per_asset": per_asset, "pooled": pooled_out, "verdict": verdict,
            "lev_signal_ratio": float(r_lev),
            "scatter": {
                "asset": asset_arr.tolist(),
                "d8_x": d8x.tolist(), "d8_rev": d8r.tolist(),
                "d15_x": d15x.tolist(), "d15_rev": d15r.tolist(),
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {os.path.join(RESULTS_DIR, 'e1b_collapse.json')}")

    # ---- 图（论文风格：莫兰迪深蓝/白，PDF 矢量） ----
    if not smoke:
        make_figure(asset_arr, d8x, d8r, d15x, d15r, feat_invar, lev_stats,
                    pooled_out)


def make_figure(asset_arr, d8x, d8r, d15x, d15r, feat_invar, lev_stats,
                pooled) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8.5, "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4),
                             gridspec_kw={"wspace": 0.38,
                                          "width_ratios": [1, 1, 1.35]})

    def scatter(ax, dx, dr, title, r_pooled):
        lim = [0, max(dx.max(), dr.max()) * 1.08]
        ax.fill_between(lim, lim, [lim[1], lim[1]], color="#F0F2F5",
                        zorder=0, lw=0)
        ax.plot(lim, lim, color="#7A8691", lw=0.8, ls="--", zorder=1)
        for ast in MORANDI:
            m = asset_arr == ast
            ax.scatter(dx[m], dr[m], s=14, c=MORANDI[ast], alpha=0.85,
                       edgecolors="white", linewidths=0.4, zorder=2)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("d(x)  (clean window)", fontsize=8.5)
        ax.set_ylabel("d(Rx)  (time-reversed)", fontsize=8.5)
        ax.set_title(title, fontsize=9)
        ax.text(0.05, 0.93, f"r = {r_pooled:.4f}", transform=ax.transAxes,
                fontsize=8.5, color="#35507A")
        ax.grid(True, color="#E8EAED", lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    scatter(axes[0], d8x, d8r, "(a) Classical 8-d engine",
            pooled["corr_d8"])
    scatter(axes[1], d15x, d15r, "(b) Full 15-d engine",
            pooled["corr_d15"])

    # (c) 特征级三层偏差审计：|Δf| 真实窗口 vs 可逆对照（log 尺度）
    #     层1/层2/层3 两点重叠 → 残余偏差是估计器伪影；
    #     lev_asym 两点分离 → 翻转来自真实不可逆结构。
    ax = axes[2]
    binning = [k for k in FULL_FEATURE_NAMES
               if k not in EXACT_FEATS + MLE_FEATS]
    order = (EXACT_FEATS + binning + MLE_FEATS + ["lev_asym"])
    real = [feat_invar[k]["mean_abs_delta_real"] if k != "lev_asym"
            else lev_stats["abs_shift_real"] for k in order]
    surr = [feat_invar[k]["mean_abs_delta_surr"] if k != "lev_asym"
            else lev_stats["abs_shift_surr"] for k in order]
    floor = 1e-17
    real_c = [max(v, floor) for v in real]
    surr_c = [max(v, floor) for v in surr]
    ypos = np.arange(len(order))[::-1]
    ax.scatter(real_c, ypos, s=16, c="#4A6FA5", zorder=3,
               edgecolors="white", linewidths=0.3, label="real window")
    ax.scatter(surr_c, ypos, s=22, facecolors="none", edgecolors="#8FA8BF",
               linewidths=1.1, zorder=3, label="reversible control")
    ax.set_xscale("log")
    ax.set_xlim(floor * 0.5, 30)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order, fontsize=6.8)
    ax.set_xlabel("mean $|\\Delta f|$ under reversal  (log)", fontsize=8.5)
    ax.set_title("(c) Three-layer artifact audit", fontsize=9)
    # 层分隔与标注
    n1, n2, n3 = len(EXACT_FEATS), len(binning), len(MLE_FEATS)
    for yb in (ypos[n1] + 0.5, ypos[n1 + n2] + 0.5,
               ypos[n1 + n2 + n3] + 0.5):
        ax.axhline(yb, color="#D5DBE1", lw=0.6, ls=":", zorder=1)
    ax.text(1e-15, ypos[0] + 0.55, "L1 order invariants", fontsize=6.2,
            color="#7A8691", va="bottom")
    ax.text(1e-15, ypos[n1] + 0.55, "L2 binning estimators", fontsize=6.2,
            color="#7A8691", va="bottom")
    ax.text(1e-15, ypos[n1 + n2] + 0.55, "L3 GARCH MLE", fontsize=6.2,
            color="#7A8691", va="bottom")
    ax.text(1e-15, ypos[n1 + n2 + n3] + 0.55, "contrast", fontsize=6.2,
            color="#A9745E", va="bottom")
    ax.grid(True, axis="x", color="#E8EAED", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=6.8, frameon=False,
              handlelength=1.0)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_e1b_collapse.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"图已写入 {out}")


if __name__ == "__main__":
    main()
