"""E3 小参考域功效分析（H3 关卡）：e 值 vs p 值 vs 保形序统计阈值。

设计依据：整体研究方案 §六 E3 + §9 H3；推导报告推进四（推断货币改革）。

要回答的问题：当参考域只有 n_ref 个样本时（现实：gold/copper 校准层
仅 3 窗，fx 6 窗），三种推断货币各自还剩多少效力？

  e 值（排除形）  W = m·e^{s(x)}/Σ_{i=1}^m e^{s(S_i)}，认证判据 W ≥ 1/α。
     E_0[W] = 1（轨道可交换性，推导见 docstring 末注）→ Markov 水平
     ≤ α 对任意 m 成立：无格点、无最小样本量；且证据上限无界
     （对比包含形 W ≤ m+1，m<19 时结构性失明，E2 用的正是包含形）。
  mid-p 置换      p = (b+0.5)/m，b = #{s(S_i) ≥ s(x)}；格点下限 0.5/m，
     m<10 时盲（现稿 midrank 修正的准确形态，牺牲精确有效性）。
  严格置换        p = (b+1)/(m+1)；格点下限 1/(m+1)，m<19 时盲
     （现稿原始协议，recall 0.000 的来源）。
  保形序统计      阈值 t = 升序第 ⌈(n+1)(1-α)⌉ 位（越界回退 max）；
     水平有效需 n ≥ 19，更小时回退 max 的真实水平 1/(n+1) > α。

两种参考域范式同场对比（同一 A0 编码器分数，货币是唯一变量）：
  轨道范式（e 值 / p 值在候选自己的轨道上），条件检验，免真实参考
     窗，对时期漂移免疫（E2 D1 结论）；
  真实参考范式（mid-p / 严格 / 保形在测试前真实窗分数上），现稿
     谱系，受格点与漂移双重约束。参考池 = 测试层之前全部真实窗
     （含训练跨度；编码器在跨度内见过相似数据，该范式自身的弱点）。

协议：
  候选 = 测试层真实窗 + N1/N3/N4/N5 伪造（伪造种子与 E2 逐位一致；
     N1 为弱伪造诊断，bench 中 N1 是乘性数值扰动非置换，保留真实
     依赖结构；N2 略去，E2 已证 e 规则 N2 recall = 1.0）；
  A0 编码器与 E2 v2 逐位一致（同种子同协议重训）；
  零假设成员水平模拟：真实窗 40 轨道中随机取 2 成员作候选 × 各自
     新鲜 n_ref 轨道，可交换性的直接检验，H3a 的测量对象；
  MDE（最小可检出效应量）= 满足判据所需的分数间隙（logit 单位，
     相对参考均值），按方法 × n_ref 报告，盲区记 inf。

H3 判定（预注册）：n_ref ∈ {3,5,10} 时
  (a) e 值零假设成员认证率 ≤ α + 0.02；
  (b) e 值 N3/4/5 检出功效 ≥ mid-p 真实参考范式。
  另报货币隔离对照：e 值真实窗认证率 ≥ 轨道 mid-p（同轨道同分数，
  唯一差异是货币），隔离格点效应的来源。

排除形 e 值有效性注记：H0 下 (x, S_1..S_m) 可交换 ⟹
  E[e^{s(x)}/Σ_S e^{s(S)}] = E[e^{s(S_1)}/Σ_S] = 1/m（求和恒等式）
  ⟹ E[W] = 1 ⟹ P(W ≥ 1/α) ≤ α，对任意 m ≥ 1 成立。

产出：results/e3_power.json + figures/fig_e3_power.pdf
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import defaultdict

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

import torch  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from bench import HALLUCINATION_BUILDERS, n5_phase_forge  # noqa: E402
from features import extract_features  # noqa: E402
from mint.model import (  # noqa: E402
    Encoder,
    ScoreHead,
    bce_loss,
    gaussianize_batch,
    info_nce,
)
from mint.operators import ORBIT_NAMES, generate_orbit  # noqa: E402

WINDOW, STEP = 1000, 50
SPLIT_SPEC = {
    "equity":  {"n_anchor": 15, "cal": (34, 43), "test": (63, 99), "cap": 8},
    "bond":    {"n_anchor": 15, "cal": (34, 43), "test": (63, 94), "cap": 8},
    "fx":      {"n_anchor": 11, "cal": (31, 36), "test": (37, 42), "cap": 6},
    "gold":    {"n_anchor": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
    "copper":  {"n_anchor": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
}
FAM_IDX = {"N1数值置换": 0, "N3标度破坏": 2, "N4跨域嫁接": 3, "N5相位伪造": 4}
FAM_E3 = {"N1数值置换": "N1", "N3标度破坏": "N3",
          "N4跨域嫁接": "N4", "N5相位伪造": "N5"}
ALPHA = 0.05
N_REF_GRID = [3, 5, 10, 19, 40]

TRAIN_LEN, N_SLICES = 700, 64
POOL_SLICES, POOL_PER, K_TRAIN = 150, 20, 12
EPOCHS, TAU, LR, CROP = int(os.environ.get("MINT_E3_EPOCHS", "250")), \
    0.1, 1e-3, 620
SEED = 20260820

ORBIT_METHODS = ["e", "e_inc", "midp_o", "strict_o"]
REAL_METHODS = ["midp_r", "strict_r", "conf"]
MAH_METHODS = ["midp_mah", "strict_mah", "conf_mah"]
ALL_METHODS = ORBIT_METHODS + REAL_METHODS + MAH_METHODS
METHOD_DESC = {
    "e":         "e值·排除形(轨道)",
    "e_inc":     "e值·包含形(轨道,E2口径)",
    "midp_o":    "mid-p置换(轨道)",
    "strict_o":  "严格置换(轨道)",
    "midp_r":    "mid-p置换(真实参考)",
    "strict_r":  "严格置换(真实参考)",
    "conf":      "保形序统计(真实参考)",
    "midp_mah":  "mid-p置换(现稿马氏)",
    "strict_mah": "严格置换(现稿马氏)",
    "conf_mah":  "保形序统计(现稿马氏)",
}

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
FIGURES_DIR = os.path.join(MINT_ROOT, "figures")


# ---------------------------------------------------------------------------
# 数据、伪造与训练（与 e2_main 逐位一致）
# ---------------------------------------------------------------------------
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


def crop_noise(w, rng):
    start = int(rng.integers(0, len(w) - CROP + 1))
    out = w[start:start + CROP].copy()
    return out + 0.02 * out.std() * rng.standard_normal(len(out))


def build_pool(r, train_end, rng, n_slices, per):
    starts = rng.integers(0, train_end - TRAIN_LEN + 1, size=n_slices)
    rows = []
    for s in starts:
        w = r[int(s):int(s) + TRAIN_LEN]
        surrs, _ = generate_orbit(w, per, names=ORBIT_NAMES, rng=rng)
        rows.append(surrs)
    return np.vstack(rows)


def train_a0(r, train_end, seed, epochs=EPOCHS):
    """A0 完整 MINT：轨道负样本（五算子池）+ 秩归一化，与 E2 v2 一致。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, head = Encoder(), ScoreHead()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    pool = build_pool(r, train_end, np.random.default_rng(seed + 77),
                      POOL_SLICES, POOL_PER)
    pool = gaussianize_batch(pool)

    t0 = time.time()
    last = {}
    for epoch in range(epochs):
        starts = rng.integers(0, train_end - TRAIN_LEN + 1, size=N_SLICES)
        wins = [r[s:s + TRAIN_LEN] for s in starts]
        X = torch.tensor(gaussianize_batch(np.array(wins)),
                         dtype=torch.float32)
        X_pos = torch.tensor(
            gaussianize_batch(np.array([crop_noise(w, rng) for w in wins])),
            dtype=torch.float32)
        need = N_SLICES * K_TRAIN
        idx = rng.choice(len(pool), size=need, replace=len(pool) < need)
        N = torch.tensor(pool[idx], dtype=torch.float32)

        opt.zero_grad()
        z, z_pos, z_neg = enc(X), enc(X_pos), enc(N)
        loss = info_nce(z, z_pos, z_neg, tau=TAU) + bce_loss(
            head(z), head(z_neg), pos_weight=float(K_TRAIN))
        loss.backward()
        opt.step()
        sched.step()
        last = {"loss": round(loss.item(), 4)}
    enc.eval(), head.eval()
    return enc, head, last, round(time.time() - t0, 1)


@torch.no_grad()
def logits(enc, head, wins):
    """秩归一化后编码器分数（变长窗口按长度分批）。"""
    out = np.empty(len(wins))
    groups = defaultdict(list)
    for i, w in enumerate(wins):
        groups[len(w)].append(i)
    for _, idxs in groups.items():
        X = torch.tensor(gaussianize_batch(
            np.array([wins[i] for i in idxs])), dtype=torch.float32)
        out[idxs] = head(enc(X)).numpy()
    return out


# ---------------------------------------------------------------------------
# 七种推断货币
# ---------------------------------------------------------------------------
def orbit_decisions(s_x, s_S, alpha=ALPHA):
    """轨道范式四种货币。返回认证布尔（W 大 / p 小 = 认证为真）。"""
    m = len(s_S)
    lse = logsumexp(s_S)
    dec = {}
    dec["e"] = np.log(m) + s_x - lse >= np.log(1.0 / alpha)
    dec["e_inc"] = (np.log(m + 1) + s_x
                    - np.logaddexp(s_x, lse) >= np.log(1.0 / alpha))
    b = int(np.sum(s_S >= s_x))
    dec["midp_o"] = (b + 0.5) / m <= alpha
    dec["strict_o"] = (b + 1) / (m + 1) <= alpha
    return dec


def orbit_mde(s_S, alpha=ALPHA):
    """满足各判据所需的 s(x) 间隙（相对轨道均值，logit 单位）；盲区 inf。"""
    m = len(s_S)
    lse = logsumexp(s_S)
    mu = float(np.mean(s_S))
    out = {"e": np.log(1.0 / alpha) - np.log(m) + lse - mu}
    denom = alpha * (m + 1) - 1.0
    out["e_inc"] = lse - np.log(denom) - mu if denom > 0 else np.inf
    srt = np.sort(s_S)[::-1]
    for key, bstar in (("midp_o", np.floor(alpha * m - 0.5)),
                       ("strict_o", np.floor(alpha * (m + 1) - 1))):
        out[key] = srt[int(bstar)] - mu if bstar >= 0 else np.inf
    return out


def realref_decisions(d_x, d_ref, alpha=ALPHA):
    """真实参考范式三种货币。返回标记伪造布尔（d 大 = 可疑），通用键。"""
    n = len(d_ref)
    b = int(np.sum(d_ref >= d_x))
    dec = {}
    dec["midp"] = (b + 0.5) / n <= alpha
    dec["strict"] = (b + 1) / (n + 1) <= alpha
    ds = np.sort(d_ref)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    dec["conf"] = d_x > ds[min(k, n) - 1]
    return dec


def realref_mde(d_ref, alpha=ALPHA):
    n = len(d_ref)
    mu = float(np.mean(d_ref))
    out = {}
    srt = np.sort(d_ref)[::-1]
    for key, bstar in (("midp", np.floor(alpha * n - 0.5)),
                       ("strict", np.floor(alpha * (n + 1) - 1))):
        out[key] = srt[int(bstar)] - mu if bstar >= 0 else np.inf
    ds = np.sort(d_ref)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    out["conf"] = ds[min(k, n) - 1] - mu
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, (c - h) / d), 4), round(min(1.0, (c + h) / d), 4))


def mahal_fit(anchors, full=True):
    A = np.array([extract_features(w, full=full) for w in anchors])
    A = A[np.isfinite(A).all(axis=1)]
    mu = A.mean(axis=0)
    inv = np.linalg.pinv(np.cov(A, rowvar=False)
                         + 1e-6 * np.eye(A.shape[1]))
    return mu, inv


def mahal_d(wins, mu, inv):
    out = np.full(len(wins), np.nan)
    for i, w in enumerate(wins):
        f = extract_features(w, full=True)
        if np.isfinite(f).all():
            dev = f - mu
            out[i] = float(np.sqrt(dev @ inv @ dev))
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    cats = ["real", "N1", "N3", "N4", "N5", "null"]
    # agg[方法][n_ref][类别] = 认证(轨道)/标记(真实参考)布尔列表
    agg = {m: {nr: {c: [] for c in cats} for nr in N_REF_GRID}
           for m in ALL_METHODS}
    mde_store = {m: {nr: [] for nr in N_REF_GRID} for m in ALL_METHODS}
    per_asset = {}

    for ai, (asset, r) in enumerate(returns.items()):
        t_asset = time.time()
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW]
                for i in range(0, len(r) - WINDOW + 1, STEP)]
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        train_end = sp["cal"][0] * STEP
        pool_wins = wins[:sp["test"][0]]

        enc, head, info, t_train = train_a0(r, train_end, SEED + ai * 1000)

        # 候选：真实 + N1/N3/N4/N5（种子与 E2 逐位一致）
        cands = [("real", w) for w in test_wins]
        for fam, cat in FAM_E3.items():
            for wi, w in enumerate(test_wins):
                cands.append((cat, forge(w, fam, seed=1000 + wi * 10
                                         + FAM_IDX[fam])))
        s_cand = logits(enc, head, [w for _, w in cands])
        s_pool = logits(enc, head, pool_wins)
        d_pool = -s_pool

        pa = {"n_test": len(test_wins), "pool_size": len(pool_wins),
              "train": info, "t_train_s": t_train,
              "n_ref_eff": {}, "curves": {m: {} for m in ALL_METHODS}}

        # ---- 轨道范式：主测试 + 零假设成员水平模拟 ----
        for ci, (fam, w) in enumerate(cands):
            for ni, nr in enumerate(N_REF_GRID):
                rng = np.random.default_rng(
                    700000 + ai * 100000 + ci * 100 + ni * 7)
                surrs, _ = generate_orbit(w, nr, names=ORBIT_NAMES, rng=rng)
                s_S = logits(enc, head, list(surrs))
                dec = orbit_decisions(s_cand[ci], s_S)
                for m in ORBIT_METHODS:
                    agg[m][nr][fam].append(bool(dec[m]))
                for m, g in orbit_mde(s_S).items():
                    mde_store[m][nr].append(float(g))

        rng_null = np.random.default_rng(800000 + ai * 10000)
        for wi, w in enumerate(test_wins):
            orbit_a, _ = generate_orbit(w, 40, names=ORBIT_NAMES,
                                        rng=np.random.default_rng(
                                            810000 + ai * 10000 + wi))
            members = rng_null.choice(40, size=2, replace=False)
            for mi in members:
                c = orbit_a[mi]
                s_c = logits(enc, head, [c])[0]
                for ni, nr in enumerate(N_REF_GRID):
                    rng = np.random.default_rng(
                        820000 + ai * 100000 + wi * 1000 + int(mi) * 100
                        + ni * 7)
                    surrs, _ = generate_orbit(c, nr, names=ORBIT_NAMES,
                                              rng=rng)
                    s_S = logits(enc, head, list(surrs))
                    dec = orbit_decisions(s_c, s_S)
                    for m in ORBIT_METHODS:
                        agg[m][nr]["null"].append(bool(dec[m]))

        # ---- 真实参考范式：参考集按 (资产, n_ref) 抽一次（部署语义），
        #      编码器分数与现稿马氏引擎共用同一参考窗集合 ----
        anchors = wins[:sp["n_anchor"]]
        mu, inv = mahal_fit(anchors)
        d_pool_m = mahal_d(pool_wins, mu, inv)
        d_cand_m = mahal_d([w for _, w in cands], mu, inv)
        pa["mah_nan"] = int(np.sum(~np.isfinite(d_cand_m)))

        for ni, nr in enumerate(N_REF_GRID):
            rng = np.random.default_rng(600000 + ai * 100 + ni)
            n_eff = min(nr, len(pool_wins))
            refs = rng.choice(len(pool_wins), size=n_eff, replace=False)
            pa["n_ref_eff"][nr] = int(n_eff)

            d_ref = d_pool[refs]
            for gk, m in (("midp", "midp_r"), ("strict", "strict_r"),
                          ("conf", "conf")):
                mde_store[m][nr].append(
                    float(realref_mde(d_ref)[gk]))
            for ci, (fam, _) in enumerate(cands):
                dec = realref_decisions(-s_cand[ci], d_ref)
                for gk, m in (("midp", "midp_r"), ("strict", "strict_r"),
                              ("conf", "conf")):
                    agg[m][nr][fam].append(bool(dec[gk]))

            drm = d_pool_m[refs]
            drm = drm[np.isfinite(drm)]
            if len(drm) >= 3:
                mde_m = realref_mde(drm)
                for gk, m in (("midp", "midp_mah"), ("strict", "strict_mah"),
                              ("conf", "conf_mah")):
                    mde_store[m][nr].append(float(mde_m[gk]))
                for ci, (fam, _) in enumerate(cands):
                    if not np.isfinite(d_cand_m[ci]):
                        continue
                    dec = realref_decisions(d_cand_m[ci], drm)
                    for gk, m in (("midp", "midp_mah"),
                                  ("strict", "strict_mah"),
                                  ("conf", "conf_mah")):
                        agg[m][nr][fam].append(bool(dec[gk]))

        for m in ALL_METHODS:
            for nr in N_REF_GRID:
                cert = agg[m][nr]["real"]
                f345 = (agg[m][nr]["N3"] + agg[m][nr]["N4"]
                        + agg[m][nr]["N5"])
                cr = round(float(np.mean(cert)), 3) if cert else None
                if f345:
                    if m in ORBIT_METHODS:
                        d3 = round(float(np.mean([not v for v in f345])), 3)
                    else:
                        d3 = round(float(np.mean(f345)), 3)
                else:
                    d3 = None
                pa["curves"][m][nr] = {"certify_real": cr, "detect_N345": d3}
        per_asset[asset] = pa
        print(f"[{asset}] test={len(test_wins)} pool={len(pool_wins)} "
              f"train={t_train}s loss={info['loss']} "
              f"total={time.time() - t_asset:.0f}s", flush=True)

    # ---- 池化汇总 ----
    def rate(m, nr, cs, invert=False):
        vals = []
        for c in cs:
            vals.extend(agg[m][nr][c])
        if not vals:
            return None, (float("nan"), float("nan"))
        n = len(vals)
        k = sum(1 for v in vals if (not v if invert else v))
        lo, hi = wilson(k, n)
        return round(k / n, 4), (lo, hi)

    pooled = {}
    for m in ALL_METHODS:
        pooled[m] = {}
        for nr in N_REF_GRID:
            n345 = ["N3", "N4", "N5"]
            cert_real, ci_cr = rate(m, nr, ["real"])
            det345, ci_d3 = rate(m, nr, n345,
                                 invert=m in ORBIT_METHODS)
            detn1, ci_d1 = rate(m, nr, ["N1"], invert=m in ORBIT_METHODS)
            flag_real = (round(1 - cert_real, 4)
                         if cert_real is not None and m in ORBIT_METHODS
                         else rate(m, nr, ["real"])[0])
            null_lv = rate(m, nr, ["null"])[0] if m in ORBIT_METHODS else None
            md = mde_store[m][nr]
            md_f = [x for x in md if np.isfinite(x)]
            pooled[m][nr] = {
                "certify_real": cert_real, "ci_certify_real": ci_cr,
                "detect_N345": det345, "ci_detect_N345": ci_d3,
                "detect_N1": detn1, "ci_detect_N1": ci_d1,
                "flag_real": flag_real,
                "null_level": null_lv,
                "mde_median": round(float(np.median(md_f)), 3)
                if md_f else None,
                "mde_blind": bool(any(not np.isfinite(x) for x in md)),
                "n_null": len(agg[m][nr]["null"]),
            }

    h3a = all(pooled["e"][nr]["null_level"] is not None
              and pooled["e"][nr]["null_level"] <= ALPHA + 0.02
              for nr in (3, 5, 10))

    def _det(m, nr):
        v = pooled[m][nr]["detect_N345"]
        return -1.0 if v is None else v

    # steel-man：对比编码器分数版与现稿马氏引擎版 midrank 的较强者
    h3b = all(_det("e", nr) >= max(_det("midp_r", nr), _det("midp_mah", nr))
              for nr in (3, 5, 10))
    h3b_detail = {str(nr): {"e": _det("e", nr),
                            "midp_r": _det("midp_r", nr),
                            "midp_mah": _det("midp_mah", nr)}
                  for nr in (3, 5, 10)}
    currency = all(pooled["e"][nr]["certify_real"]
                   >= pooled["midp_o"][nr]["certify_real"]
                   for nr in (3, 5, 10))
    verdict = {
        "H3a_level_valid_small_n": bool(h3a),
        "H3b_power_ge_midrank_steelman": bool(h3b),
        "H3b_detail": h3b_detail,
        "currency_check_e_ge_midp_orbit": bool(currency),
        "H3_pass": bool(h3a and h3b),
    }

    result = {
        "experiment": "E3_small_reference_power",
        "alpha": ALPHA, "n_ref_grid": N_REF_GRID,
        "methods": METHOD_DESC,
        "protocol": {
            "encoder": "A0 (orbit negatives, five operators, rank norm), "
                       "identical to E2 v2 (seed per asset)",
            "candidates": "test real windows + N1/N3/N4/N5 forgeries "
                          "(E2 seeds)",
            "null_sim": "2 random members of a 40-orbit per real window, "
                        "each with fresh n_ref orbit",
            "real_ref_pool": "all real windows before test layer",
            "note_N1": "N1 is multiplicative perturbation (not permutation): "
                       "retains real dependence, a weak forgery diagnostic",
        },
        "pooled": pooled,
        "per_asset": {
            a: {"n_test": v["n_test"], "pool_size": v["pool_size"],
                "n_ref_eff": v["n_ref_eff"], "train": v["train"],
                "mah_nan_candidates": v.get("mah_nan", 0)}
            for a, v in per_asset.items()
        },
        "verdict": verdict,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e3_power.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    # ---- 控制台摘要 ----
    print("\n=== E3 池化功效表（列: n_ref = 3/5/10/19/40）===")
    hdr = f"{'方法':<22}{'指标':<14}" + "".join(f"{nr:>9}" for nr in N_REF_GRID)
    print(hdr)
    for m in ALL_METHODS:
        for key, label in (("detect_N345", "检出N3/4/5"),
                           ("certify_real", "认证真实窗"),
                           ("flag_real", "真实窗误标"),
                           ("null_level", "零假设认证率"),
                           ("mde_median", "MDE(logit)")):
            vals = []
            for nr in N_REF_GRID:
                v = pooled[m][nr][key]
                vals.append("-" if v is None else
                            (f"{v:.3f}" if isinstance(v, float) else str(v)))
            if all(v == "-" for v in vals):
                continue
            print(f"{METHOD_DESC[m]:<22}{label:<14}"
                  + "".join(f"{v:>9}" for v in vals))
    print("\n判定:", json.dumps(verdict, ensure_ascii=False))

    # ---- 图 4 ----
    make_figure(pooled)
    print(f"\n输出: {os.path.join(RESULTS_DIR, 'e3_power.json')}")


# ---------------------------------------------------------------------------
# 图 4：小参考域功效曲线
# ---------------------------------------------------------------------------
def make_figure(pooled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.log2(np.array(N_REF_GRID, dtype=float))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    C = {"e": "#2F5D7C", "e_inc": "#7FA6BF", "midp_o": "#B0895E",
         "strict_o": "#D4B483", "midp_r": "#A9745E", "strict_r": "#C99B8F",
         "conf": "#6E8B6E", "midp_mah": "#8C5A4E", "strict_mah": "#B5837A",
         "conf_mah": "#4E6E5A"}

    def curve(ax, m, key, **kw):
        ys = [pooled[m][nr][key] for nr in N_REF_GRID]
        if all(y is None for y in ys):
            return
        ax.plot(x, [0.0 if y is None else y for y in ys], "-o",
                color=C[m], ms=3.5, lw=1.5, **kw)

    ax = axes[0]
    for m in ("e", "midp_r", "midp_mah", "strict_r", "conf_mah"):
        ls = "-" if m in ("e", "midp_mah") else "--"
        curve(ax, m, "detect_N345", label=METHOD_DESC[m], ls=ls)
    ax.axvline(np.log2(10), color="#999999", lw=0.7, ls=":")
    ax.axvline(np.log2(19), color="#999999", lw=0.7, ls=":")
    ax.text(np.log2(10) - 0.08, 0.06, "mid-p\n盲区", fontsize=6.5,
            color="#777777", ha="right")
    ax.text(np.log2(19) - 0.08, 0.30, "严格置换/保形\n格点盲区", fontsize=6.5,
            color="#777777", ha="right")
    ax.set_title("(a) N3/N4/N5 检出功效", fontsize=9)
    ax.set_ylim(-0.03, 1.05)

    ax = axes[1]
    for m in ("e", "e_inc", "midp_o", "strict_o"):
        curve(ax, m, "certify_real", label=METHOD_DESC[m],
              ls="--" if m == "e_inc" else "-")
    for m in ("e", "midp_o"):
        ys = [pooled[m][nr]["null_level"] for nr in N_REF_GRID]
        ax.plot(x, [0.0 if y is None else y for y in ys], "-s", ms=2.5,
                lw=1.0, color=C[m], alpha=0.55,
                label=f"{METHOD_DESC[m]}·零假设水平")
    ax.axhline(ALPHA, color="#A9745E", lw=0.9, ls="--")
    ax.text(0.4, ALPHA + 0.012, "α=0.05", fontsize=7, color="#A9745E")
    ax.set_title("(b) 真实窗认证功效与零假设水平", fontsize=9)
    ax.set_ylim(-0.03, 1.05)

    ax = axes[2]
    for m in ("e", "midp_o", "midp_r", "midp_mah", "conf", "conf_mah"):
        curve(ax, m, "flag_real", label=METHOD_DESC[m],
              ls="--" if m in ("strict_r", "conf_mah") else "-")
    ax.axhline(ALPHA, color="#A9745E", lw=0.9, ls="--")
    ax.text(0.4, ALPHA + 0.015, "名义 α", fontsize=7, color="#A9745E")
    ax.set_title("(c) 真实窗误标率（检测器口径 FPR）", fontsize=9)
    ax.set_ylim(-0.03, 1.05)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in N_REF_GRID])
        ax.set_xlabel("参考域规模 n_ref", fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("rate", fontsize=8)
    for ax in (axes[0], axes[1], axes[2]):
        ax.legend(fontsize=6.2, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "fig_e3_power.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
