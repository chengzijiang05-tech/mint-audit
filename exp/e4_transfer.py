"""E4 跨市场转移（H4/T4 关卡）：冻结编码器 + 本地轨道校准 vs 阈值搬运。

设计依据：整体研究方案 §六 E4 + §9 H4；推导报告 T4（有效性无条件
转移、功效条件转移）。

三种策略（同一 A0 编码器族，策略是唯一变量）：
  S1 阈值搬运      源市场编码器 + 源市场 cal 层 q95 冻结阈值（直接分
     数 anom=-s）→ 目标市场测试层。R6-B 古典引擎版的 MINT 复刻
     （古典参照 CN→US 0.80 / CN→HK 0.75 直接引用 R6 存档）。
  S2 冻结+本地轨道  源市场编码器 + 目标窗口自身 K=32 轨道（五算子，
     逐窗本地生成）→ 双向 e 值规则：
        W_cert(x) = (K+1)·e^{s(x)} / Σ e^{s(y_i)}    认证为真（E2 v2 同式）
        W_flag(x) = (K+1)·e^{-s(x)} / Σ e^{-s(y_i)}  判为伪造
     阈值 1/α=20；两者互斥（softmax 权重不可能同时 >20/33），
     其间为弃权区（三区语义：认证 / 检出 / 弃权）。W_flag 的零假设
     期望为 1，Markov 水平 ≤ α 对任意市场成立，T4(i) 的可检验预言。
  S3 全本地重训    目标市场自训编码器 + 同轨道规则 = 轨道矩阵对角线。

转移矩阵 3×3：源 {CN(equity), US(spy), HK(hsi)} × 目标 {CN, US, HK}。
CN→US→HK 双向全覆盖；另加 CN 五资产编码器系综 → US/HK（系统级
转移主张，E2 v2 各资产编码器逐位复现）。

协议（R6 存档逐位对齐，保证与 R6-B 古典矩阵可比）：
  CN  fit equity[0,14]  cal equity(34,43)=10  test equity(63,99) cap10
  US  fit spy[0,14]     cal spy(34,43)=10     test spy(63,99) cap10
  HK  fit hsi[0,11]     cal hscei(12,19)=8    test hsi+hscei(32,43) cap10×2
  伪造族 N1–N5，种子 2000+wi*10+fi（R6 系列）
  MINT 训练跨度：CN/US = cal 起点前全部历史（train_end=1700，E2 协议）；
  HK 数据不足 700 切片，放宽为测试层前全部历史（train_end=1550，
  HSI+HSCEI 两序列池化切片；与 HSCEI cal 层日历重叠跨序列，披露，
  MINT 路径不使用 cal 层，仅 S1 HK→HK 阈值受影响）。

T4(ii) δ 实测：δ̂(E,M) = TV(编码器分数空间中，市场 M 真实窗分数分布,
其轨道成员分数分布)（8 箱直方图估计；另报 AUC 稳健口径）。δ̂ 大 =
真实结构与零假设族可分 → 功效保留。报告 δ̂–功效关系的 Spearman
相关与逐单元散点（图 5d）。

H4 判定（预注册）：S2 的 CN→US 与 CN→HK 真实窗误标率 FPR ≤ 0.15
→ PASS；任一 > 0.30 → FAIL。另报零假设成员 W_flag 率（T4(i) 结构
有效性：任何源×目标组合应 ≤ α+容差）。

产出：results/e4_transfer.json + figures/fig_e4_transfer.pdf
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
from scipy import stats  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from bench import HALLUCINATION_BUILDERS, n5_phase_forge  # noqa: E402
from mint.model import (  # noqa: E402
    Encoder,
    ScoreHead,
    bce_loss,
    gaussianize_batch,
    info_nce,
)
from mint.operators import ORBIT_NAMES, generate_orbit  # noqa: E402

WINDOW, STEP = 1000, 50
FAMILIES = ["N1数值置换", "N2时间倒置", "N3标度破坏", "N4跨域嫁接",
            "N5相位伪造"]
FAM_SHORT = ["N1", "N2", "N3", "N4", "N5"]
ALPHA = 0.05
K_EVAL = 32
E_VAL = 1.0 / ALPHA

TRAIN_LEN, N_SLICES = 700, 64
POOL_SLICES, POOL_PER, K_TRAIN = 150, 20, 12
EPOCHS = int(os.environ.get("MINT_E4_EPOCHS", "250"))
QUICK = os.environ.get("MINT_E4_QUICK", "") == "1"
TAU, LR, CROP = 0.1, 1e-3, 620
SEED = 20260820

MARKETS = ["CN", "US", "HK"]
LAYERS = {
    "CN": {"train": ["equity"], "train_end": 1700,
           "cal": ("equity", 34, 43),
           "test": [("equity", 63, 99, 10)]},
    "US": {"train": ["us_spy"], "train_end": 1700,
           "cal": ("us_spy", 34, 43),
           "test": [("us_spy", 63, 99, 10)]},
    "HK": {"train": ["hk_hsi", "hk_hscei"], "train_end": 1550,
           "cal": ("hk_hscei", 12, 19),
           "test": [("hk_hsi", 32, 43, 10), ("hk_hscei", 32, 43, 10)]},
}
MKT_SEED = {"CN": SEED, "US": SEED + 100000, "HK": SEED + 200000}
R6_CLASSICAL = {  # R6-B 存档（古典 FSCT 引擎，同协议）论文对照
    "CN->CN": {"fpr": 0.10, "recall": 0.66},
    "CN->US": {"fpr": 0.80, "recall": 0.92},
    "CN->HK": {"fpr": 0.75, "recall": 0.91},
    "US->CN": {"fpr": 0.00, "recall": 0.46},
    "US->US": {"fpr": 0.00, "recall": 0.44},
    "US->HK": {"fpr": 0.00, "recall": 0.41},
    "HK->CN": {"fpr": 0.20, "recall": 0.62},
    "HK->US": {"fpr": 0.60, "recall": 0.76},
    "HK->HK": {"fpr": 0.15, "recall": 0.57},
}

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
R6_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "r6_returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
FIGURES_DIR = os.path.join(MINT_ROOT, "figures")


# ---------------------------------------------------------------------------
# 数据与伪造
# ---------------------------------------------------------------------------
def load_all_returns() -> dict:
    out = {}
    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        out.update({k: d[k] for k in d.files})
    with np.load(R6_NPZ, allow_pickle=True) as d:
        out.update({k: d[k] for k in d.files})
    return out


def windows_of(r):
    return [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]


def span(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


def forge(x, family, seed):
    if family == "N5相位伪造":
        return n5_phase_forge(x, seed=seed)
    return HALLUCINATION_BUILDERS[family](x, seed=seed)


# ---------------------------------------------------------------------------
# A0 训练（E2 v2 协议；支持多序列池化切片，HK 用）
# ---------------------------------------------------------------------------
def crop_noise(w, rng):
    start = int(rng.integers(0, len(w) - CROP + 1))
    out = w[start:start + CROP].copy()
    return out + 0.02 * out.std() * rng.standard_normal(len(out))


def build_pool(series_list, train_end, rng, n_slices, per):
    rows = []
    for _ in range(n_slices):
        r = series_list[int(rng.integers(0, len(series_list)))]
        s = int(rng.integers(0, train_end - TRAIN_LEN + 1))
        w = r[s:s + TRAIN_LEN]
        surrs, _ = generate_orbit(w, per, names=ORBIT_NAMES, rng=rng)
        rows.append(surrs)
    return np.vstack(rows)


def train_a0(series_list, train_end, seed, epochs):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, head = Encoder(), ScoreHead()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pool = gaussianize_batch(
        build_pool(series_list, train_end, np.random.default_rng(seed + 77),
                   POOL_SLICES, POOL_PER))
    t0 = time.time()
    last = {}
    for _ in range(epochs):
        starts = rng.integers(0, train_end - TRAIN_LEN + 1, size=N_SLICES)
        sidx = rng.integers(0, len(series_list), size=N_SLICES)
        wins = [series_list[j][s:s + TRAIN_LEN]
                for s, j in zip(starts, sidx)]
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
    out = np.empty(len(wins))
    groups = defaultdict(list)
    for i, w in enumerate(wins):
        groups[len(w)].append(i)
    for _, idxs in groups.items():
        X = torch.tensor(gaussianize_batch(
            np.array([wins[i] for i in idxs])), dtype=torch.float32)
        out[idxs] = head(enc(X)).numpy()
    return out


def e_pair(s_x, s_S):
    """双向 e 值（包含形，x 计入分母；E2 v2 同式）。

    W_cert = (K+1)·softmax(s)_x   大 → 结构真实
    W_flag = (K+1)·softmax(-s)_x  大 → 相对自身轨道失真（判伪造）
    零假设（x 与轨道可交换）下两者期望均为 1，Markov 水平 ≤ α。"""
    m = len(s_S)
    lse_p = logsumexp(np.r_[s_x, s_S])
    lse_n = logsumexp(np.r_[-s_x, -s_S])
    return (float((m + 1) * np.exp(s_x - lse_p)),
            float((m + 1) * np.exp(-s_x - lse_n)))


def tv_dist(a, b, bins=8):
    """一维 TV 距离（共享分箱直方图估计）。"""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max()) + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    pa = np.histogram(a, bins=edges)[0] / len(a)
    pb = np.histogram(b, bins=edges)[0] / len(b)
    return float(0.5 * np.abs(pa - pb).sum())


def auc_sep(a, b):
    """AUC = P(score_real > score_orbit)（Mann-Whitney）。"""
    s = np.r_[a, b]
    ranks = stats.rankdata(s)
    u = ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2.0
    return float(u / (len(a) * len(b)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, (c - h) / d), 4), round(min(1.0, (c + h) / d), 4))


# ---------------------------------------------------------------------------
# 单元评测：S1 阈值搬运 + S2/S3 轨道规则 + δ̂ + 零假设水平模拟
# ---------------------------------------------------------------------------
def eval_cell(enc, head, tgt, seed_base):
    """tgt: {'clean': [...], 'forged': {fam: [...]}, 'cal': [...]}。"""
    clean, forged = tgt["clean"], tgt["forged"]
    s_clean = logits(enc, head, clean)
    s_fam = {f: logits(enc, head, forged[f]) for f in FAMILIES}
    s_cal = logits(enc, head, tgt["cal"]) if tgt["cal"] else None

    out = {"n_clean": len(clean),
           "n_forged": sum(len(v) for v in forged.values())}

    # ---- S1 阈值搬运（源 cal q95，直接分数 anom=-s）----
    if s_cal is not None and len(s_cal) >= 3:
        thr = float(np.quantile(-s_cal, 1 - ALPHA))
        out["S1"] = {
            "threshold": round(thr, 3),
            "fpr": round(float(np.mean(-s_clean > thr)), 4),
            "fpr_ci": wilson(int(np.sum(-s_clean > thr)), len(clean)),
            "recall": {fs: round(float(np.mean(-s_fam[f] > thr)), 4)
                       for f, fs in zip(FAMILIES, FAM_SHORT)},
        }
        rec = [np.mean(-s_fam[f] > thr) for f in FAMILIES]
        out["S1"]["macro_recall"] = round(float(np.mean(rec)), 4)
    else:
        out["S1"] = None

    # ---- S2 轨道规则（逐窗本地轨道，双向 e 值）----
    flag_r, cert_r = [], []
    fam_stats = {}
    s_real_all, s_orb_all = [], []
    for wi, w in enumerate(clean):
        surrs, _ = generate_orbit(w, K_EVAL,
                                  rng=np.random.default_rng(seed_base + wi))
        s = logits(enc, head, [w] + list(surrs))
        w_cert, w_flag = e_pair(s[0], s[1:])
        flag_r.append(w_flag >= E_VAL)
        cert_r.append(w_cert >= E_VAL)
        s_real_all.append(float(s[0]))
        s_orb_all.extend(s[1:].tolist())
    for f, fs in zip(FAMILIES, FAM_SHORT):
        fl, ce, ab = [], [], []
        for wi, w in enumerate(forged[f]):
            surrs, _ = generate_orbit(
                w, K_EVAL,
                rng=np.random.default_rng(seed_base + 1000 + wi))
            s = logits(enc, head, [w] + list(surrs))
            w_cert, w_flag = e_pair(s[0], s[1:])
            fl.append(w_flag >= E_VAL)
            ce.append(w_cert >= E_VAL)
            ab.append((w_flag < E_VAL) and (w_cert < E_VAL))
        fam_stats[fs] = {
            "flag_recall": round(float(np.mean(fl)), 4),
            "not_certified": round(float(np.mean([not c for c in ce])), 4),
            "abstain": round(float(np.mean(ab)), 4),
        }
    nf = sum(len(forged[f]) for f in FAMILIES)
    out["S2"] = {
        "fpr": round(float(np.mean(flag_r)), 4),
        "fpr_ci": wilson(int(np.sum(flag_r)), len(flag_r)),
        "certify_real": round(float(np.mean(cert_r)), 4),
        "recall_flag": fam_stats,
        "macro_flag_recall": round(float(np.mean(
            [v["flag_recall"] for v in fam_stats.values()])), 4),
        "macro_not_certified": round(float(np.mean(
            [v["not_certified"] for v in fam_stats.values()])), 4),
        "n_forged": nf,
    }

    # ---- T4(ii) δ̂：真实窗 vs 轨道成员的分数分布距离 ----
    s_real_all = np.array(s_real_all)
    s_orb_all = np.array(s_orb_all)
    out["delta"] = {
        "tv8": round(tv_dist(s_real_all, s_orb_all), 4),
        "auc": round(auc_sep(s_real_all, s_orb_all), 4),
        "mean_gap": round(float(np.mean(s_real_all)
                                - np.mean(s_orb_all)), 3),
    }

    # ---- T4(i) 零假设成员水平：轨道成员作候选，W_flag 率 ≤ α ----
    rng = np.random.default_rng(seed_base + 90000)
    null_flags = []
    for wi, w in enumerate(clean):
        orbit40, _ = generate_orbit(
            w, 40, rng=np.random.default_rng(seed_base + 91000 + wi))
        for mi in rng.choice(40, size=2, replace=False):
            c = orbit40[mi]
            surrs, _ = generate_orbit(
                c, K_EVAL,
                rng=np.random.default_rng(seed_base + 92000 + wi * 10
                                          + int(mi)))
            s = logits(enc, head, [c] + list(surrs))
            _, w_flag = e_pair(s[0], s[1:])
            null_flags.append(w_flag >= E_VAL)
    out["null_flag_rate"] = {
        "rate": round(float(np.mean(null_flags)), 4),
        "ci": wilson(int(np.sum(null_flags)), len(null_flags)),
        "n": len(null_flags),
    }
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    returns = load_all_returns()

    # ---- 三市场测试层与 cal 层（R6 协议）----
    layers = {}
    for mkt in MARKETS:
        spec = LAYERS[mkt]
        clean = []
        for sym, lo, hi, cap in spec["test"]:
            clean += span(windows_of(returns[sym]), lo, hi, cap=cap)
        if QUICK:
            clean = clean[:2]
        forged = {f: [forge(w, f, seed=2000 + wi * 10 + fi)
                      for wi, w in enumerate(clean)]
                  for fi, f in enumerate(FAMILIES)}
        cs, cl, ch = spec["cal"]
        cal = span(windows_of(returns[cs]), cl, ch)
        layers[mkt] = {"clean": clean, "forged": forged, "cal": cal}
        print(f"[{mkt}] test clean={len(clean)} "
              f"forged/族={len(clean)} cal={len(cal)}", flush=True)

    # ---- 训练三个市场编码器（CN equity 逐位复现 E2 v2）----
    encoders = {}
    for mi, mkt in enumerate(MARKETS):
        spec = LAYERS[mkt]
        series = [returns[s] for s in spec["train"]]
        enc, head, info, t = train_a0(series, spec["train_end"],
                                      MKT_SEED[mkt], EPOCHS)
        encoders[mkt] = (enc, head)
        print(f"[{mkt}] A0 train {t}s loss={info['loss']}", flush=True)

    # ---- 3×3 转移矩阵 ----
    matrix = {}
    for si, src in enumerate(MARKETS):
        enc, head = encoders[src]
        for ti, tgt_mkt in enumerate(MARKETS):
            t0 = time.time()
            seed_base = 500000 + si * 100000 + ti * 10000
            cell = eval_cell(enc, head, layers[tgt_mkt], seed_base)
            matrix[f"{src}->{tgt_mkt}"] = cell
            print(f"  {src}->{tgt_mkt}: S1_FPR="
                  f"{(cell['S1'] or {}).get('fpr', '-')} "
                  f"S2_FPR={cell['S2']['fpr']} "
                  f"null={cell['null_flag_rate']['rate']} "
                  f"δTV={cell['delta']['tv8']} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- CN 五资产系综 → US/HK（系统级转移）----
    ensemble = {}
    if not QUICK:
        cn_assets = ["equity", "bond", "fx", "gold", "copper"]
        for ai, asset in enumerate(cn_assets):
            if asset == "equity":
                enc, head = encoders["CN"]
            else:
                enc, head, _, _ = train_a0(
                    [returns[asset]], LAYERS["CN"]["train_end"],
                    SEED + ai * 1000, EPOCHS)
            for ti, tgt_mkt in enumerate(("US", "HK")):
                seed_base = 3000000 + ai * 20000 + ti * 5000
                cell = eval_cell(enc, head, layers[tgt_mkt], seed_base)
                ensemble[f"{asset}->{tgt_mkt}"] = {
                    "fpr": cell["S2"]["fpr"],
                    "certify_real": cell["S2"]["certify_real"],
                    "macro_flag_recall": cell["S2"]["macro_flag_recall"],
                    "null": cell["null_flag_rate"]["rate"],
                    "delta_tv": cell["delta"]["tv8"]}
            print(f"  ens {asset}->US/HK done", flush=True)

    # ---- 汇总与判定 ----
    def fpr_of(cell):
        return cell["S2"]["fpr"]

    h4 = {"CN->US": fpr_of(matrix["CN->US"]),
          "CN->HK": fpr_of(matrix["CN->HK"])}
    h4_pass = all(v <= 0.15 for v in h4.values())
    h4_fail = any(v > 0.30 for v in h4.values())

    null_ok = all(matrix[c]["null_flag_rate"]["rate"] <= ALPHA + 0.02
                  for c in matrix)
    deltas = [matrix[c]["delta"]["tv8"] for c in matrix]
    powers = [matrix[c]["S2"]["certify_real"] for c in matrix]
    keys = list(matrix.keys())
    rho, rho_p = stats.spearmanr(deltas, powers)
    rho_r, rho_rp = stats.spearmanr(
        deltas, [matrix[c]["S2"]["macro_flag_recall"] for c in keys])

    verdict = {
        "H4_transfer_fpr": h4,
        "H4_pass": bool(h4_pass and not h4_fail),
        "H4_fail": bool(h4_fail),
        "T4i_null_validity_all_cells": bool(null_ok),
        "T4ii_spearman_delta_certify": {"rho": round(float(rho), 3),
                                        "p": round(float(rho_p), 4)},
        "T4ii_spearman_delta_flag_recall": {"rho": round(float(rho_r), 3),
                                            "p": round(float(rho_rp), 4)},
    }

    result = {
        "experiment": "E4_cross_market_transfer",
        "alpha": ALPHA, "k_orbit": K_EVAL, "epochs": EPOCHS,
        "seed": SEED, "quick": QUICK,
        "protocol": {
            "layers": {m: {"test_n": len(layers[m]["clean"]),
                           "cal_n": len(layers[m]["cal"])}
                       for m in MARKETS},
            "forge_seeds": "2000+wi*10+fi (R6 series)",
            "train_span": {m: {"series": LAYERS[m]["train"],
                               "end_day": LAYERS[m]["train_end"]}
                           for m in MARKETS},
            "notes": [
                "HK train span = pre-test history (deviation from E2's "
                "train<cal ordering; HK data too short; MINT paths do "
                "not use the cal layer)",
                "HK cal layer (HSCEI 12-19) partially overlaps test in "
                "calendar within HSCEI (inherited R6 protocol; affects "
                "only S1 HK->HK threshold cell)",
                "R6 classical reference numbers cited from archived "
                "r6_cross_market.json (same protocol)",
                "delta estimated in encoder score space (1-D projection "
                "of invariant-law TV distance)",
            ],
        },
        "matrix": matrix,
        "ensemble_cn_five": ensemble,
        "r6_classical_reference": R6_CLASSICAL,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e4_transfer.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    # ---- 控制台摘要 ----
    print("\n=== E4 转移矩阵（S2 轨道规则 FPR / 认证率 | S1 阈值搬运 FPR）===")
    print(f"{'单元':<10}{'S1_FPR':>8}{'S2_FPR':>8}{'认证':>7}"
          f"{'检出':>7}{'弃权*':>7}{'零假设':>7}{'δTV':>7}{'AUC':>7}")
    for c in [f"{s}->{t}" for s in MARKETS for t in MARKETS]:
        e = matrix[c]
        s1 = (e["S1"] or {}).get("fpr")
        ab = 1 - e["S2"]["certify_real"] - e["S2"]["fpr"]
        print(f"{c:<10}{(s1 if s1 is not None else '-'):>8}"
              f"{e['S2']['fpr']:>8}{e['S2']['certify_real']:>7}"
              f"{e['S2']['macro_flag_recall']:>7}{ab:>7}"
              f"{e['null_flag_rate']['rate']:>7}"
              f"{e['delta']['tv8']:>7}{e['delta']['auc']:>7}")
    print("* 弃权 = 1 − 认证 − 检出（真实窗口径为 1−认证−误标）")
    print("\n判定:", json.dumps(verdict, ensure_ascii=False))
    print(f"\n输出: {os.path.join(RESULTS_DIR, 'e4_transfer.json')}")

    make_figure(matrix, ensemble, verdict)


# ---------------------------------------------------------------------------
# 图 5：E4 转移地图
# ---------------------------------------------------------------------------
def make_figure(matrix, ensemble, verdict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4))

    def heat(ax, key, title):
        M = np.full((3, 3), np.nan)
        for si, s in enumerate(MARKETS):
            for ti, t in enumerate(MARKETS):
                e = matrix[f"{s}->{t}"]
                if key == "S1":
                    v = (e["S1"] or {}).get("fpr")
                elif key == "S2":
                    v = e["S2"]["fpr"]
                elif key == "rec":
                    v = e["S2"]["macro_flag_recall"]
                else:
                    v = e["delta"]["tv8"]
                M[si, ti] = np.nan if v is None else v
        if key in ("S1", "S2"):
            ax.imshow(M, cmap="Reds", vmin=0, vmax=0.8)
            tc = "#5A2A20"
        elif key == "rec":
            ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
            tc = "#1F3A5C"
        else:
            ax.imshow(M, cmap="Purples", vmin=0, vmax=1)
            tc = "#3D2A5C"
        for si in range(3):
            for ti in range(3):
                if np.isfinite(M[si, ti]):
                    ax.text(ti, si, f"{M[si, ti]:.2f}", ha="center",
                            va="center", fontsize=9, color=tc,
                            fontweight="bold")
        ax.set_xticks(range(3), MARKETS)
        ax.set_yticks(range(3), MARKETS)
        ax.set_xlabel("目标市场", fontsize=8)
        ax.set_ylabel("源市场", fontsize=8)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=8)

    heat(axes[0], "S1", "(a) S1 阈值搬运 FPR\n(古典 R6: CN→US 0.80)")
    heat(axes[1], "S2", "(b) S2 冻结编码器+本地轨道 FPR\n(H4: CN→外 ≤0.15)")
    heat(axes[2], "rec", "(c) S2 检出功效 macro recall")
    heat(axes[3], "delta", "(d) δ̂ TV(真实, 轨道)")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "fig_e4_transfer.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
