"""E2 追加敌手族 N6：真 GARCH(1,1) 合成序列。

问题：五个伪造族中 N2/N5 与训练轨道/算子族高度相关，MINT 的优势
是否只是"对训练空集内的伪造敏感"？N6 完全在五算子并集之外：
GARCH(1,1) 参数从各资产拟合层真实估计，条件方差递归全新模拟，
非任何算子的轨道成员。

对照组（无需训练，先行验证协议）：
  A5b  古典 8 维马氏
  LAonly 有向杠杆不对称单变量
主组（重训，协议同 e2_main）：
  A0   完整 MINT（轨道对比 + 秩归一化 + e 值）

产出：results/e2_n6_garch.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

MINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSCT_ROOT = os.path.join(MINT_ROOT, "shared_infra", "fractal_consistency")
for p in (MINT_ROOT, FSCT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch  # noqa: E402

from features import extract_features  # noqa: E402
from features.phase_ext import leverage_asym  # noqa: E402
from mint.model import (  # noqa: E402
    Encoder, ScoreHead, bce_loss, gaussianize_batch, info_nce,
)
from mint.operators import ORBIT_NAMES, generate_orbit  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

WINDOW, STEP = 1000, 50
SPLIT_SPEC = {
    "equity":  {"n_fit": 15, "cal": (34, 43), "test": (63, 99), "cap": 8},
    "bond":    {"n_fit": 15, "cal": (34, 43), "test": (63, 94), "cap": 8},
    "fx":      {"n_fit": 11, "cal": (31, 36), "test": (37, 42), "cap": 6},
    "gold":    {"n_fit": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
    "copper":  {"n_fit": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
}
ALPHA = 0.05
TRAIN_LEN, N_SLICES = 700, 64
POOL_SLICES, POOL_PER, K_TRAIN = 150, 20, 12
K_EVAL = 32
EPOCHS = 250
TAU, LR, CROP = 0.1, 1e-3, 620
SEED = 20260820
RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
OUT_JSON = os.path.join(MINT_ROOT, "results", "e2_n6_garch.json")


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


def fit_garch(r: np.ndarray):
    """MLE 拟合 GARCH(1,1)，返回 (w, a, b)。"""
    r = np.asarray(r, dtype=float)
    r = r - r.mean()
    n = len(r)
    init_var = np.var(r)

    def nll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or (a + b) >= 1.0 or (a + b) <= 0:
            return 1e12
        s2 = np.empty(n)
        s2[0] = init_var
        for t in range(1, n):
            s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
        eps = 1e-12
        return float(np.sum(np.log(np.maximum(s2[1:], eps))
                            + r[1:] ** 2 / np.maximum(s2[1:], eps)))

    best, best_val = None, np.inf
    for a0, b0 in [(0.05, 0.90), (0.08, 0.85), (0.10, 0.80)]:
        try:
            res = minimize(nll, x0=np.array([init_var * 0.05, a0, b0]),
                           method="L-BFGS-B",
                           bounds=[(1e-12, None), (0.0, 1.0), (0.0, 1.0)],
                           options={"maxiter": 300})
            if res.fun < best_val:
                best_val, best = res.fun, res.x
        except Exception:
            continue
    return tuple(map(float, best)) if best is not None else (None, None, None)


def garch_simulate(w, a, b, n, sigma2_0, rng):
    """GARCH(1,1) 条件方差递归全新模拟（学生 t 创新以匹配厚尾）。"""
    s2 = np.empty(n)
    s2[0] = sigma2_0
    df = 5.0
    r = np.empty(n)
    r[0] = np.sqrt(sigma2_0) * rng.standard_t(df) / np.sqrt(df / (df - 2))
    for t in range(1, n):
        s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * rng.standard_t(df) / np.sqrt(df / (df - 2))
    return r


def build_n6(fit_wins, test_wins, seed):
    """N6：以各资产拟合层估得的 GARCH 参数生成同长度合成窗，
    标准差与真实测试窗对齐（消除平凡尺度线索）。"""
    rng = np.random.default_rng(seed)
    allfit = np.concatenate(fit_wins)
    w, a, b = fit_garch(allfit)
    out = []
    for i in range(len(test_wins)):
        x = garch_simulate(w, a, b, WINDOW, np.var(allfit), rng)
        x = x - x.mean()
        x = x * (np.std(test_wins[i]) / max(np.std(x), 1e-12))
        out.append(x)
    return out, (w, a, b)


# ---- 复用 e2_main 的训练与评分（A0 配置，逐位一致） ----
def crop_noise(w, rng):
    start = int(rng.integers(0, len(w) - CROP + 1))
    out = w[start:start + CROP].copy()
    return out + 0.02 * out.std() * rng.standard_normal(len(out))


def prep(X, rank=True):
    if rank:
        return gaussianize_batch(X)
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=1, keepdims=True)) / (
        X.std(axis=1, keepdims=True) + 1e-12)


def build_pool(r, train_end, rng, n_slices, per):
    starts = rng.integers(0, train_end - TRAIN_LEN + 1, size=n_slices)
    rows = []
    for s in starts:
        w = r[int(s):int(s) + TRAIN_LEN]
        surrs, _ = generate_orbit(w, per, names=ORBIT_NAMES, rng=rng)
        rows.append(surrs)
    return np.vstack(rows)


def train_model(r, train_end, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, head = Encoder(), ScoreHead()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    pool = build_pool(r, train_end, np.random.default_rng(seed + 77),
                      POOL_SLICES, POOL_PER)
    pool = prep(pool, True)
    last = {}
    for epoch in range(EPOCHS):
        starts = rng.integers(0, train_end - TRAIN_LEN + 1, size=N_SLICES)
        wins = [r[s:s + TRAIN_LEN] for s in starts]
        X = torch.tensor(prep(np.array(wins), True), dtype=torch.float32)
        X_pos = torch.tensor(
            prep(np.array([crop_noise(w, rng) for w in wins]), True),
            dtype=torch.float32)
        need = N_SLICES * K_TRAIN
        idx = rng.choice(len(pool), size=need, replace=len(pool) < need)
        N = torch.tensor(pool[idx], dtype=torch.float32)
        opt.zero_grad()
        z, z_pos, z_neg = enc(X), enc(X_pos), enc(N)
        loss_nce = info_nce(z, z_pos, z_neg, tau=TAU)
        s_pos, s_neg = head(z), head(z_neg)
        loss_bce = bce_loss(s_pos, s_neg, pos_weight=float(K_TRAIN))
        loss = loss_nce + loss_bce
        loss.backward()
        opt.step()
        sched.step()
        last = {"loss": round(loss.item(), 4)}
    enc.eval(), head.eval()
    return enc, head, last


@torch.no_grad()
def logits_var(enc, head, wins, rank=True):
    out = np.empty(len(wins))
    from collections import defaultdict
    groups = defaultdict(list)
    for i, w in enumerate(wins):
        groups[len(w)].append(i)
    for _, idxs in groups.items():
        X = torch.tensor(prep(np.array([wins[i] for i in idxs]), rank),
                         dtype=torch.float32)
        out[idxs] = head(enc(X)).numpy()
    return out


def e_values(enc, head, wins, k, seed):
    W = np.empty(len(wins))
    for i, w in enumerate(wins):
        surrs, _ = generate_orbit(w, k, rng=np.random.default_rng(seed + i))
        s = logits_var(enc, head, [w] + list(surrs))
        m = float(np.max(s))
        lse = m + float(np.log(np.sum(np.exp(s - m))))
        W[i] = float((k + 1) * np.exp(float(s[0]) - lse))
    return W


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = rankdata(np.concatenate([neg, pos]))
    rp = r[len(neg):]
    return float((rp.sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def wilson(k, n, zz=1.96):
    p = k / n
    d = 1 + zz * zz / n
    c = p + zz * zz / (2 * n)
    h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n))
    return (float((c - h) / d), float((c + h) / d))


def main():
    t0 = time.time()
    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    res = {"seed": SEED, "families": "N6=GARCH(1,1) t(5) simulated, "
           "params fit on calibration-free fit layer, std aligned",
           "per_asset": {}, "methods": {}}
    anom = {"A0": {"cal": [], "real": [], "n6": []},
            "A5b": {"cal": [], "real": [], "n6": []},
            "LAonly": {"cal": [], "real": [], "n6": []}}
    Wraw = {"real": [], "n6": []}

    for ai, (asset, r) in enumerate(returns.items()):
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        fit_wins = wins[:sp["n_fit"]]
        cal_wins = span_windows(wins, *sp["cal"])
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        train_end = sp["cal"][0] * STEP
        n6, (w, a, b) = build_n6(fit_wins, test_wins, seed=7000 + ai)
        print(f"[{asset}] fit{len(fit_wins)} cal{len(cal_wins)} "
              f"test{len(test_wins)} | GARCH w={w:.2e} a={a:.3f} b={b:.3f}",
              flush=True)
        res["per_asset"][asset] = {"garch": [w, a, b],
                                   "n_test": len(test_wins)}

        # A0 训练 + e 值评分
        enc, head, loss = train_model(r, train_end,
                                      SEED + ai * 1000)
        print(f"    A0 trained, loss={loss['loss']:.3f}", flush=True)
        base = 800000 + ai * 10000
        w_cal = e_values(enc, head, cal_wins, K_EVAL, base)
        w_real = e_values(enc, head, test_wins, K_EVAL, base + 100)
        w_n6 = e_values(enc, head, n6, K_EVAL, base + 300)
        anom["A0"]["cal"].append(-w_cal)
        anom["A0"]["real"].append(-w_real)
        anom["A0"]["n6"].append(-w_n6)
        Wraw["real"].append(w_real)
        Wraw["n6"].append(w_n6)

        # A5b 古典 8 维马氏
        A = np.array([extract_features(x) for x in fit_wins])
        mu = np.nanmean(A, axis=0)
        inv = np.linalg.pinv(np.nan_to_num(np.cov(A, rowvar=False)))

        def msc(ws):
            B = np.array([extract_features(x) for x in ws])
            D = B - mu
            return np.einsum("ij,jk,ik->i", D, inv, D)

        anom["A5b"]["cal"].append(msc(cal_wins))
        anom["A5b"]["real"].append(msc(test_wins))
        anom["A5b"]["n6"].append(msc(n6))

        # LAonly 有向单变量
        anom["LAonly"]["cal"].append(
            np.array([leverage_asym(x) for x in cal_wins]))
        anom["LAonly"]["real"].append(
            np.array([leverage_asym(x) for x in test_wins]))
        anom["LAonly"]["n6"].append(np.array([leverage_asym(x) for x in n6]))

    for tag in anom:
        real = np.concatenate(anom[tag]["real"])
        n6 = np.concatenate(anom[tag]["n6"])
        # 逐资产 cal q95 → 池化判定
        flags_real, flags_n6 = [], []
        for ai, asset in enumerate(returns):
            cal = anom[tag]["cal"][ai]
            thr = np.quantile(cal[np.isfinite(cal)], 1 - ALPHA)
            flags_real.append(np.asarray(anom[tag]["real"][ai]) > thr)
            flags_n6.append(np.asarray(anom[tag]["n6"][ai]) > thr)
        fr, fn = np.concatenate(flags_real), np.concatenate(flags_n6)
        k_r, k_n = int(fr.sum()), int(fn.sum())
        res["methods"][tag] = {
            "n6_recall": float(fn.mean()),
            "n6_recall_wilson": list(wilson(k_n, len(fn))),
            "fpr": float(fr.mean()),
            "fpr_wilson": list(wilson(k_r, len(fr))),
            "n6_auc": auc(n6, real),
        }
        print(f"{tag:7s} N6 recall={fn.mean():.3f} "
              f"(Wilson {wilson(k_n, len(fn))[0]:.3f}-"
              f"{wilson(k_n, len(fn))[1]:.3f})  FPR={fr.mean():.3f}  "
              f"N6 AUC={auc(n6, real):.3f}")

    # e 值认证：N6 是否被认证为真（不应）
    Wr = np.concatenate(Wraw["real"])
    Wn = np.concatenate(Wraw["n6"])
    res["e_rule"] = {
        "real_certify_rate": float(np.mean(Wr >= 1 / ALPHA)),
        "n6_certify_rate": float(np.mean(Wn >= 1 / ALPHA)),
        "n6_flag_rate_echannel": float(np.mean(Wn < 1 / ALPHA)),
    }
    print(f"e-rule: real certify {res['e_rule']['real_certify_rate']:.3f} | "
          f"N6 certify {res['e_rule']['n6_certify_rate']:.3f} "
          f"(flag {res['e_rule']['n6_flag_rate_echannel']:.3f})")

    res["elapsed_s"] = time.time() - t0
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"saved: {OUT_JSON}  ({res['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
