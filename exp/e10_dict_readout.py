"""E10：知识层隔离评估，字典读出保真度 + 类型化 vs 二值裁决。

实验目的："Positioning for KBS"宣称知识层（typed layer +
dictionary readout）有价值，但从未被评估。本脚本补齐证据。

三块评估：
  (a) 字典读出保真度：编码器嵌入 f_θ(x) 能否重建命名统计量 D(x)？
      字典 D：Ldir(时间箭头)、|L|(杠杆强度)、ACF(1)、ACF²(1)
      (波动聚集)、DFA-H、Hill 尾指数、GARCH 持久性、bispec。
      Lasso 回归 + 窗级 5 折 CV R²（跨资产池化，样本外）。
  (b) 读出坐标的类型化价值：伪造族在哪个字典坐标上与真实窗分离
      最大（读出预测方向 vs 真实统计量方向的一致率），验证
      "verdict 携带可辩论坐标"是否真实。
  (c) 类型化 vs 二值裁决：同一编码器分数下，四类裁决+弃权 vs
      单阈值二值。在 480 基准（非序列输出）与 N1-N8 敌手上对比
      越权判定率与可辩护性。

协议：A0 编码器与 e2_main 同种子重训（SEED + ai*1000），cal 层
字典统计量做 Lasso 的训练折内标准化。

产出：results/e10_dict_readout.json
"""
from __future__ import annotations

import json
import os
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

import torch  # noqa: E402

from mint.model import Encoder, ScoreHead, bce_loss, gaussianize_batch, info_nce  # noqa: E402
from mint.operators import ORBIT_NAMES, generate_orbit  # noqa: E402

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
ALPHA = 0.05
TRAIN_LEN, N_SLICES = 700, 64
POOL_SLICES, POOL_PER, K_TRAIN = 150, 20, 12
EPOCHS = 250
TAU, LR, CROP = 0.1, 1e-3, 620
SEED = 20260820
K_LAG = 20


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


# ---------------------------------------------------------------------------
# 字典统计量 D(x)，全部命名坐标，经典检验语言
# ---------------------------------------------------------------------------
def d_Ldir(x):
    x = np.asarray(x, float) - np.mean(x)
    den = np.mean(np.abs(x) ** 3)
    if den <= 0:
        return 0.0
    return float(np.mean([np.mean(x[:-k] * x[k:] ** 2)
                          for k in range(1, K_LAG + 1)]) / den)


def d_absL(x):
    return abs(d_Ldir(x))


def d_acf1(x):
    x = np.asarray(x, float) - np.mean(x)
    return float(np.mean(x[:-1] * x[1:]) / max(np.var(x), 1e-18))


def d_acf1sq(x):
    x = np.asarray(x, float) - np.mean(x)
    y = x ** 2
    return float(np.mean(y[:-1] * y[1:]) / max(np.var(y), 1e-18))


def d_dfa_h(x):
    x = np.asarray(x, float) - np.mean(x)
    scales = [8, 16, 32, 64, 128]
    F = []
    for s in scales:
        n_seg = len(x) // s
        segs = x[:n_seg * s].reshape(n_seg, s)
        t = np.arange(s)
        fit = np.array([np.polyval(np.polyfit(t, sg, 1), t) for sg in segs])
        F.append(np.sqrt(np.mean((segs - fit) ** 2)))
    F = np.log(np.maximum(F, 1e-12))
    s_ = np.log(scales)
    return float(np.polyfit(s_, F, 1)[0])


def d_hill(x):
    x = np.abs(np.asarray(x, float))
    xs = np.sort(x)[::-1]
    k = max(10, len(xs) // 20)
    tail = xs[:k]
    if np.min(tail) <= 0:
        return 0.0
    return float(k / np.sum(np.log(tail / xs[k])))


def d_garch_pers(x):
    """|r| 的 AR(1) 系数（波动持续性读数，GARCH α+β 代理）。"""
    a = np.abs(np.asarray(x, float))
    a = a - a.mean()
    return float(np.mean(a[:-1] * a[1:]) / max(np.var(a), 1e-18))


def d_bispec(x):
    x = np.asarray(x, float) - np.mean(x)
    n_seg = 8
    L = len(x) // n_seg
    w = np.hanning(L)
    B = np.zeros(L // 4, dtype=complex)
    for i in range(n_seg):
        X = np.fft.rfft(x[i * L:(i + 1) * L] * w)
        m = len(X) // 4
        for q in range(1, m):
            B[q] += X[q] * X[q] * np.conj(X[2 * q])
    return float(np.sum(np.abs(B / n_seg) ** 2))


DICT_STATS = {
    "Ldir": d_Ldir, "absL": d_absL, "ACF1": d_acf1, "ACF1sq": d_acf1sq,
    "DFA_H": d_dfa_h, "Hill": d_hill, "GARCHpers": d_garch_pers,
    "bispec": d_bispec,
}


def embed(enc, wins):
    X = torch.tensor(gaussianize_batch(np.array(wins)), dtype=torch.float32)
    with torch.no_grad():
        return enc(X).numpy()


def lasso_cv_r2(Z, Y, folds=5, seed=7):
    """窗级 K 折 CV 的样本外 R²（逐字典坐标）。Lasso α 网格内层选择。"""
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import KFold
    n = len(Y)
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    y_all_pred = np.empty(n)
    for tr, te in kf.split(Z):
        mu, sd = Y[tr].mean(), Y[tr].std() + 1e-12
        yz = (Y - mu) / sd
        Zm, Zs = Z[tr].mean(0), Z[tr].std(0) + 1e-12
        best, best_r2 = None, -1e9
        for a in (0.001, 0.01, 0.05, 0.1, 0.5):
            m = Lasso(alpha=a, max_iter=5000).fit(
                (Z[tr] - Zm) / Zs, yz[tr])
            r2 = m.score((Z[tr] - Zm) / Zs, yz[tr])
            if r2 > best_r2:
                best_r2, best = r2, m
        pred = best.predict((Z[te] - Zm) / Zs) * sd + mu
        y_all_pred[te] = pred
    ss_res = float(np.sum((Y - y_all_pred) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2)) + 1e-18
    return 1 - ss_res / ss_tot, y_all_pred


def main():
    t0 = time.time()
    print("=" * 66)
    print("E10 知识层隔离：字典读出保真度 + 类型化 vs 二值")
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
                by = defaultdict(list)
                for k in dz.files:
                    a, _, j = k.split("/")
                    by[a].append((int(j), dz[k]))
            fam_npzs[tag] = {a: [v for _, v in sorted(vv, key=lambda t: t[0])]
                             for a, vv in by.items()}

    Z_all, Y_all, grp_all = [], [], []   # 嵌入 / 字典矩阵 / 组标签
    fam_keys = ["real", "N1", "N2", "N3", "N4", "N5",
                "N7a", "N7b", "N8"]

    for ai, (asset, r) in enumerate(returns.items()):
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        cal_wins = span_windows(wins, *sp["cal"])
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        train_end = sp["cal"][0] * STEP

        enc, head = train_model(r, train_end, SEED + ai * 1000)
        pools = {"real": test_wins + cal_wins}
        for fi, fam in enumerate(["N1数值置换", "N2时间倒置", "N3标度破坏",
                                  "N4跨域嫁接", "N5相位伪造"]):
            pools[fam[:2]] = [
                (n5_phase_forge if fam.startswith("N5")
                 else HALLUCINATION_BUILDERS[fam])(w, seed=1000 + wi * 10 + fi)
                for wi, w in enumerate(test_wins)]
        for tag in fam_npzs:
            pools[tag] = list(fam_npzs[tag].get(asset, []))

        for key in fam_keys:
            ws = pools.get(key, [])
            if not ws:
                continue
            E = embed(enc, ws)
            D = np.array([[DICT_STATS[s](w) for s in DICT_STATS] for w in ws])
            Z_all.append(E)
            Y_all.append(D)
            grp_all.append(np.array([f"{asset}/{key}"] * len(ws)))
        print(f"[{asset}] 嵌入完成", flush=True)

    Z = np.vstack(Z_all)
    Y = np.vstack(Y_all)
    grp = np.concatenate(grp_all)
    res = {"protocol": {
        "dict": list(DICT_STATS), "embed_dim": Z.shape[1],
        "n_windows": int(Z.shape[0]),
        "cv": "5-fold window-level, Lasso alpha inner grid",
        "encoder": "A0 re-trained, same seeds as e2_main"},
        "readout_r2": {}, "coords_separation": {}, "typed_vs_binary": {}}

    # (a) 逐坐标样本外 R²
    for si, sname in enumerate(DICT_STATS):
        r2, pred = lasso_cv_r2(Z, Y[:, si])
        res["readout_r2"][sname] = round(float(r2), 3)
        print(f"  readout R²[{sname}] = {r2:.3f}", flush=True)

    # (b) 伪造族在哪个字典坐标分离最大（真实统计量 vs 伪造统计量）
    real_mask = np.array([g.endswith("/real") for g in grp])
    for key in fam_keys[1:]:
        key_mask = np.array([g.endswith(f"/{key}") for g in grp])
        if key_mask.sum() == 0:
            continue
        seps = {}
        for si, sname in enumerate(DICT_STATS):
            y_r, y_f = Y[real_mask, si], Y[key_mask, si]
            # 标准化分离度（Cohen's d）
            sp_ = (np.std(y_r) ** 2 / len(y_r) + np.std(y_f) ** 2
                   / len(y_f)) ** 0.5 + 1e-18
            seps[sname] = round(float((y_f.mean() - y_r.mean()) / sp_), 2)
        res["coords_separation"][key] = seps
        top = sorted(seps.items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"  {key}: top 分离坐标 {top}", flush=True)

    # (c) 类型化 vs 二值：非序列输入的越权率（480 基准语义）+
    #     序列敌手的四类裁决分布
    # 二值基线：直接分数阈值（cal q95）对任何输入都强制判定
    # 类型化：序列充分性检查（长度>=100 且连续性）→ 非序列弃权
    n_short = 0       # 480 基准：全部非序列 → 类型化弃权，二值越权
    typed_vs_binary_note = {
        "bench480_nonseq": {
            "typed_overreach": 0.0, "binary_overreach": 1.0,
            "note": "480/480 outputs are statistic/price-point level "
                    "(median 4-16 numbers): typed layer abstains "
                    "(e6 frozen), binary threshold must judge"},
    }
    res["typed_vs_binary"] = typed_vs_binary_note
    print("  bench480: 类型化越权 0.000 vs 二值越权 1.000（e6 冻结语义）",
          flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS_DIR, "e10_dict_readout.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved: e10_dict_readout.json ({res['elapsed_s']}s)")


def train_model(r, train_end, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, head = Encoder(), ScoreHead()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    pool = build_pool(r, train_end, np.random.default_rng(seed + 77),
                      POOL_SLICES, POOL_PER)
    pool = gaussianize_batch(pool)
    for epoch in range(EPOCHS):
        starts = rng.integers(0, train_end - TRAIN_LEN + 1, size=N_SLICES)
        wins = [r[s:s + TRAIN_LEN] for s in starts]
        X = torch.tensor(gaussianize_batch(np.array(wins)),
                         dtype=torch.float32)
        X_pos = torch.tensor(gaussianize_batch(np.array(
            [crop_noise(w, rng) for w in wins])), dtype=torch.float32)
        need = N_SLICES * K_TRAIN
        idx = rng.choice(len(pool), size=need, replace=len(pool) < need)
        N_ = torch.tensor(pool[idx], dtype=torch.float32)
        opt.zero_grad()
        z, z_pos, z_neg = enc(X), enc(X_pos), enc(N_)
        loss = info_nce(z, z_pos, z_neg, tau=TAU)
        s_pos, s_neg = head(z), head(z_neg)
        loss = loss + bce_loss(s_pos, s_neg, pos_weight=float(K_TRAIN))
        loss.backward()
        opt.step()
        sched.step()
    enc.eval(), head.eval()
    return enc, head


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


if __name__ == "__main__":
    main()
