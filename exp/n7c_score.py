"""N7c 评分：Chronos-T5 锻造路径过 MINT 全协议（R7 当代生成器）。

协议与 e7_generators.py 逐位一致（A0 种子、三层切分、cal q95 冻结
阈值、K=32、α=0.05）；e 值随机流用 base+900 段位，与 e7 的
real(base)/fam(base+700) 均不重叠。

方法（对齐 tab:real-gen 列）：A0 MINT e 值、A5b 古典 8 维马氏、
LAonly 有向杠杆不对称。

产出：results/n7c_score.json
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

from features import extract_features  # noqa: E402
from features.phase_ext import leverage_asym  # noqa: E402
from mint.model import (  # noqa: E402
    Encoder, ScoreHead, bce_loss, gaussianize_batch, info_nce,
)
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
K_EVAL = 32
EPOCHS = 250
TAU, LR, CROP = 0.1, 1e-3, 620
SEED = 20260820
N_BOOT = 2000
NPZ_IN = os.path.join(RESULTS_DIR, "n7c_chronos_forge.npz")


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


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
        loss = info_nce(z, z_pos, z_neg, tau=TAU)
        s_pos, s_neg = head(z), head(z_neg)
        loss_bce = bce_loss(s_pos, s_neg, pos_weight=float(K_TRAIN))
        loss = loss + loss_bce
        loss.backward()
        opt.step()
        sched.step()
        last = {"loss": round(loss.item(), 4)}
    enc.eval(), head.eval()
    return enc, head, last


@torch.no_grad()
def logits_var(enc, head, wins):
    out = np.empty(len(wins))
    groups = defaultdict(list)
    for i, w in enumerate(wins):
        groups[len(w)].append(i)
    for _, idxs in groups.items():
        X = torch.tensor(prep(np.array([wins[i] for i in idxs])), dtype=torch.float32)
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


def auc_pair(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.r_[pos, neg]
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = len(pos), len(neg)
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def cluster_boot_auc(fam_cells, real_cells, reps=N_BOOT, seed=535353):
    rng = np.random.default_rng(seed)
    n_w = len(fam_cells)
    aucs = np.empty(reps)
    for b in range(reps):
        idx = rng.integers(0, n_w, n_w)
        pos = np.concatenate([np.atleast_1d(fam_cells[i]) for i in idx])
        neg = np.concatenate([np.atleast_1d(real_cells[i]) for i in idx])
        aucs[b] = auc_pair(pos, neg)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def wilson(k, n, zz=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + zz * zz / n
    c = p + zz * zz / (2 * n)
    h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def main():
    smoke = "--smoke" in sys.argv
    epochs = 15 if smoke else EPOCHS
    print("=" * 66)
    print("N7c Chronos-T5 锻造路径评分（与 e7 协议逐位一致）"
          + ("  [冒烟]" if smoke else ""))
    print("=" * 66, flush=True)
    t0 = time.time()

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    with np.load(NPZ_IN, allow_pickle=True) as d:
        by_asset = defaultdict(dict)
        for key in d.files:
            asset, rest = key.split("/", 1)
            by_asset[asset][rest] = d[key]
    n_tot = sum(len(v) for v in by_asset.values())
    print(f"N7c: {n_tot} 条（{dict((a, len(b)) for a, b in by_asset.items())}）",
          flush=True)

    anom = {m: {"cal": [], "real": [], "fam": []}
            for m in ["A0", "A5b", "LAonly"]}
    cluster_cells = {m: [] for m in anom}
    real_cells = []
    Wraw = {"real": [], "fam": []}

    for ai, (asset, r) in enumerate(returns.items()):
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        fit_wins = wins[:sp["n_fit"]]
        cal_wins = span_windows(wins, *sp["cal"])
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        train_end = sp["cal"][0] * STEP
        fw = list(by_asset[asset].values())
        print(f"\n[{asset}] cal{len(cal_wins)} test{len(test_wins)} "
              f"N7c {len(fw)}", flush=True)

        enc, head, last = train_model(r, train_end, SEED + ai * 1000)
        print(f"    A0 trained loss={last['loss']}", flush=True)
        base = 950000 + ai * 10000
        w_cal = e_values(enc, head, cal_wins, K_EVAL, base)
        w_real = e_values(enc, head, test_wins, K_EVAL, base + 500)
        w_f = e_values(enc, head, fw, K_EVAL, base + 900)
        anom["A0"]["cal"].append(-w_cal)
        anom["A0"]["real"].append(-w_real)
        anom["A0"]["fam"].append(-w_f)
        Wraw["real"].append(w_real)
        Wraw["fam"].append(w_f)
        real_cells.append(np.asarray(-w_real))

        A = np.array([extract_features(x) for x in fit_wins])
        mu = np.nanmean(A, axis=0)
        inv = np.linalg.pinv(np.nan_to_num(np.cov(A, rowvar=False)))

        def msc(ws):
            B = np.array([extract_features(x) for x in ws])
            D = B - mu
            return np.einsum("ij,jk,ik->i", D, inv, D)

        anom["A5b"]["cal"].append(msc(cal_wins))
        anom["A5b"]["real"].append(msc(test_wins))
        anom["A5b"]["fam"].append(msc(fw))
        anom["LAonly"]["cal"].append(
            np.array([leverage_asym(x) for x in cal_wins]))
        anom["LAonly"]["real"].append(
            np.array([leverage_asym(x) for x in test_wins]))
        anom["LAonly"]["fam"].append(
            np.array([leverage_asym(x) for x in fw]))

        for m in anom:
            cells = [[] for _ in range(len(test_wins))]
            vf = np.asarray(anom[m]["fam"][ai])
            for j in range(len(fw)):
                cells[j % len(test_wins)].append(vf[j])
            cluster_cells[m].append(cells)

    res = {"protocol": {
        "A0_seed": SEED, "epochs": epochs, "K_eval": K_EVAL,
        "threshold": "per-asset cal q95 frozen", "alpha": ALPHA,
        "cluster_bootstrap": f"source-window level, {N_BOOT} reps",
        "e_value_seed_base": "950000+ai*10000+900"},
        "methods": {}}

    for m in anom:
        real_all = np.concatenate(anom[m]["real"])
        flags_real = []
        flags_f = []
        for ai in range(len(returns)):
            cal = anom[m]["cal"][ai]
            thr = np.quantile(cal[np.isfinite(cal)], 1 - ALPHA)
            flags_real.append(np.asarray(anom[m]["real"][ai]) > thr)
            flags_f.append(np.asarray(anom[m]["fam"][ai]) > thr)
        fr = np.concatenate(flags_real)
        ff = np.concatenate(flags_f)
        fa = np.concatenate(anom[m]["fam"])
        auc_v = auc_pair(fa, real_all)
        cells = [c for arr in cluster_cells[m] for c in arr if len(c)]
        r_flat = [np.array([x]) for ai in range(len(anom[m]["real"]))
                  for x in np.atleast_1d(anom[m]["real"][ai])]
        lo, hi = cluster_boot_auc(cells, r_flat)
        res["methods"][m] = {
            "fpr": float(fr.mean()), "n_real": int(len(fr)),
            "auc": round(float(auc_v), 3),
            "auc_cluster_ci": [round(lo, 3), round(hi, 3)],
            "recall": float(ff.mean()),
            "recall_wilson": [round(x, 3) for x in
                              wilson(int(ff.sum()), len(ff))],
            "n_forge": int(len(ff))}
        v = res["methods"][m]
        print(f"[{m}] FPR={v['fpr']:.3f} AUC={v['auc']} "
              f"CI{v['auc_cluster_ci']} recall={v['recall']:.3f}",
              flush=True)

    Wr = np.concatenate(Wraw["real"])
    Wf = np.concatenate(Wraw["fam"])
    res["e_rule"] = {
        "real_certify": float(np.mean(Wr >= 1 / ALPHA)),
        "n7c_certify": float(np.mean(Wf >= 1 / ALPHA)),
        "n7c_certified_count": int((Wf >= 1 / ALPHA).sum()),
        "n_forge": int(len(Wf))}
    print("\ne-rule:", json.dumps(res["e_rule"], indent=1), flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    out = os.path.join(RESULTS_DIR, "n7c_score.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved: n7c_score.json ({res['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
