"""E7：真实生成器审计主评估，N7a LLM 锻造 / N7b QuantGAN / N8 GJR-EGARCH。

实验目的：标题级主张 auditing machine-generated
financial series 需要真实生成器敌手；N6 泛化证据与 N2 同签名需正交敌手。

协议（与 e2_main 逐位一致）：
  - A0 训练：SEED + ai*1000 + 0（与 e2_main A0 相同种子与配置）；
  - e 值：K=32，seed base = 900000 + ai*10000（与 e2_main real/fam 同源
    错开：本脚本用 base+500 段位避免与既有结果共用随机流）；
  - 三层切分、cal q95 冻结阈值、ALPHA=0.05 同 e2_main。

对比方法：
  A0    MINT e 值（-W 异常分）+ 认证规则
  A5b   古典 8 维马氏（与 e2_main 同实现）
  LAonly 有向杠杆不对称单变量
  Gen2  第二代正统（TR 统计量+IAAFT surrogate 校准，读数 e9 JSON 的 p 值，
        判伪造 = p>0.5 免训练规则）

推断口径（吸取方法论 W2 教训，一步到位）：
  AUC 点估计 + 源窗级 cluster bootstrap 95% CI（窗口为重抽单元，
  族内窗口与族间同窗联动）；DeLong 仅作参考，主报 cluster bootstrap。

产出：results/e7_generators.json
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
FAM_TAGS = ["N7a", "N7b", "N8"]
FAM_FILES = {
    "N7a": "n7a_llm_forge.npz",
    "N7b": "n7b_quantgan.npz",
    "N8": "n8_gjr_egarch.npz",
}
N_BOOT = 2000


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


def cluster_boot_auc(fam_anom_by_win, real_anom_by_win, reps=N_BOOT,
                     seed=424242):
    """源窗级 cluster bootstrap：窗口为重抽单元（真实窗与各伪造族内
    对应窗联动），返回 AUC 95% CI。fam_anom_by_win: list per source win。
    """
    rng = np.random.default_rng(seed)
    n_w = len(fam_anom_by_win)
    aucs = np.empty(reps)
    for b in range(reps):
        idx = rng.integers(0, n_w, n_w)
        pos, neg = [], []
        for i in idx:
            pos.append(np.atleast_1d(fam_anom_by_win[i]))
            neg.append(np.atleast_1d(real_anom_by_win[i]))
        pos = np.concatenate(pos)
        neg = np.concatenate(neg)
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
    print("E7 真实生成器审计：N7a LLM锻造 / N7b QuantGAN / N8 GJR-EGARCH"
          + ("  [冒烟]" if smoke else ""))
    print("=" * 66, flush=True)
    t0 = time.time()

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    # 敌手 npz 读入
    fams = {}
    for tag, fn in FAM_FILES.items():
        path = os.path.join(RESULTS_DIR, fn)
        with np.load(path, allow_pickle=True) as d:
            by_asset = defaultdict(dict)
            for key in d.files:
                asset, rest = key.split("/", 1)
                by_asset[asset][rest] = d[key]
            fams[tag] = by_asset
        n_tot = sum(len(v) for v in by_asset.values())
        print(f"{tag}: {n_tot} 条（{dict((a, len(b)) for a, b in by_asset.items())}）",
              flush=True)

    # 第二代 p 值（e9 JSON 已算好，逐族逐资产）
    e9_path = os.path.join(RESULTS_DIR, "e9_gen2_surrogate.json")
    gen2 = None
    if os.path.exists(e9_path):
        with open(e9_path, encoding="utf-8") as fh:
            e9 = json.load(fh)
        gen2 = e9["pvalues"]

    # 池化容器
    anom = {m: {"cal": [], "real": [], "fam": {t: [] for t in FAM_TAGS}}
            for m in ["A0", "A5b", "LAonly"]}
    per_asset = {}
    cluster_cells = {m: {t: [] for t in FAM_TAGS} for m in anom}
    real_cells = []
    Wraw = {"real": [], "fam": {t: [] for t in FAM_TAGS}}

    for ai, (asset, r) in enumerate(returns.items()):
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        fit_wins = wins[:sp["n_fit"]]
        cal_wins = span_windows(wins, *sp["cal"])
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        train_end = sp["cal"][0] * STEP
        # 对齐：伪造条数取 min(可用条数, len(test_wins)*2) 不强求
        fam_wins = {}
        for tag in FAM_TAGS:
            avail = list(fams[tag][asset].values())
            fam_wins[tag] = avail
        n_f = {t: len(fam_wins[t]) for t in FAM_TAGS}
        print(f"\n[{asset}] cal{len(cal_wins)} test{len(test_wins)} "
              f"敌手 {n_f}", flush=True)

        enc, head, last = train_model(r, train_end, SEED + ai * 1000)
        print(f"    A0 trained loss={last['loss']}", flush=True)
        base = 950000 + ai * 10000
        w_cal = e_values(enc, head, cal_wins, K_EVAL, base)
        w_real = e_values(enc, head, test_wins, K_EVAL, base + 500)
        anom["A0"]["cal"].append(-w_cal)
        anom["A0"]["real"].append(-w_real)
        Wraw["real"].append(w_real)
        real_cells.append(-w_real)

        A = np.array([extract_features(x) for x in fit_wins])
        mu = np.nanmean(A, axis=0)
        inv = np.linalg.pinv(np.nan_to_num(np.cov(A, rowvar=False)))

        def msc(ws):
            B = np.array([extract_features(x) for x in ws])
            D = B - mu
            return np.einsum("ij,jk,ik->i", D, inv, D)

        anom["A5b"]["cal"].append(msc(cal_wins))
        anom["A5b"]["real"].append(msc(test_wins))
        anom["LAonly"]["cal"].append(
            np.array([leverage_asym(x) for x in cal_wins]))
        anom["LAonly"]["real"].append(
            np.array([leverage_asym(x) for x in test_wins]))

        for tag in FAM_TAGS:
            fw = fam_wins[tag]
            w_f = e_values(enc, head, fw, K_EVAL, base + 700)
            anom["A0"]["fam"][tag].append(-w_f)
            anom["A5b"]["fam"][tag].append(msc(fw))
            anom["LAonly"]["fam"][tag].append(
                np.array([leverage_asym(x) for x in fw]))
            Wraw["fam"][tag].append(w_f)
            # 聚类单元：伪造条按源轮转归窗（每窗 ≤2 条时轮转分派）
            n_t = len(fw)
            cells = [[] for _ in range(len(test_wins))]
            for j in range(n_t):
                cells[j % len(test_wins)].append(-w_f[j])
            cluster_cells["A0"][tag].append(cells)
            a5 = msc(fw)
            cells5 = [[] for _ in range(len(test_wins))]
            for j in range(n_t):
                cells5[j % len(test_wins)].append(a5[j])
            cluster_cells["A5b"][tag].append(cells5)
            la = np.array([leverage_asym(x) for x in fw])
            cellsl = [[] for _ in range(len(test_wins))]
            for j in range(n_t):
                cellsl[j % len(test_wins)].append(la[j])
            cluster_cells["LAonly"][tag].append(cellsl)

    # ---- 汇总 ----
    res = {"protocol": {
        "A0_seed": SEED, "epochs": epochs, "K_eval": K_EVAL,
        "threshold": "per-asset cal q95 frozen", "alpha": ALPHA,
        "cluster_bootstrap": f"source-window level, {N_BOOT} reps"},
        "methods": {}, "per_asset": per_asset}

    for tag_m in anom:
        real_all = np.concatenate(anom[tag_m]["real"])
        # 阈值逐资产 cal q95
        flags_real = []
        for ai, asset in enumerate(returns):
            cal = anom[tag_m]["cal"][ai]
            thr = np.quantile(cal[np.isfinite(cal)], 1 - ALPHA)
            flags_real.append(
                np.asarray(anom[tag_m]["real"][ai]) > thr)
        fr = np.concatenate(flags_real)
        s = {"fpr": float(fr.mean()), "n_real": int(len(fr))}
        for tag_f in FAM_TAGS:
            flags_f = []
            for ai, asset in enumerate(returns):
                cal = anom[tag_m]["cal"][ai]
                thr = np.quantile(cal[np.isfinite(cal)], 1 - ALPHA)
                v = np.asarray(anom[tag_m]["fam"][tag_f][ai])
                flags_f.append(v > thr)
            ff = np.concatenate(flags_f)
            fa = np.concatenate(anom[tag_m]["fam"][tag_f])
            auc_v = auc_pair(fa, real_all)
            # 聚类 bootstrap CI（源窗为重抽单元）
            cells = [c for arr in cluster_cells[tag_m][tag_f]
                     for c in arr if len(c)]
            r_flat = []
            for ai in range(len(anom[tag_m]["real"])):
                for x in np.atleast_1d(anom[tag_m]["real"][ai]):
                    r_flat.append(np.array([x]))
            lo, hi = cluster_boot_auc(cells, r_flat)
            s[tag_f] = {
                "auc": round(float(auc_v), 3),
                "auc_cluster_ci": [round(lo, 3), round(hi, 3)],
                "recall": float(ff.mean()),
                "recall_wilson": [round(x, 3) for x in
                                  wilson(int(ff.sum()), len(ff))],
                "n_forge": int(len(ff))}
        res["methods"][tag_m] = s
        print(f"\n[{tag_m}] FPR={s['fpr']:.3f}")
        for tag_f in FAM_TAGS:
            v = s[tag_f]
            print(f"  {tag_f}: AUC={v['auc']} CI{v['auc_cluster_ci']} "
                  f"recall={v['recall']:.3f}", flush=True)

    # e 值认证
    Wr = np.concatenate(Wraw["real"])
    res["e_rule"] = {"real_certify": float(np.mean(Wr >= 1 / ALPHA))}
    for tag_f in FAM_TAGS:
        Wf = np.concatenate(Wraw["fam"][tag_f])
        res["e_rule"][f"{tag_f}_certify"] = float(np.mean(Wf >= 1 / ALPHA))
    print("\ne-rule:", json.dumps(res["e_rule"], indent=1), flush=True)

    # 第二代基线汇总（引用 e9）
    if gen2:
        res["gen2_Ldir"] = {}
        for tag_f in FAM_TAGS:
            pv_f, pv_r = [], []
            for asset in returns:
                blk = gen2["Ldir"].get(asset, {})
                pv_r.extend(blk.get("real", []))
                pv_f.extend(blk.get(tag_f, []))
            if pv_f:
                res["gen2_Ldir"][tag_f] = {
                    "auc": round(auc_pair(pv_f, pv_r), 3),
                    f"recall@p>={ALPHA}": round(
                        float(np.mean(np.array(pv_f) >= ALPHA)), 3),
                    f"fpr@p>={ALPHA}": round(
                        float(np.mean(np.array(pv_r) >= ALPHA)), 3)}
        print("Gen2(Ldir):", json.dumps(res["gen2_Ldir"], indent=1),
              flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS_DIR, "e7_generators.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved: e7_generators.json ({res['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
