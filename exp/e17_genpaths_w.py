"""E17：真实生成器 128 条路径的逐条 W 落盘 + 零计数 cluster-robust 界。

动机：0/128 的 Wilson 界假设路径独立，而 80/128 条共享
同参数 GJR 递归、20 条共享 5 个 GAN 训练、10 条共享 2 个 LLM、18 条
共享 Chronos 条件运行。本脚本逐位复现 e7/n7c 的冻结协议（训练种子、
e 值种子段位均相同），落盘每条路径的 W 值与聚类归属，供 cluster-robust
重分析与零边距（max W vs 阈值 20）核验。

聚类单元 = 生成拟合单元（LLM 解码 × 资产、GAN 训练、Chronos 条件
运行、拟合递归 × 资产），共 40 单元发出 128 条路径。
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

from mint.model import (  # noqa: E402
    Encoder,
    ScoreHead,
    bce_loss,
    gaussianize_batch,
    info_nce,
)
from mint.operators import ORBIT_NAMES, generate_orbit  # noqa: E402

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
ALPHA = 0.05
E_VAL = 1.0 / ALPHA
TRAIN_LEN, N_SLICES = 700, 64
POOL_SLICES, POOL_PER, K_TRAIN = 150, 20, 12
K_EVAL = 32
EPOCHS = 250
TAU, LR, CROP = 0.1, 1e-3, 620
SEED = 20260820

FAM_FILES = {
    "N7a": "n7a_llm_forge.npz",
    "N7b": "n7b_quantgan.npz",
    "N7c": "n7c_chronos_forge.npz",
    "N8": "n8_gjr_egarch.npz",
}


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
    enc.eval(), head.eval()
    return enc, head


@torch.no_grad()
def logits_var(enc, head, wins):
    out = np.empty(len(wins))
    groups = defaultdict(list)
    for i, w in enumerate(wins):
        groups[len(w)].append(i)
    for _, idxs in groups.items():
        X = torch.tensor(prep(np.array([wins[i] for i in idxs])),
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


def main() -> None:
    t0 = time.time()
    print("=" * 66)
    print("E17 生成器路径逐条 W 落盘（复现 e7/n7c 冻结协议）")
    print("=" * 66, flush=True)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

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
            print(f"{tag}: {n_tot} 条", flush=True)

    rows = []          # 机器路径记录
    real_rows = []     # 真实测试窗记录
    per_asset = {}

    for ai, (asset, r) in enumerate(returns.items()):
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        cal_wins = span_windows(wins, *sp["cal"])
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        train_end = sp["cal"][0] * STEP
        base = 950000 + ai * 10000

        enc, head = train_model(r, train_end, SEED + ai * 1000)
        w_cal = e_values(enc, head, cal_wins, K_EVAL, base)
        w_real = e_values(enc, head, test_wins, K_EVAL, base + 500)
        for j, w in enumerate(w_real):
            real_rows.append({"asset": asset, "win_idx": j, "W": float(w),
                              "certified": bool(w >= E_VAL)})
        per_asset[asset] = {
            "n_test": len(test_wins),
            "real_cert": int(np.sum(w_real >= E_VAL)),
            "real_max_w": float(np.max(w_real)),
            "cal_q95_w": float(np.quantile(w_cal, 0.95)),
        }
        print(f"[{asset}] real cert {per_asset[asset]['real_cert']}"
              f"/{len(test_wins)} maxW={per_asset[asset]['real_max_w']:.2f}",
              flush=True)

        for tag in FAM_FILES:
            avail = fams[tag][asset]
            paths = list(avail.items())     # (key, array)
            ws = e_values(enc, head, [p[1] for p in paths], K_EVAL,
                          base + 900 if tag == "N7c" else base + 700)
            for (key, _arr), w in zip(paths, ws):
                unit = f"{tag}|{asset}|{key.rsplit('/', 1)[0]}"
                rows.append({
                    "asset": asset, "family": tag,
                    "path_key": key.replace("/", ":"), "unit": unit,
                    "W": float(w), "certified": bool(w >= E_VAL)})

    n_tot = len(rows)
    n_cert = sum(r_["certified"] for r_ in rows)
    units = sorted({r_["unit"] for r_ in rows})
    unit_cert = {u: sum(r_["certified"] for r_ in rows if r_["unit"] == u)
                 for u in units}
    asset_cert = {a: sum(r_["certified"] for r_ in rows if r_["asset"] == a)
                  for a in returns}
    max_w = max(r_["W"] for r_ in rows)

    # cluster-robust 零计数上界：单元级 Clopper-Pearson（0 计数）
    def cp_upper(c_units):
        return float(1 - 0.05 ** (1 / c_units))

    out = {
        "protocol": {"A0_seed": SEED, "epochs": EPOCHS, "K_eval": K_EVAL,
                     "alpha": ALPHA, "e_value_seed_base":
                         "950000+ai*10000 (+500 real, +700 fam, +900 n7c)"},
        "n_paths": n_tot, "n_certified": n_cert,
        "n_units": len(units), "unit_cert": unit_cert,
        "asset_cert": asset_cert,
        "max_w_paths": max_w, "threshold": E_VAL,
        "per_asset_real": per_asset,
        "bounds": {
            "wilson_path_level": None,
            "cluster_unit_cp95": cp_upper(len(units)),
            "cluster_asset_cp95": cp_upper(len(asset_cert)),
            "n_units_by_family": {
                t: len({r_["unit"] for r_ in rows
                        if r_["family"] == t}) for t in FAM_FILES},
        },
        "paths": rows, "real_windows": real_rows,
    }
    try:
        from scipy.stats import beta as _beta
        lo_w = 0.0
        hi_w = float(_beta.ppf(0.95, n_cert + 1, n_tot - n_cert))
        out["bounds"]["wilson_path_level"] = hi_w
    except Exception:
        pass

    with open(os.path.join(RESULTS_DIR, "e17_genpaths_w.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\n路径 {n_tot} 条 | 认证 {n_cert} | 单元 {len(units)} 个 | "
          f"max W = {max_w:.3f} (阈值 {E_VAL})")
    print(f"cluster-robust 95% 上界（单元级）= "
          f"{out['bounds']['cluster_unit_cp95']:.4f}")
    print(f"saved: e17_genpaths_w.json  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
