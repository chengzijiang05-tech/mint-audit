"""N7c Chronos 路径的 Gen2 四统计量读数补算（对齐 e9_gen2_surrogate.py 协议）。

f_idx=8（e9 的 fam_series 枚举为 N1..N5, N7a, N7b, N8 共 8 族，N7c 追加第 9 位）。
真实窗 p 值直接取 e9_gen2_surrogate.json（不重算，保证单一口径）。
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

from mint.operators import iaaft  # noqa: E402

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
N7C_NPZ = os.path.join(RESULTS_DIR, "n7c_chronos_forge.npz")

WINDOW, STEP = 1000, 50
SPLIT_SPEC = {
    "equity":  {"n_fit": 15, "cal": (34, 43), "test": (63, 99), "cap": 8},
    "bond":    {"n_fit": 15, "cal": (34, 43), "test": (63, 94), "cap": 8},
    "fx":      {"n_fit": 11, "cal": (31, 36), "test": (37, 42), "cap": 6},
    "gold":    {"n_fit": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
    "copper":  {"n_fit": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
}
B_SURR = 99
K_LAG = 20
SEED = 20260821
ALPHA = 0.05
F_IDX_N7C = 8


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


def stat_L(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    d = np.mean(x ** 2)
    if d <= 0:
        return 0.0
    ls = []
    for k in range(1, K_LAG + 1):
        num = np.mean(x[:-k] * x[k:] ** 2) - np.mean(x[:-k] ** 2 * x[k:])
        ls.append(num / (d * d))
    return float(np.mean(np.abs(ls)))


def stat_Ldir(x):
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


def stat_cck(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    d = np.mean(x ** 2)
    tr = []
    for k in range(1, K_LAG + 1):
        tr.append((np.mean(x[:-k] * x[k:] ** 2)
                   - np.mean(x[:-k] ** 2 * x[k:])) / (d * d))
    tr = np.array(tr)
    return float(np.sum(tr ** 2) / max(len(tr), 1))


def stat_bispec(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    n_seg = 8
    L = len(x) // n_seg
    B = np.zeros(L // 4, dtype=complex)
    w = np.hanning(L)
    for i in range(n_seg):
        seg = x[i * L:(i + 1) * L] * w
        X = np.fft.rfft(seg)
        m = len(X) // 4
        for q in range(1, m):
            B[q] += X[q] * X[q] * np.conj(X[2 * q])
    return float(np.sum(np.abs(B / max(n_seg, 1)) ** 2))


STATS = {"L": stat_L, "Ldir": stat_Ldir, "CCK": stat_cck,
         "bispec": stat_bispec}


def surr_pvalue(x, stat_fn, seed, b=B_SURR):
    rng = np.random.default_rng(seed)
    t_obs = stat_fn(x)
    t_s = np.empty(b)
    for i in range(b):
        s = iaaft(x, rng)
        t_s[i] = stat_fn(s)
    p_hi = (1 + np.sum(t_s >= t_obs)) / (b + 1)
    p_lo = (1 + np.sum(t_s <= t_obs)) / (b + 1)
    return float(min(p_hi, p_lo))


def auc_pair(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if np.std(pos) == 0 and np.std(neg) == 0:
        return 0.5
    s = np.r_[pos, neg]
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = len(pos), len(neg)
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def main():
    t0 = time.time()
    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    with open(os.path.join(RESULTS_DIR, "e9_gen2_surrogate.json"),
              encoding="utf-8") as fh:
        e9 = json.load(fh)

    with np.load(N7C_NPZ, allow_pickle=True) as d:
        n7c = {}
        for key in d.files:
            asset, cfg, j = key.split("/")
            n7c.setdefault(asset, []).append((cfg, int(j), d[key]))
    for asset in n7c:
        n7c[asset].sort(key=lambda t: (t[0], t[1]))

    out = {}
    for stat_name, stat_fn in STATS.items():
        pv_f, pv_r = [], []
        for ai, (asset, r) in enumerate(returns.items()):
            blk = e9["pvalues"][stat_name][asset]
            real_p = blk["real"]
            pv_r.extend(real_p)
            for wi, (_cfg, _j, w) in enumerate(n7c.get(asset, [])):
                p = surr_pvalue(w, stat_fn, SEED + 600000 + ai * 100
                                + F_IDX_N7C * 97 + wi)
                pv_f.append(p)
        a = auc_pair(pv_f, pv_r)
        fl_f = float(np.mean(np.array(pv_f) >= ALPHA))
        fl_r = float(np.mean(np.array(pv_r) >= ALPHA))
        out[stat_name] = {
            "auc": round(a, 3),
            f"recall@p>={ALPHA}": round(fl_f, 3),
            f"fpr@p>={ALPHA}": round(fl_r, 3),
            "n_forge": len(pv_f), "n_real": len(pv_r)}
        print(f"[{stat_name}] auc={out[stat_name]['auc']} "
              f"recall={fl_f:.3f} fpr={fl_r:.3f}", flush=True)

    res = {
        "protocol": {
            "stats": list(STATS),
            "surrogates": f"IAAFT B={B_SURR}",
            "seed": SEED, "family_index": F_IDX_N7C,
            "real_p_from": "e9_gen2_surrogate.json",
            "n_forge": 18, "n_real": 28},
        "summary": out,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps(res["summary"], indent=1))
    with open(os.path.join(RESULTS_DIR, "n7c_gen2_ldir.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"saved: n7c_gen2_ldir.json ({res['elapsed_s']}s)")


if __name__ == "__main__":
    main()
