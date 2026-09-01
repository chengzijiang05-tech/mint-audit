"""E2 诊断（G2 失败归因）：分离校准漂移 / 长度错配 / e 值域压缩。

G2 三门槛全败但池化 AUC 全胜，判别力在、工作点坏。三个嫌疑：
  D1 时期漂移   cal(1000) vs test(1000)：同长度跨时期。cal q05 冻结
                阈值的实测 FPR（fx 0.33、equity 0.12）由它决定；
  D2 长度错配   训练切片 700（正样本裁到 620）vs 评测窗 1000：对
                train 期窗与 test 期窗各测 @1000 与 @700 纯裁剪，
                配对差把长度效应从时期效应中分离出来；
  D3 e 值域压缩 同一模型同一轨道，sigmoid 域 W（现行）与 logit 域
                W=(K+1)·softmax(s)_x（exp 权重，同样可交换有效）的
                认证率对比，sigmoid 饱和压平大 logit 差，是真实窗
                认证率 0 的第一嫌疑（认证需 logit 分离约 ±4，logit
                域只需 ±2）。

修复候选验证：A0-L（TRAIN_LEN=1000 / CROP=880，其余协议不动）与
A0-S（现行 700/620，种子与 e2_main A0 逐位一致）对比，分离度、
认证率、逐族 recall、FPR、AUC 全表，直接决定 E2 是否以 1000 长度
重训 + e 值换 logit 域。

产出：results/e2_diag.json
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

from bench import HALLUCINATION_BUILDERS, n5_phase_forge  # noqa: E402
from mint.model import (  # noqa: E402
    Encoder,
    ScoreHead,
    bce_loss,
    gaussianize_batch,
    info_nce,
    orbit_e_value,
)
from mint.operators import ORBIT_NAMES, generate_orbit  # noqa: E402
from scipy import stats  # noqa: E402

WINDOW, STEP = 1000, 50
SPLIT_SPEC = {
    "equity":  {"n_anchor": 15, "cal": (34, 43), "test": (63, 99), "cap": 8},
    "bond":    {"n_anchor": 15, "cal": (34, 43), "test": (63, 94), "cap": 8},
    "fx":      {"n_anchor": 11, "cal": (31, 36), "test": (37, 42), "cap": 6},
    "gold":    {"n_anchor": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
    "copper":  {"n_anchor": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
}
FAMILIES = ["N1数值置换", "N2时间倒置", "N3标度破坏", "N4跨域嫁接",
            "N5相位伪造"]
FAM_SHORT = ["N1", "N2", "N3", "N4", "N5"]
ALPHA = 0.05
READOUTS = ("direct", "Wsig", "Wlogit")

N_SLICES = 64
POOL_SLICES, POOL_PER, K_TRAIN = 150, 20, 12
K_EVAL = 32
EPOCHS = 250
TAU, LR = 0.1, 1e-3
SEED = 20260820
VARIANTS = (("S", 700, 620), ("L", 1000, 880))

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


def forge(x: np.ndarray, family: str, seed: int) -> np.ndarray:
    if family == "N5相位伪造":
        return n5_phase_forge(x, seed=seed)
    return HALLUCINATION_BUILDERS[family](x, seed=seed)


def crop_noise_len(w, rng, n):
    start = int(rng.integers(0, len(w) - n + 1))
    out = w[start:start + n].copy()
    return out + 0.02 * out.std() * rng.standard_normal(len(out))


def build_pool(r, train_end, names, rng, n_slices, per, train_len):
    starts = rng.integers(0, train_end - train_len + 1, size=n_slices)
    rows = []
    for s in starts:
        w = r[int(s):int(s) + train_len]
        surrs, _ = generate_orbit(w, per, names=names, rng=rng)
        rows.append(surrs)
    return np.vstack(rows)


def train_a0(r, train_end, train_len, crop_len, seed):
    """与 e2_main.train_model(A0) 逐位一致（S 变体时），仅长度参数化。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, head = Encoder(), ScoreHead()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    pool = build_pool(r, train_end, ORBIT_NAMES,
                      np.random.default_rng(seed + 77),
                      POOL_SLICES, POOL_PER, train_len)
    pool = gaussianize_batch(pool)

    t0 = time.time()
    for _epoch in range(EPOCHS):
        starts = rng.integers(0, train_end - train_len + 1, size=N_SLICES)
        wins = [r[s:s + train_len] for s in starts]
        X = torch.tensor(gaussianize_batch(np.array(wins)),
                         dtype=torch.float32)
        X_pos = torch.tensor(gaussianize_batch(np.array(
            [crop_noise_len(w, rng, crop_len) for w in wins])),
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
    enc.eval(), head.eval()
    return enc, head, round(time.time() - t0, 1)


@torch.no_grad()
def logits_var(enc, head, wins):
    out = np.empty(len(wins))
    groups = defaultdict(list)
    for i, w in enumerate(wins):
        groups[len(w)].append(i)
    for _, idxs in groups.items():
        X = torch.tensor(gaussianize_batch(
            np.array([wins[i] for i in idxs])), dtype=torch.float32)
        out[idxs] = head(enc(X)).numpy()
    return out


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def orbit_anatomy(enc, head, wins, seed):
    """逐窗轨道解剖：s(x)、轨道均分、sigmoid/logit 两域 W、逐算子均分。"""
    out = {k: np.empty(len(wins))
           for k in ("s_x", "s_orb_mean", "W_sig", "W_logit")}
    op_scores = defaultdict(list)
    for i, w in enumerate(wins):
        surrs, used = generate_orbit(w, K_EVAL,
                                     rng=np.random.default_rng(seed + i))
        s = logits_var(enc, head, [w] + list(surrs))
        sx, sy = float(s[0]), s[1:]
        out["s_x"][i] = sx
        out["s_orb_mean"][i] = float(np.mean(sy))
        p = sigmoid_np(s)
        out["W_sig"][i] = orbit_e_value(float(p[0]), p[1:])
        m = float(np.max(s))
        lse = m + np.log(np.sum(np.exp(s - m)))
        out["W_logit"][i] = float((K_EVAL + 1) * np.exp(sx - lse))
        for j, op in enumerate(used):
            op_scores[op].append(float(sy[j]))
    return out, dict(op_scores)


def crop_avg(enc, head, wins, n, rng, reps=3):
    out = np.empty(len(wins))
    for i, w in enumerate(wins):
        vals = []
        for _ in range(reps):
            st = int(rng.integers(0, len(w) - n + 1))
            vals.append(logits_var(enc, head, [w[st:st + n]])[0])
        out[i] = float(np.mean(vals))
    return out


def fpr_at(s_cal, s_test):
    thr = float(np.quantile(np.asarray(s_cal, float), ALPHA))
    return float(np.mean(np.asarray(s_test, float) < thr))


def auc_pair(pos, neg) -> float:
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    s = np.r_[pos, neg]
    ranks = stats.rankdata(s)
    n1, n0 = len(pos), len(neg)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def auc_finite(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return auc_pair(pos[np.isfinite(pos)], neg[np.isfinite(neg)])


def pooled_decisions(seg):
    """逐资产 cal q95 阈值 → 池化 recall / FPR。seg 值为 anom（高=可疑）。"""
    hit = {k: 0 for k in FAM_SHORT}
    tot = {k: 0 for k in FAM_SHORT}
    flags_real = []
    for i in range(len(seg["cal"])):
        cal = np.asarray(seg["cal"][i], float)
        cal = cal[np.isfinite(cal)]
        thr = float(np.quantile(cal, 1 - ALPHA))
        real = np.asarray(seg["real"][i], float)
        flags_real.append(real > thr)
        for k2 in FAM_SHORT:
            f = np.asarray(seg[k2][i], float)
            hit[k2] += int(np.nansum(f > thr))
            tot[k2] += len(f)
    rec = {k: hit[k] / tot[k] for k in FAM_SHORT}
    return rec, float(np.mean(np.concatenate(flags_real)))


def main() -> None:
    t0 = time.time()
    print("=" * 66)
    print("E2 诊断：D1 时期漂移 | D2 长度错配 | D3 e 值域压缩 | A0-L 修复候选")
    print(f"协议：R1 三层切分 | α={ALPHA} | K_eval={K_EVAL} | epochs={EPOCHS}")
    print("=" * 66, flush=True)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    ACC = {v: {ro: defaultdict(list) for ro in READOUTS} for v, _, _ in VARIANTS}
    OPS = {v: defaultdict(list) for v, _, _ in VARIANTS}
    report = {}

    for ai, (asset, r) in enumerate(returns.items()):
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        cal_wins = span_windows(wins, *sp["cal"])
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        train_end = sp["cal"][0] * STEP
        forged = {fam: [forge(w, fam, seed=1000 + wi * 10 + fi)
                        for wi, w in enumerate(test_wins)]
                  for fi, fam in enumerate(FAMILIES)}
        tr_starts = np.linspace(0, train_end - WINDOW, 6).astype(int)
        tr_wins = [r[s:s + WINDOW] for s in tr_starts]
        print(f"\n[{asset}] cal{len(cal_wins)} test{len(test_wins)} "
              f"train_end={train_end}", flush=True)

        blk = {}
        for var, tlen, clen in VARIANTS:
            enc, head, tt = train_a0(r, train_end, tlen, clen,
                                     SEED + ai * 1000)
            rngc = np.random.default_rng(555 + ai * 10)
            s_cal = logits_var(enc, head, cal_wins)
            s_test = logits_var(enc, head, test_wins)
            s_cal700 = crop_avg(enc, head, cal_wins, 700, rngc)
            s_test700 = crop_avg(enc, head, test_wins, 700, rngc)
            s_tr = logits_var(enc, head, tr_wins)
            s_tr700 = crop_avg(enc, head, tr_wins, 700, rngc)

            base = 900000 + ai * 10000
            an_cal, _ = orbit_anatomy(enc, head, cal_wins, base)
            an_test, op_test = orbit_anatomy(enc, head, test_wins, base + 100)
            an_fam = {}
            for fi, fam in enumerate(FAMILIES):
                an_fam[fam], _ = orbit_anatomy(enc, head, forged[fam],
                                               base + 200 + fi * 100)
            s_fam = {fam: logits_var(enc, head, forged[fam])
                     for fam in FAMILIES}

            ks = stats.ks_2samp(s_cal, s_test)
            d_test = s_test - s_test700
            d_cal = s_cal - s_cal700
            d_tr = s_tr - s_tr700
            cert_sig = an_test["W_sig"] >= 1 / ALPHA
            cert_logit = an_test["W_logit"] >= 1 / ALPHA

            ACC[var]["direct"]["cal"].append(-s_cal)
            ACC[var]["direct"]["real"].append(-s_test)
            ACC[var]["Wsig"]["cal"].append(-an_cal["W_sig"])
            ACC[var]["Wsig"]["real"].append(-an_test["W_sig"])
            ACC[var]["Wlogit"]["cal"].append(-an_cal["W_logit"])
            ACC[var]["Wlogit"]["real"].append(-an_test["W_logit"])
            for fam in FAMILIES:
                k2 = fam[:2]
                ACC[var]["direct"][k2].append(-s_fam[fam])
                ACC[var]["Wsig"][k2].append(-an_fam[fam]["W_sig"])
                ACC[var]["Wlogit"][k2].append(-an_fam[fam]["W_logit"])
            for op, vals in op_test.items():
                OPS[var][op].append(float(np.mean(vals)))

            blk[var] = {
                "train_s": tt,
                "D1_drift": {"mean_cal": float(np.mean(s_cal)),
                             "mean_test": float(np.mean(s_test)),
                             "ks_p": float(ks.pvalue)},
                "D2_length": {"test_1000_700": float(np.mean(d_test)),
                              "cal_1000_700": float(np.mean(d_cal)),
                              "train_1000_700": float(np.mean(d_tr))},
                "D3_certify": {
                    "sig": f"{int(cert_sig.sum())}/{len(test_wins)}",
                    "logit": f"{int(cert_logit.sum())}/{len(test_wins)}",
                    "W_sig_med": float(np.median(an_test["W_sig"])),
                    "W_sig_max": float(np.max(an_test["W_sig"])),
                    "W_logit_med": float(np.median(an_test["W_logit"])),
                    "W_logit_max": float(np.max(an_test["W_logit"]))},
                "sep_gap": float(np.mean(an_test["s_x"]
                                         - an_test["s_orb_mean"])),
                "fpr_direct": fpr_at(s_cal, s_test),
                "fpr_direct_700": fpr_at(s_cal700, s_test700),
            }
            print(f"  [{var}] {tt}s 漂移Δ={np.mean(s_test) - np.mean(s_cal):+.2f}"
                  f"(KS p={ks.pvalue:.2f}) | 长度Δ test={np.mean(d_test):+.2f}"
                  f" cal={np.mean(d_cal):+.2f} train={np.mean(d_tr):+.2f} | "
                  f"认证 sig={int(cert_sig.sum())}/{len(test_wins)} "
                  f"logit={int(cert_logit.sum())}/{len(test_wins)} | "
                  f"W_logit med={np.median(an_test['W_logit']):.1f}"
                  f" max={np.max(an_test['W_logit']):.1f}", flush=True)
        report[asset] = blk

    # ---- 池化对比表 ----
    print("\n" + "=" * 66)
    print("池化对比（28 真实窗 + 5×28 伪造；cal q95 冻结阈值；anom 高=可疑）")
    print(f"{'读出':<10}{'N1':>6}{'N2':>6}{'N3':>6}{'N4':>6}{'N5':>6}"
          f"{'macro':>8}{'FPR':>7}{'AUC全':>7}{'认证real':>9}")
    print("-" * 78)
    pooled_out = {}
    for var, _, _ in VARIANTS:
        pooled_out[var] = {}
        for ro in READOUTS:
            seg = ACC[var][ro]
            rec, fpr = pooled_decisions(seg)
            real = np.concatenate(seg["real"])
            auc_all = auc_finite(np.concatenate(
                [np.concatenate(seg[k2]) for k2 in FAM_SHORT]), real)
            entry = {"recall": rec, "macro": float(np.mean(list(rec.values()))),
                     "fpr": fpr, "auc_all": auc_all,
                     "auc_family": {k2: auc_finite(np.concatenate(seg[k2]),
                                                   real)
                                    for k2 in FAM_SHORT}}
            cert_str = "-"
            if ro in ("Wsig", "Wlogit"):
                cert_real = float(np.mean(-real >= 1 / ALPHA))
                entry["e_rule"] = {
                    "certify_real": cert_real,
                    "recall": {k2: float(np.mean(
                        -np.concatenate(seg[k2]) < 1 / ALPHA))
                        for k2 in FAM_SHORT}}
                cert_str = f"{cert_real:.3f}"
            pooled_out[var][ro] = entry
            print(f"{var + '/' + ro:<10}"
                  + "".join(f"{rec[k]:>6.3f}" for k in FAM_SHORT)
                  + f"{entry['macro']:>8.3f}{fpr:>7.3f}{auc_all:>7.3f}"
                  f"{cert_str:>9}")
        print()

    print("逐算子代理均分（test 窗轨道 logit；越高=越像真实）")
    for var, _, _ in VARIANTS:
        m = {op: float(np.mean(v)) for op, v in OPS[var].items()}
        print(f"  [{var}] " + " ".join(f"{op}={m[op]:+.2f}"
                                       for op in ORBIT_NAMES))
        pooled_out[var]["op_mean"] = m

    res = {"seed": SEED, "epochs": EPOCHS, "k_eval": K_EVAL,
           "variants": {v: {"train_len": tl, "crop": cl}
                        for v, tl, cl in VARIANTS},
           "per_asset": report, "pooled": pooled_out,
           "elapsed_s": round(time.time() - t0, 1)}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, "e2_diag.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out_json}")


if __name__ == "__main__":
    main()
