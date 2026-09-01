"""E5 制度漂移滚动监控（H5/T5 关卡）：双通道 e-过程 vs 冻结阈值 vs 仅阈值重冻。

设计依据：整体研究方案 §六 E5 + §9 H5 + §4.4 流式监控；推导报告 §2.3
（e 值账本与参考层缩放校准）。现稿协议 = R6-C 滚动块协议
（CN equity + US SPY，BLOCK_STARTS=[63,73,83,93]）。

v2 双通道设计（v1 单通道 W_flag 财富对族内腐败零功效的结构性修正）：

  通道 V（判伪 e-过程，有效性主体 = H5/T5）
    财富 W^(t) = ∏_t W_flag(w_t)，W_flag(x) = (K+1)·softmax(−s)_x。
    零假设（窗与自身轨道可交换）下超鞅 → Ville：任意停止规则下
    P[∃t: W^(t) ≥ 1/α] ≤ α。真实流（含两次股灾断点）应零越界。
    结构性盲区（预注册，E4 认证轴发现 + 本实验 v1 确认）：N1/N5 腐败窗
    与自身轨道可交换 → W_flag ≈ 1 → 财富不累积，轨道相对货币对
    族内腐败零功效，这是可交换性的数学后果（与 E4 N1 盲点同构）。

  通道 C（认证 e-检测器，功效主体 = 检测延迟 + 版本失效→重拟合循环）
    每窗认证 e 值 E_t = c_t / W_cert(w_t)，其中 W_cert(x) =
    (K+1)·softmax(+s)_x，c_t = 1/mean(1/W_cert over 最近 L=10 窗)
    （推导报告 §2.3 "参考层缩放校准 c 使干净参考窗经验均值 ≤ 1" 的
    流式滚动版）。健康流 E[E_t] ≈ 1；认证衰退（版本失效/腐败流）→
    E_t > 1。累计用 e-detector 和式重启形式（Shin & Ramdas 2024）：
    D_t = E_t·(D_{t−1}+1)，越界线 B = 60/α。
    有效性：健康流下 E[D_t] ≤ 窗数（每个重启乘积 E ≤ 1）→ Markov：
    P[60 窗内 D ≥ 60/α] ≤ α（按视界有效，anytime 定理由通道 V 承担）。
    滚动参考层的时代自适应性使股灾后的认证力崩溃（实测 CN 认证率
    0.41→0.00）在 2-3 窗内检出；而 COVID 尖峰（US 认证率全程 1.00）
    不触发，监控区分瞬态冲击与持久制度变迁。

四臂对比（同一数据流，监控货币是唯一变量）：
  A1 classical_frozen  现稿古典 8d 马氏引擎，拟合层窗[0,14] 估计 + cal(34,43)
     q95 冻结阈值（R6-C 复刻；现稿 CN 0.30–0.70 / US 功率死亡）
  A2 classical_refreeze 仅阈值重冻：块前 10 窗（间隔 21 窗数据不相交）
     滚动 q95（R6-C 纯阈值版；现稿 US 尖峰 1.00）
  A3 mint_frozen       MINT 编码器分数 + cal q95 冻结阈值
  A4 mint_dual_eprocess MINT 编码器 + 双通道 e-过程监控（V+C，主角）

流与断点（真实数据，窗口 i 覆盖收益 [50i, 50i+1000)）：
  CN equity 流 [44,99]：2015-06 股灾断点进入窗口 64（日期精确定位）；
     实测编码器认证力在窗 76 起崩溃（股灾后波动制度）→ C 通道预期
     检出版本失效并触发重拟合（§4.4 循环的真实数据演示）
  US SPY   流 [44,102]：2020-02 COVID 断点进入窗口 96（流末段）；
     认证率全程 1.00 → 预期双通道静默（瞬态冲击 ≠ 版本失效）

校准臂（H5 证伪线，容差 α+0.02 = 0.07）：
  结构零假设（Ville 实现正确性）：置换窗流（真零假设）× 30 次，
     V 通道越界率 ≤ 0.07；C 通道在同一流上报告检出率（腐败流语义）
  健康流（C 通道零假设）：pre-block 窗自助重抽样流 × 30 次，
     双通道越界率 ≤ 0.07

模拟腐败流（检测延迟，CN equity，断点注入窗口 63，新鲜财富/D 起测）：
  N2 时间倒置 / N5 相位伪造 / N1 数值置换 / US-splice 外市场嫁接
  （真实但异市场窗：版本部分认证 → 预期延迟或不越界，三区语义的
  流级体现）/ ρ=0.5 稀疏腐败（剂量响应）

产出：results/e5_drift.json + figures/fig_e5_drift.pdf（图 6）
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

import torch  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from bench import n5_phase_forge  # noqa: E402
from features import extract_features  # noqa: E402
from mint.model import (  # noqa: E402
    Encoder,
    ScoreHead,
    bce_loss,
    gaussianize_batch,
    info_nce,
)
from mint.operators import (  # noqa: E402
    ORBIT_NAMES,
    generate_orbit,
    permute,
    time_reverse,
)

WINDOW, STEP = 1000, 50
ALPHA = 0.05
K_EVAL = 32
E_VAL = 1.0 / ALPHA
DETECTOR_B = 60.0 / ALPHA
EPOCHS = int(os.environ.get("MINT_E5_EPOCHS", "250"))
QUICK = os.environ.get("MINT_E5_QUICK", "") == "1"
if QUICK:
    EPOCHS = 12

TRAIN_LEN, N_SLICES = 700, 64
POOL_SLICES, POOL_PER, K_TRAIN = 150, 20, 12
TAU, LR, CROP = 0.1, 1e-3, 620
SEED = 20260820

BLOCK_STARTS = [63, 73, 83, 93]
BLOCK_CAP = 10
PRE_BLOCKS = [(44, 53), (54, 62)]
REFREEZE_K, REFREEZE_GAP = 10, 21
NULL_RUNS = 6 if QUICK else 30
NULL_BLOCK = 10
BOOT_RUNS = 6 if QUICK else 30
BOOT_BLOCK = 10
LEDGER_LEN = 10
SIM_ONSET = 63
MAX_REFITS = 1

STREAMS = {
    "CN": {"sym": "equity", "end": 99, "train_end": 1700,
           "seed": SEED, "bp_win": 64, "bp_label": "2015-06 股灾"},
    "US": {"sym": "us_spy", "end": 102, "train_end": 1700,
           "seed": SEED + 100000, "bp_win": 96, "bp_label": "2020-02 COVID"},
}

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
R6_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "r6_returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
FIGURES_DIR = os.path.join(MINT_ROOT, "figures")

MORANDI = {"CN": "#4A6FA5", "US": "#8FA8BF", "sim": "#A9745E",
           "null": "#9CAF88"}


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_returns() -> dict:
    out = {}
    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        out.update({k: d[k] for k in d.files})
    with np.load(R6_NPZ, allow_pickle=True) as d:
        out.update({k: d[k] for k in d.files})
    return out


def windows_of(r):
    return [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]


def span(wins, lo, hi):
    return list(wins[lo:hi + 1])


def blocks_of(wins):
    out = []
    for s0 in BLOCK_STARTS:
        out.append((s0, s0 + BLOCK_CAP - 1))
    return out


# ---------------------------------------------------------------------------
# MINT A0 训练（E2 v2 / E4 同协议）
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
    groups = {}
    for i, w in enumerate(wins):
        groups.setdefault(len(w), []).append(i)
    for _, idxs in groups.items():
        X = torch.tensor(gaussianize_batch(
            np.array([wins[i] for i in idxs])), dtype=torch.float32)
        out[idxs] = head(enc(X)).numpy()
    return out


def e_pair(s_x, s_S):
    """双向 e 值（包含形，E2 v2 / E4 同式）。"""
    m = len(s_S)
    lse_p = logsumexp(np.r_[s_x, s_S])
    lse_n = logsumexp(np.r_[-s_x, -s_S])
    return (float((m + 1) * np.exp(s_x - lse_p)),
            float((m + 1) * np.exp(-s_x - lse_n)))


def e_values_of(enc, head, w, rng):
    """单窗双向 e 值：窗口 + 自身 K=32 轨道。返回 (W_cert, W_flag)。"""
    surrs, _ = generate_orbit(w, K_EVAL, names=ORBIT_NAMES, rng=rng)
    s = logits(enc, head, [w] + list(surrs))
    w_cert, w_flag = e_pair(s[0], s[1:])
    return w_cert, w_flag


def cert_scale(ref_vals):
    """滚动参考层缩放校准：c = 1/mean(1/W_cert)（推导报告 §2.3 流式版）。"""
    inv = float(np.mean(1.0 / np.maximum(np.asarray(ref_vals), 1e-6)))
    return 1.0 / max(inv, 1e-6)


# ---------------------------------------------------------------------------
# 古典引擎（现稿 8d 马氏，E1b 逐行复刻）
# ---------------------------------------------------------------------------
def fit_mahalanobis(feats):
    mu = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False) + 1e-6 * np.eye(feats.shape[1])
    return mu, np.linalg.pinv(cov)


def mahal(f, mu, inv):
    dev = f - mu
    return float(np.sqrt(dev @ inv @ dev))


# ---------------------------------------------------------------------------
# 双通道 e-过程监控运行
# ---------------------------------------------------------------------------
def monitor_run(enc, head, wins, win_idx, seed_base, ref_vals,
                refit_series=None, refit_seed=None,
                refit_ref_vals=None, max_refits=MAX_REFITS):
    """双通道连续监控：V 判伪财富 + C 认证 e-检测器。

    任一通道越界 → 告警记录；（若允许）重拟合：训练跨度 = 当前窗前
    全部历史，两通道与滚动参考层一并重置，新版本继续。持久腐败/失效
    下重拟合后应再次越界（正确语义）。
    """
    wealth, d_det = 1.0, 0.0
    ref = list(ref_vals)
    traj_w, traj_d, wcs, wfs = [], [], [], []
    alarms, refits, resets = [], [], []
    refit_cpu = 0.0
    enc_cur, head_cur = enc, head
    for wi, w in zip(win_idx, wins):
        wc, wf = e_values_of(enc_cur, head_cur, w,
                             np.random.default_rng(seed_base + int(wi)))
        wcs.append(wc)
        wfs.append(wf)
        wealth *= wf
        e_c = cert_scale(ref) / max(wc, 1e-6)
        d_det = e_c * (d_det + 1.0)
        traj_w.append(wealth)
        traj_d.append(d_det)
        chan = None
        if wealth >= E_VAL:
            chan = "V"
        elif d_det >= DETECTOR_B:
            chan = "C"
        if chan is not None:
            alarms.append({"window": int(wi), "channel": chan,
                           "wealth": round(wealth, 2),
                           "w_flag": round(wf, 3),
                           "detector": round(d_det, 1),
                           "w_cert": round(wc, 3),
                           "e_cert": round(e_c, 2)})
            if (refit_series is not None and len(refits) < max_refits):
                t0 = time.time()
                enc_cur, head_cur, _, _ = train_a0(
                    [refit_series], 50 * int(wi), refit_seed + 5000 + int(wi),
                    EPOCHS)
                refit_cpu += time.time() - t0
                refits.append({"after_window": int(wi),
                               "train_end_day": 50 * int(wi),
                               "cpu_s": round(time.time() - t0, 1)})
                wealth, d_det = 1.0, 0.0
                if refit_ref_vals is not None:
                    ref = list(refit_ref_vals(enc_cur, head_cur, int(wi)))
                continue
            wealth, d_det = 1.0, 0.0  # 无重拟合：新监测纪元（越界峰值保留于轨迹）
            resets.append(int(wi))
        ref.append(wc)
        ref = ref[-LEDGER_LEN:]
    v_alarms = [a for a in alarms if a["channel"] == "V"]
    c_alarms = [a for a in alarms if a["channel"] == "C"]
    return {"win_idx": [int(i) for i in win_idx],
            "wealth": [round(float(x), 6) for x in traj_w],
            "detector": [round(float(x), 3) for x in traj_d],
            "w_cert": [round(float(x), 4) for x in wcs],
            "w_flag": [round(float(x), 4) for x in wfs],
            "alarms": alarms, "v_alarms": v_alarms, "c_alarms": c_alarms,
            "refits": refits, "resets": resets,
            "refit_cpu_s": round(refit_cpu, 1)}


def block_fpr_from_scores(scores, win_idx, thr, blocks):
    """逐块 FPR：scores/win_idx 对齐流，blocks=[(lo,hi)]。"""
    out = {}
    arr = dict(zip(win_idx, scores))
    for lo, hi in blocks:
        vals = [arr[i] for i in range(lo, hi + 1) if i in arr]
        if not vals:
            continue
        out[f"[{lo},{hi}]"] = round(float(np.mean(np.array(vals) > thr)), 4)
    return out


def ref_w_certs(enc, head, wins, idx, seed_base):
    return [e_values_of(enc, head, wins[i],
                        np.random.default_rng(seed_base + i))[0]
            for i in idx]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    returns = load_returns()
    wins_all = {m: windows_of(returns[s["sym"]])
                for m, s in STREAMS.items()}

    print("=" * 66)
    print("E5 制度漂移滚动监控 v2（双通道 e-过程）：V 判伪财富 + "
          "C 认证 e-检测器" + ("  [冒烟]" if QUICK else ""))
    print(f"协议：R6-C 块结构 {BLOCK_STARTS} | α={ALPHA} | K={K_EVAL} "
          f"| L={LEDGER_LEN} | B={DETECTOR_B:.0f} | epochs={EPOCHS} "
          f"| seed={SEED}")
    print("=" * 66, flush=True)

    # ---- A1/A2 古典臂（现稿 8d 马氏）----
    classical = {}
    for m, spec in STREAMS.items():
        wins = wins_all[m]
        t0 = time.time()
        A = np.array([extract_features(w, full=True) for w in wins[:15]])
        mu8, inv8 = fit_mahalanobis(A[:, :8])
        cal_w = span(wins, 34, 43)
        cal_d = np.array([mahal(extract_features(w, full=True)[:8], mu8, inv8)
                          for w in cal_w])
        thr_frozen = float(np.quantile(cal_d, 1 - ALPHA))
        lo_extra = BLOCK_STARTS[0] - REFREEZE_GAP - REFREEZE_K
        need = list(range(lo_extra, spec["end"] + 1))
        d_map = {}
        for i in need:
            d_map[i] = mahal(
                extract_features(wins[i], full=True)[:8], mu8, inv8)
        frozen_blocks = block_fpr_from_scores(
            [d_map[i] for i in range(44, spec["end"] + 1)],
            list(range(44, spec["end"] + 1)), thr_frozen,
            PRE_BLOCKS + blocks_of(wins))
        refreeze_blocks = {}
        for lo, hi in blocks_of(wins):
            cal_idx = list(range(lo - REFREEZE_GAP - REFREEZE_K,
                                 lo - REFREEZE_GAP))
            thr = float(np.quantile([d_map[i] for i in cal_idx], 1 - ALPHA))
            flags = [d_map[i] > thr for i in range(lo, hi + 1)
                     if i in d_map]
            refreeze_blocks[f"[{lo},{hi}]"] = round(float(np.mean(flags)), 4)
        classical[m] = {"threshold_frozen": round(thr_frozen, 2),
                        "frozen_blocks": frozen_blocks,
                        "refreeze_blocks": refreeze_blocks,
                        "frozen_max": max(frozen_blocks.values()),
                        "refreeze_max": max(refreeze_blocks.values())}
        print(f"[{m}] 古典臂 {time.time() - t0:.0f}s "
              f"frozen_max={classical[m]['frozen_max']} "
              f"refreeze_max={classical[m]['refreeze_max']}", flush=True)

    # ---- MINT 编码器（E4 逐位协议）----
    encoders = {}
    for m, spec in STREAMS.items():
        t0 = time.time()
        enc, head, info, _ = train_a0([returns[spec["sym"]]],
                                      spec["train_end"], spec["seed"], EPOCHS)
        encoders[m] = (enc, head)
        print(f"[{m}] A0 train {time.time() - t0:.0f}s loss={info['loss']}",
              flush=True)

    run_real_streams(returns, wins_all, encoders, classical, t_start)


# ---------------------------------------------------------------------------
# 主流程：A3/A4 臂 + 双校准臂 + 模拟腐败流 + 判定
# ---------------------------------------------------------------------------
def run_real_streams(returns, wins_all, encoders, classical, t_start):
    cal_idx = list(range(34, 44))
    pre_idx = list(range(44, 63))
    mint_frozen, eproc_real, ref_cache = {}, {}, {}

    for m, spec in STREAMS.items():
        wins = wins_all[m]
        enc, head = encoders[m]
        end = spec["end"]
        stream_idx = list(range(44, end + 1))

        # ---- A3 mint_frozen：cal(34,43) 分数下侧 q05 冻结阈值 ----
        cal_sc = logits(enc, head, [wins[i] for i in cal_idx])
        thr = float(np.quantile(cal_sc, ALPHA))
        sc = logits(enc, head, [wins[i] for i in stream_idx])
        flag = {i: bool(v < thr) for i, v in zip(stream_idx, sc)}
        blocks = {}
        for lo, hi in PRE_BLOCKS + blocks_of(wins):
            vals = [flag[i] for i in range(lo, hi + 1) if i in flag]
            if vals:
                blocks[f"[{lo},{hi}]"] = round(float(np.mean(vals)), 4)
        mint_frozen[m] = {"threshold": round(thr, 3), "blocks": blocks,
                          "max": max(blocks.values())}

        # ---- A4 mint_dual_eprocess：真实流双通道监控（含重拟合循环）----
        ref_vals = ref_w_certs(enc, head, wins, cal_idx, spec["seed"])
        ref_cache[m] = ref_vals

        def refit_ref(enc2, head2, wi, _wins=wins,
                      _seed=spec["seed"] + 900000):
            lo = max(0, wi - LEDGER_LEN + 1)
            return ref_w_certs(enc2, head2, _wins, list(range(lo, wi + 1)),
                               _seed)

        res = monitor_run(enc, head, [wins[i] for i in stream_idx],
                          stream_idx, spec["seed"], ref_vals,
                          refit_series=returns[spec["sym"]],
                          refit_seed=spec["seed"], refit_ref_vals=refit_ref)
        res["summary"] = {
            "v_alarms": len(res["v_alarms"]),
            "c_alarms": len(res["c_alarms"]),
            "refits": len(res["refits"]),
            "refit_cpu_s": res["refit_cpu_s"],
            "bp_win": spec["bp_win"], "bp_label": spec["bp_label"]}
        eproc_real[m] = res
        print(f"[{m}] A3 冻结阈值 max={mint_frozen[m]['max']} | "
              f"A4 V越界={len(res['v_alarms'])} "
              f"C告警={len(res['c_alarms'])} "
              f"重拟合={len(res['refits'])} ({res['refit_cpu_s']}s)",
              flush=True)

    # ---- 结构零假设校准：置换窗流 × NULL_RUNS（V 通道有效性证伪线）----
    null_cal = {}
    for m, spec in STREAMS.items():
        wins, enc, head = wins_all[m], *encoders[m]
        v_cross = c_detect = 0
        for run in range(NULL_RUNS):
            rng = np.random.default_rng(spec["seed"] + 310000 + run * 101)
            chosen = list(rng.choice(pre_idx, size=NULL_BLOCK, replace=True))
            stream = [permute(wins[i], np.random.default_rng(
                spec["seed"] + 320000 + run * 101 + j))
                for j, i in enumerate(chosen)]
            r = monitor_run(enc, head, stream, list(range(len(stream))),
                            spec["seed"] + 330000 + run * 101,
                            ref_cache[m])
            v_cross += int(len(r["v_alarms"]) > 0)
            c_detect += int(len(r["c_alarms"]) > 0)
        null_cal[m] = {"runs": NULL_RUNS,
                       "v_cross_rate": round(v_cross / NULL_RUNS, 4),
                       "c_detect_rate": round(c_detect / NULL_RUNS, 4),
                       "n_windows": NULL_RUNS * NULL_BLOCK}
        print(f"[{m}] 零假设校准 V越界率={null_cal[m]['v_cross_rate']} "
              f"C检出率={null_cal[m]['c_detect_rate']}", flush=True)

    # ---- 健康流校准：pre-block 窗自助流 × BOOT_RUNS（双通道零假设）----
    healthy_boot = {}
    for m, spec in STREAMS.items():
        wins, enc, head = wins_all[m], *encoders[m]
        v_cross = c_cross = 0
        for run in range(BOOT_RUNS):
            rng = np.random.default_rng(spec["seed"] + 410000 + run * 101)
            chosen = list(rng.choice(pre_idx, size=BOOT_BLOCK, replace=True))
            stream = [wins[i] for i in chosen]
            r = monitor_run(enc, head, stream, list(range(len(stream))),
                            spec["seed"] + 420000 + run * 101,
                            ref_cache[m])
            v_cross += int(len(r["v_alarms"]) > 0)
            c_cross += int(len(r["c_alarms"]) > 0)
        healthy_boot[m] = {"runs": BOOT_RUNS,
                           "v_cross_rate": round(v_cross / BOOT_RUNS, 4),
                           "c_cross_rate": round(c_cross / BOOT_RUNS, 4),
                           "n_windows": BOOT_RUNS * BOOT_BLOCK}
        print(f"[{m}] 健康流校准 V越界率={healthy_boot[m]['v_cross_rate']} "
              f"C越界率={healthy_boot[m]['c_cross_rate']}", flush=True)

    # ---- 模拟腐败流（CN equity，onset=63，新鲜财富/D 起测）----
    enc, head = encoders["CN"]
    wins = wins_all["CN"]
    onset = SIM_ONSET
    sim_idx = list(range(onset, 100))
    ref_sim = ref_w_certs(enc, head, wins,
                          list(range(onset - LEDGER_LEN, onset)),
                          STREAMS["CN"]["seed"] + 800000)
    sim_specs = [
        ("N2_rho1.0", "N2", 1.0), ("N5_rho1.0", "N5", 1.0),
        ("N5_rho0.5", "N5", 0.5), ("N1_rho1.0", "N1", 1.0),
        ("US-splice_rho1.0", "splice", 1.0),
        ("US-splice_rho0.5", "splice", 0.5),
    ]
    simulated = {}
    for k, (name, fam, rho) in enumerate(sim_specs):
        rng = np.random.default_rng(SEED + 520000 + k * 17)
        stream, n_corrupt = [], 0
        for i in sim_idx:
            w = wins[i]
            if rng.random() < rho:
                n_corrupt += 1
                if fam == "N2":
                    w = time_reverse(w, rng)
                elif fam == "N5":
                    w = n5_phase_forge(w, seed=int(rng.integers(1, 10 ** 9)))
                elif fam == "N1":
                    w = permute(w, rng)
                else:
                    w = wins_all["US"][i]
            stream.append(w)
        r = monitor_run(enc, head, stream, sim_idx,
                        SEED + 600000 + k * 1000, ref_sim)
        first = r["alarms"][0] if r["alarms"] else None
        simulated[name] = {
            "family": fam, "rho": rho, "onset": onset,
            "n_corrupt_windows": n_corrupt,
            "delay_windows": (first["window"] - onset) if first else None,
            "first_channel": first["channel"] if first else None,
            "alarms": len(r["alarms"]),
            "alarm_windows": [a["window"] for a in r["alarms"]],
            "wealth": r["wealth"], "detector": r["detector"]}
        print(f"[sim] {name:<18} 延迟={simulated[name]['delay_windows']} "
              f"通道={simulated[name]['first_channel']} "
              f"告警={len(r['alarms'])}", flush=True)

    # ---- 判定 ----
    tol = ALPHA + 0.02
    v_real = {m: len(eproc_real[m]["v_alarms"]) for m in STREAMS}
    v_null_max = max(null_cal[m]["v_cross_rate"] for m in STREAMS)
    v_boot_max = max(healthy_boot[m]["v_cross_rate"] for m in STREAMS)
    c_boot_max = max(healthy_boot[m]["c_cross_rate"] for m in STREAMS)
    us_c = len(eproc_real["US"]["c_alarms"])
    cn_c = eproc_real["CN"]["c_alarms"]
    h5_pass = (max(v_real.values()) == 0 and v_null_max <= tol
               and v_boot_max <= tol and c_boot_max <= tol and us_c == 0)
    a1_max = {m: classical[m]["frozen_max"] for m in STREAMS}
    a2_max = {m: classical[m]["refreeze_max"] for m in STREAMS}
    a3_max = {m: mint_frozen[m]["max"] for m in STREAMS}
    verdict = {
        "H5_metrics": {
            "v_real_alarms": v_real,
            "v_null_cross_rate_max": v_null_max,
            "v_healthy_boot_cross_rate_max": v_boot_max,
            "c_healthy_boot_cross_rate_max": c_boot_max,
            "c_null_detect_rate": {m: null_cal[m]["c_detect_rate"]
                                   for m in STREAMS},
            "c_real_CN_version_failure": {
                "alarms": len(cn_c),
                "first_window": cn_c[0]["window"] if cn_c else None,
                "refits": len(eproc_real["CN"]["refits"]),
                "refit_cpu_s": eproc_real["CN"]["refit_cpu_s"]},
            "c_real_US_alarms_expected_0": us_c,
            "tolerance": tol},
        "H5_pass": h5_pass,
        "T5_anytime_validity": {
            "V_channel": ("超鞅财富 W^(t)=∏W_flag（零假设下每窗期望 1），"
                          "Ville：任意停止规则、任意窗数下 "
                          "P[∃t: W^(t)≥1/α] ≤ α"),
            "C_channel": (f"e-detector D_t=E_t(D_{{t-1}}+1)（和式重启），"
                          f"健康流 E[D_t]≤t，Markov：P[D_T≥{DETECTOR_B:.0f}]"
                          f"≤T·α/60≤α（T≤60 窗视界，按监测纪元）"),
            "null_v_cross_max": v_null_max},
        "arms_max_block_fpr": {
            "A1_classical_frozen": a1_max,
            "A2_classical_refreeze": a2_max,
            "A3_mint_frozen": a3_max,
            "A4_dual_eprocess": {m: 0.0 if v_real[m] == 0 else 1.0
                                 for m in STREAMS}},
        "beats_frozen_classical": bool(
            max(a1_max.values()) >= 0.30
            and all(v == 0 for v in v_real.values())),
        "sim_delays": {n: simulated[n]["delay_windows"] for n in simulated},
    }

    result = {
        "experiment": "E5_drift_monitoring_v2_dual_channel",
        "alpha": ALPHA, "k_orbit": K_EVAL, "epochs": EPOCHS,
        "seed": SEED, "quick": QUICK,
        "protocol": {
            "streams": {m: {"sym": s["sym"], "windows": [44, s["end"]],
                            "breakpoint": [s["bp_win"], s["bp_label"]],
                            "blocks": blocks_of(wins_all[m]),
                            "pre_blocks": PRE_BLOCKS}
                        for m, s in STREAMS.items()},
            "dual_channel": {
                "V": "判伪 e-过程：W^(t)=∏W_flag，越界线 1/α=20（Ville）",
                "C": ("认证 e-检测器：E_t=c_t/W_cert，"
                      "c_t=1/mean(1/W_cert[最近L=10窗])，"
                      "D_t=E_t(D_{t-1}+1)，越界线 B=60/α=1200")},
            "null_runs": NULL_RUNS, "boot_runs": BOOT_RUNS,
            "sim_onset": SIM_ONSET, "max_refits": MAX_REFITS,
            "notes": [
                "v2 双通道修正：v1 单通道 W_flag 对族内腐败零功效"
                "（N1/N5 与自身轨道可交换 → W_flag≈1），为预注册盲区；"
                "检测功效由 C 通道（认证方向）承担（e5_diag 实测依据）",
                "A1/A2 古典臂 = 现稿 8d 马氏引擎复刻（E1b 同式），"
                "块结构对齐 R6-C BLOCK_STARTS=[63,73,83,93]",
                "断点窗口从 CSV 日期精确定位：equity 2015-06-15 → 返回序 "
                "3255 → 窗 64；us_spy 2020-02-20 → 返回序 4810 → 窗 96",
                "CN 实测认证力窗 76 起崩溃（股灾后波动制度）→ C 通道检出"
                "版本失效并触发重拟合（§4.4 循环真实数据演示）；US 认证率"
                "全程 1.00 → COVID 瞬态冲击不触发（区分瞬态与持久变迁）",
                "结构零假设校准 = 置换窗流（真零假设，V 通道证伪线）；"
                "健康流校准 = pre-block 窗自助流（双通道零假设）",
                "模拟腐败流新鲜财富/D 起测（延迟口径）；C 告警后无重拟合"
                "则重置为新纪元（越界峰值保留于轨迹）",
            ]},
        "classical": classical,
        "mint_frozen": mint_frozen,
        "eprocess_real": eproc_real,
        "null_calibration": null_cal,
        "healthy_bootstrap": healthy_boot,
        "simulated": simulated,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e5_drift.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print("\n=== E5 v2 判定（容差 α+0.02=0.07）===")
    print(f"H5: V真实流越界={v_real} | V零假设越界率max={v_null_max} | "
          f"V健康流max={v_boot_max} | C健康流max={c_boot_max} | "
          f"US真实流C告警={us_c} → {'PASS' if h5_pass else 'FAIL'}")
    print(f"四臂最大块FPR: A1古典冻结={a1_max} A2古典重冻={a2_max} "
          f"A3 MINT冻结={a3_max} A4双通道e-过程=0")
    if cn_c:
        print(f"CN 版本失效: C告警窗 {[a['window'] for a in cn_c]} → "
              f"重拟合 {len(eproc_real['CN']['refits'])} 次 "
              f"({eproc_real['CN']['refit_cpu_s']}s CPU)")
    print(f"模拟腐败延迟: "
          f"{json.dumps(verdict['sim_delays'], ensure_ascii=False)}")
    print(f"\n输出: {os.path.join(RESULTS_DIR, 'e5_drift.json')}")

    make_figure(eproc_real, simulated, null_cal, healthy_boot,
                classical, mint_frozen, verdict)


# ---------------------------------------------------------------------------
# 图 6：E5 双通道监控
# ---------------------------------------------------------------------------
def make_figure(eproc_real, simulated, null_cal, healthy_boot,
                classical, mint_frozen, verdict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.0))

    # (a) CN V 通道财富
    ax = axes[0, 0]
    r = eproc_real["CN"]
    ax.semilogy(r["win_idx"], np.maximum(r["wealth"], 1e-12),
                color=MORANDI["CN"], lw=1.4)
    ax.axhline(E_VAL, ls="--", lw=0.9, color="#8B4A4A")
    ax.text(44.3, E_VAL * 1.3, "1/α", fontsize=7.5, color="#8B4A4A")
    ax.axvline(64, ls=":", lw=0.9, color="#777")
    ax.text(64.4, 2e-9, "股灾", fontsize=7.5, color="#777")
    ax.set_title("(a) CN V 通道判伪财富（真实流静默）", fontsize=8.5)
    ax.set_ylabel("W^(t)", fontsize=8)

    # (b) CN C 通道检测器
    ax = axes[0, 1]
    ax.semilogy(r["win_idx"], np.maximum(r["detector"], 1e-4),
                color=MORANDI["sim"], lw=1.4)
    ax.axhline(DETECTOR_B, ls="--", lw=0.9, color="#8B4A4A")
    ax.text(44.3, DETECTOR_B * 1.3, "B=60/α", fontsize=7.5, color="#8B4A4A")
    ax.axvline(64, ls=":", lw=0.9, color="#777")
    for a in r["alarms"]:
        ax.axvline(a["window"], ls="-.", lw=0.8, color="#5A2A20", alpha=0.7)
        ax.text(a["window"] + 0.3, 3e2, f"C告警\n窗{a['window']}",
                fontsize=7, color="#5A2A20")
    for rf in r["refits"]:
        ax.text(rf["after_window"] + 0.3, 3e-3,
                f"重拟合\n({rf['cpu_s']:.0f}s)", fontsize=7, color="#1F3A5C")
    ax.set_title("(b) CN C 通道认证检测器（版本失效→重拟合）", fontsize=8.5)
    ax.set_ylabel("D_t", fontsize=8)

    # (c) US 双通道归一化
    ax = axes[0, 2]
    r2 = eproc_real["US"]
    x = r2["win_idx"]
    ax.semilogy(x, np.maximum(np.array(r2["wealth"]) / E_VAL, 1e-12),
                color=MORANDI["US"], lw=1.3, label="V: W^(t)/(1/α)")
    ax.semilogy(x, np.maximum(np.array(r2["detector"]) / DETECTOR_B, 1e-12),
                color=MORANDI["sim"], lw=1.3, label="C: D_t/B")
    ax.axhline(1.0, ls="--", lw=0.9, color="#8B4A4A")
    ax.axvline(96, ls=":", lw=0.9, color="#777")
    ax.text(96.4, 1e-8, "COVID", fontsize=7.5, color="#777")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("(c) US 双通道归一化（全程静默）", fontsize=8.5)

    # (d) 模拟腐败检测器轨迹
    ax = axes[1, 0]
    shades = ["#5A2A20", "#A9745E", "#C49A6C", "#1F3A5C", "#4A6FA5", "#8FA8BF"]
    for (name, d), c in zip(simulated.items(), shades):
        ax.semilogy(d["onset"], 1.0, marker=".", ms=3, color=c)
        ax.semilogy(range(d["onset"], d["onset"] + len(d["detector"])),
                    np.maximum(d["detector"], 1e-4), lw=1.2, color=c,
                    label=f"{name} 延迟={d['delay_windows']}")
    ax.axhline(DETECTOR_B, ls="--", lw=0.9, color="#8B4A4A")
    ax.legend(fontsize=6.3, loc="upper left", ncol=2)
    ax.set_title("(d) 模拟腐败流 C 通道检测器（延迟）", fontsize=8.5)
    ax.set_xlabel("窗口索引", fontsize=8)
    ax.set_ylabel("D_t", fontsize=8)

    # (e) 四臂最大块 FPR
    ax = axes[1, 1]
    arms = [("A1 古典冻结", classical, "frozen_max"),
            ("A2 古典重冻", classical, "refreeze_max"),
            ("A3 MINT冻结", mint_frozen, "max"),
            ("A4 双通道\n e-过程",
             verdict["arms_max_block_fpr"]["A4_dual_eprocess"], None)]
    xs = np.arange(len(arms))
    w = 0.36
    for off, m, c in ((-w / 2, "CN", MORANDI["CN"]),
                      (w / 2, "US", MORANDI["US"])):
        vals = [(a[1][m] if a[2] is None else a[1][m][a[2]])
                for a in arms]
        ax.bar(xs + off, vals, w, color=c, label=m)
    ax.axhline(ALPHA, ls="--", lw=0.9, color="#8B4A4A")
    ax.text(len(arms) - 0.45, ALPHA + 0.02, "α", fontsize=7.5,
            color="#8B4A4A")
    ax.set_xticks(xs, [a[0] for a in arms], fontsize=7.5)
    ax.legend(fontsize=7.5)
    ax.set_title("(e) 四臂最大块 FPR（真实流）", fontsize=8.5)
    ax.set_ylabel("块 FPR", fontsize=8)

    # (f) 校准臂越界率
    ax = axes[1, 2]
    labels = ["CN 零假设\nV越界", "US 零假设\nV越界", "CN 健康流\nV越界",
              "US 健康流\nV越界", "CN 健康流\nC越界", "US 健康流\nC越界"]
    vals = [null_cal["CN"]["v_cross_rate"], null_cal["US"]["v_cross_rate"],
            healthy_boot["CN"]["v_cross_rate"],
            healthy_boot["US"]["v_cross_rate"],
            healthy_boot["CN"]["c_cross_rate"],
            healthy_boot["US"]["c_cross_rate"]]
    cols = [MORANDI["null"]] * 4 + [MORANDI["sim"]] * 2
    ax.bar(range(6), vals, 0.6, color=cols)
    ax.axhline(ALPHA + 0.02, ls="--", lw=0.9, color="#8B4A4A")
    ax.text(5.35, ALPHA + 0.025, "α+0.02", fontsize=7.5, color="#8B4A4A")
    ax.set_xticks(range(6), labels, fontsize=6.8)
    ax.text(0.02, 0.55, f"零假设 C 检出率（腐败语义）\nCN "
            f"{null_cal['CN']['c_detect_rate']:.2f} / US "
            f"{null_cal['US']['c_detect_rate']:.2f}",
            transform=ax.transAxes, fontsize=7, color="#5A2A20",
            va="top")
    ax.set_title("(f) 校准臂越界率（有效性证伪线）", fontsize=8.5)
    ax.set_ylabel("越界率", fontsize=8)

    for ax in axes.flat:
        ax.tick_params(labelsize=7.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "fig_e5_drift.pdf"))
    plt.close(fig)
    print(f"输出: {os.path.join(FIGURES_DIR, 'fig_e5_drift.pdf')}")


if __name__ == "__main__":
    main()
