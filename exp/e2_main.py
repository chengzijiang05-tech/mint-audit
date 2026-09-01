"""E2 主实验 + E7 消融矩阵（G2 关卡）：五资产 × N1–N5 全伪造族。

设计依据：整体研究方案 §六 E2/E7 + §4.5 消融矩阵 + §9 G2 关卡。

协议（R1 三层切分原样，SPLIT_SPEC 与 e1b_collapse / r1_checkpoint_audit
逐位一致）：
  拟合层    前 n_anchor 窗（马氏基线拟合）
  校准层  cal 跨度真实窗（各方法异常分 q95 阈值冻结）
  测试层  test 跨度（cap 限窗）真实窗 + N1–N5 伪造，
          种子 1000+wi*10+fi（R1 审计协议）
  MINT 训练跨度：cal 层起点之前全部历史（随机切片反记忆），
  三层时间严格有序 train < cal < test。

评分方向约定（全脚本统一）：异常分 = 判伪造倾向，越高越可疑。
  e 值法   anom = -W（W 大 = 相对自身轨道极端 = 结构真实）
  直接分数 anom = -s_θ（s 高 = 真实窗式）
  马氏引擎 anom = d(x)
  W 为 logit 域 exp 权重（v2；E2 诊断 D3：sigmoid 域饱和使真实窗
  认证率塌缩为 0，logit 域 equity 0/8 → 5/8）

消融矩阵（E7 可执行行；类型化门控/词典读出为系统层组件，留 E6/E8）：
  A0  完整 MINT        轨道负样本（5 算子池）+ 秩归一化 + e 值
  A1  - 轨道负样本     训练负样本换随机增强（TS2Vec 式）
  A2  - 多算子并集     训练轨道仅 IAAFT
  A3  - 秩归一化       输入 z-score 原始收益（边际形状可访问）
  A4  - e 值层         A0 编码器 + 直接分数 + cal 阈值
  A5  - 学习表征       Φ_C 15 维马氏引擎（现稿引擎）
  A5b 古典 8 维块      现稿古典块引擎（E1b 主角，参照）

显著性：DeLong 配对 z 检验（逐族 + 全族池化 AUC）+ McNemar 精确
检验（池化判定）。MINT e 规则（免校准，T3 语义）：W ≥ 1/α 认证
为真，未认证判可疑；报告真实窗认证率与逐族 recall。

产出：results/e2_main.json + figures/fig_e2_main.pdf
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
from features import extract_features  # noqa: E402
from mint.model import (  # noqa: E402
    Encoder,
    ScoreHead,
    bce_loss,
    gaussianize_batch,
    info_nce,
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

TRAIN_LEN, N_SLICES = 700, 64
POOL_SLICES, POOL_PER, K_TRAIN = 150, 20, 12
K_EVAL = 32
EPOCHS = 250
TAU, LR, CROP = 0.1, 1e-3, 620
SEED = 20260820

CONFIGS = {
    "A0": {"neg": "orbit", "pool": "five", "rank": True},
    "A1": {"neg": "augment", "pool": None, "rank": True},
    "A2": {"neg": "orbit", "pool": "iaaft", "rank": True},
    "A3": {"neg": "orbit", "pool": "five", "rank": False},
}
TRAINABLE = list(CONFIGS)
EV_METHODS = TRAINABLE                      # e 值评分的方法
ALL_METHODS = TRAINABLE + ["A4", "A5", "A5b"]
METHOD_DESC = {
    "A0": "完整 MINT", "A1": "-轨道负样本(TS2Vec式)",
    "A2": "-多算子并集(仅IAAFT)", "A3": "-秩归一化",
    "A4": "-e值层(直接分数)", "A5": "-学习表征(15d马氏)",
    "A5b": "古典8d马氏(参照)",
}

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
FIGURES_DIR = os.path.join(MINT_ROOT, "figures")


# ---------------------------------------------------------------------------
# 数据、伪造与增强
# ---------------------------------------------------------------------------
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


def crop_noise(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    start = int(rng.integers(0, len(w) - CROP + 1))
    out = w[start:start + CROP].copy()
    return out + 0.02 * out.std() * rng.standard_normal(len(out))


def augment_negative(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = w.copy()
    op = rng.integers(0, 4)
    if op == 0:
        out = np.roll(out, int(rng.integers(-150, 151)))
    elif op == 1:
        out = out * rng.uniform(0.7, 1.3)
    elif op == 2:
        out = out + 0.15 * out.std() * rng.standard_normal(len(out))
    else:
        lo = int(rng.integers(0, 120))
        hi = len(out) - int(rng.integers(0, 120))
        seg = out[lo:hi]
        out = np.concatenate([
            np.full(lo, seg[0]), seg, np.full(len(out) - hi, seg[-1])])
    return out


# ---------------------------------------------------------------------------
# 训练（A0–A3 共用循环，负样本源 / 轨道算子集 / 预处理三参数化）
# ---------------------------------------------------------------------------
def prep(X: np.ndarray, rank: bool) -> np.ndarray:
    """行级预处理：秩归一化（A3 之外）或 z-score（A3，保留边际形状）。"""
    if rank:
        return gaussianize_batch(X)
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=1, keepdims=True)) / (
        X.std(axis=1, keepdims=True) + 1e-12)


def build_pool(r, train_end, names, rng, n_slices, per):
    starts = rng.integers(0, train_end - TRAIN_LEN + 1, size=n_slices)
    rows = []
    for s in starts:
        w = r[int(s):int(s) + TRAIN_LEN]
        surrs, _ = generate_orbit(w, per, names=names, rng=rng)
        rows.append(surrs)
    return np.vstack(rows)


def train_model(r, train_end, cfg, seed, epochs, pool_slices, pool_per):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, head = Encoder(), ScoreHead()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rank = cfg["rank"]

    pool = None
    if cfg["neg"] == "orbit":
        names = ORBIT_NAMES if cfg["pool"] == "five" else ["iaaft"]
        pool = build_pool(r, train_end, names,
                          np.random.default_rng(seed + 77),
                          pool_slices, pool_per)
        pool = prep(pool, rank)

    t0 = time.time()
    last = {}
    for epoch in range(epochs):
        starts = rng.integers(0, train_end - TRAIN_LEN + 1, size=N_SLICES)
        wins = [r[s:s + TRAIN_LEN] for s in starts]
        X = torch.tensor(prep(np.array(wins), rank), dtype=torch.float32)
        X_pos = torch.tensor(
            prep(np.array([crop_noise(w, rng) for w in wins]), rank),
            dtype=torch.float32)
        if cfg["neg"] == "orbit":
            need = N_SLICES * K_TRAIN
            idx = rng.choice(len(pool), size=need,
                             replace=len(pool) < need)
            N = torch.tensor(pool[idx], dtype=torch.float32)
        else:
            neg = np.array([augment_negative(w, rng)
                            for w in wins for _ in range(K_TRAIN)])
            N = torch.tensor(prep(neg, rank), dtype=torch.float32)

        opt.zero_grad()
        z, z_pos, z_neg = enc(X), enc(X_pos), enc(N)
        loss_nce = info_nce(z, z_pos, z_neg, tau=TAU)
        s_pos, s_neg = head(z), head(z_neg)
        loss_bce = bce_loss(s_pos, s_neg, pos_weight=float(K_TRAIN))
        loss = loss_nce + loss_bce
        loss.backward()
        opt.step()
        sched.step()
        last = {"loss": round(loss.item(), 4),
                "acc_pos": round((s_pos > 0).float().mean().item(), 3),
                "acc_neg": round((s_neg < 0).float().mean().item(), 3)}
    enc.eval(), head.eval()
    return enc, head, last, round(time.time() - t0, 1)


# ---------------------------------------------------------------------------
# 评分（变长窗口按长度分批）
# ---------------------------------------------------------------------------
@torch.no_grad()
def logits_var(enc, head, wins, rank):
    out = np.empty(len(wins))
    groups = defaultdict(list)
    for i, w in enumerate(wins):
        groups[len(w)].append(i)
    for _, idxs in groups.items():
        X = torch.tensor(prep(np.array([wins[i] for i in idxs]), rank),
                         dtype=torch.float32)
        out[idxs] = head(enc(X)).numpy()
    return out


def e_values(enc, head, wins, rank, k, seed):
    """逐窗轨道 e 值 W（K 条轨道，logit 域 exp 权重）。

    W = (K+1)·e^{s(x)} / Σ_{i=0}^{K} e^{s(z_i)} = (K+1)·softmax(s)_x，
    可交换性下 E[W]=1 与非负权重域无关；sigmoid 域在大 logit 差处
    饱和（认证需 ±4 分离，logit 域只需 ±2，E2 诊断 D3：equity
    真实窗认证 0/8 → 5/8）。logsumexp 数值稳定，W 有界 [0, K+1]。"""
    W = np.empty(len(wins))
    for i, w in enumerate(wins):
        surrs, _ = generate_orbit(w, k, rng=np.random.default_rng(seed + i))
        s = logits_var(enc, head, [w] + list(surrs), rank)
        m = float(np.max(s))
        lse = m + float(np.log(np.sum(np.exp(s - m))))
        W[i] = float((k + 1) * np.exp(float(s[0]) - lse))
    return W


def mahal_fit(anchors, full):
    A = np.array([extract_features(w, full=full) for w in anchors])
    A = A[np.isfinite(A).all(axis=1)]
    mu = A.mean(axis=0)
    inv = np.linalg.pinv(np.cov(A, rowvar=False)
                         + 1e-6 * np.eye(A.shape[1]))
    return mu, inv


def mahal_scores(wins, mu, inv, full):
    out = np.empty(len(wins))
    for i, w in enumerate(wins):
        f = extract_features(w, full=full)
        if not np.isfinite(f).all():
            out[i] = np.nan
            continue
        dev = f - mu
        out[i] = float(np.sqrt(dev @ inv @ dev))
    return out


# ---------------------------------------------------------------------------
# 指标与统计
# ---------------------------------------------------------------------------
def auc_pair(pos, neg) -> float:
    """AUC = P(anom_forge > anom_real)（Mann-Whitney U）。"""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    s = np.r_[pos, neg]
    ranks = stats.rankdata(s)
    n1, n0 = len(pos), len(neg)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def delong_paired(pos_a, neg_a, pos_b, neg_b):
    """配对 DeLong：同一批对象上两方法 AUC 差的 z 检验。

    pos_* = 伪造窗异常分，neg_* = 真实窗异常分（越高越判伪造）。
    返回 (auc_a, auc_b, z, p, degen)。degen=True 表示渐近方差退化
    （方法 b 的分数对 pos/neg 无区分 → 组件近似常数 → 方差 0，
    典型见于马氏引擎对 N2 的 T2 塌缩），此时 z/p 无意义，调用方
    改用配对 bootstrap。"""
    n1, n0 = len(pos_a), len(neg_a)

    def comps(pos, neg):
        d = pos[:, None] - neg[None, :]
        psi = np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))
        return psi.mean(axis=1), psi.mean(axis=0)

    V10a, V01a = comps(np.asarray(pos_a, float), np.asarray(neg_a, float))
    V10b, V01b = comps(np.asarray(pos_b, float), np.asarray(neg_b, float))
    A_a, A_b = float(V10a.mean()), float(V10b.mean())
    c1 = np.cov(V10a, V10b, ddof=1)[0, 1] / n1
    c0 = np.cov(V01a, V01b, ddof=1)[0, 1] / n0
    var = float(c1 + c0)
    if var <= 0:
        return A_a, A_b, 0.0, 1.0, True
    z = (A_a - A_b) / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return A_a, A_b, float(z), p, False


def boot_paired(pos_a, neg_a, pos_b, neg_b, reps=2000, seed=12345):
    """配对 bootstrap：伪造/真实索引联合重抽（两方法共享同一重抽
    → 配对保持），AUC 差的双侧 p 与 95% CI。DeLong 退化时的兜底。"""
    pos_a = np.asarray(pos_a, float)
    neg_a = np.asarray(neg_a, float)
    pos_b = np.asarray(pos_b, float)
    neg_b = np.asarray(neg_b, float)
    rng = np.random.default_rng(seed)
    n1, n0 = len(pos_a), len(neg_a)
    diffs = np.empty(reps)
    for t in range(reps):
        i1 = rng.integers(0, n1, n1)
        i0 = rng.integers(0, n0, n0)
        diffs[t] = (auc_pair(pos_a[i1], neg_a[i0])
                    - auc_pair(pos_b[i1], neg_b[i0]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    phi = float(np.mean(diffs <= 0))
    p = float(min(1.0, 2 * min(phi, 1 - phi)))
    return {"boot_p": max(p, 1.0 / reps),
            "boot_ci_lo": float(lo), "boot_ci_hi": float(hi)}


def mcnemar_exact(flags_a, flags_b):
    """配对 McNemar 精确检验（a、b 为同批伪造窗的判定布尔向量）。"""
    a = np.asarray(flags_a, bool)
    b = np.asarray(flags_b, bool)
    n01 = int(np.sum(a & ~b))   # 仅 a 检出
    n10 = int(np.sum(~a & b))   # 仅 b 检出
    n_d = n01 + n10
    if n_d == 0:
        return n01, n10, 1.0
    p = float(stats.binomtest(min(n01, n10), n_d, 0.5).pvalue)
    return n01, n10, p


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    smoke = "--smoke" in sys.argv
    seed_run = SEED
    for a in sys.argv:
        if a.startswith("--seed="):
            seed_run = int(a.split("=", 1)[1])
    t0 = time.time()
    epochs = 15 if smoke else EPOCHS
    pool_slices = 30 if smoke else POOL_SLICES
    pool_per = 8 if smoke else POOL_PER
    k_eval = 10 if smoke else K_EVAL
    print("=" * 66)
    print("E2 主实验 + E7 消融矩阵（五资产 × N1–N5，G2 关卡）"
          + ("  [冒烟]" if smoke else ""))
    print(f"协议：R1 三层切分 | cal q95 冻结阈值 | α={ALPHA} | "
          f"K_eval={k_eval} | epochs={epochs} | seed={seed_run}")
    print("=" * 66)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    # 池化容器：异常分（跨资产拼接）与 e 值原始 W
    anom = {m: {"cal": [], "real": [], "fam": {f: [] for f in FAMILIES}}
            for m in ALL_METHODS}
    Wraw = {m: {"real": [], "fam": {f: [] for f in FAMILIES}}
            for m in EV_METHODS}
    per_asset = {}
    train_info = {}

    for ai, (asset, r) in enumerate(returns.items()):
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        anchors = wins[:sp["n_anchor"]]
        cal_wins = span_windows(wins, *sp["cal"])
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        train_end = sp["cal"][0] * STEP
        forged = {fam: [forge(w, fam, seed=1000 + wi * 10 + fi)
                        for wi, w in enumerate(test_wins)]
                  for fi, fam in enumerate(FAMILIES)}
        print(f"\n[{asset}] fit{sp['n_anchor']} cal{len(cal_wins)} "
              f"test{len(test_wins)}×(1+5) | 训练跨度 day 0–{train_end - 1}",
              flush=True)

        # ---- 训练 A0–A3 ----
        models = {}
        for ci, tag in enumerate(TRAINABLE):
            enc, head, last, t = train_model(
                r, train_end, CONFIGS[tag], seed_run + ai * 1000 + ci,
                epochs, pool_slices, pool_per)
            models[tag] = (enc, head)
            train_info[f"{asset}/{tag}"] = {**last, "s": t}
            print(f"    [{tag}] {t}s loss={last['loss']} "
                  f"acc={last['acc_pos']}/{last['acc_neg']}", flush=True)

        # ---- e 值评分（A0–A3）----
        for ci, tag in enumerate(EV_METHODS):
            enc, head = models[tag]
            rank = CONFIGS[tag]["rank"]
            base = 900000 + ai * 10000 + ci * 1000 + (seed_run - SEED)
            w_cal = e_values(enc, head, cal_wins, rank, k_eval, base)
            w_real = e_values(enc, head, test_wins, rank, k_eval, base + 100)
            w_fam = {fam: e_values(enc, head, forged[fam], rank, k_eval,
                                   base + 200 + fi * 100)
                     for fi, fam in enumerate(FAMILIES)}
            anom[tag]["cal"].append(-w_cal)
            anom[tag]["real"].append(-w_real)
            Wraw[tag]["real"].append(w_real)
            for fam in FAMILIES:
                anom[tag]["fam"][fam].append(-w_fam[fam])
                Wraw[tag]["fam"][fam].append(w_fam[fam])

        # ---- A4：A0 编码器直接分数 ----
        enc0, head0 = models["A0"]
        rank0 = CONFIGS["A0"]["rank"]
        anom["A4"]["cal"].append(-logits_var(enc0, head0, cal_wins, rank0))
        anom["A4"]["real"].append(-logits_var(enc0, head0, test_wins, rank0))
        for fam in FAMILIES:
            anom["A4"]["fam"][fam].append(
                -logits_var(enc0, head0, forged[fam], rank0))

        # ---- A5 / A5b：马氏引擎 ----
        for tag, full in (("A5", True), ("A5b", False)):
            mu, inv = mahal_fit(anchors, full)
            anom[tag]["cal"].append(mahal_scores(cal_wins, mu, inv, full))
            anom[tag]["real"].append(mahal_scores(test_wins, mu, inv, full))
            for fam in FAMILIES:
                anom[tag]["fam"][fam].append(
                    mahal_scores(forged[fam], mu, inv, full))

        # ---- 逐资产阈值与判定 ----
        seg = len(anom["A0"]["cal"]) - 1
        per_asset[asset] = {}
        for tag in ALL_METHODS:
            cal = np.concatenate(anom[tag]["cal"][seg:seg + 1])
            cal = cal[np.isfinite(cal)]
            thr = float(np.quantile(cal, 1 - ALPHA))
            real = np.concatenate(anom[tag]["real"][seg:seg + 1])
            entry = {"n_test": len(test_wins),
                     "fpr": float(np.nanmean(real > thr))}
            for fam in FAMILIES:
                f = np.concatenate(anom[tag]["fam"][fam][seg:seg + 1])
                entry[fam[:2]] = float(np.nanmean(f > thr))
            per_asset[asset][tag] = entry

    # ---- 池化指标 ----
    print("\n" + "=" * 66)
    print("池化指标（28 真实测试窗 + 5×28 伪造，cal q95 冻结阈值）")
    print(f"{'配置':<22}{'N1':>6}{'N2':>6}{'N3':>6}{'N4':>6}{'N5':>6}"
          f"{'macro':>8}{'FPR':>7}{'AUC全':>7}")
    print("-" * 78)
    pooled = {}
    for tag in ALL_METHODS:
        real = np.concatenate(anom[tag]["real"])
        ok_r = np.isfinite(real)
        # 逐资产 cal q95 阈值 → 池化 recall / FPR / 判定向量
        recalls, flags_real, flags_fam = pooled_decisions(anom, tag, returns)
        fam_auc = {}
        for fam in FAMILIES:
            f = np.concatenate(anom[tag]["fam"][fam])
            f = f[np.isfinite(f)]
            fam_auc[fam[:2]] = auc_pair(f, real[ok_r])
        all_f = np.concatenate([np.concatenate(anom[tag]["fam"][f])
                                for f in FAMILIES])
        all_f = all_f[np.isfinite(all_f)]
        macro = float(np.mean(list(recalls.values())))
        pooled[tag] = {
            "recall": recalls,
            "macro_recall": macro,
            "fpr": float(np.mean(flags_real)),
            "auc_family": fam_auc,
            "auc_all": auc_pair(all_f, real[ok_r]),
            "flags_real": flags_real,
            "flags_fam": flags_fam,
        }
        r = recalls
        print(f"{tag+' '+METHOD_DESC[tag]:<22}"
              f"{r['N1']:.3f} {r['N2']:.3f} {r['N3']:.3f} "
              f"{r['N4']:.3f} {r['N5']:.3f} "
              f"{macro:>8.3f}{pooled[tag]['fpr']:>7.3f}"
              f"{pooled[tag]['auc_all']:>7.3f}")

    # ---- e 规则（免校准，W ≥ 1/α 认证为真）----
    print("\ne 规则（W ≥ 1/α = 20 认证为真；未认证判可疑）")
    e_rule = {}
    for tag in EV_METHODS:
        w_real = np.concatenate(Wraw[tag]["real"])
        cert_real = float(np.mean(w_real >= 1 / ALPHA))
        rec = {}
        for fam in FAMILIES:
            w_f = np.concatenate(Wraw[tag]["fam"][fam])
            rec[fam[:2]] = float(np.mean(w_f < 1 / ALPHA))
        e_rule[tag] = {"real_certify_rate": cert_real, "recall": rec}
        print(f"  [{tag}] 真实窗认证率={cert_real:.3f} | "
              f"recall: " + " ".join(f"{k}={v:.3f}" for k, v in rec.items()))

    # ---- 显著性检验 ----
    print("\n显著性（DeLong 配对 z 检验 / McNemar 精确检验）")
    stats_out = {"delong": {}, "mcnemar": {}}
    real_all = {t: np.concatenate(anom[t]["real"])
                for t in ALL_METHODS}
    pairs = [("A0", "A5"), ("A0", "A1"), ("A0", "A2"), ("A0", "A3"),
             ("A0", "A4"), ("A0", "A5b")]
    for a, b in pairs:
        for scope in FAM_SHORT + ["ALL"]:
            if scope == "ALL":
                pos_a = np.concatenate(
                    [np.concatenate(anom[a]["fam"][f]) for f in FAMILIES])
                pos_b = np.concatenate(
                    [np.concatenate(anom[b]["fam"][f]) for f in FAMILIES])
            else:
                fam = FAMILIES[FAM_SHORT.index(scope)]
                pos_a = np.concatenate(anom[a]["fam"][fam])
                pos_b = np.concatenate(anom[b]["fam"][fam])
            # 配对联合有限掩码（保持窗口一一对应）
            okp = np.isfinite(pos_a) & np.isfinite(pos_b)
            ra, rb = real_all[a], real_all[b]
            okr = np.isfinite(ra) & np.isfinite(rb)
            A_a, A_b, z, p, degen = delong_paired(pos_a[okp], ra[okr],
                                                  pos_b[okp], rb[okr])
            entry = {"auc_a": A_a, "auc_b": A_b, "z": z, "p": p}
            if degen:
                entry.update(boot_paired(pos_a[okp], ra[okr],
                                         pos_b[okp], rb[okr]))
                entry["p"] = entry["boot_p"]
                entry["degenerate"] = True
            stats_out["delong"][f"{a}vs{b}/{scope}"] = entry
            if scope in ("N2", "N5", "ALL"):
                extra = ""
                if degen:
                    extra = (f" [DeLong 退化→boot p={entry['boot_p']:.2e}"
                             f" CI {entry['boot_ci_lo']:.3f},"
                             f"{entry['boot_ci_hi']:.3f}]")
                print(f"  DeLong {a}vs{b}[{scope}]: "
                      f"AUC {A_a:.3f} vs {A_b:.3f} z={z:+.2f} "
                      f"p={entry['p']:.2e}{extra}")
        n01, n10, p = mcnemar_exact(pooled[a]["flags_fam"]["ALL"],
                                    pooled[b]["flags_fam"]["ALL"])
        stats_out["mcnemar"][f"{a}vs{b}"] = {"only_a": n01, "only_b": n10,
                                             "p": p}
        print(f"  McNemar {a}vs{b}: 仅a={n01} 仅b={n10} p={p:.2e}")

    # ---- 判定 ----
    m = pooled["A0"]
    baselines = {t: pooled[t]["macro_recall"]
                 for t in ("A1", "A4", "A5", "A5b")}
    best_base = max(baselines, key=baselines.get)
    dl = stats_out["delong"][f"A0vs{best_base}/ALL"]["p"]
    mn = stats_out["mcnemar"][f"A0vs{best_base}"]["p"]
    h1 = bool(m["macro_recall"] > 0.693 and min(dl, mn) < 0.05)
    h2 = bool(m["recall"]["N2"] >= 0.75)
    n5_hold = bool(m["recall"]["N5"] >= 0.886)
    verdict = {
        "h1_macro_beats_0693_significant": h1,
        "h2_n2_recall_ge_075": h2,
        "n5_hold_ge_0886": n5_hold,
        "best_baseline": best_base,
        "delong_p_vs_best": dl,
        "mcnemar_p_vs_best": mn,
        "g2_pass": bool(h1 and h2),
    }
    print("\n" + "-" * 66)
    for k, v in verdict.items():
        if isinstance(v, bool):
            print(f"G2[{k}]: {'PASS' if v else 'FAIL'}")
    print(f"G2 总体: {'通过' if verdict['g2_pass'] else '未通过'}"
          + ("（冒烟口径，非正式）" if smoke else ""))

    # ---- 输出 ----
    res = {
        "smoke": smoke, "epochs": epochs, "k_eval": k_eval,
        "alpha": ALPHA, "seed": seed_run,
        "protocol": {"window": WINDOW, "step": STEP,
                     "split": SPLIT_SPEC, "families": FAMILIES},
        "pooled": {t: {k: v for k, v in pooled[t].items()
                       if k not in ("flags_real", "flags_fam")}
                   for t in ALL_METHODS},
        "e_rule": e_rule,
        "stats": stats_out,
        "per_asset": per_asset,
        "train_info": train_info,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "" if seed_run == SEED else f"_s{seed_run}"
    out_json = os.path.join(RESULTS_DIR, f"e2_main{suffix}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out_json}")

    # 原始分数落盘（默认种子）：供事后重分析，免重训
    if seed_run == SEED:
        arrays = {"assets": np.array(list(returns))}
        kinds = [("cal", "cal"), ("real", "real")] + [
            (f, f[:2]) for f in FAMILIES]
        for m in ALL_METHODS:
            for kind, key in kinds:
                src = (anom[m]["cal"] if kind == "cal"
                       else anom[m]["real"] if kind == "real"
                       else anom[m]["fam"][kind])
                for seg_i, arr in enumerate(src):
                    arrays[f"anom/{m}/{key}/{seg_i}"] = np.asarray(arr)
        for m in EV_METHODS:
            for kind, key in kinds[1:]:
                src = (Wraw[m]["real"] if kind == "real"
                       else Wraw[m]["fam"][kind])
                for seg_i, arr in enumerate(src):
                    arrays[f"W/{m}/{key}/{seg_i}"] = np.asarray(arr)
        npz_path = os.path.join(RESULTS_DIR, "e2_main_scores.npz")
        np.savez_compressed(npz_path, **arrays)
        print(f"原始分数已写入 {npz_path}")

    if not smoke and seed_run == SEED:
        make_figure(pooled, e_rule, verdict)


def pooled_decisions(anom, tag, returns):
    """逐资产 cal q95 阈值 → 池化 recall / FPR / 判定向量。

    anom[tag]["cal"/"real"/"fam"][k] 为第 k 个资产的分数数组（每资产
    一项，顺序与 returns 迭代一致）。"""
    hit = {f[:2]: [] for f in FAMILIES}
    tot = {f[:2]: [] for f in FAMILIES}
    flags_real, flags_fam = [], {f[:2]: [] for f in FAMILIES}
    for seg, _asset in enumerate(returns):
        cal = np.concatenate(anom[tag]["cal"][seg:seg + 1])
        cal = cal[np.isfinite(cal)]
        thr = float(np.quantile(cal, 1 - ALPHA))
        real = np.concatenate(anom[tag]["real"][seg:seg + 1])
        flags_real.append(real > thr)
        for fam in FAMILIES:
            f = np.concatenate(anom[tag]["fam"][fam][seg:seg + 1])
            fl = f > thr
            hit[fam[:2]].append(int(np.nansum(fl)))
            tot[fam[:2]].append(len(fl))
            flags_fam[fam[:2]].append(fl)
    rec = {k: float(np.sum(hit[k]) / np.sum(tot[k])) for k in hit}
    fr = np.concatenate(flags_real)
    ff = {k: np.concatenate(v) for k, v in flags_fam.items()}
    ff["ALL"] = np.concatenate(list(ff.values()))
    return rec, fr, ff


def make_figure(pooled, e_rule, verdict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.3),
                             gridspec_kw={"wspace": 0.36})

    # (a) 逐族 recall（四方法）
    ax = axes[0]
    show = ["A0", "A1", "A4", "A5"]
    colors = ["#4A6FA5", "#9CAF88", "#8FA8BF", "#C9A66B"]
    x = np.arange(len(FAM_SHORT))
    for i, tag in enumerate(show):
        vals = [pooled[tag]["recall"][k] for k in FAM_SHORT]
        ax.bar(x + (i - 1.5) * 0.2, vals, width=0.19, color=colors[i],
               label=f"{tag} {METHOD_DESC[tag]}", edgecolor="white",
               linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(FAM_SHORT)
    ax.set_ylabel("recall @ FPR 0.05", fontsize=8.5)
    ax.set_title("(a) Per-family recall (pooled 5 assets)", fontsize=9)
    ax.axhline(0.75, color="#A9745E", lw=0.8, ls="--")
    ax.text(0.02, 0.77, "H2 gate 0.75", fontsize=7, color="#A9745E")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", color="#E8EAED", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")

    # (b) 消融矩阵 macro recall
    ax = axes[1]
    tags = ["A0", "A1", "A2", "A3", "A4", "A5", "A5b"]
    vals = [pooled[t]["macro_recall"] for t in tags]
    n2v = [pooled[t]["recall"]["N2"] for t in tags]
    ypos = np.arange(len(tags))[::-1]
    ax.barh(ypos + 0.18, vals, height=0.34, color="#4A6FA5",
            label="macro recall", edgecolor="white", linewidth=0.4)
    ax.barh(ypos - 0.18, n2v, height=0.34, color="#9CAF88",
            label="N2 recall", edgecolor="white", linewidth=0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{t} {METHOD_DESC[t]}" for t in tags], fontsize=7.5)
    ax.axvline(0.693, color="#A9745E", lw=0.9, ls="--")
    ax.text(0.70, ypos[-1] - 0.55, "H1: 0.693", fontsize=7, color="#A9745E")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("recall @ FPR 0.05", fontsize=8.5)
    ax.set_title("(b) Ablation matrix (E7)", fontsize=9)
    ax.grid(True, axis="x", color="#E8EAED", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    # (c) 实测 FPR vs 名义 α + e 规则真实窗认证率
    ax = axes[2]
    tags = ["A0", "A1", "A2", "A3", "A4", "A5", "A5b"]
    fpr = [pooled[t]["fpr"] for t in tags]
    ypos = np.arange(len(tags))[::-1]
    ax.barh(ypos, fpr, height=0.62, color="#8FA8BF",
            edgecolor="white", linewidth=0.4)
    ax.axvline(ALPHA, color="#A9745E", lw=0.9, ls="--")
    ax.text(ALPHA + 0.008, ypos[-1] - 0.55, "nominal α=0.05",
            fontsize=7, color="#A9745E")
    ax.set_yticks(ypos)
    ax.set_yticklabels(tags, fontsize=8)
    for y, v in zip(ypos, fpr):
        ax.text(v + 0.006, y, f"{v:.2f}", va="center", fontsize=6.5,
                color="#5A6B7A")
    ax.set_xlim(0, max(0.22, max(fpr) * 1.25))
    ax.set_xlabel("realized FPR on 28 real windows", fontsize=8.5)
    ax.set_title("(c) Validity: realized FPR vs α", fontsize=9)
    ax.grid(True, axis="x", color="#E8EAED", lw=0.5)
    ax.set_axisbelow(True)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_e2_main.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"图已写入 {out}")


if __name__ == "__main__":
    main()