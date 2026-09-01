"""G1 后半：MINT 小规模原型（单资产 equity，N2/N5 跑通，目标 AUC>0.75）。

设计依据：整体研究方案 §4.2 训练算法 + §9 G1 关卡。

协议（时间分块冻结，与 R1 三层切分一致的精神）：
  训练切片  测试层之前全部历史（day 0–3149），每 epoch 随机起点切片
  测试窗    窗口索引 63–99（day 3150–5949）stride 50 → 37 窗
  两层时间严格不相交。

v2 关键修正（G1 诊断 _diag_g1.py 的结论）：
  1. AUC 公式 bug：分母 n_r·(n_r+n_f) → n_r·n_f（旧版全部 AUC 被压低
     ~2 倍；手工 lev 的 N2 AUC 实为 1.000、vol_excess 的 N5 AUC 实为
     1.000，任务在协议下完全可解，上界已校准）；
  2. 编码器 stride 4×4 → 2,2,2,4：旧架构池化前仅剩 2 个时间步，
     时间不对称结构被抹平（model.py 已改）；
  3. 分数头改 logit 域：sigmoid 饱和压缩测试分数序；
  4. 正样本增强去 np.roll（循环移位边界伪影是正样本独有捷径特征，
     测试分布不含该伪影）→ 时间裁剪 + 加性噪声（TS2Vec 式 cropping）；
  5. 反记忆：固定 87 个重叠 97.5% 的训练窗 → 每 epoch 随机起点切片
     （TRAIN_LEN=700，起点 U(0, 2450)），位置不可记忆；
  6. 轨道负样本池：150 个随机切片 × 20 条五算子轨道预生成，
     每 epoch 无放回抽样（跨窗轨道与真实切片的边际/谱分布相同
     → 判别力被迫落向轨道唯一破坏的东西：依赖结构）。

对照（同架构同循环，隔离"轨道负样本"组件）：
  AUG 版：负样本换成随机增强（循环移位/幅度缩放/加性噪声/子窗截取）。

评估（G1 验收）：
  测试 37 真窗 vs N2 时间倒置 / N5 相位伪造（种子 1000+wi*10+fi，
  对齐 R1 审计协议）；
  分数：s_θ logit 与 轨道 e 值 W（K=32）；
  基线：现稿 15 维马氏引擎 + 手工统计量上界（lev / vol_excess）；
  诊断：模型分数与手工统计量的 Spearman 相关（模型读出了什么）；
  通过标准：MINT s_θ AUC(N2) > 0.75 且 AUC(N5) > 0.75。

产出：results/g1_prototype.json + figures/fig_g1_prototype.pdf
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

from bench import HALLUCINATION_BUILDERS, n5_phase_forge  # noqa: E402
from features import extract_features  # noqa: E402
from mint.model import (  # noqa: E402
    Encoder,
    ScoreHead,
    bce_loss,
    gaussianize,
    info_nce,
    orbit_e_value,
)
from mint.operators import generate_orbit  # noqa: E402
from scipy import stats  # noqa: E402

WINDOW = 1000              # 测试窗长（协议固定）
TRAIN_LEN = 700            # 训练切片长（< 测试窗，位置多样性优先）
N_SLICES = 64              # 每 epoch 真实切片数
POOL_SLICES = 150          # 轨道池基准切片数
POOL_PER = 20              # 每基准切片轨道条数
K_TRAIN = 12               # 每 epoch 每切片负样本数（从池抽样）
K_EVAL = 32
EPOCHS = 250
TAU = 0.1
LR = 1e-3
CROP = 620                 # 正样本裁剪长度（of 700）
SEED = 20260820
N_ANCHOR = 15
TEST_SPAN = (63, 99)
TEST_STEP = 50

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
FIGURES_DIR = os.path.join(MINT_ROOT, "figures")


# ---------------------------------------------------------------------------
# 数据、伪造与增强
# ---------------------------------------------------------------------------
def forge(x: np.ndarray, family: str, seed: int) -> np.ndarray:
    if family.startswith("N5"):
        return n5_phase_forge(x, seed=seed)
    return HALLUCINATION_BUILDERS[family](x, seed=seed)


def crop_noise(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """正样本增强：随机时间裁剪 + 加性噪声（无循环移位，roll 的
    首尾拼接伪影是正样本独有捷径特征，G1 v2 移除）。"""
    start = int(rng.integers(0, len(w) - CROP + 1))
    out = w[start:start + CROP].copy()
    return out + 0.02 * out.std() * rng.standard_normal(len(out))


def augment_negative(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """随机增强负样本（TS2Vec 式，AUG 对照版）。"""
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
# 手工统计量（可解性上界校准 + 模型读出诊断）
# ---------------------------------------------------------------------------
def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def handcrafted(z: np.ndarray) -> tuple[float, float]:
    """(lev, vol_excess)：lev 攻 N2（时间箭头），vol_excess 攻 N5
    （幅值聚集超出线性部分）。"""
    z2 = z ** 2
    az = np.abs(z)
    lev = corr(z2[:-1], z[1:])
    acf_z = np.mean([abs(corr(z[:-k], z[k:])) for k in range(1, 11)])
    acf_az = np.mean([corr(az[:-k], az[k:]) for k in range(1, 11)])
    return lev, float(acf_az - acf_z)


# ---------------------------------------------------------------------------
# 训练（MINT 与 AUG 共用循环，仅负样本源不同）
# ---------------------------------------------------------------------------
def build_orbit_pool(r: np.ndarray, train_end: int,
                     rng: np.random.Generator) -> np.ndarray:
    """轨道池：POOL_SLICES 个随机切片 × POOL_PER 条五算子轨道。"""
    starts = rng.integers(0, train_end - TRAIN_LEN + 1, size=POOL_SLICES)
    rows = []
    for s in starts:
        w = r[int(s):int(s) + TRAIN_LEN]
        surrs, _ = generate_orbit(w, POOL_PER, rng=rng)
        rows.append(surrs)
    pool = np.vstack(rows)
    return gaussianize_batch(pool)


def gaussianize_batch(X: np.ndarray) -> np.ndarray:
    return np.vstack([gaussianize(row) for row in X])


def train_model(r: np.ndarray, train_end: int, negative_source: str,
                epochs: int = EPOCHS, seed: int = SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, head = Encoder(), ScoreHead()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    pool = build_orbit_pool(r, train_end, np.random.default_rng(seed + 77))
    print(f"    轨道池 {pool.shape}（{'轨道' if negative_source=='orbit' else '增强'}源）")
    n_pool = len(pool)
    lo_rng = train_end - TRAIN_LEN

    hist = []
    t0 = time.time()
    for epoch in range(epochs):
        starts = rng.integers(0, lo_rng + 1, size=N_SLICES)
        wins = [r[s:s + TRAIN_LEN] for s in starts]
        X = torch.tensor(gaussianize_batch(np.array(wins)),
                         dtype=torch.float32)
        X_pos = torch.tensor(
            gaussianize_batch(np.array([crop_noise(w, rng) for w in wins])),
            dtype=torch.float32)

        if negative_source == "orbit":
            idx = rng.choice(n_pool, size=N_SLICES * K_TRAIN, replace=False)
            N = torch.tensor(pool[idx], dtype=torch.float32)
        else:
            neg = np.array([augment_negative(w, rng)
                            for w in wins for _ in range(K_TRAIN)])
            N = torch.tensor(gaussianize_batch(neg), dtype=torch.float32)

        opt.zero_grad()
        z, z_pos, z_neg = enc(X), enc(X_pos), enc(N)
        loss_nce = info_nce(z, z_pos, z_neg, tau=TAU)
        s_pos, s_neg = head(z), head(z_neg)
        loss_bce = bce_loss(s_pos, s_neg, pos_weight=float(K_TRAIN))
        loss = loss_nce + loss_bce
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 25 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                acc_p = (s_pos > 0).float().mean().item()
                acc_n = (s_neg < 0).float().mean().item()
            hist.append({"epoch": epoch, "loss": round(loss.item(), 4),
                         "nce": round(loss_nce.item(), 4),
                         "bce": round(loss_bce.item(), 4),
                         "acc_pos": round(acc_p, 3),
                         "acc_neg": round(acc_n, 3)})
    enc.eval(), head.eval()
    return enc, head, hist, round(time.time() - t0, 1)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_logit(enc, head, X: list[np.ndarray]) -> np.ndarray:
    Z = torch.tensor(gaussianize_batch(np.array(X)), dtype=torch.float32)
    return head(enc(Z)).numpy()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def orbit_e(enc, head, x: np.ndarray, k: int, seed: int) -> float:
    surrs, _ = generate_orbit(x, k, rng=np.random.default_rng(seed))
    s_all = score_logit(enc, head, [x] + list(surrs))
    p = sigmoid(s_all)
    return orbit_e_value(float(p[0]), p[1:])


def auc(real_scores, forge_scores) -> float:
    """AUC = P(score_real > score_forge)（Mann-Whitney U）。

    v2 修正：分母 n_r·n_f（旧版误作 n_r·(n_r+n_f)，AUC 被压低 ~2 倍）。"""
    real = np.asarray(real_scores, dtype=float)
    forge = np.asarray(forge_scores, dtype=float)
    s = np.r_[real, forge]
    ranks = stats.rankdata(s)
    n_r, n_f = len(real), len(forge)
    u = ranks[:n_r].sum() - n_r * (n_r + 1) / 2.0
    return float(u / (n_r * n_f))


def baseline_mahal_auc(anchors, real_wins, forge_wins) -> float:
    """现稿 15 维马氏引擎基线，拟合层窗库估计，-d(x) 为分数。"""
    A = np.array([extract_features(w, full=True) for w in anchors])
    A = A[np.isfinite(A).all(axis=1)]
    mu = A.mean(axis=0)
    inv = np.linalg.pinv(np.cov(A, rowvar=False) + 1e-6 * np.eye(A.shape[1]))

    def d(x):
        f = extract_features(x, full=True)
        if not np.isfinite(f).all():
            return np.nan
        dev = f - mu
        return float(np.sqrt(dev @ inv @ dev))

    dr = np.array([d(x) for x in real_wins])
    df_ = np.array([d(x) for x in forge_wins])
    return auc(-dr[np.isfinite(dr)], -df_[np.isfinite(df_)])


def handcrafted_auc(real_z, n2_z, n5_z) -> dict[str, float]:
    """手工统计量上界：lev 攻 N2，vol_excess 攻 N5。"""
    out = {}
    for key, fam in ((0, "n2"), (1, "n5")):
        vr = np.array([handcrafted(z)[key] for z in real_z])
        vf = np.array([handcrafted(z)[key] for z in
                       (n2_z if fam == "n2" else n5_z)])
        out["lev" if key == 0 else "vol_excess"] = auc(vr, vf)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_seed() -> int:
    for a in sys.argv[1:]:
        if a.startswith("--seed="):
            return int(a.split("=", 1)[1])
    return SEED


def main() -> None:
    smoke = "--smoke" in sys.argv
    seed = parse_seed()
    t0 = time.time()
    epochs = 40 if smoke else EPOCHS
    print("=" * 66)
    print("G1 后半：MINT 小规模原型 v2（equity，N2/N5，AUC>0.75 验收）"
          + ("  [冒烟]" if smoke else "")
          + (f"  [seed={seed}]" if seed != SEED else ""))
    print("=" * 66)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        r = d["equity"]

    lo, hi = TEST_SPAN
    train_end = lo * TEST_STEP
    test_wins = [r[i * TEST_STEP:i * TEST_STEP + WINDOW]
                 for i in range(lo, hi + 1)]
    print(f"训练跨度 day 0–{train_end - 1}（随机切片 len={TRAIN_LEN}，"
          f"{N_SLICES}/epoch）| 测试窗 {len(test_wins)}"
          f"（day {lo * TEST_STEP}–{hi * TEST_STEP + WINDOW - 1}）")

    n2_wins = [forge(w, "N2时间倒置", seed=1000 + wi * 10 + 1)
               for wi, w in enumerate(test_wins)]
    n5_wins = [forge(w, "N5相位伪造", seed=1000 + wi * 10 + 4)
               for wi, w in enumerate(test_wins)]

    # ---- 手工统计量上界（任务可解性校准）----
    real_z = [gaussianize(w) for w in test_wins]
    n2_z = [gaussianize(w) for w in n2_wins]
    n5_z = [gaussianize(w) for w in n5_wins]
    hc = handcrafted_auc(real_z, n2_z, n5_z)
    print(f"[0/4] 手工上界: lev→AUC(N2)={hc['lev']:.3f}  "
          f"vol_excess→AUC(N5)={hc['vol_excess']:.3f}")

    # ---- 训练两个版本 ----
    print("\n[1/4] 训练 MINT（轨道负样本池）...")
    enc_m, head_m, hist_m, t_m = train_model(r, train_end, "orbit", epochs,
                                             seed=seed)
    print(f"  耗时 {t_m}s | 末轮 loss={hist_m[-1]['loss']} "
          f"acc_pos={hist_m[-1]['acc_pos']} acc_neg={hist_m[-1]['acc_neg']}")
    print("[2/4] 训练 AUG 对照（随机增强负样本）...")
    enc_a, head_a, hist_a, t_a = train_model(r, train_end, "augment", epochs,
                                             seed=seed + 500)
    print(f"  耗时 {t_a}s | 末轮 loss={hist_a[-1]['loss']} "
          f"acc_pos={hist_a[-1]['acc_pos']} acc_neg={hist_a[-1]['acc_neg']}")

    # ---- 评估 ----
    print("\n[3/4] 评估 s_θ logit 与轨道 e 值（K=32）...")
    res = {"epochs": epochs, "n_test": len(test_wins), "smoke": smoke,
           "handcrafted_ceiling": hc}

    hc_real = np.array([handcrafted(z) for z in real_z])
    hc_n2 = np.array([handcrafted(z) for z in n2_z])
    hc_n5 = np.array([handcrafted(z) for z in n5_z])
    lev_all = np.r_[hc_real[:, 0], hc_n2[:, 0], hc_n5[:, 0]]
    vol_all = np.r_[hc_real[:, 1], hc_n2[:, 1], hc_n5[:, 1]]

    for tag, enc, head in (("MINT", enc_m, head_m), ("AUG", enc_a, head_a)):
        s_real = score_logit(enc, head, test_wins)
        s_n2 = score_logit(enc, head, n2_wins)
        s_n5 = score_logit(enc, head, n5_wins)
        e_real = [orbit_e(enc, head, w, K_EVAL, seed + i)
                  for i, w in enumerate(test_wins)]
        e_n2 = [orbit_e(enc, head, w, K_EVAL, seed + 1000 + i)
                for i, w in enumerate(n2_wins)]
        e_n5 = [orbit_e(enc, head, w, K_EVAL, seed + 2000 + i)
                for i, w in enumerate(n5_wins)]
        s_all = np.r_[s_real, s_n2, s_n5]
        res[tag] = {
            "s_auc_n2": auc(s_real, s_n2),
            "s_auc_n5": auc(s_real, s_n5),
            "e_auc_n2": auc(e_real, e_n2),
            "e_auc_n5": auc(e_real, e_n5),
            "s_real_mean": float(np.mean(s_real)),
            "s_n2_mean": float(np.mean(s_n2)),
            "s_n5_mean": float(np.mean(s_n5)),
            "spearman_s_lev": float(stats.spearmanr(s_all, lev_all).statistic),
            "spearman_s_vol": float(stats.spearmanr(s_all, vol_all).statistic),
        }
        print(f"  [{tag}] s_auc: N2={res[tag]['s_auc_n2']:.3f} "
              f"N5={res[tag]['s_auc_n5']:.3f} | "
              f"e_auc: N2={res[tag]['e_auc_n2']:.3f} "
              f"N5={res[tag]['e_auc_n5']:.3f} | "
              f"ρ(s,lev)={res[tag]['spearman_s_lev']:+.2f} "
              f"ρ(s,vol)={res[tag]['spearman_s_vol']:+.2f}")

    print("[4/4] 基线：现稿 15 维马氏引擎...")
    anchors = [r[i * TEST_STEP:i * TEST_STEP + WINDOW] for i in range(N_ANCHOR)]
    res["BASELINE_15d"] = {
        "s_auc_n2": baseline_mahal_auc(anchors, test_wins, n2_wins),
        "s_auc_n5": baseline_mahal_auc(anchors, test_wins, n5_wins),
    }
    b = res["BASELINE_15d"]
    print(f"  [BASELINE] s_auc: N2={b['s_auc_n2']:.3f} "
          f"N5={b['s_auc_n5']:.3f}")

    # ---- G1 判定 ----
    m = res["MINT"]
    g1_pass = bool(m["s_auc_n2"] > 0.75 and m["s_auc_n5"] > 0.75)
    orbit_necessary = bool(
        m["s_auc_n2"] - res["AUG"]["s_auc_n2"] > 0.05
        or m["s_auc_n5"] - res["AUG"]["s_auc_n5"] > 0.05)
    res["verdict"] = {
        "mint_n2_auc_gt_075": bool(m["s_auc_n2"] > 0.75),
        "mint_n5_auc_gt_075": bool(m["s_auc_n5"] > 0.75),
        "orbit_negatives_necessary": orbit_necessary,
        "g1_pass": g1_pass,
    }
    print("\n" + "-" * 66)
    for k, v in res["verdict"].items():
        print(f"G1[{k}]: {'PASS' if v else 'FAIL'}")
    print(f"G1 总体: {'通过，11 周计划全速' if g1_pass else '未通过，回退评估'}"
          f"{'（冒烟口径，非正式）' if smoke else ''}")
    res["elapsed_s"] = round(time.time() - t0, 1)
    res["train_hist_mint"] = hist_m
    res["train_hist_aug"] = hist_a
    res["seed"] = seed

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "" if seed == SEED else f"_s{seed}"
    out_json = os.path.join(RESULTS_DIR, f"g1_prototype{suffix}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {out_json}")

    if not smoke and seed == SEED:
        make_figure(res)


def make_figure(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8.5, "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2),
                             gridspec_kw={"wspace": 0.34})

    # (a) AUC 对比条形
    ax = axes[0]
    rows = ["MINT", "AUG", "BASELINE_15d"]
    labels = ["MINT (orbit negatives)", "AUG (random aug.)",
              "Manhalanobis 15-d"]
    n2 = [res[t]["s_auc_n2"] for t in rows]
    n5 = [res[t]["s_auc_n5"] for t in rows]
    ypos = np.arange(len(rows) + 1)[::-1]
    labels = labels + ["handcrafted ceiling"]
    n2 = n2 + [res["handcrafted_ceiling"]["lev"]]
    n5 = n5 + [res["handcrafted_ceiling"]["vol_excess"]]
    ax.barh(ypos + 0.18, n2, height=0.32, color="#4A6FA5", alpha=0.92,
            label="N2 time-reversal", edgecolor="white", linewidth=0.4)
    ax.barh(ypos - 0.18, n5, height=0.32, color="#9CAF88", alpha=0.92,
            label="N5 phase-forge", edgecolor="white", linewidth=0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.75, color="#A9745E", lw=0.9, ls="--")
    ax.text(0.755, ypos[-1] - 0.52, "G1 gate 0.75", fontsize=7,
            color="#A9745E")
    ax.axvline(0.5, color="#7A8691", lw=0.7, ls=":")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUC (real vs forged)", fontsize=8.5)
    ax.set_title("(a) Detection AUC, equity prototype", fontsize=9)
    ax.grid(True, axis="x", color="#E8EAED", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7, frameon=False)

    # (b) 训练曲线
    ax = axes[1]
    hm, ha = res["train_hist_mint"], res["train_hist_aug"]
    ax.plot([h["epoch"] for h in hm], [h["loss"] for h in hm],
            color="#4A6FA5", lw=1.2, label="MINT loss")
    ax.plot([h["epoch"] for h in ha], [h["loss"] for h in ha],
            color="#8FA8BF", lw=1.2, ls="--", label="AUG loss")
    ax.set_xlabel("epoch", fontsize=8.5)
    ax.set_ylabel("InfoNCE + BCE", fontsize=8.5)
    ax.set_title("(b) Training loss", fontsize=9)
    ax.grid(True, color="#E8EAED", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_g1_prototype.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"图已写入 {out}")


if __name__ == "__main__":
    main()
