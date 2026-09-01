"""E2 深度基线（表 2 独立行）：TS2Vec+马氏 / Anomaly Transformer /
DCdetector / TimesNet，紧凑 CPU 实现，协议与 e2_main.py 逐位一致。

协议对齐（R1 三层切分原样，SPLIT_SPEC 与 e2_main 逐字一致）：
  拟合层    前 n_anchor 窗（D1 马氏拟合，同 A5 协议）
  校准层  cal 跨度真实窗（q95 阈值冻结）
  测试层  test 跨度（cap 限窗）真实窗 + N1–N5 伪造，
          种子 1000+wi*10+fi（与 e2_main 确定性一致 → 可配对检验）
  训练    随机 700 日切片（train_end 之前），逐窗 z-score

输入口径披露：四种深度模型均采用逐窗 z-score（实例归一化），
这是四模型文献的标准预处理（TS2Vec/AT/DCdetector/TimesNet 均如此），
与 MINT 的秩归一化（边际歼灭）口径不同，深度基线保留边际形状
访问权，对它们公平；差异在报告中明示。

紧凑实现披露（相对原论文的简化，全部登记）：
  D1 TS2Vec    6 层空洞 TCN 编码器；层级对比简化为实例对比（随机
               起点双裁剪 maxpool）+ 对齐时间对比（同起点双长度、
               位置内/跨样本双负样本）；读出=冻结表征 + 拟合层马氏
               （pinv+岭，同 A5 协议）
  D2 AT        conv 嵌入 stride=4（点→段），单层 2 头，先验为可学习
               σ 的高斯邻接；minimax 关联差异（λ=3，detach 双侧）；
               评分=逐点重构误差 × softmax(AssDis)
  D3 DCdet     patch=25；patch 间注意力 + patch 内注意力双分支；
               InfoNCE 正对=同实例双分支，负样本=批内跨样本 +
               patch 置换位移（原文机制）；评分=实例级双分支表征
               差异（原文元素级，紧凑简化）
  D4 TimesNet  FFT 周期 top-2（自输入谱，两 block 共用）；Inception
               简化为三分支和（3/5/1 conv）；残差折叠-展开；评分=MSE

显著性：A0 配对 DeLong / McNemar 用 e2_main_scores.npz 冻结分数
（伪造窗种子确定性 → 逐窗配对成立）；仅默认种子有 npz，多种子
运行只报指标稳定性。

产出：results/e2_deep.json（--seed=* → e2_deep_s*.json）
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bench import HALLUCINATION_BUILDERS, n5_phase_forge  # noqa: E402
from scipy import stats as sstats  # noqa: E402

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

TRAIN_LEN = 700
N_SLICES_D = 64          # 与 e2_main MINT 训练预算逐位一致
EPOCHS_D = 250
LR_D = 1e-3
SEED = 20260820

DEEP_METHODS = ["D1", "D2", "D3", "D4"]
DEEP_DESC = {
    "D1": "TS2Vec frozen + Mahalanobis",
    "D2": "Anomaly Transformer (assoc. discrepancy)",
    "D3": "DCdetector (dual-attention contrastive)",
    "D4": "TimesNet (reconstruction)",
}

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")


# ---------------------------------------------------------------------------
# 协议（与 e2_main 逐位一致）
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


def zscore(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return (X - X.mean(axis=1, keepdims=True)) / (
        X.std(axis=1, keepdims=True) + 1e-12)


def slices_batch(r, train_end, n, rng):
    starts = rng.integers(0, train_end - TRAIN_LEN + 1, size=n)
    return np.stack([r[int(s):int(s) + TRAIN_LEN] for s in starts])


# ---------------------------------------------------------------------------
# D1 TS2Vec
# ---------------------------------------------------------------------------
class TS2VecEncoder(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.convs = nn.ModuleList()
        d_in = 1
        for dil in (1, 2, 4, 8, 16, 32):
            self.convs.append(nn.Conv1d(d_in, ch, 3, padding=dil,
                                        dilation=dil))
            d_in = ch
    def forward(self, x):                    # (B, T) -> (B, ch, T)
        h = x.unsqueeze(1)
        for i, c in enumerate(self.convs):
            r = F.gelu(c(h))
            h = r if i == 0 else h + r
        return h


def info_nce_sym(g1, g2, tau=0.1):
    logits = g1 @ g2.T / tau
    labels = torch.arange(len(g1), device=g1.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.T, labels))


def ts2vec_losses(enc, X, rng):
    B, T = X.shape
    L = int(rng.integers(T // 2, T + 1))
    s1, s2 = int(rng.integers(0, T - L + 1)), int(rng.integers(0, T - L + 1))
    z1, z2 = enc(X[:, s1:s1 + L]), enc(X[:, s2:s2 + L])
    loss = info_nce_sym(z1.max(-1).values, z2.max(-1).values, 0.1)
    # 对齐时间对比：同起点双长度，位置内 + 跨样本双负样本
    L2 = int(rng.integers(L // 2, L + 1))
    s = int(rng.integers(0, T - L + 1))
    zA, zB = enc(X[:, s:s + L2]), enc(X[:, s:s + L])
    ks = rng.integers(0, L2, size=6)
    for k in ks:
        k = int(k)
        q = zA[:, :, k]                                   # (B, C)
        pos = (q * zB[:, :, k]).sum(-1, keepdim=True)     # (B, 1)
        neg_same = torch.einsum("bc,bct->bt", q, zB)      # (B, L)
        neg_same[:, k] = -1e9
        neg_cross = q @ zB[:, :, k].T                     # (B, B)
        neg_cross.fill_diagonal_(-1e9)
        logits = torch.cat([pos, neg_same, neg_cross], dim=1) / 0.1
        loss = loss + F.cross_entropy(
            logits, torch.zeros(B, dtype=torch.long, device=X.device))
    return loss + info_nce_sym(zA.mean(-1), zB[:, :L2].mean(-1), 0.1)


def d1_fit_mahal(enc, anchors):
    with torch.no_grad():
        X = torch.tensor(zscore(np.array(anchors)), dtype=torch.float32)
        R = enc(X).max(-1).values.numpy()                 # (n, ch)
    R = R[np.isfinite(R).all(axis=1)]
    mu = R.mean(axis=0)
    inv = np.linalg.pinv(np.cov(R, rowvar=False)
                         + 1e-6 * np.eye(R.shape[1]))
    return mu, inv


@torch.no_grad()
def d1_score(enc, wins, mu, inv):
    X = torch.tensor(zscore(np.array(wins)), dtype=torch.float32)
    R = enc(X).max(-1).values.numpy()
    dev = R - mu
    return np.sqrt(np.einsum("bi,ij,bj->b", dev, inv, dev))


# ---------------------------------------------------------------------------
# D2 Anomaly Transformer
# ---------------------------------------------------------------------------
def posenc(N, d):
    pos = torch.arange(N, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d, 2, dtype=torch.float32)
    ang = pos * torch.exp(-np.log(10000.0) * i / d)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)[:, :d]


class ATNet(nn.Module):
    def __init__(self, d=64, heads=2):
        super().__init__()
        self.embed = nn.Conv1d(1, d, 4, 4)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.dec = nn.Linear(d, 4)
        self.sigma = nn.Parameter(torch.ones(heads) * 2.0)
        self.d, self.heads = d, heads

    def forward(self, x):                       # (B, T) -> (B, T), (B,), (B,N)
        B, T = x.shape
        N = T // 4
        h = self.embed(x.unsqueeze(1)).transpose(1, 2) + posenc(N, self.d)
        qkv = (self.qkv(h).view(B, N, 3, self.heads, self.d // self.heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        dh = self.d // self.heads
        attn = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(dh),
                             dim=-1)            # (B, H, N, N)
        idx = torch.arange(N)
        dist2 = (idx[:, None] - idx[None, :]).float() ** 2
        sig = self.sigma.view(1, -1, 1, 1) ** 2
        prior = torch.exp(-dist2 / (2 * sig))
        prior = prior / prior.sum(-1, keepdim=True)
        kl = (prior * (torch.log(prior + 1e-9)
                       - torch.log(attn + 1e-9))).sum(-1)   # (B, H, N)
        assdis = kl.mean(dim=(1, 2))                       # (B,)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, self.d)
        recon = self.dec(self.proj(out)).reshape(B, T)
        return recon, assdis, kl.mean(1)                   # (B,T),(B,),(B,N)

    def loss(self, x, lam=3.0):
        # 单次手工展开（需 attn/prior 分量做 detach 双侧 minimax）
        B, T = x.shape
        N = T // 4
        h = self.embed(x.unsqueeze(1)).transpose(1, 2) + posenc(N, self.d)
        qkv = (self.qkv(h).view(B, N, 3, self.heads, self.d // self.heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        dh = self.d // self.heads
        attn = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(dh), dim=-1)
        idx = torch.arange(N)
        dist2 = (idx[:, None] - idx[None, :]).float() ** 2
        sig = self.sigma.view(1, -1, 1, 1) ** 2
        prior = torch.exp(-dist2 / (2 * sig))
        prior = prior / prior.sum(-1, keepdim=True)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, self.d)
        recon = self.dec(self.proj(out)).reshape(B, T)
        logp, loga = torch.log(prior + 1e-9), torch.log(attn + 1e-9)
        term_series = (prior.detach() * (logp.detach() - loga)).sum(-1).mean()
        term_prior = (prior * (logp - loga.detach())).sum(-1).mean()
        return F.mse_loss(recon, x) + lam * (term_series - term_prior)


@torch.no_grad()
def d2_score(model, wins):
    X = torch.tensor(zscore(np.array(wins)), dtype=torch.float32)
    recon, _, kl_pos = model(X)
    err = (X - recon).pow(2).view(len(X), -1, 4).mean(-1)   # (B, N)
    w = torch.softmax(kl_pos, dim=-1)
    return (w * err).sum(-1).numpy()


# ---------------------------------------------------------------------------
# D3 DCdetector
# ---------------------------------------------------------------------------
class DCNet(nn.Module):
    def __init__(self, d=64, P=25, heads=2):
        super().__init__()
        self.patch_emb = nn.Linear(P, d)
        self.point_emb = nn.Linear(1, d)
        self.mha_patch = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mha_in = nn.MultiheadAttention(d, heads, batch_first=True)
        self.P = P

    def rep(self, x):                             # (B, T) -> pa/ia (B, N, d)
        B, T = x.shape
        N = T // self.P
        patches = x.view(B, N, self.P)
        pe = self.patch_emb(patches)
        pa, _ = self.mha_patch(pe, pe, pe)
        ie = self.point_emb(patches.unsqueeze(-1)).view(B * N, self.P, -1)
        ia, _ = self.mha_in(ie, ie, ie)
        ia = ia.view(B, N, self.P, -1).mean(2)
        return pa, ia

    def loss(self, x, perm):
        B, T = x.shape
        N = T // self.P
        pa, ia = self.rep(x)
        xp = x.view(B, N, self.P)[:, perm].reshape(B, T)
        _, iap = self.rep(xp)
        z1 = F.normalize(pa.mean(1), dim=-1)
        z2 = F.normalize(ia.mean(1), dim=-1)
        z2p = F.normalize(iap.mean(1), dim=-1)
        logits = torch.cat([z1 @ z2.T, z1 @ z2p.T], dim=1) / 0.2
        labels = torch.arange(B, device=x.device)
        return F.cross_entropy(logits, labels)


def d3_perm(n, seed):
    return np.random.default_rng(seed).permutation(n)


@torch.no_grad()
def d3_score(model, wins):
    X = torch.tensor(zscore(np.array(wins)), dtype=torch.float32)
    pa, ia = model.rep(X)
    z1 = F.normalize(pa.mean(1), dim=-1)
    z2 = F.normalize(ia.mean(1), dim=-1)
    return (z1 - z2).pow(2).sum(-1).numpy()


# ---------------------------------------------------------------------------
# D4 TimesNet
# ---------------------------------------------------------------------------
class Inception(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.b1 = nn.Conv2d(d, d, 3, padding=1)
        self.b2 = nn.Conv2d(d, d, 5, padding=2)
        self.b3 = nn.Conv2d(d, d, 1)
    def forward(self, h):
        return F.gelu(self.b1(h)) + F.gelu(self.b2(h)) + F.gelu(self.b3(h))


class TimesNet(nn.Module):
    def __init__(self, d=32, blocks=2, topk=2):
        super().__init__()
        self.embed = nn.Linear(1, d)
        self.blocks = nn.ModuleList([Inception(d) for _ in range(blocks)])
        self.head = nn.Linear(d, 1)
        self.topk = topk

    @staticmethod
    def periods(x):                     # (B, T) -> list[(period, idx)]
        X = x.detach().cpu().numpy()
        out = []
        for xi in X:
            spec = np.abs(np.fft.rfft(xi))
            T = len(xi)
            bins = np.arange(2, T // 2 + 1)
            amp = spec[2:T // 2 + 1]
            top = bins[np.argsort(amp)[::-1][:2]]
            out.append([int(T // b) for b in top])
        # 返回逐样本 top-2 周期列表
        return out

    def _fold_conv(self, blk, h, p):
        # h: (b, d, T) -> fold (b, d, r, p) -> conv -> unfold
        b, d, T = h.shape
        r = int(np.ceil(T / p))
        Tp = r * p
        if Tp > T:
            pad = h[:, :, -1:].expand(b, d, Tp - T)
            h = torch.cat([h, pad], dim=-1)
        h2 = h.reshape(b, d, r, p)
        h2 = blk(h2)
        h2 = h2.reshape(b, d, Tp)
        return h2[:, :, :T]

    def forward(self, x):               # (B, T) -> (B, T)
        B, T = x.shape
        h = self.embed(x.unsqueeze(-1)).transpose(1, 2)     # (B, d, T)
        pers = self.periods(x)
        for blk in self.blocks:
            out = torch.zeros_like(h)
            # 逐周期分支（top-2，等权聚合，幅度加权的紧凑简化）
            for k in range(self.topk):
                by_p = {}
                for i, pl in enumerate(pers):
                    by_p.setdefault(pl[k], []).append(i)
                for p, idx in by_p.items():
                    p = max(2, min(p, T // 2))
                    sub = h[idx]
                    out[idx] = out[idx] + self._fold_conv(blk, sub, p)
            h = h + out / self.topk
        return self.head(h.transpose(1, 2)).squeeze(-1)


# ---------------------------------------------------------------------------
# 统一训练与评分
# ---------------------------------------------------------------------------
def train_deep(kind, r, train_end, seed, epochs, n_slices):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if kind == "D1":
        model = TS2VecEncoder()
        loss_fn = lambda m, X: ts2vec_losses(m, X, rng)
    elif kind == "D2":
        model = ATNet()
        loss_fn = lambda m, X: m.loss(X)
    elif kind == "D3":
        model = DCNet()
        def loss_fn(m, X):
            N = X.shape[1] // m.P
            perm = d3_perm(N, seed + int(rng.integers(1 << 30)))
            return m.loss(X, perm)
    else:
        model = TimesNet()
        loss_fn = lambda m, X: F.mse_loss(m(X), X)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_D, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()
    last = 0.0
    for _ in range(epochs):
        X = torch.tensor(zscore(slices_batch(r, train_end, n_slices, rng)),
                         dtype=torch.float32)
        opt.zero_grad()
        loss = loss_fn(model, X)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        last = loss.item()
    model.eval()
    return model, {"loss": round(last, 4), "s": round(time.time() - t0, 1)}


@torch.no_grad()
def score_deep(kind, model, wins, ctx):
    """异常分（越高越可疑）。wins 同长度（评估窗全部 1000）。"""
    if kind == "D1":
        return d1_score(model, wins, ctx["mu"], ctx["inv"])
    if kind == "D2":
        return d2_score(model, wins)
    if kind == "D3":
        return d3_score(model, wins)
    X = torch.tensor(zscore(np.array(wins)), dtype=torch.float32)
    return (model(X) - X).pow(2).mean(dim=1).numpy()


# ---------------------------------------------------------------------------
# 指标（与 e2_main 相同公式）
# ---------------------------------------------------------------------------
def auc_pair(pos, neg) -> float:
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    s = np.r_[pos, neg]
    ranks = sstats.rankdata(s)
    n1, n0 = len(pos), len(neg)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def delong_paired(pos_a, neg_a, pos_b, neg_b):
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
    p = float(2 * sstats.norm.sf(abs(z)))
    return A_a, A_b, float(z), p, False


def boot_paired(pos_a, neg_a, pos_b, neg_b, reps=2000, seed=12345):
    pos_a, neg_a = np.asarray(pos_a, float), np.asarray(neg_a, float)
    pos_b, neg_b = np.asarray(pos_b, float), np.asarray(neg_b, float)
    rng = np.random.default_rng(seed)
    n1, n0 = len(pos_a), len(neg_a)
    diffs = np.empty(reps)
    for t in range(reps):
        i1, i0 = rng.integers(0, n1, n1), rng.integers(0, n0, n0)
        diffs[t] = (auc_pair(pos_a[i1], neg_a[i0])
                    - auc_pair(pos_b[i1], neg_b[i0]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    phi = float(np.mean(diffs <= 0))
    p = float(min(1.0, 2 * min(phi, 1 - phi)))
    return {"boot_p": max(p, 1.0 / reps), "boot_ci_lo": float(lo),
            "boot_ci_hi": float(hi)}


def mcnemar_exact(flags_a, flags_b):
    a = np.asarray(flags_a, bool)
    b = np.asarray(flags_b, bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n_d = n01 + n10
    if n_d == 0:
        return n01, n10, 1.0
    p = float(sstats.binomtest(min(n01, n10), n_d, 0.5).pvalue)
    return n01, n10, p


def pooled_decisions(anom, tag, assets):
    hit = {k: [] for k in FAM_SHORT}
    tot = {k: [] for k in FAM_SHORT}
    flags_real, flags_fam = [], {k: [] for k in FAM_SHORT}
    for seg in range(len(assets)):
        cal = np.concatenate(anom[tag]["cal"][seg:seg + 1])
        cal = cal[np.isfinite(cal)]
        thr = float(np.quantile(cal, 1 - ALPHA))
        real = np.concatenate(anom[tag]["real"][seg:seg + 1])
        flags_real.append(real > thr)
        for fam, key in zip(FAMILIES, FAM_SHORT):
            f = np.concatenate(anom[tag]["fam"][fam][seg:seg + 1])
            fl = f > thr
            hit[key].append(int(np.nansum(fl)))
            tot[key].append(len(fl))
            flags_fam[key].append(fl)
    rec = {k: float(np.sum(hit[k]) / np.sum(tot[k])) for k in hit}
    fr = np.concatenate(flags_real)
    ff = {k: np.concatenate(v) for k, v in flags_fam.items()}
    ff["ALL"] = np.concatenate(list(ff.values()))
    return rec, fr, ff


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
    epochs = 6 if smoke else EPOCHS_D
    n_slices = 12 if smoke else N_SLICES_D
    print("=" * 66)
    print("E2 深度基线（TS2Vec+马氏 / AT / DCdetector / TimesNet）"
          + ("  [冒烟]" if smoke else ""))
    print(f"协议：R1 三层切分 | cal q95 冻结 | α={ALPHA} | "
          f"epochs={epochs} | slices={n_slices} | seed={seed_run}")
    print("=" * 66)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}
    assets = list(returns)

    anom = {m: {"cal": [], "real": [], "fam": {f: [] for f in FAMILIES}}
            for m in DEEP_METHODS}
    per_asset, train_info = {}, {}

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

        for mi, tag in enumerate(DEEP_METHODS):
            mseed = seed_run + 10000 + ai * 1000 + mi
            model, info = train_deep(tag, r, train_end, mseed,
                                     epochs, n_slices)
            train_info[f"{asset}/{tag}"] = info
            ctx = {}
            if tag == "D1":
                ctx["mu"], ctx["inv"] = d1_fit_mahal(model, anchors)
            anom[tag]["cal"].append(score_deep(tag, model, cal_wins, ctx))
            anom[tag]["real"].append(score_deep(tag, model, test_wins, ctx))
            for fam in FAMILIES:
                anom[tag]["fam"][fam].append(
                    score_deep(tag, model, forged[fam], ctx))
            print(f"    [{tag}] {info['s']}s loss={info['loss']}",
                  flush=True)

        per_asset[asset] = {}
        for tag in DEEP_METHODS:
            seg = len(anom[tag]["cal"]) - 1
            cal = np.concatenate(anom[tag]["cal"][seg:seg + 1])
            cal = cal[np.isfinite(cal)]
            thr = float(np.quantile(cal, 1 - ALPHA))
            real = np.concatenate(anom[tag]["real"][seg:seg + 1])
            entry = {"n_test": len(test_wins),
                     "fpr": float(np.nanmean(real > thr))}
            for fam, key in zip(FAMILIES, FAM_SHORT):
                f = np.concatenate(anom[tag]["fam"][fam][seg:seg + 1])
                entry[key] = float(np.nanmean(f > thr))
            per_asset[asset][tag] = entry

    # ---- 池化指标 ----
    print("\n" + "=" * 66)
    print("池化指标（28 真实测试窗 + 5×28 伪造，cal q95 冻结阈值）")
    print(f"{'配置':<40}{'N1':>6}{'N2':>6}{'N3':>6}{'N4':>6}{'N5':>6}"
          f"{'macro':>8}{'FPR':>7}{'AUC全':>7}")
    print("-" * 96)
    pooled = {}
    flags_store = {}
    for tag in DEEP_METHODS:
        real = np.concatenate(anom[tag]["real"])
        ok_r = np.isfinite(real)
        recalls, flags_real, flags_fam = pooled_decisions(anom, tag, assets)
        flags_store[tag] = flags_fam
        fam_auc = {}
        for fam, key in zip(FAMILIES, FAM_SHORT):
            f = np.concatenate(anom[tag]["fam"][fam])
            f = f[np.isfinite(f)]
            fam_auc[key] = auc_pair(f, real[ok_r])
        all_f = np.concatenate([np.concatenate(anom[tag]["fam"][f])
                                for f in FAMILIES])
        all_f = all_f[np.isfinite(all_f)]
        macro = float(np.mean(list(recalls.values())))
        pooled[tag] = {"recall": recalls, "macro_recall": macro,
                       "fpr": float(np.mean(flags_real)),
                       "auc_family": fam_auc,
                       "auc_all": auc_pair(all_f, real[ok_r])}
        rr = recalls
        print(f"{tag + ' ' + DEEP_DESC[tag]:<40}"
              f"{rr['N1']:.3f} {rr['N2']:.3f} {rr['N3']:.3f} "
              f"{rr['N4']:.3f} {rr['N5']:.3f} "
              f"{macro:>8.3f}{pooled[tag]['fpr']:>7.3f}"
              f"{pooled[tag]['auc_all']:>7.3f}")

    # ---- A0 配对显著性（默认种子，npz 冻结分数） ----
    stats_out = {}
    npz_path = os.path.join(RESULTS_DIR, "e2_main_scores.npz")
    if seed_run == SEED and os.path.exists(npz_path):
        z = np.load(npz_path, allow_pickle=True)
        assets_a0 = [str(a) for a in z["assets"]]
        assert assets_a0 == assets, "资产顺序与 e2_main_scores 不一致"
        a0 = {"A0": {
            "cal": [z[f"anom/A0/cal/{i}"] for i in range(len(assets))],
            "real": [z[f"anom/A0/real/{i}"] for i in range(len(assets))],
            "fam": {fam: [z[f"anom/A0/{fam[:2]}/{i}"]
                          for i in range(len(assets))]
                    for fam in FAMILIES}}}
        # A0 池化判定向量（逐资产 cal q95）
        _, _, a0_flags_fam = pooled_decisions(a0, "A0", assets)
        real_a0_raw = np.concatenate(a0["A0"]["real"])
        print("\nA0 配对显著性（DeLong / McNemar，e2_main 冻结分数）")
        for tag in DEEP_METHODS:
            real_b = np.concatenate(anom[tag]["real"])
            okr = np.isfinite(real_a0_raw) & np.isfinite(real_b)
            real_a0, real_b = real_a0_raw[okr], real_b[okr]
            for scope in FAM_SHORT + ["ALL"]:
                if scope == "ALL":
                    pos_a = np.concatenate(
                        [np.concatenate(a0["A0"]["fam"][f])
                         for f in FAMILIES])
                    pos_b = np.concatenate(
                        [np.concatenate(anom[tag]["fam"][f])
                         for f in FAMILIES])
                else:
                    fam = FAMILIES[FAM_SHORT.index(scope)]
                    pos_a = np.concatenate(a0["A0"]["fam"][fam])
                    pos_b = np.concatenate(anom[tag]["fam"][fam])
                okp = np.isfinite(pos_a) & np.isfinite(pos_b)
                A_a, A_b, zsc, p, degen = delong_paired(
                    pos_a[okp], real_a0, pos_b[okp], real_b)
                entry = {"auc_a": A_a, "auc_b": A_b, "z": zsc, "p": p}
                if degen:
                    entry.update(boot_paired(pos_a[okp], real_a0,
                                             pos_b[okp], real_b))
                    entry["p"] = entry["boot_p"]
                    entry["degenerate"] = True
                stats_out[f"A0vs{tag}/{scope}"] = entry
                if scope in ("N2", "ALL"):
                    print(f"  DeLong A0vs{tag}[{scope}]: "
                          f"AUC {A_a:.3f} vs {A_b:.3f} z={zsc:+.2f} "
                          f"p={entry['p']:.2e}")
            n01, n10, p = mcnemar_exact(a0_flags_fam["ALL"],
                                        flags_store[tag]["ALL"])
            stats_out[f"mcnemar/A0vs{tag}"] = {"only_a": n01,
                                               "only_b": n10, "p": p}
            print(f"  McNemar A0vs{tag}: 仅a={n01} 仅b={n10} p={p:.2e}")

    # ---- 原始分数落盘（默认种子）：供源窗级 cluster 重分析，免重训 ----
    if seed_run == SEED:
        arrays = {"assets": np.array(assets)}
        kinds = [("cal", "cal"), ("real", "real")] + [
            (f, f[:2]) for f in FAMILIES]
        for m in DEEP_METHODS:
            for kind, key in kinds:
                src = (anom[m]["cal"] if kind == "cal"
                       else anom[m]["real"] if kind == "real"
                       else anom[m]["fam"][kind])
                for seg_i, arr in enumerate(src):
                    arrays[f"anom/{m}/{key}/{seg_i}"] = np.asarray(arr)
        deep_npz = os.path.join(RESULTS_DIR, "e2_deep_scores.npz")
        np.savez_compressed(deep_npz, **arrays)
        print(f"原始分数已写入 {deep_npz}")

    # ---- 输出 ----
    res = {
        "smoke": smoke, "epochs": epochs, "n_slices": n_slices,
        "seed": seed_run, "alpha": ALPHA,
        "input_norm": "per-window z-score (instance)",
        "protocol": {"window": WINDOW, "step": STEP, "split": SPLIT_SPEC,
                     "families": FAMILIES, "train_len": TRAIN_LEN},
        "methods": DEEP_DESC,
        "compact_notes": {
            "D1": "6层空洞TCN；实例+对齐时间对比（双负样本）；"
                  "冻结表征+拟合层马氏（pinv+岭，同A5）",
            "D2": "conv嵌入stride4；单层2头；可学习σ高斯先验；"
                  "minimax λ=3；评分=重构误差×softmax(AssDis)",
            "D3": "patch=25；patch间+patch内双注意力分支；InfoNCE"
                  "正对=同实例双分支，负样本=批内+patch置换位移；"
                  "评分=实例级双分支差异",
            "D4": "FFT周期top-2（自输入谱，两block共用）；Inception"
                  "三分支和；等权聚合；评分=MSE"},
        "pooled": pooled,
        "per_asset": per_asset,
        "train_info": train_info,
        "stats": stats_out,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if smoke:
        out_json = os.path.join(RESULTS_DIR, "e2_deep_smoke.json")
    else:
        suffix = "" if seed_run == SEED else f"_s{seed_run}"
        out_json = os.path.join(RESULTS_DIR, f"e2_deep{suffix}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out_json}（elapsed {res['elapsed_s']}s）")


if __name__ == "__main__":
    main()
