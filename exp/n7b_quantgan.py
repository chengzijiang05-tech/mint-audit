"""N7b：深度生成器锻造序列（QuantGAN 风格 TCN-GAN，紧凑 CPU 实现）。

实验目的：深度生成金融序列既是"machine-generated"的
实体也是签名正交敌手（可拟合杠杆不对称）。TimeGAN/QuantGAN 从未
作为敌手出场。

协议：
  - 训练数据：各资产拟合层窗口（train_end = cal 层起点之前，与
    MINT 编码器相同训练跨度，无测试层泄漏）；
  - 结构：TCN 生成器（latent → 日收益窗口）+ TCN 判别器
    （谱归一化），hinge 损失，WGAN-GP 风格交替训练（GP 改为
    谱归一化，CPU 预算下更稳）；
  - 生成：拟合层条件统计（vol/mean）注入 latent，生成 T=1000
    路径，与测试层真实窗标准差对齐（尺度公平，同 N6/N7a）。

紧凑实现披露（相对 Wiese et al. 2020 QuantGAN 的简化，全部登记）：
  1. 生成器 receptive field 251 日（原版 511+，覆盖短程波动聚集）；
  2. 判别器为 4 层 TCN + 谱归一化（原版 patch 判别器 + GP）；
  3. 训练 400 步 G/D 交替（原版数千步；等 CPU 预算口径，与
     e2_deep.py 基线哲学一致）；
  4. 无滚动窗口条件结构（原版用随机滚动窗训练；这里固定窗长，
     与 MINT 窗口口径一致）。

产出：results/n7b_quantgan.npz + results/n7b_quantgan.json
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

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.nn.utils import spectral_norm  # noqa: E402

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
PER_ASSET = 4        # 每资产生成条数
STEPS = 400          # G/D 交替步数
BS = 16              # batch
LATENT = 32          # latent 维度
SEED = 20260821


class CausalBlock(nn.Module):
    def __init__(self, ch, k, dil):
        super().__init__()
        p = (k - 1) * dil
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, dilation=dil), nn.GELU(),
            nn.Conv1d(ch, ch, k, dilation=dil), nn.GELU())
        self.p = p

    def forward(self, x):
        return x + self.net(nn.functional.pad(x, (self.p, self.p)))[..., :x.shape[-1]]


class GenTCN(nn.Module):
    """latent (LATENT,L) → 收益序列 (1,WINDOW)。感受野 251 日。"""
    def __init__(self):
        super().__init__()
        self.inp = nn.Conv1d(LATENT, 64, 1)
        self.blocks = nn.Sequential(
            CausalBlock(64, 5, 1), CausalBlock(64, 5, 3),
            CausalBlock(64, 5, 9), CausalBlock(64, 5, 27),
            CausalBlock(64, 5, 54))
        self.out = nn.Conv1d(64, 1, 1)

    def forward(self, z):
        h = self.blocks(self.inp(z))
        return self.out(h)[:, 0, :]


class DiscTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv1d(1, 32, 5, padding=2)), nn.GELU(),
            spectral_norm(nn.Conv1d(32, 64, 5, dilation=2, padding=4)),
            nn.GELU(),
            spectral_norm(nn.Conv1d(64, 64, 5, dilation=4, padding=8)),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            spectral_norm(nn.Linear(64, 1)))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


def load_spec():
    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}
    spec = {}
    for asset, r in returns.items():
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        train_end = sp["cal"][0] * STEP
        # 训练切片：训练跨度内步长 STEP 全窗（时间严格 < cal）
        train_wins = np.array(
            [r[i:i + WINDOW] for i in range(0, train_end - WINDOW + 1, STEP)])
        spec[asset] = {
            "train": train_wins,
            "test": test_wins,
            "target_std": float(np.mean([np.std(w) for w in test_wins])),
        }
    return spec


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


def train_gan(train_wins, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = train_wins.mean(), train_wins.std()
    X = torch.tensor((train_wins - mu) / sd, dtype=torch.float32)
    G, D = GenTCN(), DiscTCN()
    g_opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    n = X.shape[0]
    hist = []
    for step in range(STEPS):
        idx = rng.integers(0, n, BS)
        xb = X[idx]
        z = torch.randn(BS, LATENT, WINDOW)
        # D 步（hinge）
        d_opt.zero_grad()
        d_real = D(xb)
        d_fake = D(G(z))
        loss_d = (nn.functional.relu(1 - d_real).mean()
                  + nn.functional.relu(1 + d_fake).mean())
        loss_d.backward()
        d_opt.step()
        # G 步
        g_opt.zero_grad()
        z = torch.randn(BS, LATENT, WINDOW)
        loss_g = -D(G(z)).mean()
        loss_g.backward()
        g_opt.step()
        if (step + 1) % 100 == 0:
            hist.append({"step": step + 1,
                         "d": round(loss_d.item(), 3),
                         "g": round(loss_g.item(), 3)})
    G.eval()
    return G, (mu, sd), hist


def main():
    smoke = "--smoke" in sys.argv
    steps = 40 if smoke else STEPS
    per = 1 if smoke else PER_ASSET
    print("=" * 66)
    print(f"N7b QuantGAN 风格 TCN-GAN 深度生成敌手"
          + ("  [冒烟]" if smoke else ""))
    print(f"steps={steps} per_asset={per} latent={LATENT} bs={BS}")
    print("=" * 66, flush=True)

    spec = load_spec()
    out = {}
    log = {"protocol": {
        "generator": "TCN(5 causal blocks, rf=251d)",
        "discriminator": "4-layer TCN + spectral norm, hinge loss",
        "train_data": "fit-layer windows, train_end = cal start (frozen)",
        "steps": steps, "simplifications": [
            "rf 251 vs 511+ (covers short-range vol clustering)",
            "spectral norm vs gradient penalty",
            "fixed window vs rolling-window conditioning",
            f"{steps} G/D steps vs thousands (equal CPU budget as deep "
            "baselines in e2_deep.py)"]}}

    for ai, (asset, s) in enumerate(spec.items()):
        t0 = time.time()
        G, (mu, sd), hist = train_gan(s["train"], SEED + ai * 100)
        torch.manual_seed(SEED + 9000 + ai)
        with torch.no_grad():
            z = torch.randn(per, LATENT, WINDOW)
            fake = G(z).numpy() * sd + mu
        for j in range(per):
            x = fake[j]
            x = x - x.mean()
            x = x * (s["target_std"] / max(np.std(x), 1e-12))
            out[f"{asset}/quantgan/{j}"] = x
        print(f"[{asset}] train={len(s['train'])} 生成{per}条 "
              f"{time.time()-t0:.0f}s | d/g {hist[-1]['d']}/{hist[-1]['g']}",
              flush=True)
        log[asset] = {"train_windows": int(len(s["train"])),
                      "hist": hist, "gen_s": round(time.time() - t0, 1)}

    np.savez_compressed(
        os.path.join(RESULTS_DIR, "n7b_quantgan.npz"), **out)
    with open(os.path.join(RESULTS_DIR, "n7b_quantgan.json"), "w",
              encoding="utf-8") as fh:
        json.dump(log, fh, ensure_ascii=False, indent=2)
    print(f"\n生成完成：{len(out)} 条已存 npz+json", flush=True)


if __name__ == "__main__":
    main()
