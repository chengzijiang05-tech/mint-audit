"""Fig 2 (v4, single panel, 跨栏 190 mm): 证据货币，方法节的实验落地。

单图版（无 (a)/(b) 分面板）：每窗 e 值对数点云一图呈现。免校准规则
W ≥ 1/α = 20 认证 1/28 真实窗（最强 equity 窗 W=24.7）；N1 数值扰动
结构完好，按类型化语义与真实窗同率认证（1/28，灰空心）；N2–N5 共
112 条结构伪造零认证，N5 相位伪造全部贴在公平线 E[W]=1 附近。每列
顶部认证计数行（1/28 · 1/28 · 0/28 ×4）是全图唯一叙事层，荧光黄带
= 认证区，E[W]=1 为公平赌注参考线。
布局约束：计数行 y=58 完整落在认证带内（不骑轴顶框线）；E[W]=1
标签置于 N4 列中心的线下方（该列 28 点全部位于 1.30 之上，零遮挡）；
阈值标签贴线右端上方，无白底框。
数据：figdata['fig2_ev']（e2_main_scores.npz W/A0 主运行冻结，20260831 口径审计）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fig_style import (NAVY, CINNABAR, FLUOR, INK, INK2, GRAY, save)

D = json.load(open(Path(__file__).parent / "figdata.json", encoding="utf-8"))
EV = D["fig2_ev"]

fig = plt.figure(figsize=(7.48, 3.05))
ax = fig.add_axes([0.065, 0.175, 0.918, 0.685])
ax.set_yscale("log")
ax.set_xlim(-0.7, 5.7)
ax.set_ylim(0.07, 85)
ax.set_yticks([0.1, 1, 10])
ax.set_yticklabels(["0.1", "1", "10"])
ax.grid(axis="x", visible=False)

thr = EV["threshold"]
rng = np.random.default_rng(20260820)

ax.axhspan(thr, 85, color=FLUOR, alpha=0.10, lw=0, zorder=0)
ax.axhline(1, color=GRAY, lw=0.8, ls=(0, (1, 2)), zorder=1)
ax.axhline(thr, color=CINNABAR, lw=1.0, ls=(0, (4, 2)), zorder=1)

# real windows: certified solid, uncertified hollow
xs, ys, cert = [], [], []
for a in EV["assets"]:
    for v in EV["real_by_asset"][a]:
        xs.append(rng.uniform(-0.15, 0.15))
        ys.append(v)
        cert.append(v >= thr)
xs, ys, cert = np.array(xs), np.array(ys), np.array(cert)
ax.scatter(xs[cert], ys[cert], s=17, c=NAVY, lw=0, zorder=3)
ax.scatter(xs[~cert], ys[~cert], s=13, facecolors="white", edgecolors=NAVY,
           linewidths=0.8, zorder=3)
imax = int(np.argmax(ys))
ax.annotate("$W{=}24.7$", xy=(xs[imax], ys[imax]), xytext=(0.62, 40.0),
            fontsize=6.2, color=NAVY, weight="bold", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.6,
                            connectionstyle="arc3,rad=-0.25"), zorder=4)

# N1 numeric perturbations: structurally intact, hollow grey
n1 = np.array(EV["fam"]["N1"])
ax.scatter(1 + rng.uniform(-0.15, 0.15, len(n1)), n1, s=13, facecolors="white",
           edgecolors=INK2, linewidths=0.8, zorder=3)

# N2-N5 structural forgeries: solid cinnabar
for k, f in enumerate(("N2", "N3", "N4", "N5")):
    v = np.array(EV["fam"][f])
    ax.scatter(2 + k + rng.uniform(-0.15, 0.15, len(v)), v, s=15, c=CINNABAR,
               alpha=0.88, lw=0, zorder=3)

# certification count per column, the single narrative layer
cnt = [sum(1 for a in EV["assets"] for v in EV["real_by_asset"][a] if v >= thr),
       sum(1 for v in EV["fam"]["N1"] if v >= thr),
       sum(1 for v in EV["fam"]["N2"] if v >= thr),
       sum(1 for v in EV["fam"]["N3"] if v >= thr),
       sum(1 for v in EV["fam"]["N4"] if v >= thr),
       sum(1 for v in EV["fam"]["N5"] if v >= thr)]
col = [NAVY, INK2, CINNABAR, CINNABAR, CINNABAR, CINNABAR]
for xc, c, cl in zip(range(6), cnt, col):
    ax.text(xc, 58, f"{c}/28", fontsize=7.2, color=cl, weight="bold",
            ha="center", va="center", zorder=4)

ax.text(5.6, 22.5, "$W = 1/\\alpha = 20$", fontsize=6.5, color=CINNABAR,
        ha="right", va="bottom", zorder=4)
ax.text(3.0, 0.92, "$\\mathbb{E}[W] = 1$", fontsize=6.2, color=INK2,
        ha="center", va="top", zorder=4)
ax.text(5.6, 0.55, "N5 sits on\nthe fair line", fontsize=5.6, color=CINNABAR,
        ha="right", va="center", zorder=4, linespacing=1.3)

ax.set_xticks(range(6))
ax.set_xticklabels(["Real\nwindows", "N1\nnumeric pert.", "N2\ntime rev.",
                    "N3\nscale brk", "N4\nsplice", "N5\nphase forge"],
                   fontsize=6.8)
ax.set_ylabel("per-window e-value $W(x)$", fontsize=7.6)
ax.set_title("The e-value as evidence currency: one certification, "
             "zero false trust", fontsize=8.5, color=INK,
             loc="left", pad=5)

save(fig, "fig2_evidence")
