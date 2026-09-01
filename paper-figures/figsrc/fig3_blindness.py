"""Fig. 3 - Structural blindness of the classical fingerprint kit: the
collapse scatter. Real data: Mahalanobis distance of each of the 28 test
windows vs its own time reversal (8-feature engine), frozen in figdata."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (ASSET_COLORS, CINN, CINN_MID, GRAY, INK, NAVY,
                       apply_style, load_figdata, save_all)
import matplotlib.pyplot as plt

d = load_figdata()["fig4"]
apply_style()

fig, ax = plt.subplots(figsize=(3.42, 3.05))
x = np.asarray(d["d8_x"])
y = np.asarray(d["d8_rev"])
assets = d["asset"]

lim = [0, max(x.max(), y.max()) * 1.06]
ax.fill_between(lim, [v * 0.85 for v in lim], [v * 1.15 for v in lim],
                color="#E4E9F0", zorder=0, lw=0)
ax.plot(lim, lim, color=INK, lw=0.9, zorder=2)

order = ["equity", "bond", "fx", "gold", "copper"]
for a in order:
    m = [i for i, s in enumerate(assets) if s == a]
    ax.scatter(x[m], y[m], s=15, facecolor=ASSET_COLORS[a], edgecolor="white",
               linewidth=0.5, zorder=4, label=a)

ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Mahalanobis distance of window $w$  (8-feature engine)")
ax.set_ylabel("distance of its time reversal $\\tilde{w}$")
ax.set_aspect("equal")

r8, r15 = d["corr_d8"], d["corr_d15"]
ax.annotate(f"$r = {r8:.3f}$", xy=(11.5, 13.5), xytext=(15.5, 9.2),
            fontsize=7.6, fontweight="bold", color=INK,
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6,
                            connectionstyle="arc3,rad=0.2"))

box = ("three layers of the collapse\n"
       "rank invariants: shift $\\leq 3{\\times}10^{-14}$\n"
       "segment estimators: artifact ratio\n"
       "1.4–1.8$\\times$ on reversible controls\n"
       "leverage asymmetry: $51.2\\times$ genuine\n"
       "arrow-of-time signal (Wilcoxon $p{=}7.5{\\times}10^{-9}$)")
ax.text(0.035, 0.965, box, transform=ax.transAxes, fontsize=5.9,
        va="top", ha="left", color=INK, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor=CINN, linewidth=0.8))

ax.text(0.965, 0.13, f"15-feature engine: $r = {r15:.3f}$",
        transform=ax.transAxes, fontsize=6.2, va="bottom", ha="right",
        color=NAVY)
ax.text(0.965, 0.03, "recall $-$ FPR $= +0.036$\nat the frozen threshold",
        transform=ax.transAxes, fontsize=6.2, va="bottom", ha="right",
        color=CINN, fontweight="bold")

leg = ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.145),
                handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25,
                ncol=2, columnspacing=0.8)
for t in leg.get_texts():
    t.set_fontsize(5.8)

fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.98))
save_all(fig, "fig3_blindness")
