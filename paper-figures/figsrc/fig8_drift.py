"""Fig. 8 - Deployment drift in one canvas: the certification e-detector D_t
on the CN stream (lollipop, alarms in cinnabar) against the silent US control,
with an inset dumbbell chart of the four deployment arms' worst-block FPR."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (CINN, CINN_BG, FLUOR, GRAY, GRAY_L, INK, NAVY,
                       NAVY_L, apply_style, load_figdata, save_all)
import matplotlib.pyplot as plt

d = load_figdata()["fig8"]
apply_style()

CN, US = d["CN"], d["US"]
B = d["threshold_B"]
arms = d["arms_max_block_fpr"]
GOLD = "#B8860B"
JADE = "#3A7D44"

wcn = np.asarray(CN["win"], dtype=float)
dcn = np.asarray(CN["detector"], dtype=float)
wus = np.asarray(US["win"], dtype=float)
dus = np.asarray(US["detector"], dtype=float)
alarm = set(CN["alarms"])

fig = plt.figure(figsize=(6.77, 2.72))
ax = fig.add_axes([0.075, 0.175, 0.899, 0.745])

# persistent-drift span and crash marker
ax.axvspan(75, 84, color=CINN_BG, alpha=0.55, zorder=0, lw=0)
ax.axvline(64, color=GRAY, lw=0.8, ls=(0, (1.5, 2)), zorder=1)

# CN lollipop
ax.vlines(wcn, 0.1, dcn, color=NAVY_L, lw=0.85, zorder=2)
for w, v in zip(wcn, dcn):
    if w in alarm:
        ax.plot([w], [v], "o", ms=4.8, color=CINN, zorder=5,
                markeredgecolor="white", markeredgewidth=0.7)
    else:
        ax.plot([w], [v], "o", ms=2.1, color=NAVY, zorder=4,
                markeredgecolor="white", markeredgewidth=0.4)

# US control stream
ax.plot(wus, dus, color=GRAY, lw=1.1, zorder=3)

# budget line in two segments (the gap hides behind the inset)
ax.hlines(B, 43.5, 46.0, color=CINN, lw=0.9, ls=(0, (4, 2)), zorder=2)
ax.hlines(B, 62.6, 103.5, color=CINN, lw=0.9, ls=(0, (4, 2)), zorder=2)
ax.text(103.2, B * 0.80, "budget $B = 1200$", fontsize=5.6, color=CINN,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.12,rounding_size=0.2",
                  facecolor="white", edgecolor="none", alpha=0.85))

# refit marker
ax.plot([85], [0.75], "o", ms=6.4, markerfacecolor="none",
        markeredgecolor=JADE, markeredgewidth=1.1, zorder=6)

ax.set_yscale("log")
ax.set_xlim(43.5, 103.5)
ax.set_ylim(0.1, 7000)
ax.set_yticks([0.1, 1, 10, 100, 1000])
ax.set_xticks([45, 55, 65, 75, 85, 95])
ax.set_xlabel("deployment window $t$")
ax.set_ylabel("certification e-detector $D_t$")
ax.spines[["top", "right"]].set_visible(False)

# ---------------------------------------------------------------- events
ax.text(65.4, 5900, "2015-06 crash\n(transient, no alarm)", fontsize=5.5,
        color=GRAY, ha="left", va="top", linespacing=1.35)
ax.text(79.5, 5600, "persistent drift\n$W_{\\mathrm{cert}}$: $25.1 \\to 1.4$",
        fontsize=5.2, color=CINN, ha="center", va="top", linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.2",
                  facecolor="white", edgecolor="none", alpha=0.85))
ax.annotate("alarm $t{=}84$\n($D{=}1207$)", xy=(83.6, 1330),
            xytext=(71.0, 1450), fontsize=5.2, color=CINN, ha="center",
            va="top", fontweight="bold", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.2",
                      facecolor="white", edgecolor="none", alpha=0.9),
            arrowprops=dict(arrowstyle="-|>", color=CINN, lw=0.7,
                            connectionstyle="arc3,rad=-0.15"))
ax.annotate("refit (164 s),\ncertification recovers to 16–27",
            xy=(85.2, 1.1), xytext=(87.2, 75), fontsize=5.5, color=JADE,
            ha="center", linespacing=1.35,
            arrowprops=dict(arrowstyle="-|>", color=JADE, lw=0.75,
                            connectionstyle="arc3,rad=0.2"))
ax.annotate("2nd alarm $t{=}94$\n($D{=}2851$)", xy=(94.0, 3120),
            xytext=(88.5, 4700), fontsize=5.5, color=CINN, ha="center",
            linespacing=1.35,
            arrowprops=dict(arrowstyle="-|>", color=CINN, lw=0.75))
ax.annotate("silent through COVID:\n$D_{\\max} = 41$, a $29\\times$ margin",
            xy=(100.0, 40), xytext=(97.8, 150), fontsize=5.5, color=GRAY,
            ha="center", linespacing=1.35,
            arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.7))

handles = [
    plt.Line2D([], [], color=NAVY, lw=0, marker="o", ms=3.4,
               markerfacecolor=NAVY, label="CN stream $D_t$ (drift watch)"),
    plt.Line2D([], [], color=GRAY, lw=1.1, label="US stream $D_t$ (control)"),
]
ax.legend(handles=handles, loc="upper right", fontsize=5.6, frameon=True,
          framealpha=0.95, edgecolor="#DDD", handlelength=1.4,
          borderpad=0.35, labelspacing=0.3, bbox_to_anchor=(0.995, 1.0))

# ------------------------------------------------- inset: four-arm FPR
axi = ax.inset_axes([0.045, 0.560, 0.270, 0.390])
axi.set_facecolor("white")
keys = ["A1_classical_frozen", "A2_classical_refreeze",
        "A3_mint_frozen", "A4_dual_eprocess"]
labels = ["classical\nfrozen", "classical\nrefreeze", "MINT\nfrozen",
          "MINT dual\ne-process"]
cnv = [arms[k]["CN"] for k in keys]
usv = [arms[k]["US"] for k in keys]
x = np.arange(4)

axi.axvspan(2.62, 3.38, color=FLUOR, alpha=0.40, zorder=0, lw=0)
axi.axhline(0.05, color=CINN, lw=0.7, ls=(0, (3, 2)), zorder=2)
axi.text(-0.52, 0.075, "$\\alpha$", fontsize=5.0, color=CINN, va="bottom")

for xi, c, u in zip(x, cnv, usv):
    axi.plot([xi, xi], [c, u], color=GRAY_L, lw=0.9, zorder=2)
    axi.plot([xi], [c], "o", ms=4.6, color=NAVY, zorder=4,
             markeredgecolor="white", markeredgewidth=0.5)
    axi.plot([xi], [u], "o", ms=4.6, markerfacecolor="white",
             markeredgecolor=GOLD, markeredgewidth=1.0, zorder=5)
    axi.text(xi - 0.12, c, f"{c:.2f}".rstrip("0").rstrip("."), fontsize=4.8,
             color=NAVY, ha="right", va="center")
    axi.text(xi + 0.12, u, f"{u:.2f}".rstrip("0").rstrip("."), fontsize=4.8,
             color=GOLD, ha="left", va="center")
axi.text(3.0, 0.42, "only arm safe\non both streams", fontsize=4.7,
         color=INK, ha="center", va="center", linespacing=1.3)

axi.scatter([-0.30], [0.955], s=10, color=NAVY, zorder=4)
axi.text(-0.22, 0.955, "CN", fontsize=4.8, color=INK, va="center")
axi.scatter([-0.30], [0.845], s=10, facecolor="white", edgecolor=GOLD,
            linewidth=0.9, zorder=5)
axi.text(-0.22, 0.845, "US", fontsize=4.8, color=INK, va="center")

axi.set_xlim(-0.55, 3.55)
axi.set_ylim(-0.07, 1.06)
axi.set_xticks(x)
axi.set_xticklabels(labels, fontsize=5.0)
axi.set_yticks([0, 0.5, 1.0])
axi.tick_params(length=1.6, pad=1.5, labelsize=4.8)
axi.set_title("worst-block FPR", fontsize=5.6, color=INK, fontweight="bold",
              pad=2.5)
for s in axi.spines.values():
    s.set_linewidth(0.55)
    s.set_color(GRAY)

save_all(fig, "fig8_drift")
