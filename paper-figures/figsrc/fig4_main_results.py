"""Fig. 4 - Main comparison heatmap + ablation dot matrix. All values from
the frozen protocol (figdata.json + Table 2 of the manuscript)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (CINN, CINN_MID, FLUOR, GRAY, GRAY_L, INK, NAVY,
                       NAVY_MID, apply_style, heat_cmap, load_figdata,
                       save_all)
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle

d = load_figdata()["fig5"]
apply_style()

fams = ["N1\niid perm.", "N2\ntime rev.", "N3\nscale break",
        "N4\ncross-splice", "N5\nphase forge", "Pooled"]
rows = [
    ("MINT (ours)", [0.499, 0.653, 0.517, 0.536, 0.876, 0.616]),
    ("Classical 8-feat Mahal.", [0.522, 0.500, 0.987, 1.000, 0.587, 0.719]),
    ("Classical 15-feat Mahal.", [0.499, 0.540, 0.949, 0.963, 0.634, 0.717]),
    ("TS2Vec + Mahalanobis", [0.517, 0.542, 0.827, 0.633, 0.356, 0.575]),
    ("Anomaly Transformer", [0.527, 0.514, 0.679, 0.352, 0.302, 0.475]),
    ("DCdetector", [0.506, 0.448, 0.230, 0.126, 0.430, 0.348]),
    ("TimesNet", [0.528, 0.510, 0.629, 0.602, 0.402, 0.534]),
]
M = np.array([r[1] for r in rows])

fig = plt.figure(figsize=(6.77, 2.52))
ax1 = fig.add_axes([0.145, 0.175, 0.40, 0.665])
ax2 = fig.add_axes([0.585, 0.115, 0.405, 0.735])

# ---------------------------------------------------------------- heatmap
norm = TwoSlopeNorm(vmin=0.34, vcenter=0.5, vmax=1.0)
im = ax1.imshow(M, cmap=heat_cmap(), norm=norm, aspect="auto")
ax1.set_xticks(range(6))
ax1.set_xticklabels(fams, fontsize=5.6)
ax1.xaxis.set_ticks_position("top")
ax1.set_yticks(range(len(rows)))
ax1.set_yticklabels([r[0] for r in rows], fontsize=6.1)
for i, (label, _) in enumerate(rows):
    if i == 0:
        ax1.get_yticklabels()[i].set_color(CINN)
        ax1.get_yticklabels()[i].set_fontweight("bold")
    else:
        ax1.get_yticklabels()[i].set_color(INK)
ax1.tick_params(length=0)
for s in ax1.spines.values():
    s.set_visible(False)

best = M.argmax(axis=0)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        dark = abs(v - 0.5) > 0.20
        ax1.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1.00",
                 ha="center", va="center", fontsize=5.9,
                 color="white" if dark else INK,
                 fontweight="bold" if i == best[j] else "normal")
ax1.add_patch(Rectangle((-0.5, -0.5), 6, 1, fill=False, edgecolor=FLUOR,
                        lw=2.0, zorder=5))
ax1.add_patch(Rectangle((0.5, -0.5), 1, 7.5, fill=False, edgecolor=CINN,
                        lw=1.1, linestyle=(0, (3, 2)), zorder=5))
ax1.text(1.0, 1.16, "structural blindness axis", ha="center", va="bottom",
         fontsize=5.9, color=CINN, fontweight="bold",
         transform=ax1.get_xaxis_transform())
ax1.annotate("", xy=(1.0, 1.10), xytext=(1.0, 1.15),
             xycoords=ax1.get_xaxis_transform(),
             arrowprops=dict(arrowstyle="-|>", color=CINN, lw=0.8))

cb = fig.colorbar(im, ax=ax1, orientation="horizontal", fraction=0.055,
                  pad=0.16, aspect=26, shrink=0.85)
cb.set_ticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
cb.set_label("AUC of candidate-versus-real ranking", fontsize=5.6)
cb.ax.tick_params(labelsize=5.4, length=1.5, pad=1.5)
cb.outline.set_linewidth(0.5)

# ------------------------------------------------------- ablation dots
sa = d["seed_agg"]
ARMS = [
    ("Full MINT", "A0", None),
    ("random augmentations", "A1", "fpr"),
    ("IAAFT-only operators", "A2", "fpr"),
    ("z-score normalization", "A3", "fpr"),
    ("absolute score", "A4", "auc_all"),
]
METRICS = [("auc_all", "Pooled AUC"), ("n2", "N2"), ("fpr", "FPR"),
           ("macro", "Macro")]
n_r = len(ARMS)
strip_w = 0.8
gap = 0.5
x0 = 2.6
y_top = n_r - 0.5

for mi, (key, title) in enumerate(METRICS):
    xc = x0 + mi * (strip_w + gap)
    vals = [sa[a][key + "_mean"] for _, a, _ in ARMS]
    sds = [sa[a][key + "_sd"] for _, a, _ in ARMS]
    lo = min(v - s for v, s in zip(vals, sds))
    hi = max(v + s for v, s in zip(vals, sds))
    pad = 0.12 * (hi - lo + 0.02)
    lo, hi = lo - pad, hi + pad
    ax2.plot([xc - strip_w / 2, xc + strip_w / 2], [y_top + 0.62] * 2,
             color=GRAY_L, lw=0.6)
    ax2.text(xc, y_top + 0.85, title, ha="center", va="bottom", fontsize=5.0,
             fontweight="bold", color=INK)
    ax2.plot([xc - strip_w / 2, xc + strip_w / 2], [-0.62] * 2,
             color=GRAY_L, lw=0.6)
    for t in (0.5,) if key in ("auc_all", "n2", "macro") else (0.05,):
        if lo < t < hi:
            xx = xc - strip_w / 2 + (t - lo) / (hi - lo) * strip_w
            ax2.plot([xx, xx], [-0.5, y_top], color=GRAY, lw=0.5,
                     linestyle=(0, (2, 2)), zorder=1)
            ax2.text(xx, -0.55, f"{t:.2f}" if t != 0.05 else "0.05\nnominal",
                     ha="center", va="top", fontsize=4.8, color=GRAY)
    for ri, (label, a, brk) in enumerate(ARMS):
        y = y_top - ri
        v = sa[a][key + "_mean"]
        s = sa[a][key + "_sd"]
        xv = xc - strip_w / 2 + (v - lo) / (hi - lo) * strip_w
        xlo = xc - strip_w / 2 + (max(v - s, lo) - lo) / (hi - lo) * strip_w
        xhi = xc - strip_w / 2 + (min(v + s, hi) - lo) / (hi - lo) * strip_w
        ax2.plot([xlo, xhi], [y, y], color=NAVY if ri else CINN, lw=0.7,
                 zorder=3)
        if brk == key and ri:
            ax2.add_patch(FancyBboxPatch((xc - strip_w / 2 - 0.09, y - 0.34),
                                         strip_w + 0.18, 0.68,
                                         boxstyle="round,pad=0.02",
                                         facecolor="none", edgecolor=CINN,
                                         lw=1.0, linestyle="-", zorder=5))
        ax2.scatter([xv], [y], s=17 if ri else 22,
                    facecolor=CINN if ri == 0 else "white",
                    edgecolor=CINN if ri == 0 else NAVY, linewidth=0.9,
                    zorder=6)
        if ri == 0:
            ax2.text(xv, y + 0.30, f"{v:.3f}".rstrip("0").rstrip("."),
                     ha="center", va="bottom", fontsize=4.9, color=CINN,
                     fontweight="bold")

x_right = x0 + 4 * (strip_w + gap) - gap
ax2.set_xlim(-3.55, x_right + 0.8)
ax2.set_ylim(-1.15, n_r + 1.15)
ax2.axis("off")
for ri, (label, a, brk) in enumerate(ARMS):
    y = y_top - ri
    ax2.text(-3.45, y, label, ha="left", va="center", fontsize=5.4,
             color=CINN if ri == 0 else INK,
             fontweight="bold" if ri == 0 else "normal")
    if ri:
        te = {"fpr": "calib. breaks", "auc_all": "score collapses"}.get(brk, "")
        ax2.text(x0 - strip_w / 2 - 0.3, y, te, ha="right", va="center",
                 fontsize=4.9, color=CINN)

fig.text(0.012, 0.965, "(a)", fontsize=7.6, fontweight="bold", color=INK)
fig.text(0.545, 0.965, "(b)", fontsize=7.6, fontweight="bold", color=INK)

save_all(fig, "fig4_main_results")
