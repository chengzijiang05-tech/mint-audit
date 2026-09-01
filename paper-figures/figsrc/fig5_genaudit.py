"""Fig. 5 - The trust margin: all 168 machine-generated paths and the 28 real
test windows against the certification line W = 1/alpha = 20.
(a) every path as a dot on log W: machine paths sit on the fair line E[W]=1,
each generator's strongest excursion (solid marker, dotted lead to the line)
stops short of it, and the only certification in either direction is a real
equity window; (b) best attempt per machine x market as a bubble matrix.
Data: figdata['fig5_genaudit'] (e17_genpaths_w.json frozen)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (CINN, CINN_MID, FLUOR, GRAY, INK, NAVY, NAVY_BG,
                       apply_style, load_figdata, save_all)
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

D = load_figdata()["fig5_genaudit"]
apply_style()

THR = D["threshold"]
assets = D["assets"]
var_order = sorted(D["variants"], key=lambda v: -D["variant_max"][v])
rows = [("real", None)] + [("var", v) for v in var_order]

rng = np.random.default_rng(20260901)

MM = 2.8346                                 # pt per mm
W_B_MM = 0.385 * 6.77 * 25.4                # panel (b) width in mm
H_MM = 0.680 * 2.80 * 25.4                  # axes height in mm
B_X_MM2U = 5.1 / W_B_MM                     # panel (b) x units per mm
B_Y_MM2U = 11.05 / H_MM                     # panel (b) y units per mm
Y_LO, Y_HI = 9.9, -1.15                     # inverted: row 0 on top


def bubble_r(w):
    return min(0.45 + 1.35 * np.log10(max(w, 1.0)), 2.35)


fig = plt.figure(figsize=(6.77, 2.80))
axa = fig.add_axes([0.185, 0.150, 0.300, 0.680])
axb = fig.add_axes([0.580, 0.150, 0.385, 0.680])
for ax in (axa, axb):
    ax.set_ylim(Y_LO, Y_HI)
    ax.tick_params(length=1.6, pad=1.5, labelsize=5.8)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color(GRAY)
axb.spines["left"].set_visible(False)

# ---------------------------------------------------------------- panel (a)
axa.set_xlim(0.055, 48)
axa.set_xscale("log")
axa.set_xticks([0.1, 1, 10, 40])
axa.set_xticklabels(["0.1", "1", "10", "40"])
axa.axhspan(-0.5, 0.5, color=NAVY_BG, alpha=0.55, lw=0, zorder=0)
axa.axvspan(THR, 48, color=FLUOR, alpha=0.10, lw=0, zorder=0)
axa.axvline(1.0, color=GRAY, lw=0.7, ls=(0, (1, 1.8)), zorder=1)
axa.axvline(THR, color=CINN, lw=1.0, ls=(0, (4, 2)), zorder=2)

for i, (kind, v) in enumerate(rows):
    if kind == "real":
        ws = np.asarray(D["real_w"], dtype=float)
        jit = rng.uniform(-0.10, 0.10, len(ws))
        for w, dy in zip(ws, jit):
            axa.plot([w], [i + dy], "o", ms=2.6, markerfacecolor="white",
                     markeredgecolor=NAVY, markeredgewidth=0.5, zorder=4)
        w = float(ws.max())
        axa.hlines(i, THR, w, color=NAVY, lw=0.9, ls=(0, (1, 1.6)), zorder=2)
        axa.plot([w], [i], "o", ms=4.4, color=NAVY, zorder=5,
                 markeredgecolor="white", markeredgewidth=0.6)
    else:
        ws = np.asarray(D["paths_w"][v], dtype=float)
        jit = rng.uniform(-0.12, 0.12, len(ws))
        for w, dy in zip(ws, jit):
            axa.plot([w], [i + dy], "o", ms=2.2, color=CINN, alpha=0.72,
                     markeredgecolor="none", zorder=4)
        w = D["variant_max"][v]
        axa.hlines(i, w, THR, color=CINN_MID, lw=0.75, ls=(0, (1, 1.6)),
                   zorder=2)
        axa.plot([w], [i], "o", ms=4.0, color=CINN, zorder=5,
                 markeredgecolor="white", markeredgewidth=0.55)

axa.text(18.5, -0.70, "certification line\n$W = 1/\\alpha = 20$",
         fontsize=5.8, color=CINN, ha="right", va="top", linespacing=1.25)
axa.text(1.30, 9.35, "$\\mathbb{E}[W]{=}1$", fontsize=5.2, color=GRAY,
         ha="left", va="top")
axa.text(47.0, 0.55, "the one certification", fontsize=5.2, color=NAVY,
         weight="bold", ha="right", va="center", zorder=6,
         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5))
axa.plot([24.65, 24.65], [0.10, 0.30], color=NAVY, lw=0.7, zorder=5)
axa.text(19.2, 1.50, "3% short of the line", fontsize=5.4, color=CINN,
         ha="right", va="center")
axa.text(1.75, 8.55, "17$\\times$ below", fontsize=5.4, color=CINN,
         ha="left", va="center")

axa.set_yticks(range(len(rows)))
axa.set_yticklabels(
    ["Real windows (%d)" % D["n_real"]] +
    ["%s %s (%d)" % (D["variant_family"][v], D["variant_label"][v],
                     D["variant_n"][v]) for v in var_order], fontsize=6.0)
for lab, (k, _) in zip(axa.get_yticklabels(), rows):
    lab.set_color(NAVY if k == "real" else INK)
    if k == "real":
        lab.set_fontweight("bold")
axa.set_xlabel("orbit e-value $W$ (log scale)", fontsize=7.0)
axa.set_title("(a) closest approach: 0/168 certified",
              fontsize=7.2, color=INK, fontweight="bold", loc="left", pad=4)

# ---------------------------------------------------------------- panel (b)
axb.set_xlim(-0.55, 4.55)
axb.set_xticks(range(5))
axb.set_xticklabels(["Equity", "Bond", "FX", "Gold", "Copper"], fontsize=6.2)
axb.tick_params(axis="y", length=0, labelleft=False)
axb.grid(axis="x", lw=0.4, color="#E8E8E8", zorder=0)
axb.axhspan(-0.5, 0.5, color=NAVY_BG, alpha=0.55, lw=0, zorder=0)
axb.set_title("(b) best attempt by machine and market",
              fontsize=7.2, color=INK, fontweight="bold", loc="left", pad=4)

for i, (kind, v) in enumerate(rows):
    for j, a in enumerate(assets):
        w = D["real_cell_max"][a] if kind == "real" else D["variant_cell_max"][v][a]
        r = bubble_r(w)
        col = NAVY if kind == "real" else CINN
        axb.scatter([j], [i], s=np.pi * (r * MM) ** 2, color=col, alpha=0.92,
                    linewidths=0.45, edgecolors="white", zorder=3)

ring_r = bubble_r(D["real_cell_max"]["equity"]) + 0.45
axb.add_patch(Ellipse((0, 0), 2 * ring_r * B_X_MM2U, 2 * ring_r * B_Y_MM2U,
                      facecolor="none", edgecolor=CINN, lw=0.9,
                      ls=(0, (2, 1.4)), zorder=4))

lab_cells = [(0, 0, "24.65", NAVY), (1, 1, "19.45", CINN),
             (2, 0, "14.66", CINN), (1, 2, "12.75", CINN)]
for i, j, s, col in lab_cells:
    w = (D["real_cell_max"][assets[j]] if i == 0
         else D["variant_cell_max"][var_order[i - 1]][assets[j]])
    xoff = bubble_r(w) * B_X_MM2U + 0.12
    axb.text(j + xoff, i, s, fontsize=5.2, color=col, ha="left",
             va="center", fontweight="bold", zorder=5)

# size legend strip + dashed-ring key
yL = 9.15
for w, xj in [(1.0, 0.10), (5.0, 0.66), (20.0, 1.55)]:
    r = bubble_r(w)
    axb.add_patch(Ellipse((xj, yL), 2 * r * B_X_MM2U, 2 * r * B_Y_MM2U,
                          facecolor=CINN, edgecolor=CINN, lw=0.7, zorder=4))
    axb.text(xj + r * B_X_MM2U + 0.06, yL, "%g" % w, fontsize=5.4, color=INK,
             ha="left", va="center")
axb.add_patch(Ellipse((2.50, yL), 2 * ring_r * B_X_MM2U, 2 * ring_r * B_Y_MM2U,
                      facecolor="none", edgecolor=CINN, lw=0.9,
                      ls=(0, (2, 1.4)), zorder=4))
axb.text(2.76, yL, "dashed ring: certified", fontsize=5.2, color=INK,
         ha="left", va="center")

save_all(fig, "fig5_genaudit")
