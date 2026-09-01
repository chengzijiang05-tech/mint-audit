"""Fig. 7 - Cross-market transfer in one canvas: the transfer law
(certification rate vs invariant separation, 19 frozen points) with an inset
slope chart showing the classical FPR collapse against MINT's 0.000."""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (CINN, CINN_MID, GRAY, INK, NAVY, NAVY_MID,
                       apply_style, load_figdata, save_all)
import matplotlib.pyplot as plt

d = load_figdata()["fig7"]
apply_style()

mk = d["markets"]
fc = np.asarray(d["fpr_classical"], dtype=float)
pts = d["delta_certify_points"]
rho = d["spearman"]["rho"]
pval = d["spearman"]["p"]

GOLD = "#B8860B"
MKT = {"CN": NAVY, "US": NAVY_MID, "HK": GOLD}


def dodge_x(points, step):
    counts = Counter((round(p["delta_tv"], 4), p["certify_real"]) for p in points)
    seen = {}
    out = []
    for p in points:
        key = (round(p["delta_tv"], 4), p["certify_real"])
        n = seen.get(key, 0)
        seen[key] = n + 1
        tot = counts[key]
        out.append(p["delta_tv"] + (n - (tot - 1) / 2) * step)
    return out


fig = plt.figure(figsize=(6.77, 2.66))
ax = fig.add_axes([0.062, 0.160, 0.918, 0.780])

ax.axhspan(-0.075, 0.045, color="#F4F1E6", zorder=0)
ax.text(1.058, -0.015, "abstention floor", fontsize=5.2, color=GRAY,
        va="center", ha="right", style="italic")

xs = np.array([p["delta_tv"] for p in pts])
ys = np.array([p["certify_real"] for p in pts])
k, b = np.polyfit(xs, ys, 1)
xr = np.linspace(0.12, 1.02, 60)
ax.plot(xr, k * xr + b, color=INK, lw=0.8, ls=(0, (4, 2)), zorder=2)

mx = [p for p in pts if p["kind"] == "matrix"]
en = [p for p in pts if p["kind"] == "ensemble"]
ex = dodge_x(en, 0.016)
mxx = dodge_x(mx, 0.028)
ax.scatter(ex, [p["certify_real"] for p in en], s=20, marker="s",
           facecolor=CINN_MID, edgecolor="white", linewidth=0.5, zorder=4,
           label="CN five-asset ensembles")
ax.scatter(mxx, [p["certify_real"] for p in mx], s=26, marker="o",
           facecolor=NAVY, edgecolor="white", linewidth=0.6, zorder=5,
           label="3$\\times$3 matrix cells")

ax.annotate("local refit US$\\to$US:\ncertify 100% at $\\delta{=}1.0$",
            xy=(0.99, 1.0), xytext=(0.842, 0.760), fontsize=5.6, color=NAVY,
            linespacing=1.35, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=0.7,
                            connectionstyle="arc3,rad=-0.15"))
ax.annotate("HK$\\to$US: 0.90 certified", xy=(1.0, 0.90),
            xytext=(0.872, 0.340), fontsize=5.6, color=INK, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.7,
                            connectionstyle="arc3,rad=0.2"))
ax.annotate("CN$\\to$HK at $\\delta_{\\mathrm{TV}}{=}0.50$:\nzero certification,\nzero false trust",
            xy=(0.497, 0.03), xytext=(0.40, 0.33), fontsize=5.6,
            color=CINN, linespacing=1.35, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.2",
                      facecolor="white", edgecolor="none", alpha=0.9),
            arrowprops=dict(arrowstyle="-|>", color=CINN, lw=0.7,
                            connectionstyle="arc3,rad=-0.2"))

ax.set_xlim(0.10, 1.07)
ax.set_ylim(-0.075, 1.10)
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("invariant separation $\\delta_{\\mathrm{TV}}$ "
              "(target law vs.\\ training operator mixture)")
ax.set_ylabel("real-window certification rate")
ax.spines[["top", "right"]].set_visible(False)

ax.text(1.062, 1.052, f"Spearman $\\rho = {rho:.3f}$, $p = {pval:.4f}$, "
        "$n = 19$", fontsize=6.0, color=INK, fontweight="bold", ha="right")
ax.legend(loc="lower right", fontsize=5.6, frameon=True, framealpha=0.95,
          edgecolor="#DDD", handlelength=1.1, borderpad=0.35,
          labelspacing=0.28, bbox_to_anchor=(0.995, 0.045))

# ------------------------------------------------- inset: FPR slope chart
axi = ax.inset_axes([0.058, 0.500, 0.300, 0.455])
axi.set_xlim(-0.62, 1.58)
axi.set_ylim(-0.045, 0.97)
axi.set_xticks([0, 1])
axi.set_xticklabels(["classical\nthreshold transport", "MINT\nlocal orbits"],
                    fontsize=5.2)
axi.set_yticks([0, 0.4, 0.8])
axi.tick_params(length=1.6, pad=1.5, labelsize=5.0)
axi.axhline(0.05, color=CINN, lw=0.7, ls=(0, (3, 2)), zorder=2)
axi.text(-0.58, 0.085, "$\\alpha$", fontsize=5.2, color=CINN, va="bottom")

for i, a in enumerate(mk):
    for j in range(len(mk)):
        v0 = fc[i][j]
        axi.plot([0, 1], [v0, 0.0], color=MKT[a],
                 lw=1.15 if v0 >= 0.6 else 0.75,
                 alpha=0.95 if v0 >= 0.6 else 0.55, zorder=3,
                 marker="o", ms=2.6, markerfacecolor="white",
                 markeredgewidth=0.7)
for lbl, y in [("CN$\\to$US 0.80", 0.832), ("CN$\\to$HK 0.75", 0.718),
               ("HK$\\to$US 0.60", 0.60)]:
    axi.text(-0.055, y, lbl, fontsize=4.9, color=INK, ha="right", va="center")
axi.annotate("0.000\nin all 9 cells", xy=(1.0, 0.035), xytext=(1.10, 0.36),
             fontsize=5.0, color=CINN, fontweight="bold", ha="left",
             va="center", linespacing=1.3,
             arrowprops=dict(arrowstyle="-|>", color=CINN, lw=0.7,
                             connectionstyle="arc3,rad=-0.2"))
axi.set_title("FPR on real target windows", fontsize=5.6, color=INK,
              fontweight="bold", pad=2.5)
for s in axi.spines.values():
    s.set_linewidth(0.55)
    s.set_color(GRAY)

save_all(fig, "fig7_transfer")
