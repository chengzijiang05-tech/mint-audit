"""Fig. 6 - (a) small-reference power curves; (b) raincloud of LLM-invented
path amplitudes vs real CSI 300 monthly amplitudes. Frozen data."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (CINN, CINN_BG, CINN_MID, GRAY, GRAY_L, INK, NAVY,
                       NAVY_L, NAVY_MID, apply_style, load_figdata, save_all)
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

d = load_figdata()["fig6"]
apply_style()

fig = plt.figure(figsize=(6.77, 2.42))
axa = fig.add_axes([0.075, 0.180, 0.375, 0.700])
axb = fig.add_axes([0.575, 0.155, 0.395, 0.745])

# ---------------------------------------------------------------- (a) power
n = d["n_ref_grid"]
pc = d["power_curves"]
axa.axvspan(2.2, 19, color="#F2EFE6", zorder=0)
axa.text(5.6, 0.44, "granularity wall\n$p_{\\min} = 1/(n{+}1) > \\alpha$\n"
         "for $n < 19$", ha="center", va="center", fontsize=5.4, color=GRAY,
         linespacing=1.4)

series = [
    ("mid-p permutation (real ref.)", pc["mid-p permutation (real ref.)"],
     NAVY_MID, "o", "-"),
    ("conformal order-stat (real ref.)", pc["conformal order-stat (real ref.)"],
     GRAY, "s", "-"),
    ("mid-p permutation (Mahalanobis)",
     pc["mid-p permutation (Mahalanobis)"], NAVY_L, "^", "--"),
]
for label, vals, col, mk, ls in series:
    axa.plot(n, vals, marker=mk, ms=3.2, lw=1.0, color=col, ls=ls, zorder=3,
             label=label, markerfacecolor="white", markeredgewidth=0.8)
ev = pc["e-value (orbit, excl.)"]
axa.plot(n, ev, marker="o", ms=4.2, lw=1.7, color=CINN, zorder=5,
         label="orbit e-value (MINT)")
axa.fill_between(n, 0, ev, color=CINN, alpha=0.10, zorder=1, lw=0)
axa.text(9.5, 0.955, "orbit e-value: power $= 1.00$ at every size,\n"
         "null level $= 0.00$ throughout", ha="center", va="top",
         fontsize=5.6, color=CINN, fontweight="bold", linespacing=1.4)
axa.annotate("untestable at level $\\alpha$", xy=(4.0, 0.02),
             xytext=(3.1, 0.30), fontsize=5.4, color=GRAY,
             arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))

axa.set_xscale("log")
axa.set_xticks(n)
axa.set_xticklabels([str(v) for v in n])
axa.minorticks_off()
axa.set_xlim(2.2, 52)
axa.set_ylim(-0.04, 1.08)
axa.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axa.set_xlabel("reference-set size $n_{\\mathrm{ref}}$")
axa.set_ylabel("power on unseen forgeries (N3–N5)")
axa.legend(loc="center right", fontsize=5.3, frameon=True, framealpha=0.95,
           edgecolor="#DDD", handlelength=1.6, borderpad=0.35,
           labelspacing=0.3)
axa.spines[["top", "right"]].set_visible(False)

# ------------------------------------------------------------ (b) raincloud
real = np.asarray(d["real_amps_pct"], dtype=float)
groups = [("real CSI 300\nmonthly, $n{=}271$", real, NAVY)]
for g in ["deepseek-chat", "deepseek-reasoner", "qwen-max"]:
    v = np.array([p["peak_up_pct"] for p in d["llm_paths"]
                  if p["generator"] == g], dtype=float)
    groups.append((f"{g}\n$n{{=}}{len(v)}$", v, CINN_MID))

ys = np.arange(len(groups))[::-1]
for (label, v, col), y in zip(groups, ys):
    rng = np.random.default_rng(11)
    jit = (rng.random(len(v)) - 0.5) * 0.42 - 0.33
    axb.scatter(v, y + jit, s=3.2, color=col, alpha=0.65, linewidths=0,
                zorder=4)
    if len(v) < 2:
        axb.annotate("$n{=}1$", xy=(v[0], y - 0.42), fontsize=5.4, color=col,
                     ha="center", va="center")
        continue
    lg = np.log10(v)
    kde = gaussian_kde(lg)
    xs = np.linspace(lg.min() - 0.15, lg.max() + 0.15, 220)
    dens = kde(xs)
    dens = dens / dens.max() * 0.36
    axb.fill_betweenx(10 ** xs, y, y + dens, color=col, alpha=0.30, lw=0,
                      zorder=2)
    axb.plot(10 ** xs, y + dens, color=col, lw=0.9, zorder=3)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    axb.plot([q1, q3], [y - 0.05, y - 0.05], color=col, lw=3.2,
             solid_capstyle="butt", zorder=4)
    axb.plot([med, med], [y - 0.16, y + 0.06], color=INK, lw=0.9, zorder=5)

axb.set_xscale("log")
axb.set_xlim(1.8, 260)
axb.set_xticks([2, 5, 10, 25, 50, 100, 200])
axb.set_xticklabels(["2", "5", "10", "25", "50", "100", "200"])
axb.minorticks_off()
axb.set_ylim(-0.85, len(groups) - 0.15)
axb.set_yticks(ys)
axb.set_yticklabels([g[0] for g in groups], fontsize=5.8)
for t, g in zip(axb.get_yticklabels(), groups):
    t.set_color(g[2] if g[2] != CINN_MID else CINN)
axb.set_xlabel("peak-to-trough amplitude (%)")
axb.spines[["top", "right"]].set_visible(False)

rmax = real.max()
axb.axvline(rmax, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=5)
axb.text(rmax * 1.12, 2.62, "real maximum 40.3%", fontsize=5.4, color=INK,
         rotation=90, va="top", ha="left")
axb.axvspan(rmax, 260, color=CINN_BG, alpha=0.6, zorder=0, lw=0)
axb.text(58, -0.62, "all 52 invented paths live beyond\nthe real maximum",
         fontsize=5.2, color=CINN, fontweight="bold", ha="center",
         linespacing=1.3)
axb.annotate(f"median 7.9%", xy=(7.89, 3.12), xytext=(2.35, 3.05),
             fontsize=5.4, color=NAVY, va="center")

fig.text(0.012, 0.955, "(a)", fontsize=7.6, fontweight="bold", color=INK)
fig.text(0.532, 0.955, "(b)", fontsize=7.6, fontweight="bold", color=INK)

save_all(fig, "fig6_power_llm")
