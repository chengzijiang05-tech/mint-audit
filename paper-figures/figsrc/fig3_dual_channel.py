"""Fig. 3 (v2) - the dual-channel drift monitor as a structured data panel.

No plots, no formulas. Each channel states what it watches and the guarantee
it carries, then the frozen experimental record (e5_drift.json). The C panel
carries the CN operating cycle (alarm -> bounded refit -> epoch reset) as a
numbered flow chain. Layout 172 x 71 mm.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import (CINN, CINN_BG, CINN_L, CINN_MID, FLUOR, GRAY,
                       GRAY_BG, INK, NAVY, NAVY_BG, NAVY_MID, new_canvas,
                       rbox, save_all, txt)

W, H = 172, 69.5
fig, ax = new_canvas(W, H)

# ======================================================== top header bar
rbox(ax, 3, 63.7, 166, 5.8, NAVY, NAVY, lw=0, r=1.0)
txt(ax, 86, 68.3, "dual-channel drift monitor  ·  the auditor of the auditor",
    size=7.2, weight="bold", color="white")
txt(ax, 86, 65.5, "operating cycle:  monitor $\\to$ alarm $\\to$ bounded "
    "refit $\\to$ epoch reset $\\to$ monitoring continues", size=5.4,
    color="#D7E1EF")

# ================================================================ V panel
PX, PW = 3, 82
rbox(ax, PX, 14, PW, 48.4, "white", NAVY, lw=1.1, r=1.2)
rbox(ax, PX, 58.2, PW, 4.2, NAVY, NAVY, lw=0, r=0.7)
txt(ax, PX + PW / 2, 60.3, "V  ·  VALIDITY  ·  watches the outputs",
    size=7.0, weight="bold", color="white")

rbox(ax, PX + 3, 50.4, PW - 6, 6.6, NAVY_BG, NAVY_MID, lw=0.8, r=0.8)
txt(ax, PX + PW / 2, 54.9, "flag e-values compound into a running wealth · "
    "supermartingale under the null", size=5.4, color=INK)
txt(ax, PX + PW / 2, 51.9, "Ville caps false alarms at $\\alpha$ under any "
    "stopping rule · line 20", size=5.4, color=NAVY, weight="bold")

txt(ax, PX + 4, 47.9, "the frozen record", size=6.2, weight="bold",
    color=NAVY, ha="left")

rows = [
    ("real deployment · CN 56 + US 59 windows", "0 crossings"),
    ("null calibration · 60 runs × 300 windows", "0 crossings"),
    ("healthy bootstrap · 60 runs × 300 windows", "0 crossings"),
    ("peak realized wealth · Ville line 20", "0.97"),
]
ys = [40.6, 34.1, 27.6, 21.1]
for (label, value), y in zip(rows, ys):
    rbox(ax, PX + 3, y, PW - 6, 5.8, "white", NAVY_MID, lw=0.7, r=0.7)
    txt(ax, PX + 6, y + 2.9, label, size=5.4, color=INK, ha="left")
    txt(ax, PX + PW - 6, y + 2.9, value, size=6.4, weight="bold", color=NAVY,
        ha="right")

rbox(ax, PX + 3, 14.8, PW - 6, 5.4, FLUOR + "33", NAVY, lw=1.0, r=0.7)
txt(ax, PX + PW / 2, 17.5, "115 real windows  ·  120 calibration runs  ·  "
    "0 false alarms", size=5.9, weight="bold", color=INK)

# ================================================================ C panel
QX, QW = 87, 82
rbox(ax, QX, 14, QW, 48.4, "white", CINN, lw=1.1, r=1.2)
rbox(ax, QX, 58.2, QW, 4.2, CINN, CINN, lw=0, r=0.7)
txt(ax, QX + QW / 2, 60.3, "C  ·  CERTIFICATION POWER  ·  watches the "
    "auditor", size=7.0, weight="bold", color="white")

rbox(ax, QX + 3, 50.4, QW - 6, 6.6, CINN_BG, CINN_MID, lw=0.8, r=0.8)
txt(ax, QX + QW / 2, 54.9, "certification strength feeds an e-detector · "
    "CUSUM lineage · additive restart", size=5.4, color=INK)
txt(ax, QX + QW / 2, 51.9, "Markov bound over each monitoring epoch · "
    "budget 1200", size=5.4, color=CINN, weight="bold")

txt(ax, QX + 4, 47.9, "CN stream · 2015-06 crash · one operating cycle",
    size=6.2, weight="bold", color=CINN, ha="left")

chips = [
    ("1", "regime change · volatility structure rebuilt · never reverted",
     GRAY_BG, GRAY, GRAY, INK),
    ("2", "certification strength collapses · median 25.1 $\\to$ 1.4",
     "white", CINN_MID, CINN_MID, INK),
    ("3", "alarm · window 84 · detector 1207 over budget 1200",
     CINN, CINN, CINN, "white"),
    ("4", "bounded refit 164 s · strength recovers to 16–27 in 4 windows",
     NAVY, NAVY, NAVY, "white"),
    ("5", "second alarm · window 94 · epoch resets · watch continues",
     "white", CINN, CINN, INK),
]
cy = [41.4, 36.8, 32.2, 27.6, 23.0]
ax.plot([QX + 6.1, QX + 6.1], [cy[-1] + 2.1, cy[0] + 2.1], color=CINN_L,
        lw=0.8, zorder=2)
for (num, label, fc, ec, nc, tc), y in zip(chips, cy):
    rbox(ax, QX + 3, y, QW - 6, 4.2, fc, ec, lw=0.9, r=0.7)
    rbox(ax, QX + 4.6, y + 0.6, 3.0, 3.0, nc, nc, lw=0, r=0.5, z=4)
    txt(ax, QX + 6.1, y + 2.1, num, size=5.0, color="white", weight="bold",
        z=5)
    txt(ax, QX + 9.6, y + 2.1, label, size=5.3, color=tc, ha="left", z=5)

rbox(ax, QX + 3, 14.8, QW - 6, 6.0, NAVY_BG, NAVY_MID, lw=0.9, r=0.7)
txt(ax, QX + QW / 2, 18.6, "US stream · COVID transient · detector peak 41",
    size=5.5, weight="bold", color=NAVY)
txt(ax, QX + QW / 2, 16.0, "29$\\times$ below budget · zero alarms · "
    "transient absorbed", size=5.3, color=INK)

# ================================================================ synthesis
rbox(ax, 3, 1, 166, 10.7, GRAY_BG, INK, lw=1.1, r=1.0)
txt(ax, 86, 9.7, "complementary by construction  ·  V bounds the outputs, "
    "C watches the auditor", size=6.4, weight="bold", color=INK)
txt(ax, 86, 6.4, "orbit-member forgeries and value-level corruption are "
    "exchangeable-silent on V · both collapse certification power · C "
    "closes the gap", size=5.2, color=INK)
txt(ax, 86, 3.3, "injected corruption, windows to alarm:  N1 3  ·  N5 3  ·  "
    "N2 5  ·  cross-market splice 27", size=5.2, color=NAVY, weight="bold")

save_all(fig, "fig3_dual_channel")
