"""Shared style for all MINT paper figures.

Palette: deep blue / cinnabar / ink / fluorescent yellow.
All figures target Elsevier double-column width (17.2 cm) with >= 6.5 pt type.
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- palette
INK      = "#15181F"   # near-black text
NAVY     = "#1F3A5F"   # deep blue (primary)
NAVY_MID = "#3D5E8C"
NAVY_L   = "#8FA8C8"
NAVY_BG  = "#E9EFF6"   # pale blue fill
CINN     = "#B7352F"   # cinnabar (accent / contribution)
CINN_MID = "#D96453"
CINN_L   = "#EFA294"
CINN_BG  = "#FBEDEA"   # pale cinnabar fill
FLUOR    = "#E8E22E"   # fluorescent yellow (highlight only)
FLUOR_D  = "#8A8A15"
GRAY     = "#5C6068"
GRAY_L   = "#9CA1AA"
GRAY_BG  = "#F2F2F0"

SEQ_CMAP = ["#0F2440", "#1F3A5F", "#3D5E8C", "#6483AC", "#8FA8C8", "#B9C9DE", "#DDE6F1"]
HEAT_CMAP = [(0.0, "#B7352F"), (0.28, "#E2A49B"), (0.5, "#F4F1E8"),
             (0.72, "#9FB6D4"), (1.0, "#1F3A5F")]  # cinnabar -> paper -> navy
ASSET_COLORS = {
    "equity": "#1F3A5F", "bond": "#3D7EA6", "fx": "#7A9CC6",
    "gold": "#B8860B", "copper": "#B7352F",
}

FIGDATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts", "figdata.json")
FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def load_figdata():
    with open(FIGDATA, encoding="utf-8") as fh:
        return json.load(fh)


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.6,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.7,
        "xtick.color": INK, "ytick.color": INK,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "axes.grid": False,
        "grid.color": "#D8D8D4", "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 450,
        "figure.dpi": 140,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "axes.unicode_minus": False,
    })


def save_all(fig, name):
    """Save one figure as pdf + svg + png into the shared figures dir."""
    os.makedirs(FIGDIR, exist_ok=True)
    base = os.path.join(FIGDIR, name)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base + ".png", bbox_inches="tight", pad_inches=0.02)
    print("saved", base + ".{pdf,svg,png}")


def heat_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("mint_heat", HEAT_CMAP)


# ------------------------------------------------- diagram primitives (mm)
def rbox(ax, x, y, w, h, fc, ec, lw=0.9, r=1.2, z=2, ls="-", alpha=1.0):
    """Rounded rectangle in data (mm) coordinates."""
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       linestyle=ls, zorder=z, alpha=alpha,
                       mutation_aspect=1)
    ax.add_patch(p)
    return p


def dbox(ax, x, y, w, h, ec=GRAY, lw=0.8, z=1, fc="none", r=1.2):
    return rbox(ax, x, y, w, h, fc, ec, lw=lw, r=r, z=z, ls=(0, (3, 2)))


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.0, z=3, style="-|>",
          shrink=0.0, rad=0.0, ms=5.0, alpha=1.0, ls="-"):
    from matplotlib.patches import FancyArrowPatch
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                        mutation_scale=ms, linewidth=lw, color=color,
                        zorder=z, shrinkA=shrink, shrinkB=shrink,
                        connectionstyle=f"arc3,rad={rad}", alpha=alpha,
                        linestyle=ls, capstyle="round", joinstyle="round")
    ax.add_patch(a)
    return a


def txt(ax, x, y, s, size=6.8, color=INK, weight="normal", ha="center",
        va="center", z=4, style="normal", family=None, rot=0, linespacing=1.25):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=z, style=style, rotation=rot,
                   linespacing=linespacing, family=family)


def polyarrow(ax, pts, color=INK, lw=1.0, ms=5.0, z=3, ls="-"):
    """Polyline routing arrow: walks the given mm points then adds an arrowhead
    on the final segment. Lets arrows travel corridors without piercing boxes."""
    import numpy as np
    import matplotlib as mpl
    arr = np.asarray(pts, dtype=float)
    xs, ys = arr[:, 0], arr[:, 1]
    lw_line = lw if lw > 0.2 else 0.2
    ax.plot(xs, ys, color=color, lw=lw_line, zorder=z, linestyle=ls)
    ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw_line,
                                mutation_scale=ms, shrinkA=0, shrinkB=0),
                zorder=z)


def stage_label(ax, x, y, num, title, color):
    rbox(ax, x - 2.6, y - 2.6, 5.2, 5.2, color, color, lw=0, r=0.9, z=5)
    txt(ax, x, y, f"{num}", size=7.6, color="white", weight="bold", z=6)
    txt(ax, x + 4.0, y, title, size=7.4, color=color, weight="bold",
        ha="left", z=6)


def new_canvas(w_mm, h_mm):
    """Return fig+ax with equal aspect and mm data coords."""
    apply_style()
    fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm)
    ax.set_ylim(0, h_mm)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def sparkline(ax, x, y, w, h, series, color, lw=0.55, z=4, fill=None):
    """Mini line plot inside diagram coordinates."""
    import numpy as np
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n > 240:
        idx = np.linspace(0, n - 1, 240).astype(int)
        s = s[idx]
        n = 240
    t = np.linspace(x, x + w, n)
    lo, hi = float(np.min(s)), float(np.max(s))
    rng = (hi - lo) or 1.0
    yy = y + h * (s - lo) / rng
    if fill is not None:
        ax.fill_between(t, y, yy, color=fill, lw=0, zorder=z - 1, alpha=0.5)
    ax.plot(t, yy, color=color, lw=lw, zorder=z, solid_joinstyle="round")
    return t, yy
