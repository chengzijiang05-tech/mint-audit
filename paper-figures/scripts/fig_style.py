"""论文图表共享样式 v2：深蓝 + 丹红 + 黑 + 荧光黄（2026-08-24 全面重绘）。

KBS 双栏排版：单栏图 90 mm (3.54 in)，跨栏图 190 mm (7.48 in)。
所有图输出 PDF（正文嵌入）+ PNG（300 dpi 预览）+ SVG（可编辑）三种格式。
荧光黄只作强调色：原创模块描边、MINT 高亮、告警标记，克制使用。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parent.parent / "figures"

# ---- 核心配色：深蓝色系 + 丹红色系 + 黑 + 荧光黄 ----
NAVY = "#1F3864"      # 深蓝（主色）
NAVY2 = "#2E5A9E"     # 中蓝
NAVY_LT = "#7FA6D9"   # 浅蓝
NAVY_PALE = "#C9DAF2" # 淡蓝
NAVY_ICE = "#EAF0F9"  # 冰蓝（底色）
CINNABAR = "#C8372D"  # 丹红（敌手/告警）
CINN_LT = "#E08373"   # 浅丹红
CINN_PALE = "#F7D9D3" # 淡丹红
FLUOR = "#EDF82F"     # 荧光黄（强调，黑字黑边）
INK = "#141414"       # 主文字（黑）
INK2 = "#595959"      # 次文字
GRAY = "#9A9A9A"      # 中性灰（基线）
GRAY_PALE = "#E8E8E4"

SEQ = [NAVY, CINNABAR, NAVY2, CINN_LT, GRAY, NAVY_LT, INK2]

# 热力图渐变：冰蓝 -> 深蓝（低值到高值）
HEAT = ["#F3F7FC", "#DCE7F5", "#B9CFEB", "#8FB0DC", "#5F8AC4", "#3A61A0", "#1F3864"]
# 发散渐变：深蓝 -> 近白 -> 丹红
DIVERGE = ["#1F3864", "#5F8AC4", "#F5F2EC", "#E08373", "#C8372D"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.edgecolor": "#8A8A8A",
    "axes.linewidth": 0.7,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": "#E0E0DB",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SINGLE_COL = 3.54   # in
ONE_HALF = 5.6
DOUBLE_COL = 7.48   # in


def save(fig, name: str) -> None:
    """保存 PDF + PNG + SVG 三格式。"""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    print("saved", name, "->", FIGDIR)
    plt.close(fig)


# ---- 叙事图元件（节点 + 箭头，draw.io 风格） ----
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402


def _pt2data(ax, pts: float, axis: str = "x") -> float:
    """把印刷点数换算成数据坐标单位。"""
    fig = ax.figure
    if axis == "x":
        units_per_in = (ax.get_xlim()[1] - ax.get_xlim()[0]) / fig.get_size_inches()[0]
    else:
        units_per_in = (ax.get_ylim()[1] - ax.get_ylim()[0]) / fig.get_size_inches()[1]
    return pts / 72.0 * units_per_in


def text_w(ax, s: str, fs: float) -> float:
    """估算字符串宽度（数据单位）。0.62 为 Arial 粗体实测安全系数。"""
    return _pt2data(ax, 0.62 * fs * max(len(line) for line in s.split("\n")))


def node(ax, x, y, w, h, title, body="", *, face=NAVY_ICE, edge=NAVY, lw=1.0,
         tc=INK, bc=None, fs=7.0, radius=0.035, zorder=3, dashed=False):
    """圆角节点：title 加粗在上，body（可多行）居中在下。返回边缘连接点字典。"""
    bc = bc if bc is not None else tc
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(box)
    if body:
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=zorder + 1)
        ax.text(x + w / 2, y + h * 0.36, body, ha="center", va="center",
                fontsize=fs - 1.1, color=bc, zorder=zorder + 1, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, color=tc, zorder=zorder + 1, linespacing=1.3,
                fontweight="bold")
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2),
            "t": (x + w / 2, y + h), "b": (x + w / 2, y),
            "c": (x + w / 2, y + h / 2)}


def arrow(ax, p0, p1, *, color=INK, lw=1.1, style="-|>", rad=0.0,
          shrink=3.0, zorder=2, alpha=1.0):
    """节点间箭头。p0/p1 为 (x, y)。rad>0 弯曲。"""
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=9, linewidth=lw,
        color=color, zorder=zorder, alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=shrink, shrinkB=shrink,
    )
    ax.add_patch(a)
    return a


def chip(ax, x, y, text, *, face=FLUOR, edge=INK, fs=6.0, zorder=5, dy=0.0):
    """荧光黄标签芯片（黑字黑边），宽度按坐标系实测。返回 (x, y)。"""
    tw = text_w(ax, text, fs) + _pt2data(ax, 8)
    th = _pt2data(ax, fs + 5, "y")
    box = FancyBboxPatch(
        (x - tw / 2, y - th / 2), tw, th,
        boxstyle="round,pad=0,rounding_size=0.012",
        facecolor=face, edgecolor=edge, linewidth=0.8, zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold", zorder=zorder + 1)
    return (x, y)


# ---- 节点注册与连接点连线（连线显式绑定节点，杜绝坐标漂移） ----
NODES: dict = {}    # nid -> (x, y, w, h)
ARROWS: list = []   # (aid, src_nid, dst_nid, [p0, p1])


def nbox(ax, nid, x, y, w, h, *, face="white", edge=NAVY, lw=1.2,
         radius=0.35, z=3, dashed=False):
    """注册并绘制圆角节点框，nid 为连线绑定的节点 ID。"""
    NODES[nid] = (x, y, w, h)
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
                       linestyle="--" if dashed else "-")
    ax.add_patch(p)
    return p


def anch(nid, side, dx=0.0, dy=0.0):
    """节点连接点：l/r/t/b/c + 微调偏移。"""
    x, y, w, h = NODES[nid]
    if side == "l":
        return (x + dx, y + h / 2 + dy)
    if side == "r":
        return (x + w + dx, y + h / 2 + dy)
    if side == "t":
        return (x + w / 2 + dx, y + h + dy)
    if side == "b":
        return (x + w / 2 + dx, y + dy)
    return (x + w / 2 + dx, y + h / 2 + dy)


def _endpt(spec):
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
        return anch(spec[0], spec[1]), spec[0]
    if isinstance(spec, tuple) and len(spec) == 4 and isinstance(spec[0], str):
        return anch(spec[0], spec[1], spec[2], spec[3]), spec[0]
    return spec, None


def flow(ax, aid, src, dst, *, color=NAVY, lw=1.6, rad=0.0, dashed=False,
         z=4, ms=9, alpha=1.0):
    """节点间直连箭头。src/dst 取 ("nid","side") 或裸坐标 (x, y)。"""
    p0, n0 = _endpt(src)
    p1, n1 = _endpt(dst)
    ARROWS.append((aid, n0, n1, [p0, p1]))
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, color=color, zorder=z, alpha=alpha,
                        linestyle=(0, (2.4, 2.0)) if dashed else "-",
                        connectionstyle=f"arc3,rad={rad}",
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def elbow(ax, aid, pts, *, color=NAVY, lw=1.4, dashed=False, z=4, ms=9,
          src=None, dst=None):
    """折线箭头：pts 为坐标序列，仅末段带箭头。"""
    ARROWS.append((aid, src, dst, list(pts)))
    ls = (0, (2.4, 2.0)) if dashed else "-"
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=color, lw=lw, zorder=z, linestyle=ls,
                solid_joinstyle="miter")
    a = FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, color=color, zorder=z, linestyle=ls,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a
