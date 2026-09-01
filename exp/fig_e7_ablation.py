"""图 7：消融矩阵三种子稳定性条形图（E7，论文版）。

数据：results/e2_main*.json 三个正式种子的池化指标聚合（均值±标准差）。
与 fig_e2_main（单种子三联图）的分工：本图是消融矩阵的多种子
稳定性视图，全部英文标注（KBS 投稿），带误差棒。

面板：
  (a) 逐族 recall：A0 完整版 vs 三个组件消融（A1/A2/A4）
  (b) 消融矩阵全行：macro recall 与 N2 recall（7 配置）
  (c) 校准：实测 FPR vs 名义 α（7 配置，A1/A2 校准塌缩可见）

产出：figures/fig_e7_ablation.pdf
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

MINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
FIGURES_DIR = os.path.join(MINT_ROOT, "figures")

FAMS = ["N1", "N2", "N3", "N4", "N5"]
TAGS = ["A0", "A1", "A2", "A3", "A4", "A5", "A5b"]
LABEL = {
    "A0": "A0 full MINT",
    "A1": "A1 −orbit neg.",
    "A2": "A2 −operator union",
    "A3": "A3 −rank norm.",
    "A4": "A4 −e-value layer",
    "A5": "A5 15-d Mahalanobis",
    "A5b": "A5b classical 8-d",
}
COLORS = {"A0": "#4A6FA5", "A1": "#C9A66B", "A2": "#9CAF88",
          "A3": "#8FA8BF", "A4": "#B49FC2", "A5": "#A9745E",
          "A5b": "#5A6B7A"}
ALPHA = 0.05


def load_runs():
    files = sorted(
        f for f in glob.glob(os.path.join(RESULTS_DIR, "e2_main*.json"))
        if re.fullmatch(r"e2_main(_s\d+)?\.json", os.path.basename(f)))
    runs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("smoke"):
            continue
        runs.append(d)
    return runs


def agg(runs):
    out = {}
    for tag in TAGS:
        rows = []
        for d in runs:
            p = d["pooled"][tag]
            rows.append([p["macro_recall"], p["fpr"], p["auc_all"]]
                        + [p["recall"][k] for k in FAMS])
        a = np.asarray(rows, dtype=float)
        mu, sd = a.mean(axis=0), a.std(axis=0, ddof=1)
        out[tag] = {
            "macro": (mu[0], sd[0]), "fpr": (mu[1], sd[1]),
            "auc": (mu[2], sd[2]),
            "fam": {k: (mu[3 + i], sd[3 + i]) for i, k in enumerate(FAMS)},
        }
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs()
    if len(runs) < 2:
        raise SystemExit(f"需要 ≥2 个正式种子，实际 {len(runs)}")
    data = agg(runs)

    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4),
                             gridspec_kw={"wspace": 0.42})

    # (a) 逐族 recall：A0 vs 三个组件消融
    ax = axes[0]
    show = ["A0", "A1", "A2", "A4"]
    x = np.arange(len(FAMS))
    for i, tag in enumerate(show):
        mu = [data[tag]["fam"][k][0] for k in FAMS]
        sd = [data[tag]["fam"][k][1] for k in FAMS]
        ax.bar(x + (i - 1.5) * 0.2, mu, width=0.19,
               yerr=sd, error_kw={"lw": 0.7, "capsize": 1.5,
                                  "ecolor": "#7A8894"},
               color=COLORS[tag], label=LABEL[tag],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(FAMS)
    ax.set_ylabel("recall @ FPR 0.05", fontsize=8.5)
    ax.set_title(f"(a) Per-family recall (mean±SD, {len(runs)} seeds)",
                 fontsize=9)
    ax.axhline(0.75, color="#A9745E", lw=0.8, ls="--")
    ax.text(0.02, 0.775, "aspiration 0.75", fontsize=6.5, color="#A9745E")
    ax.set_ylim(0, 1.08)
    ax.grid(True, axis="y", color="#E8EAED", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")

    # (b) 消融矩阵：macro 与 N2（全 7 行）
    ax = axes[1]
    ypos = np.arange(len(TAGS))[::-1]
    mac = [data[t]["macro"][0] for t in TAGS]
    mac_sd = [data[t]["macro"][1] for t in TAGS]
    n2 = [data[t]["fam"]["N2"][0] for t in TAGS]
    n2_sd = [data[t]["fam"]["N2"][1] for t in TAGS]
    ax.barh(ypos + 0.18, mac, height=0.34, xerr=mac_sd,
            error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": "#7A8894"},
            color="#4A6FA5", label="macro recall",
            edgecolor="white", linewidth=0.4)
    ax.barh(ypos - 0.18, n2, height=0.34, xerr=n2_sd,
            error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": "#7A8894"},
            color="#9CAF88", label="N2 recall",
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([LABEL[t] for t in TAGS], fontsize=7)
    ax.axvline(0.75, color="#A9745E", lw=0.8, ls="--")
    ax.text(0.76, ypos[-1] - 0.62, "0.75", fontsize=6.5, color="#A9745E")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("recall @ FPR 0.05", fontsize=8.5)
    ax.set_title("(b) Ablation matrix (E7)", fontsize=9)
    ax.grid(True, axis="x", color="#E8EAED", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    # (c) 实测 FPR vs 名义 α
    ax = axes[2]
    fpr = [data[t]["fpr"][0] for t in TAGS]
    fpr_sd = [data[t]["fpr"][1] for t in TAGS]
    ax.barh(ypos, fpr, height=0.62, xerr=fpr_sd,
            error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": "#7A8894"},
            color=[COLORS[t] for t in TAGS],
            edgecolor="white", linewidth=0.4)
    ax.axvline(ALPHA, color="#A9745E", lw=0.9, ls="--")
    ax.text(ALPHA + 0.008, ypos[-1] - 0.62, "nominal α=0.05",
            fontsize=6.5, color="#A9745E")
    ax.set_yticks(ypos)
    ax.set_yticklabels(TAGS, fontsize=8)
    for y, v, s in zip(ypos, fpr, fpr_sd):
        ax.text(v + s + 0.012, y, f"{v:.3f}", va="center", fontsize=6.5,
                color="#5A6B7A")
    ax.set_xlim(0, max(0.42, max(f + s for f, s in zip(fpr, fpr_sd)) * 1.2))
    ax.set_xlabel("realized FPR on 28 real windows", fontsize=8.5)
    ax.set_title("(c) Validity: realized FPR vs α", fontsize=9)
    ax.grid(True, axis="x", color="#E8EAED", lw=0.5)
    ax.set_axisbelow(True)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_e7_ablation.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"图已写入 {out}（{len(runs)} 种子："
          + ", ".join(str(d.get("seed")) for d in runs) + "）")


if __name__ == "__main__":
    main()
