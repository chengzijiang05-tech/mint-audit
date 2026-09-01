"""E6：480 输出基准整合（G3 收尾，表 4 素材）。

设计依据：整体研究方案 §六 E6。四条整合线：
  1. 断言通道不动：r2_detector_eval.json 冻结数据原样引用
     （S1 P=R=1.000、S2/S4 FPR=0.000、S3 全弃权；自洽评分披露不变）。
  2. 结构通道升级 MINT：类型化门控路由验证，480 输出的数值
     序列充分性检查 + S3 无源层 119 条三点路径的结构评分语义
     （全部 abstain 保留）。
  3. 七基线文本层矩阵保留：table7.tex 发表冻结值原样搬运，
     MINT 作为结构层新行。
  4. break-even 代数更新：MINT E2 三种子实测操作点替换旧
     8-feature 点（r_FP=0.358, recall=0.693 → ρ* 系数 1.94）。

门控依据：MINT 窗口 1,000 日、轨道算子（IAAFT/块置换）在百级
以下序列无结构意义；序列充分性下限 MIN_SEQ_LEN=100（输入长度
1/10，保守口径）。四 stratum 输出均为统计量/价格点级少量数字
（中位 4–16 个），预期 480/480 结构弃权，类型化门控对非序列
输出零越权判定，S1 的 99 个数字错误由断言通道独占管辖。

数据来源（归档只读，复制冻结）：
  r2_bench.jsonl          480 输出基准（4 模型 × 4 stratum × 30）
  r2_detector_eval.json   断言通道评估（L1b）
产出：results/e6_bench480.json + figures/fig_e6_routing.pdf
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys

import numpy as np

MINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSCT_ROOT = os.path.join(MINT_ROOT, "shared_infra", "fractal_consistency")
for p in (MINT_ROOT, FSCT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ARCHIVE_DATA = os.environ.get("MINT_BENCH480_ARCHIVE", "")
LOCAL_DATA = os.path.join(FSCT_ROOT, "data", "bench480")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
FIGURES_DIR = os.path.join(MINT_ROOT, "figures")

WINDOW = 1000
MIN_SEQ_LEN = 100
ALPHA = 0.05
NUM_RE = re.compile(r"-?\d+\.?\d*")

# table7.tex 发表冻结值（DSS submission，2026-08-18 定稿）
TEXT_BASELINES = [
    # method, S1 P, S1 R, S1 F1, S2 FPR, S4 FPR, S3 flag, S1 abstain
    ("SelfCheck-numeric", 0.967, 0.898, 0.931, 0.000, 0.050, 1.000, 9),
    ("SE-numeric", 0.982, 0.571, 0.723, 0.000, 0.075, 0.975, 9),
    ("SCV (vote)", 0.964, 0.816, 0.884, 0.000, 0.100, 0.975, 9),
    ("LLM-as-judge", 0.890, 0.980, 0.933, 0.000, 0.575, 0.000, 8),
    ("RAG-naive (flat tol.)", 1.000, 0.929, 0.963, 0.000, 0.000,
     "abstain", 8),
    ("Panel-3 (contested)", 0.000, 0.000, 0.000, None, None, 0.000,
     "9/214"),
    ("FSCT-L1b (eng. tol.)", 1.000, 1.000, 1.000, 0.000, 0.000,
     "abstain", 8),
]
# 旧结构通道操作点（现稿 8-feature real-reference，body4.tex 冻结）
OLD_STRUCT_POINT = {"r_fp": 0.358, "recall": 0.693, "coef": 1.94}


def freeze_data() -> None:
    os.makedirs(LOCAL_DATA, exist_ok=True)
    for name in ("r2_bench.jsonl", "r2_detector_eval.json"):
        dst = os.path.join(LOCAL_DATA, name)
        if os.path.exists(dst) or not ARCHIVE_DATA:
            continue
        src = os.path.join(ARCHIVE_DATA, name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"已冻结 {name} → {dst}")


def load_bench():
    with open(os.path.join(LOCAL_DATA, "r2_bench.jsonl"),
              encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_assertion():
    with open(os.path.join(LOCAL_DATA, "r2_detector_eval.json"),
              encoding="utf-8") as f:
        return json.load(f)["pooled"]


# ---------------------------------------------------------------------------
# 结构层门控
# ---------------------------------------------------------------------------

def seq_numbers(text: str) -> list[float]:
    return [float(x) for x in NUM_RE.findall(text)]


def route_structural(record: dict) -> dict:
    """类型化门控：序列充分性判定（最宽松口径，全部数字计入）。"""
    nums = seq_numbers(record["output"])
    n = len(nums)
    if n >= MIN_SEQ_LEN:
        # 480 基准中预期不存在该分支；若出现则需 MINT 编码器评分
        return {"route": "score", "n_nums": n,
                "reason": "sequence-sufficient"}
    return {"route": "abstain", "n_nums": n,
            "reason": f"no_scoreable_sequence(n={n}<{MIN_SEQ_LEN})"}


S3_PATH_RE = [
    re.compile(r"期初[^0-9\-]{0,12}(\d+\.?\d*)"),
    re.compile(r"(?:月中|高峰|高点|峰值)[^0-9\-]{0,12}(\d+\.?\d*)"),
    re.compile(r"(?:月末|月底|期末|期末值)[^0-9\-]{0,12}(\d+\.?\d*)"),
]


def s3_three_point(record: dict):
    """S3 三点路径提取：期初/峰/末。结构有效性：峰须为三点最大值。"""
    text = record["output"]
    vals = []
    for pat in S3_PATH_RE:
        m = pat.search(text)
        if not m:
            return None
        vals.append(float(m.group(1)))
    start, peak, end = vals
    # 路径形式须为 start→peak→end 且峰为最大值（否则为误匹配）
    if (min(vals) <= 0 or peak < start or peak < end
            or not all(5.0 <= v <= 150.0 for v in vals)):  # VIX 量级窗
        return None
    if len(set(vals)) < 2:
        return None
    return vals


def s3_amplitude(vals: list[float]) -> dict:
    """三点路径幅度结构（描述性，不下判定）。"""
    start, peak, end = vals
    up = (peak - start) / start
    down = (peak - end) / peak if peak > 0 else 0.0
    net = (end - start) / start
    return {"start": start, "peak": peak, "end": end,
            "peak_up_pct": round(up * 100, 1),
            "peak_to_end_pct": round(down * 100, 1),
            "net_pct": round(net * 100, 1)}


def equity_monthly_amplitudes() -> dict:
    """真实 equity 月度峰谷幅度分布（22 日窗，描述性对照）。"""
    npz = np.load(os.path.join(FSCT_ROOT, "data", "market_cache",
                               "returns.npz"), allow_pickle=True)
    r = npz["equity"].astype(float)
    px = np.exp(np.cumsum(r))          # 价格水平（起点归一）
    amps, nets = [], []
    for i in range(0, len(px) - 22, 22):
        w = px[i:i + 22]
        hi, lo, s, e = w.max(), w.min(), w[0], w[-1]
        amps.append((hi - lo) / lo)
        nets.append((e - s) / s)
    a = np.asarray(amps)
    return {"n_windows": len(a),
            "hi_lo_p50": round(float(np.median(a)), 4),
            "hi_lo_p95": round(float(np.quantile(a, 0.95)), 4),
            "hi_lo_max": round(float(a.max()), 4),
            "net_p50_abs": round(float(np.median(np.abs(nets))), 4)}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    freeze_data()
    bench = load_bench()
    assertion = load_assertion()
    print(f"\n480 基准载入：{len(bench)} 条；断言通道汇总 "
          f"P={assertion['precision']} R={assertion['recall']}")

    # ---- 1. 结构层路由（480 全量） ----
    routing = {"score": 0, "abstain": 0}
    by_stratum = {}
    for rec in bench:
        r = route_structural(rec)
        routing[r["route"]] += 1
        s = rec["stratum"]
        by_stratum.setdefault(s, {"n": 0, "abstain": 0, "score": 0})
        by_stratum[s]["n"] += 1
        by_stratum[s][r["route"]] += 1
    print(f"\n结构层路由：score={routing['score']} "
          f"abstain={routing['abstain']} / {len(bench)}")
    for s, v in sorted(by_stratum.items()):
        print(f"  {s}: abstain {v['abstain']}/{v['n']}")

    # ---- 2. S3 三点路径结构读数 ----
    s3_paths = []
    for rec in bench:
        if rec["stratum"] != "s3":
            continue
        vals = s3_three_point(rec)
        if vals is not None:
            s3_paths.append({
                "generator": rec["generator"], "trial": rec["trial"],
                "key": rec["key"], **s3_amplitude(vals), "n_nums":
                len(seq_numbers(rec["output"]))})
    n_s3_parse = len(s3_paths)
    ups = np.asarray([p["peak_up_pct"] for p in s3_paths])
    downs = np.asarray([p["peak_to_end_pct"] for p in s3_paths])
    eq = equity_monthly_amplitudes()
    print(f"\nS3 三点路径提取：{n_s3_parse}/120"
          f"（可解析输出 119 + 空输出 1）")
    if n_s3_parse:
        print(f"  峰值上行幅度: 中位 {np.median(ups):.1f}% "
              f"范围 [{ups.min():.1f}, {ups.max():.1f}]%")
        print(f"  峰末回撤幅度: 中位 {np.median(downs):.1f}% "
              f"范围 [{downs.min():.1f}, {downs.max():.1f}]%")
    print(f"  对照 equity 真实月度峰谷幅度: p50={eq['hi_lo_p50']*100:.1f}% "
          f"p95={eq['hi_lo_p95']*100:.1f}% max={eq['hi_lo_max']*100:.1f}%")

    # ---- 3. break-even 操作点更新 ----
    agg_file = os.path.join(RESULTS_DIR, "e2_seed_agg.json")
    with open(agg_file, encoding="utf-8") as f:
        agg = json.load(f)["aggregate"]["A0"]
    r_fp = agg["fpr_mean"]
    recall = agg["macro_mean"]
    coef_new = recall / r_fp
    # N2 口径（E1b 盲区轴）
    n2 = agg["n2_mean"]
    coef_n2 = n2 / r_fp
    be = {
        "old_point_8feature": OLD_STRUCT_POINT,
        "mint_point": {
            "r_fp": round(r_fp, 4), "r_fp_sd": round(agg["fpr_sd"], 4),
            "macro_recall": round(recall, 4),
            "n2_recall": round(n2, 4),
            "coef_macro": round(coef_new, 2),
            "coef_n2": round(coef_n2, 2)},
        "domain_pi05": {
            "old_rho_star": OLD_STRUCT_POINT["coef"],
            "mint_rho_star_macro": round(coef_new, 2),
            "mint_rho_star_n2": round(coef_n2, 2),
            "expansion_x_macro": round(coef_new / OLD_STRUCT_POINT["coef"], 1),
            "expansion_x_n2": round(coef_n2 / OLD_STRUCT_POINT["coef"], 1)},
        "e_rule_point": {
            "note": "免校准 e 规则（W≥1/α）：r_FP=0、N2–N5 复核召回 1.00"
                    "（E2/E4 冻结），与断言通道同款严格占优",
            "strictly_dominates": True},
    }
    print(f"\nbreak-even 更新：ρ* 系数 {OLD_STRUCT_POINT['coef']} → "
          f"{coef_new:.2f}（macro）/ {coef_n2:.2f}（N2）")
    print(f"  π=0.5 适用域：ρ < {OLD_STRUCT_POINT['coef']} → "
          f"ρ < {coef_new:.1f}（macro），扩展 "
          f"{coef_new / OLD_STRUCT_POINT['coef']:.1f}×")

    # ---- 4. 表 4 素材 ----
    table4 = {
        "text_layer_baselines_frozen": [
            {"method": m, "s1_p": p, "s1_r": r, "s1_f1": f1,
             "s2_fpr": s2, "s4_fpr": s4, "s3_flag": s3, "s1_abstain": ab}
            for m, p, r, f1, s2, s4, s3, ab in TEXT_BASELINES],
        "assertion_channel_frozen": {
            "method": "FSCT-L1b (eng. tol.)",
            "s1_p": assertion["precision"], "s1_r": assertion["recall"],
            "s1_f1": 1.0, "s2_fpr": 0.0, "s4_fpr": 0.0,
            "s3_flag": "abstain (120/120)",
            "s1_abstain": 8,
            "source": "r2_detector_eval.json（自洽评分披露不变）"},
        "mint_structural_layer_new": {
            "method": "MINT structural layer (typed gating)",
            "s1_p": None, "s1_r": None, "s1_f1": None,
            "s2_fpr": None, "s4_fpr": None,
            "s3_flag": "abstain (119/119 parseable; reason-coded)",
            "s1_abstain": 120,
            "routing": "480/480 structural abstention "
                       "(no_scoreable_sequence; n_nums med 4–16 << 100)",
            "semantics": "类型化门控：非序列输出零越权判定，S1 数字"
                         "错误由断言通道独占管辖，结构证据留给序列层"
                         "（E2/E4）"},
    }

    out = {
        "experiment": "E6_480_bench_integration",
        "n_outputs": len(bench),
        "strata": {s: v for s, v in sorted(by_stratum.items())},
        "structural_routing": routing,
        "min_seq_len": MIN_SEQ_LEN,
        "s3_three_point": {
            "n_extracted": n_s3_parse, "n_parseable": 119, "n_total": 120,
            "peak_up_pct_median": round(float(np.median(ups)), 1) if n_s3_parse else None,
            "peak_up_pct_range": [round(float(ups.min()), 1),
                                  round(float(ups.max()), 1)] if n_s3_parse else None,
            "peak_to_end_pct_median": round(float(np.median(downs)), 1) if n_s3_parse else None,
            "equity_monthly_hi_lo_reference": eq,
            "paths": s3_paths},
        "break_even": be,
        "table4": table4,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, "e6_bench480.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out_json}")

    make_figure(by_stratum, routing, be, s3_paths, eq, ups, downs)


def make_figure(by_stratum, routing, be, s3_paths, eq, ups, downs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2),
                             gridspec_kw={"wspace": 0.4})

    # (a) 结构层路由矩阵（stratum × route）
    ax = axes[0]
    strs = ["s1", "s2", "s3", "s4"]
    abst = [by_stratum[s]["abstain"] for s in strs]
    scor = [by_stratum[s]["score"] for s in strs]
    x = np.arange(len(strs))
    ax.bar(x, abst, width=0.55, color="#8FA8BF",
           label="structural abstain", edgecolor="white", linewidth=0.4)
    ax.bar(x, scor, width=0.55, bottom=abst, color="#C9A66B",
           label="routed to MINT score", edgecolor="white", linewidth=0.4)
    for xi, (a, s) in enumerate(zip(abst, scor)):
        ax.text(xi, a + s + 2, f"{a + s}", ha="center", fontsize=7.5,
                color="#5A6B7A")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={by_stratum[s]['n']})" for s in strs])
    ax.set_ylim(0, 135)
    ax.set_ylabel("outputs", fontsize=8.5)
    ax.set_title("(a) Typed gating routes 480 outputs", fontsize=9)
    ax.grid(True, axis="y", color="#E8EAED", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    # (b) S3 三点路径幅度 vs equity 真实月度幅度
    ax = axes[1]
    if len(ups):
        ax.scatter(ups, downs, s=14, color="#4A6FA5", alpha=0.65,
                   edgecolor="white", linewidth=0.3)
        ax.axvline(eq["hi_lo_p95"] * 100, color="#A9745E", lw=0.8,
                   ls="--")
        ax.text(eq["hi_lo_p95"] * 100 + 4, 2, "equity monthly\np95 hi-lo",
                fontsize=6.5, color="#A9745E")
    ax.set_xlabel("peak up-move (%)", fontsize=8.5)
    ax.set_ylabel("peak-to-end drawdown (%)", fontsize=8.5)
    ax.set_title("(b) S3 hypothetical 3-point paths\n"
                 "(descriptive, no verdict)", fontsize=9)
    ax.grid(True, color="#E8EAED", lw=0.5)
    ax.set_axisbelow(True)

    # (c) break-even 适用域
    ax = axes[2]
    pts = [("8-feature (old)", OLD_STRUCT_POINT["coef"], "#5A6B7A"),
           ("MINT macro", be["mint_point"]["coef_macro"], "#4A6FA5"),
           ("MINT N2", be["mint_point"]["coef_n2"], "#9CAF88")]
    names = [p[0] for p in pts]
    vals = [p[1] for p in pts]
    cols = [p[2] for p in pts]
    ax.bar(names, vals, width=0.55, color=cols,
           edgecolor="white", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.35, f"{v:.2f}", ha="center", fontsize=7.5,
                color="#5A6B7A")
    ax.set_ylabel(r"$\rho^{*}$ coefficient  $(1-r_{FN})/r_{FP}$",
                  fontsize=8.5)
    ax.set_title("(c) Break-even domain at π=0.5", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(True, axis="y", color="#E8EAED", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7.5)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_e6_routing.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"图已写入 {out}")


if __name__ == "__main__":
    main()
