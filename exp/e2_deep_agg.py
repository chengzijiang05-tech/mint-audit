"""E2 深度基线多种子聚合（表 2 深度行数据源）。

读取 results/e2_deep.json（默认种子，含 A0 配对显著性）与
e2_deep_s*.json，报告 D1–D4 的 macro / FPR / AUC全 / 逐族 recall
均值±标准差，以及默认种子下的 A0 vs 深度基线 DeLong / McNemar
配对检验汇总。冒烟运行（smoke=true）自动剔除。
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

MINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
FAMS = ["N1", "N2", "N3", "N4", "N5"]
TAGS = ["D1", "D2", "D3", "D4"]


def main() -> None:
    files = sorted(
        f for f in glob.glob(os.path.join(RESULTS_DIR, "e2_deep*.json"))
        if re.fullmatch(r"e2_deep(_s\d+)?\.json", os.path.basename(f)))
    runs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("smoke"):
            continue
        runs.append((os.path.basename(f), d))
    if not runs:
        print("没有找到正式（非冒烟）深度基线运行结果")
        return

    print(f"E2 深度基线多种子聚合：{len(runs)} 个正式运行\n")
    print(f"{'配置':<4}{'macro':>14}{'FPR':>14}{'AUC全':>14}"
          + "".join(f"{k:>14}" for k in FAMS))
    print("-" * 104)
    out = {"n_runs": len(runs), "seeds": [d["seed"] for _, d in runs],
           "aggregate": {}}
    for tag in TAGS:
        arr = np.array([[d["pooled"][tag]["macro_recall"],
                         d["pooled"][tag]["fpr"],
                         d["pooled"][tag]["auc_all"]]
                        + [d["pooled"][tag]["recall"][k] for k in FAMS]
                        for _, d in runs])
        mu, sd = arr.mean(axis=0), (
            arr.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(arr.shape[1]))
        cells = [f"{m:.3f}±{s:.3f}" for m, s in zip(mu, sd)]
        print(f"{tag:<4}" + "".join(f"{c:>14}" for c in cells))
        out["aggregate"][tag] = {
            "macro_mean": float(mu[0]), "macro_sd": float(sd[0]),
            "fpr_mean": float(mu[1]), "fpr_sd": float(sd[1]),
            "auc_all_mean": float(mu[2]), "auc_all_sd": float(sd[2]),
            "recall_mean": {k: float(m) for k, m in zip(FAMS, mu[3:])},
            "recall_sd": {k: float(s) for k, s in zip(FAMS, sd[3:])},
            # 行向量 = [macro, fpr, auc, N1..N5] → N2 在索引 4
            "n2_mean": float(mu[4]), "n2_sd": float(sd[4]),
        }

    # 默认种子的 A0 配对显著性汇总（逐族 + ALL 的 DeLong p、McNemar p）
    base = next((d for _, d in runs if not d["stats"] == {}), None)
    if base and base.get("stats"):
        sig = {}
        for tag in TAGS:
            sig[tag] = {
                "delong_p": {s: base["stats"][f"A0vs{tag}/{s}"]["p"]
                             for s in FAMS + ["ALL"]
                             if f"A0vs{tag}/{s}" in base["stats"]},
                "delong_auc": {s: {"a0": base["stats"][f"A0vs{tag}/{s}"]["auc_a"],
                                   "deep": base["stats"][f"A0vs{tag}/{s}"]["auc_b"]}
                               for s in FAMS + ["ALL"]
                               if f"A0vs{tag}/{s}" in base["stats"]},
                "mcnemar_p": base["stats"][f"mcnemar/A0vs{tag}"]["p"],
            }
        out["stats_vs_a0"] = sig
        print("\nA0 vs 深度基线（默认种子，配对检验）")
        for tag in TAGS:
            s = sig[tag]
            print(f"  {tag}: DeLong[ALL] p={s['delong_p']['ALL']:.2e} "
                  f"AUC {s['delong_auc']['ALL']['a0']:.3f} vs "
                  f"{s['delong_auc']['ALL']['deep']:.3f} | "
                  f"McNemar p={s['mcnemar_p']:.2e}")

    out_json = os.path.join(RESULTS_DIR, "e2_deep_agg.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n聚合结果已写入 {out_json}")


if __name__ == "__main__":
    main()
