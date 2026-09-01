"""E2 多种子稳定性聚合（G2 复验④）。

读取 results/e2_main.json（默认种子）与 e2_main_s*.json（--seed=*
运行），报告 A0 及全部消融行的 macro / FPR / AUC全 / 逐族 recall 的
均值±标准差、e 规则认证率，以及逐种子 G2 判定，回答"结论是否
跨种子稳定"。冒烟运行（smoke=true）自动剔除。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

MINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
FAMS = ["N1", "N2", "N3", "N4", "N5"]
TAGS = ["A0", "A1", "A2", "A3", "A4", "A5", "A5b"]


def main() -> None:
    files = sorted(
        f for f in glob.glob(os.path.join(RESULTS_DIR, "e2_main*.json"))
        if re.fullmatch(r"e2_main(_s\d+)?\.json", os.path.basename(f)))
    runs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("smoke"):
            continue
        runs.append((os.path.basename(f), d))
    if not runs:
        print("没有找到正式（非冒烟）E2 运行结果")
        return

    print(f"E2 多种子稳定性聚合：{len(runs)} 个正式运行\n")
    print(f"{'配置':<6}{'macro':>14}{'FPR':>14}{'AUC全':>14}"
          + "".join(f"{k:>14}" for k in FAMS))
    print("-" * 104)
    for tag in TAGS:
        rows = []
        for _, d in runs:
            p = d["pooled"][tag]
            rows.append([p["macro_recall"], p["fpr"], p["auc_all"]]
                        + [p["recall"][k] for k in FAMS])
        arr = np.array(rows)
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(arr.shape[1])
        cells = [f"{m:.3f}±{s:.3f}" for m, s in zip(mu, sd)]
        print(f"{tag:<6}" + "".join(f"{c:>14}" for c in cells))

    print("\ne 规则真实窗认证率（W ≥ 1/α）")
    for tag in ("A0", "A1", "A2", "A3"):
        vals = [d["e_rule"][tag]["real_certify_rate"] for _, d in runs
                if tag in d.get("e_rule", {})]
        if vals:
            print(f"  {tag}: {np.mean(vals):.3f}"
                  + (f" ± {np.std(vals, ddof=1):.3f}" if len(vals) > 1 else "")
                  + f"  (n={len(vals)})")

    print("\n逐种子 G2 判定")
    for name, d in runs:
        v = d["verdict"]
        print(f"  {name:<26} seed={d['seed']} h1={v['h1_macro_beats_0693_significant']}"
              f" h2={v['h2_n2_recall_ge_075']} n5={v['n5_hold_ge_0886']}"
              f" → {'PASS' if v['g2_pass'] else 'FAIL'}")

    out = {"n_runs": len(runs), "seeds": [d["seed"] for _, d in runs]}
    agg = {}
    for tag in TAGS:
        arr = np.array([[d["pooled"][tag]["macro_recall"],
                         d["pooled"][tag]["fpr"],
                         d["pooled"][tag]["auc_all"]]
                        + [d["pooled"][tag]["recall"][k] for k in FAMS]
                        for _, d in runs])
        agg[tag] = {
            "macro_mean": float(arr[:, 0].mean()),
            "macro_sd": float(arr[:, 0].std(ddof=1)) if len(runs) > 1 else 0.0,
            "fpr_mean": float(arr[:, 1].mean()),
            "fpr_sd": float(arr[:, 1].std(ddof=1)) if len(runs) > 1 else 0.0,
            "auc_all_mean": float(arr[:, 2].mean()),
            "auc_all_sd": float(arr[:, 2].std(ddof=1)) if len(runs) > 1 else 0.0,
            # 行向量 = [macro, fpr, auc, N1, N2, N3, N4, N5] → N2 在索引 4
            "n2_mean": float(arr[:, 4].mean()),
            "n2_sd": float(arr[:, 4].std(ddof=1)) if len(runs) > 1 else 0.0,
        }
    out["aggregate"] = agg
    out["verdicts"] = [{"file": n, "seed": d["seed"], **d["verdict"]}
                       for n, d in runs]
    out_json = os.path.join(RESULTS_DIR, "e2_seed_agg.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n聚合结果已写入 {out_json}")


if __name__ == "__main__":
    main()
