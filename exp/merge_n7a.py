"""合并 N7a 分模型产物：n7a_llm_forge_{model}.npz/json → n7a_llm_forge.npz/json。

两模型（gpt2-medium / qwen2.5-0.5b-instruct）并行分进程生成后，
各自输出带 suffix 的 npz+json。本脚本合并为 E9/E7/E10 读取的统一文件。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

MINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(MINT_ROOT, "results")

MODELS = ["gpt2-medium", "qwen2.5-0.5b-instruct"]
OUT_NPZ = os.path.join(RESULTS_DIR, "n7a_llm_forge.npz")
OUT_JSON = os.path.join(RESULTS_DIR, "n7a_llm_forge.json")


def main():
    merged = {}
    logs = []
    protocols = {}
    for m in MODELS:
        npz = os.path.join(RESULTS_DIR, f"n7a_llm_forge_{m}.npz")
        js = os.path.join(RESULTS_DIR, f"n7a_llm_forge_{m}.json")
        if not os.path.exists(npz):
            print(f"!! 缺 {npz}，跳过 {m}")
            continue
        with np.load(npz, allow_pickle=True) as d:
            for k in d.files:
                # 原 key 已含 model 段（asset/model/j），两模型天然不冲突
                merged[k] = d[k]
        if os.path.exists(js):
            with open(js, encoding="utf-8") as fh:
                jd = json.load(fh)
            logs.extend(jd.get("log", []))
            protocols[m] = jd.get("protocol", {})

    np.savez_compressed(OUT_NPZ, **merged)
    n_ok = sum(1 for lg in logs if lg.get("ok"))
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({
            "protocol": {
                "merged": True,
                "per_model": protocols,
                "forge_len": protocols.get(
                    MODELS[0], {}).get("forge_len"),
            },
            "n_attempts": len(logs), "n_ok": n_ok,
            "log": logs,
        }, fh, ensure_ascii=False, indent=2)
    print(f"合并完成：{len(merged)} 条序列，OK {n_ok}/{len(logs)}，"
          f"已存 {os.path.basename(OUT_NPZ)}+json")


if __name__ == "__main__":
    main()