"""N7c：时序基础模型锻造序列（Chronos-T5 采样器，R7 当代生成器）。

编辑决定书 R7：N7a 需扩容至 ≥50 条存活路径，并至少加入一个当代
生成器（≥7B instruct 模型，或 Chronos/Moirai 类时序基础模型）。
Chronos-2 仅输出分位数（确定性预测，非概率采样路径），故采用
ChronosPipeline（T5 家族，forecast_type=SAMPLES）作为采样式锻造器。

协议（与 N7a 机制对齐，敌手条件于真实数据、信息优势不弱化）：
  1. 语境构造：各资产拟合层窗口前 PREFIX=100 个日收益作为条件
     上下文（与 N7a 真值前缀一致，敌手看到真实序列片段）。
  2. 分块续写：每块 CHUNK=64 点（模型训练视界内），上下文取
     [前缀+已生成] 的最近 CTX=512 点（模型上下文长度上限），
     采样 temperature=1.0、num_samples=1，迭代至 1000 点。
  3. 验收门（与 N7a 一致）：长度 ≥1000 且唯一值比例 ≥0.30；
     未达标换种子重试（RETRIES 次），全部记录在日志。
  4. 尺度对齐：与该资产测试层真实窗标准差对齐（N6/N7a/N8 同协议）。

模型：amazon/chronos-t5-base（2024，200M 参数，概率采样式时序
基础模型）。本机 CPU 推理，权重经 hf-mirror 下载入缓存。

产出：results/n7c_chronos_forge.npz + results/n7c_chronos_forge.json
键格式：{asset}/chronos-t5-base/{j}，与 e7_generators npz 读取器兼容。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

MINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSCT_ROOT = os.path.join(MINT_ROOT, "shared_infra", "fractal_consistency")
for p in (MINT_ROOT, FSCT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

RETURNS_NPZ = os.path.join(FSCT_ROOT, "data", "market_cache", "returns.npz")
RESULTS_DIR = os.path.join(MINT_ROOT, "results")

WINDOW, STEP = 1000, 50
SPLIT_SPEC = {
    "equity":  {"n_fit": 15, "cal": (34, 43), "test": (63, 99), "cap": 8},
    "bond":    {"n_fit": 15, "cal": (34, 43), "test": (63, 94), "cap": 8},
    "fx":      {"n_fit": 11, "cal": (31, 36), "test": (37, 42), "cap": 6},
    "gold":    {"n_fit": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
    "copper":  {"n_fit": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
}
PREFIX = 100            # 真值前缀长度（N7a 对齐）
FORGE_LEN = 1000        # 锻造序列长度
CHUNK = 64              # 每块新增点数（模型训练视界内）
CTX = 512               # 条件上下文长度上限（模型上下文长度）
TEMPERATURE = 1.0
ACCEPT_UNIQ = 0.30
RETRIES = 3
GEN_SEED = 30303030     # 独立于 N7a 的种子基
PER_PATH = 5            # 每资产锻造条数
MODEL_ID = "amazon/chronos-t5-base"
MODEL_TAG = "chronos-t5-base"


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


def load_windows():
    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}
    spec = {}
    for asset, r in returns.items():
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
        spec[asset] = {
            "fit": wins[:sp["n_fit"]],
            "target_std": float(np.mean([np.std(w) for w in test_wins])),
        }
    return spec


def forge_one(pipe, ctx_w, target_std, gen_seed):
    """分块采样续写：torch 全局种子控制采样可复现。"""
    import torch
    torch.manual_seed(gen_seed)
    series = list(np.asarray(ctx_w[:PREFIX], dtype=float))
    t0 = time.time()
    while len(series) < PREFIX + FORGE_LEN:
        context = torch.tensor(
            [series[-CTX:]], dtype=torch.float32)
        out = pipe.predict(
            context, prediction_length=CHUNK, num_samples=1,
            temperature=TEMPERATURE, limit_prediction_length=False)
        vals = np.asarray(out, dtype=float).squeeze()
        vals = np.atleast_1d(vals).ravel()[:CHUNK]
        if not np.all(np.isfinite(vals)):
            break
        series.extend(float(v) for v in vals)
    generated = np.array(series[PREFIX:PREFIX + FORGE_LEN], dtype=float)
    uniq = (len(set(np.round(generated, 6))) / len(generated)
            if len(generated) else 0.0)
    ok = len(generated) >= FORGE_LEN and uniq >= ACCEPT_UNIQ
    if ok:
        generated = generated - generated.mean()
        generated = generated * (target_std / max(np.std(generated), 1e-12))
    else:
        generated = None
    return generated, {
        "n_generated": len(generated) if generated is not None else
                       len(series) - PREFIX,
        "uniq_ratio": round(uniq, 3), "gen_s": round(time.time() - t0, 1),
    }


def _save(npz_path, json_path, out_series, forge_log):
    np.savez_compressed(npz_path, **out_series)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "protocol": {
                "mechanism": "chunked sample continuation (Chronos-T5 "
                             "probabilistic sampling, temperature 1.0)",
                "model": MODEL_ID,
                "forge_len": FORGE_LEN, "prefix_len": PREFIX,
                "chunk": CHUNK, "context_len": CTX,
                "temperature": TEMPERATURE,
                "acceptance": f"len>={FORGE_LEN} AND uniq_ratio>="
                               f"{ACCEPT_UNIQ}",
                "retries_per_path": RETRIES, "per_path_target": PER_PATH,
                "seed": GEN_SEED,
                "scale_align": "asset test-layer mean std",
            },
            "n_attempts": len(forge_log),
            "n_series_saved": len(out_series),
            "log": forge_log,
        }, fh, ensure_ascii=False, indent=2)


def main():
    smoke = "--smoke" in sys.argv
    retry = "--retry" in sys.argv
    per_path = 1 if smoke else PER_PATH
    npz_path = os.path.join(RESULTS_DIR, "n7c_chronos_forge.npz")
    json_path = os.path.join(RESULTS_DIR, "n7c_chronos_forge.json")
    print("=" * 66)
    print(f"N7c Chronos-T5 基础模型锻造"
          + ("  [冒烟]" if smoke else "")
          + (f"  [断点重试]" if retry else ""))
    print(f"模型：{MODEL_ID} | 每资产 {per_path} 条")
    print("=" * 66, flush=True)

    import torch
    torch.set_num_threads(int(os.environ.get("N7C_THREADS", "16")))
    from chronos import ChronosPipeline

    t0 = time.time()
    pipe = ChronosPipeline.from_pretrained(MODEL_ID)
    print(f"    已加载 {time.time()-t0:.0f}s | 上下文长度 "
          f"{pipe.model_context_length}", flush=True)

    spec = load_windows()
    out_series, forge_log = {}, []
    if retry and os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as d:
            for k in d.files:
                out_series[k] = d[k]
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as fh:
                forge_log = json.load(fh).get("log", [])
        print(f"断点续传：已有 {len(out_series)} 条，仅补缺口\n", flush=True)

    counter = 0
    for ai, (asset, s) in enumerate(spec.items()):
        for j in range(per_path):
            key = f"{asset}/{MODEL_TAG}/{j}"
            if key in out_series:
                counter += 1
                continue
            done = False
            for attempt in range(RETRIES):
                ctx_w = s["fit"][(j + attempt) % len(s["fit"])]
                gen_seed = GEN_SEED + ai * 1000 + j * 100 + (
                    attempt + 1) * 77777
                series, meta = forge_one(pipe, ctx_w, s["target_std"],
                                         gen_seed)
                meta.update({"asset": asset, "model": MODEL_TAG,
                             "gen_seed": gen_seed, "attempt": attempt})
                forge_log.append(meta)
                if series is not None:
                    out_series[key] = series
                    print(f"    [{asset}#{j}] OK {len(series)} pts "
                          f"uniq={meta['uniq_ratio']} try{attempt} "
                          f"({meta['gen_s']}s)", flush=True)
                    done = True
                    break
                print(f"    [{asset}#{j}] DROP n={meta['n_generated']} "
                      f"uniq={meta['uniq_ratio']} try{attempt} "
                      f"({meta['gen_s']}s)", flush=True)
            if not done:
                print(f"    [{asset}#{j}] {RETRIES} 次尝试均未达标",
                      flush=True)
            counter += 1
        _save(npz_path, json_path, out_series, forge_log)

    _save(npz_path, json_path, out_series, forge_log)
    print(f"\n锻造完成：npz 共 {len(out_series)} 条"
          f"（尝试 {len(forge_log)} 次），已存 npz+json", flush=True)


if __name__ == "__main__":
    main()
