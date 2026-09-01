"""N7a：真实 LLM 文本域锻造序列（SDForger 机制对齐）。

实验目的：标题承诺审计 machine-generated series，
但全部伪造族为算法变换，真实 LLM 生成的长序列从未进入结构通道。

协议（对齐 Rousseau et al. 2025 SDForger 的文本化-续写-解析三段机制，
引言钩子场景"LLM 直接吐出千日收益"的最裸暴露形态）：
  1. 语境构造：从各资产拟合层窗口提取真实摘要统计（日波动率、均值、
     最小/最大值），写入 system prompt，给敌手信息优势，使其有条件
     生成统计上合理的序列（更强的敌手，而非更弱）。
  2. 数值文本化：拟合层真实窗口前 PREFIX=100 个日收益转为 6 位小数
     逗号分隔文本（与数据合同一致的真值文本，无信息损失）。
  3. LLM 自回归续写：温度 1.0 / top-p 0.95 采样（数值约束解码，只允许
     数字/小数点/负号/逗号 token，杜绝漂移到散文），迭代分段续写至
     1000 点；段级质量门（停滞回退、多样性、离群）与全局验收（唯一值
     比例 ≥ 0.30）保证敌手强度，退化序列（如吸引子循环）不进入评估；
     不足格的尝试丢弃并记日志，换种子重试（SDForger 式 parse-and-drop）。
  4. 解析回数值：正则抽取；停滞检测（连续 STALL=20 个相同值）截断；
     不足 1000 点的尝试丢弃并记日志。
  5. 尺度对齐：与该资产测试层真实窗标准差对齐（与 N6/N8 协议一致，
     消除平凡尺度线索，敌手不因量纲暴露而输）。

锻造器（真实开源权重，本地 CPU 推理，注册可复现）：
  Qwen2.5-0.5B-Instruct   （阿里系 instruct 模型，商用 Qwen 同家族）
  gpt2-medium             （OpenAI 系 base LM，纯自回归续写机制暴露）

产出：results/n7a_llm_forge.npz + results/n7a_llm_forge.json（生成日志）
"""
from __future__ import annotations

import json
import os
import re
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
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 权重已在缓存，绕过网络校验
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
PREFIX = 100          # prompt 中的真值前缀长度
FORGE_LEN = 1000      # 锻造序列长度（与 N7b/N8、真实窗 1000 点对齐）
SEG_NEW = 100         # 每段目标新增数值数
SEG_TOKENS = 400      # 每段生成 token 上限（CPU ~8.5 tok/s，440 约 47s/段）
MAX_SEGS = 40         # 最大段数（停滞逃逸后多数 15-25 段可达 1000 点）
RECTX = 60            # 段间重续写的尾部数值数
STALL = 20            # 停滞判定：连续相同数值个数
RETRIES = 3           # 断点重试模式下每缺口最多尝试次数
BASE_TEMP = 1.0       # 基础采样温度（0.8 时数值分布过尖，吸引子固化）
TEMP_CAP = 1.4        # 逃逸温度上限（>1.5 量纲爆炸）
TEMP_ESC = 0.25       # 每次停滞/退化的升温步长
DIV_RATIO = 0.20      # 段级多样性门：新增数值唯一值比例下限
DIV_MIN_UNIQ = 10     # 段级多样性门：唯一值绝对数下限
OUTLIER_FACTOR = 8.0  # 段级离群门：超出语境极值 8 倍整段丢弃
ACCEPT_UNIQ = 0.30    # 全局验收：序列唯一值比例下限（真实窗 47%-95%）
GEN_SEED = 20260821   # 锻造种子（torch/采样）
PER_MODEL = 5         # 每资产每模型锻造条数（R7 扩容：1→5，j=0 保留原冻结条目）
MODELS = [
    ("qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct", False),
    ("gpt2-medium",
     os.path.expanduser(
         "~/.cache/modelscope/models/AI-ModelScope--gpt2-medium/"
         "snapshots/master"), False),
]
NUM_RE = re.compile(r"-?\d+\.?\d*")
NUM_TOKEN_RE = re.compile(r"\s*[-0-9.,]+\s*")


def make_numeric_processor(tok):
    """数值约束解码：只允许解码串为纯数值字符（数字/小数/负号/逗号）
    的 token，杜绝基座 LM 漂移到散文；这是 SDForger 原始数值模板
    （raw numeric template）的格式层强制，与论文口径一致。"""
    from transformers import LogitsProcessor

    class _NumProc(LogitsProcessor):
        def __init__(self, mask):
            self.mask = mask

        def __call__(self, input_ids, scores):
            m = self.mask.to(scores.device)
            if m.shape[0] != scores.shape[-1]:
                pad = scores.shape[-1] - m.shape[0]
                m = torch.cat([m, torch.full((pad,), -float("inf"),
                                             device=m.device)])
            return scores + m

    import torch
    vocab = tok.get_vocab()
    allowed = [tid for s, tid in vocab.items()
               if NUM_TOKEN_RE.fullmatch(tok.decode([tid]))]
    size = max(max(allowed) + 1, len(vocab)) if allowed else len(vocab)
    mask = torch.full((size,), -float("inf"))
    mask[torch.tensor(sorted(allowed), dtype=torch.long)] = 0.0
    return _NumProc(mask), len(allowed)
SYS_TMPL = (
    "You are a quantitative analyst generating realistic daily return "
    "series for the {asset} market. Reference statistics from real data: "
    "daily volatility {vol:.4%}, daily mean {mu:.5f}, min {mn:.4f}, "
    "max {mx:.4f}. Continue the comma-separated series with the same "
    "style, 6 decimal places, no commentary."
)


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
            "test": test_wins,
            "target_std": float(np.mean([np.std(w) for w in test_wins])),
        }
    return spec


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


def _build_prompt(tok, use_chat, asset, ctx_stats, tail_vals, first_seg):
    """构造续写 prompt：首段用真值前缀，后续段用已生成尾部重续写。"""
    tail_txt = ",".join(f"{v:.6f}" for v in tail_vals)
    if use_chat:
        vol, mu, mn, mx = ctx_stats
        sys_txt = SYS_TMPL.format(asset=asset, vol=vol, mu=mu, mn=mn, mx=mx)
        user = (tail_txt if first_seg else
                f"Continue this daily return series, same style, "
                f"6 decimal places, comma-separated, no commentary:\n"
                f"{tail_txt},")
        msgs = [{"role": "system", "content": sys_txt},
                {"role": "user", "content": user}]
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    if first_seg:
        # 纯续写模式也在首段注入资产+真实统计信息（信息优势，非弱化敌手）
        vol, mu, mn, mx = ctx_stats
        head = (f"{asset} daily returns, vol {vol:.4%} mean {mu:.5f} "
                f"min {mn:.4f} max {mx:.4f}, continue:\n")
        return head + tail_txt + ","
    return tail_txt + ","


def _parse_tail(txt):
    """解析生成文本的新增数值。停滞检测：连续 STALL 个相同值即截断，
    且整个重复游程回退，污染值不进入序列，也不进入下一段重续写
    上下文（否则模型会把自己的循环当先例固化）。"""
    vals = [float(v) for v in NUM_RE.findall(txt)]
    cut, run = len(vals), 1
    for i in range(1, len(vals)):
        run = run + 1 if vals[i] == vals[i - 1] else 1
        if run >= STALL:
            cut = i - run + 1
            break
    return vals[:cut], cut < len(vals)


def make_stall_stopper(tok):
    """提前停止：生成中每 20 步解码一次尾部，连续 6 个相同数值即停，
    省下停滞段剩余 token 的 CPU 预算。"""
    from transformers import StoppingCriteria

    class _Stop(StoppingCriteria):
        def __init__(self):
            self.n = 0

        def __call__(self, input_ids, scores, **kwargs):
            self.n += 1
            if self.n % 20:
                return False
            tail = tok.decode(input_ids[0, -60:],
                              skip_special_tokens=True)
            vals = [float(v) for v in NUM_RE.findall(tail)]
            return len(vals) >= 6 and len(set(vals[-6:])) == 1

    return _Stop()


def forge_one(model_name, tok, model, ctx_w, asset, target_std, gen_seed,
              use_chat, proc):
    """迭代分段续写锻造（SDForger 滑窗机制 + 数值约束解码）。

    三层质量门（全部作用于"是否采纳该段"，不改动任何已采纳数值）：
      1. 停滞：连续 STALL 个相同值 → 回退整个重复游程；
      2. 多样性：新增数值唯一值比例 < DIV_RATIO（吸引子循环）→ 整段丢弃；
      3. 离群：任一值超出语境极值 OUTLIER_FACTOR 倍 → 整段丢弃。
    丢弃/停滞后下段温度 +TEMP_ESC（上限 TEMP_CAP），干净段恢复 BASE_TEMP。
    全局验收：长度达标且唯一值比例 ≥ ACCEPT_UNIQ（防跨段吸引子固化）。"""
    import torch
    torch.manual_seed(gen_seed)
    ctx_stats = (float(ctx_w.std()), float(ctx_w.mean()),
                 float(ctx_w.min()), float(ctx_w.max()))
    ctx_maxabs = max(abs(ctx_stats[2]), abs(ctx_stats[3]), 1e-9)
    generated = []
    segs_used, tok_total, empty_streak = 0, 0, 0
    discards = {"empty": 0, "div": 0, "outlier": 0}
    temp = BASE_TEMP
    t0 = time.time()
    for seg in range(MAX_SEGS):
        if len(generated) >= FORGE_LEN:
            break
        tail = ctx_w[:PREFIX] if seg == 0 else np.array(
            generated[-RECTX:], dtype=float)
        prompt = _build_prompt(tok, use_chat, asset, ctx_stats,
                               tail, seg == 0)
        enc = tok(prompt, return_tensors="pt")
        n_in = enc["input_ids"].shape[1]
        stopper = make_stall_stopper(tok)
        with torch.no_grad():
            out_ids = model.generate(
                **enc, max_new_tokens=SEG_TOKENS, do_sample=True,
                temperature=temp, top_p=0.95,
                pad_token_id=tok.eos_token_id,
                logits_processor=[proc], stopping_criteria=[stopper])
        tok_total += int(out_ids.shape[1] - n_in)
        txt = tok.decode(out_ids[0][n_in:], skip_special_tokens=True)
        new_vals, stalled = _parse_tail(txt)
        segs_used += 1
        if not new_vals:
            discards["empty"] += 1
            empty_streak += 1
            temp = min(temp + TEMP_ESC, TEMP_CAP)
            if empty_streak >= 6:
                break
            continue
        n_uniq = len(set(np.round(new_vals, 6)))
        div_bad = n_uniq < max(DIV_MIN_UNIQ, DIV_RATIO * len(new_vals))
        out_bad = max(abs(v) for v in new_vals) > OUTLIER_FACTOR * ctx_maxabs
        if div_bad or out_bad:
            discards["div" if div_bad else "outlier"] += 1
            empty_streak += 1
            temp = BASE_TEMP if out_bad else min(temp + TEMP_ESC, TEMP_CAP)
            if empty_streak >= 6:
                break
            continue
        generated.extend(new_vals)
        empty_streak = 0
        temp = min(BASE_TEMP + TEMP_ESC, TEMP_CAP) if stalled else BASE_TEMP
        if len(generated) >= FORGE_LEN:
            break
    uniq_ratio = (len(set(np.round(generated, 6))) / len(generated)
                  if generated else 0.0)
    ok = len(generated) >= FORGE_LEN and uniq_ratio >= ACCEPT_UNIQ
    if ok:
        series = np.array(generated[:FORGE_LEN], dtype=float)
        series = series - series.mean()
        series = series * (target_std / max(np.std(series), 1e-12))
    else:
        series = None
    meta = {
        "asset": asset, "model": model_name, "ok": bool(ok),
        "n_generated": len(generated), "segs": segs_used,
        "uniq_ratio": round(uniq_ratio, 3), "discards": discards,
        "gen_tokens": tok_total, "gen_s": round(time.time() - t0, 1),
    }
    return series, meta


def main():
    smoke = "--smoke" in sys.argv
    retry = "--retry" in sys.argv
    salt = 0
    if "--salt" in sys.argv:
        salt = int(sys.argv[sys.argv.index("--salt") + 1])
    per_model = 1 if smoke else PER_MODEL
    only_model = None
    if "--model" in sys.argv:
        only_model = sys.argv[sys.argv.index("--model") + 1]
    run_models = [m for m in MODELS
                  if only_model is None or m[0] == only_model]
    suffix = f"_{only_model}" if only_model else ""
    if "--suffix" in sys.argv:
        suffix = sys.argv[sys.argv.index("--suffix") + 1]
    npz_path = os.path.join(RESULTS_DIR, f"n7a_llm_forge{suffix}.npz")
    json_path = os.path.join(RESULTS_DIR, f"n7a_llm_forge{suffix}.json")
    asset_filter = None
    if "--asset" in sys.argv:
        asset_filter = set(
            sys.argv[sys.argv.index("--asset") + 1].split(","))
    print("=" * 66)
    print(f"N7a LLM 文本域锻造（SDForger 机制对齐）"
          + ("  [冒烟]" if smoke else "")
          + (f"  [断点重试 salt={salt}]" if retry else ""))
    print(f"锻造器：{[m[0] for m in run_models]} | 每资产每模型 {per_model} 条"
          + (f" | 资产过滤 {sorted(asset_filter)}" if asset_filter else ""))
    print("=" * 66, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    torch.set_num_threads(int(os.environ.get("N7A_THREADS",
                                             str(os.cpu_count() or 16))))

    spec = load_windows()
    out_series = {}
    forge_log = []
    if retry and os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as d:
            for k in d.files:
                out_series[k] = d[k]
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as fh:
                forge_log = json.load(fh).get("log", [])
        print(f"断点续传：已有 {len(out_series)} 条，仅补缺口\n", flush=True)

    for model_name, hf_id, use_chat in run_models:
        print(f"\n>>> 加载 {hf_id} ...", flush=True)
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype="float32")
        model.eval()
        proc, n_num_tok = make_numeric_processor(tok)
        print(f"    已加载 {time.time()-t0:.0f}s | 参数 "
              f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M | "
              f"数值掩码 token {n_num_tok}", flush=True)
        counter = 0
        for ai, (asset, s) in enumerate(spec.items()):
            if asset_filter is not None and asset not in asset_filter:
                continue
            for j in range(per_model):
                key = f"{asset}/{model_name}/{j}"
                if key in out_series:
                    counter += 1
                    continue
                done = False
                for attempt in range(RETRIES):
                    # 每次尝试轮转拟合窗 + 独立种子（salt 使重试轮次
                    # 不同于首轮，记录在案，可复现）
                    ctx_w = s["fit"][(j + attempt + salt) % len(s["fit"])]
                    gen_seed = GEN_SEED + ai * 1000 + counter + (
                        attempt + 1 + salt * 100) * 77777
                    series, meta = forge_one(
                        model_name, tok, model, ctx_w, asset,
                        s["target_std"], gen_seed, use_chat, proc)
                    meta["gen_seed"] = gen_seed
                    meta["attempt"] = attempt
                    meta["prompt_ctx_window"] = (j + attempt) % len(s["fit"])
                    forge_log.append(meta)
                    if series is not None:
                        out_series[key] = series
                        print(f"    [{asset}#{j}] OK {len(series)} pts "
                              f"{meta['segs']}segs try{attempt} "
                              f"({meta['gen_s']}s)", flush=True)
                        done = True
                        break
                    print(f"    [{asset}#{j}] DROP n={meta['n_generated']} "
                          f"{meta['segs']}segs try{attempt} "
                          f"({meta['gen_s']}s)", flush=True)
                if not done:
                    print(f"    [{asset}#{j}] {RETRIES} 次尝试均未达标",
                          flush=True)
                counter += 1
            # 逐资产检查点：中断后 --retry 可从已存条目续跑
            _save(npz_path, json_path, out_series, forge_log,
                   run_models, per_model)
        del model

    _save(npz_path, json_path, out_series, forge_log,
           run_models, per_model)
    n_ok = sum(1 for k in out_series if k != "__meta__")
    print(f"\n锻造完成：npz 共 {n_ok} 条（尝试 {len(forge_log)} 次），已存 npz+json",
          flush=True)


def _save(npz_path, json_path, out_series, forge_log, run_models, per_model):
    np.savez_compressed(npz_path,
                        **{k: v for k, v in out_series.items()
                           if k != "__meta__"})
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "protocol": {
                "mechanism": "textualize->iterative autoregressive "
                             "re-prompting->parse (SDForger sliding-window "
                             "mechanism, raw numeric template, "
                             "numeric-constrained decoding)",
                "forge_len": FORGE_LEN,
                "prefix_len": PREFIX, "seg_new": SEG_NEW,
                "seg_tokens": SEG_TOKENS, "max_segs": MAX_SEGS,
                "recontext": RECTX,
                "temperature": BASE_TEMP, "top_p": 0.95,
                "temp_escape": f"+{TEMP_ESC} per stall/discard "
                               f"(cap {TEMP_CAP}, reset on clean segment)",
                "segment_gates": {
                    "stall": f">={STALL} identical consecutive values "
                             f"-> run cut back",
                    "diversity": f"uniq ratio < {DIV_RATIO} "
                                 f"(min {DIV_MIN_UNIQ}) -> segment dropped",
                    "outlier": f"|v| > {OUTLIER_FACTOR}x context max "
                               f"-> segment dropped"},
                "acceptance": f"len>={FORGE_LEN} AND uniq_ratio>="
                              f"{ACCEPT_UNIQ}",
                "early_stop": "generation halts when 6 identical values "
                              "appear in decoded tail",
                "retries_per_gap": RETRIES,
                "per_model_target": per_model, "seed": GEN_SEED,
                "scale_align": "asset test-layer mean std",
                "models": {m[0]: m[1] for m in run_models},
            },
            "n_attempts": len(forge_log),
            "n_series_saved": sum(1 for k in out_series
                                  if k != "__meta__"),
            "log": forge_log,
        }, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
