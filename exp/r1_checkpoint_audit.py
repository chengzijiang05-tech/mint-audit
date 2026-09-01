"""R1 检查点污染审计：全量复算 48 分数向量，逐条对照检查点。

背景：E1b 复算发现 R1-B equity 8d 的 clean 分数与协议正确的复算值
不符，但 forged 分数（条目 ≥12）逐位一致，怀疑检查点前缀条目
来自更早协议版本的运行（陈旧污染）。

方法：按 R1-B 当前协议（SPLIT_SPEC、种子公式 1000+wi*10+fi、
N1–N5 构造器、马氏引擎）全量重算每资产 8 clean + 40 forged 分数，
与检查点逐条比较，输出不匹配条目位置与陈旧前缀判定。
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

MINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSCT_ROOT = os.path.join(MINT_ROOT, "shared_infra", "fractal_consistency")
sys.path.insert(0, FSCT_ROOT)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from bench import HALLUCINATION_BUILDERS, n5_phase_forge  # noqa: E402
from features import extract_features  # noqa: E402

WINDOW, STEP = 1000, 50
SPLIT_SPEC = {
    "equity":  {"n_anchor": 15, "cal": (34, 43), "test": (63, 99), "cap": 8},
    "bond":    {"n_anchor": 15, "cal": (34, 43), "test": (63, 94), "cap": 8},
    "fx":      {"n_anchor": 11, "cal": (31, 36), "test": (37, 42), "cap": 6},
    "gold":    {"n_anchor": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
    "copper":  {"n_anchor": 11, "cal": (31, 33), "test": (34, 36), "cap": 3},
}
FORGE_FAMILIES = ["N1数值置换", "N2时间倒置", "N3标度破坏", "N4跨域嫁接",
                  "N5相位伪造"]


def span_windows(wins, lo, hi, cap=None):
    sel = wins[lo:hi + 1]
    if cap is not None and len(sel) > cap:
        idx = np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))
        sel = [sel[i] for i in idx]
    return list(sel)


def forge(x, family, seed):
    if family == "N5相位伪造":
        return n5_phase_forge(x, seed=seed)
    return HALLUCINATION_BUILDERS[family](x, seed=seed)


def main():
    with np.load(os.path.join(FSCT_ROOT, "data", "market_cache",
                              "returns.npz"), allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    report = {}
    for mode, full in (("8d", False), ("15d", True)):
        with open(os.path.join(FSCT_ROOT, "experiments", "results",
                               f"r1_checkpoint_{mode}.json"),
                  encoding="utf-8") as f:
            ck = json.load(f)

        for asset, r in returns.items():
            sp = SPLIT_SPEC[asset]
            wins = [r[i:i + WINDOW]
                    for i in range(0, len(r) - WINDOW + 1, STEP)]
            anchors = wins[:sp["n_anchor"]]
            clean = span_windows(wins, *sp["test"], cap=sp["cap"])

            A = np.array([extract_features(w, full=full) for w in anchors])
            A = A[np.isfinite(A).all(axis=1)]
            mu = A.mean(axis=0)
            inv = np.linalg.pinv(np.cov(A, rowvar=False)
                                 + 1e-6 * np.eye(A.shape[1]))

            def mah(w):
                f = extract_features(w, full=full)
                if not np.isfinite(f).all():
                    return np.nan
                dev = f - mu
                return float(np.sqrt(dev @ inv @ dev))

            items = ([("clean", w) for w in clean]
                     + [(fam, forge(w, fam, seed=1000 + wi * 10 + fi))
                        for wi, w in enumerate(clean)
                        for fi, fam in enumerate(FORGE_FAMILIES)])
            fresh = np.array([mah(x) for _, x in items])
            old = np.array(ck["scores"][asset][:len(items)])
            mism = np.where(np.abs(fresh - old) > 1e-6)[0]

            # 条目布局：n_clean 个 clean，随后每窗 5 族
            n_clean = len(clean)
            report[f"{mode}/{asset}"] = {
                "n_items": len(items),
                "n_mismatch": int(len(mism)),
                "mismatch_positions": mism.tolist(),
                "mismatch_range": [int(mism.min()), int(mism.max())]
                if len(mism) else None,
                "clean_mismatches": int(np.sum(mism < n_clean)),
                "stale_prefix_hypothesis": bool(
                    len(mism) and (mism.max() < 20)
                    and np.all(np.diff(mism) == 1)
                    and mism[0] == 0),
            }
            print(f"[{mode}/{asset}] {len(mism)}/{len(items)} 不匹配 "
                  f"clean段不匹配={report[f'{mode}/{asset}']['clean_mismatches']} "
                  f"位置范围={report[f'{mode}/{asset}']['mismatch_range']} "
                  f"陈旧前缀={report[f'{mode}/{asset}']['stale_prefix_hypothesis']}",
                  flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "r1_checkpoint_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n审计结果已写入 {out}")

    n_stale = sum(1 for v in report.values() if v["n_mismatch"] > 0)
    print(f"受污染资产-口径组合: {n_stale}/{len(report)}")


if __name__ == "__main__":
    main()
