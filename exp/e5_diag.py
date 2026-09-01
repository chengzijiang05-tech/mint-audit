"""E5 诊断：认证通道（W_cert）货币的水平测量，决定流式监控的账本构造。

v1 发现：W_flag（判伪方向）对族内腐败（N1/N2/N5/splice）结构性无功效
，伪造窗与其自身轨道可交换，W_flag < 1，财富只衰减不累积。
检测功效应在认证方向：W_cert(x) = (K+1)·e^{s(x)} / Σ_{y∈{x}∪orbit} e^{s(y)}，
干净窗 W_cert ≈ 20-33（认证），伪造窗 W_cert ≈ 1-5（不可认证）。

本脚本实测（同 E5 协议编码器，seed 一致）：
  1. 校准块 34-43、pre-block 44-53、断点前后流窗的 W_cert 分布；
  2. 腐败窗（N1/N2/N5/US-splice，断点后若干位）的 W_cert / W_flag；
  3. 腐败窗 W_cert 在「校准块账本」与「滚动账本」下的秩
     ， 决定认证通道 e 值的账本选择。
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

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

from bench import n5_phase_forge  # noqa: E402
from mint.operators import generate_orbit, permute, time_reverse  # noqa: E402

import e5_drift as E5  # noqa: E402

RETURNS = E5.load_returns()


def e_pair_of(enc, head, w, rng):
    surrs, _ = generate_orbit(w, E5.K_EVAL, names=E5.ORBIT_NAMES, rng=rng)
    s = E5.logits(enc, head, [w] + list(surrs))
    return E5.e_pair(s[0], s[1:])


def summarize(name, vals):
    a = np.array(vals)
    print(f"  {name:<28} n={len(a):<3} min={a.min():<8.3f} "
          f"med={np.median(a):<8.3f} max={a.max():<8.3f} "
          f"认证率(≥20)={np.mean(a >= 20):.2f}")


def main():
    t0 = time.time()
    print("=" * 66)
    print("E5 诊断：W_cert 认证货币水平测量")
    print("=" * 66, flush=True)

    wins_all = {m: E5.windows_of(RETURNS[s["sym"]])
                for m, s in E5.STREAMS.items()}

    for mi, (m, spec) in enumerate(E5.STREAMS.items()):
        enc, head, info, _ = E5.train_a0([RETURNS[spec["sym"]]],
                                         spec["train_end"], spec["seed"],
                                         E5.EPOCHS)
        wins = wins_all[m]
        print(f"\n[{m}] 编码器 loss={info['loss']} "
              f"({time.time() - t0:.0f}s)")

        groups = {
            "cal 34-43 (账本候选)": range(34, 44),
            "pre 44-53 (流首)": range(44, 54),
            "mid 54-75 (断点前)": range(54, 76),
            "late 76-99 (断点后)": range(76, spec["end"] + 1),
        }
        wc_store = {}
        for gname, rng_idx in groups.items():
            wcs, wfs = [], []
            for i in rng_idx:
                wc, wf = e_pair_of(enc, head, wins[i],
                                   np.random.default_rng(700000 + mi * 100000 + i))
                wcs.append(wc)
                wfs.append(wf)
            wc_store[gname] = np.array(wcs)
            summarize(gname + " W_cert", wcs)
            print(f"       W_flag med={np.median(wfs):.4f}")

        # 腐败窗：断点后 3 个位置的四种腐败
        print(f"  --- 腐败窗（断点后位置）---")
        for i in (70, 80, 90):
            if i > spec["end"]:
                continue
            w = wins[i]
            corrupted = {
                "N1 permute": permute(w, np.random.default_rng(600000 + i)),
                "N2 reverse": time_reverse(w, np.random.default_rng(0)),
                "N5 phase": n5_phase_forge(w, seed=600100 + i),
            }
            if m == "CN":
                corrupted["US-splice"] = wins_all["US"][i]
            for cname, cw in corrupted.items():
                wc, wf = e_pair_of(enc, head, cw,
                                   np.random.default_rng(700000 + mi * 100000 + i))
                cal = wc_store["cal 34-43 (账本候选)"]
                roll = wc_store["mid 54-75 (断点前)"][-10:]
                r_cal = int(np.sum(cal < wc)) + 1
                r_roll = int(np.sum(roll < wc)) + 1
                print(f"    窗{i} {cname:<12} W_cert={wc:<8.3f} "
                      f"W_flag={wf:<8.4f} 秩(cal)={r_cal}/11 "
                      f"秩(roll)={r_roll}/11")

    print(f"\n完成 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
