"""E5 敏感性：MAX_REFITS=3 下的多轮重拟合成本曲线（收敛 vs 持续失效）。

E5 报告局限 5：v2 主运行 MAX_REFITS=1，二次告警后仅纪元重置；持久
失效下的多轮重拟合行为（收敛 or 持续失效）与部署成本未测。

两分支（同一编码器、同一双通道监控协议，仅失效形态不同）：

  场景 R（收敛分支：真实制度变迁）
    CN equity 真实流 [44,99]（2015 股灾断点窗 64），MAX_REFITS=3，
    与主运行同种子（告警序列应逐位复现 84→94）。问题：窗 94 二次
    告警后的重拟合 #2（训练跨度扩至日 4700，股灾后制度数据已入训）
    能否恢复认证力并保持到流末？
    预期：重拟合把新制度纳入训练 → 认证恢复 → 收敛（无三次告警）。

  场景 S（持续失效分支：对抗性持续腐败）
    N2 ρ=1.0 腐败流 [63,99]（与主运行同种子逐位复现，首告警窗 68），
    MAX_REFITS=3。信任假设显式化：重拟合训练数据与参考层重置均取
    可信源（clean 序列/同索引 clean 窗），"重拟合只信账本外可信
    数据"的诚实部署语义。问题：对抗性腐败不是版本失效，重拟合无法
    修复 → 告警复现 → 3 次重拟合耗尽 → 纪元重置下监控是否继续诚实
    报告（锯齿告警）→ 成本是否封顶。

产出：results/e5_sens_refits.json + figures/fig_e5_sens.pdf
"""
from __future__ import annotations

import json
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

from mint.operators import time_reverse  # noqa: E402

import e5_drift as E5  # noqa: E402

MAX_REFITS = 3
RESULTS_DIR = os.path.join(MINT_ROOT, "results")
FIGURES_DIR = os.path.join(MINT_ROOT, "figures")


def refit_recovery(res):
    """每个重拟合的恢复证据：后续 ≤4 窗 W_cert 中位 + 持续到下次告警的窗数。"""
    out = []
    alarms = [a["window"] for a in res["alarms"]]
    for rf in res["refits"]:
        aw = rf["after_window"]
        pos = res["win_idx"].index(aw)
        seg = res["w_cert"][pos + 1:pos + 5]
        nxt = [w for w in alarms if w > aw]
        out.append({
            "after_window": aw,
            "train_end_day": rf["train_end_day"],
            "cpu_s": rf["cpu_s"],
            "w_cert_median_next4": (round(float(np.median(seg)), 2)
                                    if seg else None),
            "next_alarm_window": nxt[0] if nxt else None,
            "windows_held": ((nxt[0] - aw) if nxt
                             else (res["win_idx"][-1] - aw))})
    return out


def build_verdict(scenR, scenS, max_refits):
    """数据驱动的判定解读（主流程与 JSON 修补共用，保证一致）。"""
    refs = scenR["refits"]
    read_R = ""
    if not refs:
        read_R = "无告警无重拟合（编码器无认证力的冒烟态）"
    else:
        r1 = refs[0]
        read_R = (f"重拟合#1@{r1['after_window']} 恢复短暂（后 4 窗 "
                  f"W_cert 中位 {r1['w_cert_median_next4']}，保持 "
                  f"{r1['windows_held']} 窗后再度告警）；")
        if len(refs) >= 2:
            r2 = refs[-1]
            read_R += (f"重拟合#2@{r2['after_window']} 后弱认证均衡"
                       f"（后 4 窗 W_cert 中位 {r2['w_cert_median_next4']}）"
                       f"+ 参考层再校准 → 流末静默：操作性收敛"
                       f"（无三次告警），非绝对认证恢复")
        elif scenR.get("converged"):
            read_R += "末次重拟合后保持到流末"
        else:
            read_R += "仍在演化（流末前再次告警）"

    exh = scenS["refits_exhausted"]
    post = scenS["post_exhaustion_alarms"]
    peak_w = None
    if exh and post:
        idxS, detS = scenS["win_idx"], scenS["detector"]
        start = idxS.index(post[0])
        peak_w = idxS[start + int(np.argmax(detS[start:]))]
    read_S = (f"告警复现 {scenS['alarms']}；{scenS['refit_count']} 次重拟合"
              f"耗尽（{scenS['refit_cpu_total_s']}s）")
    if post:
        read_S += (f"；耗尽后纪元重置继续报告（告警 {post}），D 峰值窗 "
                   f"{peak_w} 后滚动参考层吸收持续腐败 → 静默"
                   f"（首批告警簇即部署行动信号）")
    elif exh:
        read_S += "；耗尽后监控静默（参考层吸收）"
    per_refit = ([rf["cpu_s"] for rf in scenR["refits"]]
                 + [rf["cpu_s"] for rf in scenS["refits"]])
    return {
        "R_convergence": {
            "refits": scenR["refit_count"],
            "cpu_s": scenR["refit_cpu_total_s"],
            "converged": scenR.get("converged", False),
            "refit_recovery_w_cert": [rf["w_cert_median_next4"]
                                      for rf in scenR["refits"]],
            "reading": read_R},
        "S_persistent_failure": {
            "refits": scenS["refit_count"],
            "cpu_s": scenS["refit_cpu_total_s"],
            "refits_exhausted": exh,
            "alarms_total": len(scenS["alarms"]),
            "post_exhaustion_alarms": post,
            "post_exhaustion_d_peak_window": peak_w,
            "reading": read_S},
        "shared_mechanism": (
            "滚动参考层再校准使静默语义为版本相对一致而非绝对认证力："
            "R 的流末静默（弱认证均衡）与 S 的耗尽后吸收同源于此；"
            "部署上首批告警簇（R: 84/94，S: 68-77）即行动信号"),
        "cost_curve": {
            "per_refit_cpu_s": (round(float(np.mean(per_refit)), 1)
                                if per_refit else None),
            "max_total_refit_cpu_s": round(
                scenR["refit_cpu_total_s"] + scenS["refit_cpu_total_s"], 1),
            "cap": max_refits},
    }


def main():
    t0 = time.time()
    returns = E5.load_returns()
    spec = E5.STREAMS["CN"]
    wins = E5.windows_of(returns[spec["sym"]])

    print("=" * 66)
    print("E5 敏感性：MAX_REFITS=3 多轮重拟合（收敛 vs 持续失效）"
          + ("  [冒烟]" if E5.QUICK else ""))
    print(f"协议同主运行 v2 | epochs={E5.EPOCHS} | seed={E5.SEED} | "
          f"B={E5.DETECTOR_B:.0f}")
    print("=" * 66, flush=True)

    enc, head, info, _ = E5.train_a0(
        [returns[spec["sym"]]], spec["train_end"], spec["seed"], E5.EPOCHS)
    print(f"[enc] loss={info['loss']}", flush=True)
    cal_idx = list(range(34, 44))

    # ---- 场景 R：真实制度变迁（收敛分支）----
    stream_idx = list(range(44, spec["end"] + 1))
    ref_vals = E5.ref_w_certs(enc, head, wins, cal_idx, spec["seed"])

    def refit_ref_R(enc2, head2, wi):
        lo = max(0, wi - E5.LEDGER_LEN + 1)
        return E5.ref_w_certs(enc2, head2, wins,
                              list(range(lo, wi + 1)), spec["seed"] + 900000)

    resR = E5.monitor_run(enc, head, [wins[i] for i in stream_idx],
                          stream_idx, spec["seed"], ref_vals,
                          refit_series=returns[spec["sym"]],
                          refit_seed=spec["seed"],
                          refit_ref_vals=refit_ref_R, max_refits=MAX_REFITS)
    last_alarm_R = resR["alarms"][-1]["window"] if resR["alarms"] else None
    last_refit_R = (resR["refits"][-1]["after_window"]
                    if resR["refits"] else None)
    scenR = {
        "stream": f"CN real [44,{spec['end']}]（2015 股灾断点窗 64）",
        "alarms": [a["window"] for a in resR["alarms"]],
        "alarm_channels": [a["channel"] for a in resR["alarms"]],
        "refits": refit_recovery(resR),
        "refit_count": len(resR["refits"]),
        "refit_cpu_total_s": resR["refit_cpu_s"],
        "converged": bool(last_refit_R is not None
                          and (last_alarm_R is None
                               or last_alarm_R <= last_refit_R)),
        "windows_after_last_refit": (spec["end"] - last_refit_R
                                     if last_refit_R is not None else None),
        "win_idx": resR["win_idx"],
        "detector": resR["detector"],
        "w_cert": resR["w_cert"],
    }
    print(f"[R] 告警={scenR['alarms']} 重拟合={scenR['refit_count']} "
          f"({scenR['refit_cpu_total_s']}s) 收敛={scenR['converged']}",
          flush=True)

    # ---- 场景 S：对抗性持续腐败（持续失效分支）----
    onset = E5.SIM_ONSET
    sim_idx = list(range(onset, 100))
    rng = np.random.default_rng(E5.SEED + 520000)  # k=0 → N2_rho1.0 同种子
    stream = []
    for i in sim_idx:
        w = wins[i]
        if rng.random() < 1.0:
            w = time_reverse(w, rng)
        stream.append(w)
    ref_sim = E5.ref_w_certs(enc, head, wins,
                             list(range(onset - E5.LEDGER_LEN, onset)),
                             spec["seed"] + 800000)

    def refit_ref_S(enc2, head2, wi):
        lo = max(0, wi - E5.LEDGER_LEN + 1)
        return E5.ref_w_certs(enc2, head2, wins,
                              list(range(lo, wi + 1)), E5.SEED + 950000)

    resS = E5.monitor_run(enc, head, stream, sim_idx, E5.SEED + 600000,
                          ref_sim, refit_series=returns[spec["sym"]],
                          refit_seed=spec["seed"],
                          refit_ref_vals=refit_ref_S, max_refits=MAX_REFITS)
    exhausted = len(resS["refits"]) == MAX_REFITS
    post_exh = [a["window"] for a in resS["alarms"]
                if exhausted and resS["refits"]
                and a["window"] > resS["refits"][-1]["after_window"]]
    scenS = {
        "stream": "N2 ρ=1.0 corrupted [63,99]（同主运行种子复现）",
        "trust_assumption": ("重拟合训练数据=clean 源序列；参考层重置="
                             "同索引 clean 窗（可信源语义）"),
        "alarms": [a["window"] for a in resS["alarms"]],
        "refits": refit_recovery(resS),
        "refit_count": len(resS["refits"]),
        "refit_cpu_total_s": resS["refit_cpu_s"],
        "refits_exhausted": exhausted,
        "post_exhaustion_alarms": post_exh,
        "monitor_stays_honest": bool(len(post_exh) > 0),
        "win_idx": resS["win_idx"],
        "detector": resS["detector"],
        "w_cert": resS["w_cert"],
    }
    print(f"[S] 告警={scenS['alarms']} 重拟合={scenS['refit_count']} "
          f"({scenS['refit_cpu_total_s']}s) 耗尽={exhausted} "
          f"耗尽后告警={post_exh}", flush=True)

    verdict = build_verdict(scenR, scenS, MAX_REFITS)

    result = {
        "experiment": "E5_sensitivity_max_refits3",
        "max_refits": MAX_REFITS,
        "epochs": E5.EPOCHS, "seed": E5.SEED, "quick": E5.QUICK,
        "scenario_R_real_regime": scenR,
        "scenario_S_persistent_corruption": scenS,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, "e5_sens_refits.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    print("\n=== 敏感性判定 ===")
    print(json.dumps(verdict, ensure_ascii=False, indent=1)[:1500])
    print(f"\n输出: {out_json}")

    make_figure(scenR, scenS)


def make_figure(scenR, scenS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))

    # (a) R：D_t 与重拟合
    ax = axes[0]
    ax.semilogy(scenR["win_idx"], np.maximum(scenR["detector"], 1e-4),
                color="#A9745E", lw=1.3)
    ax.axhline(E5.DETECTOR_B, ls="--", lw=0.9, color="#8B4A4A")
    ax.text(44.3, E5.DETECTOR_B * 1.3, "B", fontsize=7.5, color="#8B4A4A")
    for k, rf in enumerate(scenR["refits"]):
        ax.axvline(rf["after_window"], ls="-.", lw=0.9, color="#1F3A5C")
        ax.text(rf["after_window"] + 0.3, 2e-3, f"重拟合{k + 1}",
                fontsize=7, color="#1F3A5C")
    for w in scenR["alarms"]:
        ax.plot(w, E5.DETECTOR_B * 1.6, marker="v", ms=5, color="#5A2A20")
    ax.axvline(64, ls=":", lw=0.9, color="#777")
    ax.text(64.4, 5e-4, "股灾", fontsize=7.5, color="#777")
    ax.set_title("(a) 场景R 真实制度变迁：D_t 与重拟合循环", fontsize=8.5)
    ax.set_ylabel("D_t", fontsize=8)
    ax.set_xlabel("窗口索引", fontsize=8)

    # (b) R：W_cert 恢复
    ax = axes[1]
    ax.semilogy(scenR["win_idx"], np.maximum(scenR["w_cert"], 1e-2),
                color="#4A6FA5", lw=1.2)
    ax.axhline(20.0, ls=":", lw=0.9, color="#777")
    ax.text(44.3, 24, "认证线 20", fontsize=7, color="#777")
    for rf in scenR["refits"]:
        ax.axvline(rf["after_window"], ls="-.", lw=0.9, color="#1F3A5C")
    ax.axvline(64, ls=":", lw=0.9, color="#777")
    ax.set_title("(b) 场景R W_cert：每次重拟合后的认证恢复", fontsize=8.5)
    ax.set_ylabel("W_cert", fontsize=8)
    ax.set_xlabel("窗口索引", fontsize=8)

    # (c) S：锯齿告警与成本封顶
    ax = axes[2]
    ax.semilogy(scenS["win_idx"], np.maximum(scenS["detector"], 1e-4),
                color="#5A2A20", lw=1.2)
    ax.axhline(E5.DETECTOR_B, ls="--", lw=0.9, color="#8B4A4A")
    ax.text(63.3, E5.DETECTOR_B * 1.3, "B", fontsize=7.5, color="#8B4A4A")
    for k, rf in enumerate(scenS["refits"]):
        ax.axvline(rf["after_window"], ls="-.", lw=0.9, color="#1F3A5C")
        ax.text(rf["after_window"] + 0.3, 3e-2, f"重拟合{k + 1}",
                fontsize=7, color="#1F3A5C")
    for w in scenS["alarms"]:
        ax.plot(w, E5.DETECTOR_B * 1.6, marker="v", ms=5, color="#8B4A4A")
    ax.set_title("(c) 场景S 持续腐败：告警复现与成本封顶", fontsize=8.5)
    ax.set_ylabel("D_t", fontsize=8)
    ax.set_xlabel("窗口索引（腐败注入窗 63 起）", fontsize=8)

    for ax in axes:
        ax.tick_params(labelsize=7.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "fig_e5_sens.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
