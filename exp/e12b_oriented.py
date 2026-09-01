"""E12b：拟合层定向的乘积 e 值基线（R12 公平版）+ 未定向镜像变体。

v1（e12_product_evalue.py）的乘积 e 值读数未定向：z 自校准对称，
真实金融窗的杠杆不对称为负（代码口径），证据质量流向镜像侧，
N2 AUC 0.005 完全倒挂，把该数当基线是稻草人。公平做法与古典
引擎同等待遇：用拟合层（真实窗，每个方法都拿得到）定向每
个读数的符号，o_j = sign(mean_anchor r_j)。

本脚本复用 v1 的 Gen2cal 结果（直接复制 JSON）与完全相同的窗口
构造/伪造种子/轨道种子，只重算 ProdE 的两个变体：
  ProdE   = 拟合层定向（正式基线）
  ProdEm  = 未定向镜像（机制注脚：方向是参考知识）

输出：results/e12_oriented.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

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

from bench import HALLUCINATION_BUILDERS, n5_phase_forge  # noqa: E402
from features import extract_features  # noqa: E402
from features.phase_ext import leverage_asym  # noqa: E402
from mint.operators import generate_orbit  # noqa: E402
from e12_product_evalue import (  # noqa: E402
    RETURNS_NPZ, RESULTS_DIR, WINDOW, STEP, SPLIT_SPEC, FAM_BUILD,
    FAM_FILES, IN_NULL, OUT_UNION, K_EVAL, ALPHA, SEED,
    span_windows, forge, vol_clust, readings, auc_pair, wilson,
)

V1_JSON = os.path.join(RESULTS_DIR, "e12_product_evalue.json")
OUT_JSON = os.path.join(RESULTS_DIR, "e12_oriented.json")


def prod_e_variants(x, k, rng, mu, inv, orient):
    """同轨道双变体：s = Σ o_j z_j（定向）与 s = Σ z_j（镜像）。"""
    surrs, _ = generate_orbit(x, k, rng=rng)
    members = [x] + list(surrs)
    R = np.array([readings(m, mu, inv) for m in members])
    sd = R.std(axis=0)
    sd[sd < 1e-12] = 1e-12
    Z = (R - R.mean(axis=0)) / sd
    Ws = {}
    for tag, sgn in (("o", orient), ("m", np.ones(3))):
        s = (Z * sgn).sum(axis=1)
        m = float(np.max(s))
        lse = m + float(np.log(np.sum(np.exp(s - m))))
        Ws[tag] = float((k + 1) * np.exp(s[0] - lse))
    return Ws


def main() -> None:
    t0 = time.time()
    print("=" * 66)
    print("E12b 拟合层定向乘积 e 值 + 镜像变体（轨道种子与 v1 逐字一致）")
    print("=" * 66, flush=True)

    with open(V1_JSON, encoding="utf-8") as fh:
        v1 = json.load(fh)

    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    fams_npz = {}
    for tag, fn in FAM_FILES.items():
        path = os.path.join(RESULTS_DIR, fn)
        with np.load(path, allow_pickle=True) as d:
            by_asset = defaultdict(list)
            for key in d.files:
                asset, rest = key.split("/", 1)
                by_asset[asset].append(d[key])
        fams_npz[tag] = by_asset
    fam_tags = FAM_BUILD + list(FAM_FILES)

    anom = {"ProdE": {"cal": [], "real": [], "fam": {f: [] for f in fam_tags}},
            "ProdEm": {"cal": [], "real": [], "fam": {f: [] for f in fam_tags}}}
    Wraw = {m: {"real": [], "fam": {f: [] for f in fam_tags}}
            for m in ("ProdE", "ProdEm")}
    orients = {}

    for ai, (asset, r) in enumerate(returns.items()):
        sp = SPLIT_SPEC[asset]
        wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
        anchors = wins[:sp["n_anchor"]]
        cal_wins = span_windows(wins, *sp["cal"])
        test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])

        forged = {fam: [forge(w, fam, seed=1000 + wi * 10 + fi)
                        for wi, w in enumerate(test_wins)]
                  for fi, fam in enumerate(FAM_BUILD)}
        npz_fam = {tag: list(fams_npz[tag].get(asset, []))
                   for tag in fams_npz}
        print(f"\n[{asset}] cal{len(cal_wins)} test{len(test_wins)} "
              f"| npz 族 {({t: len(v) for t, v in npz_fam.items()})}",
              flush=True)

        A = np.array([extract_features(w, full=False) for w in anchors])
        A = A[np.isfinite(A).all(axis=1)]
        mu = A.mean(axis=0)
        inv = np.linalg.pinv(np.cov(A, rowvar=False)
                             + 1e-6 * np.eye(A.shape[1]))

        RA = np.array([readings(w, mu, inv) for w in anchors])
        orient = np.where(RA.mean(axis=0) > 0, 1.0, -1.0)
        orients[asset] = orient.tolist()
        print(f"    anchor means = {np.round(RA.mean(axis=0), 4).tolist()} "
              f"-> orient {orient.tolist()}", flush=True)

        base = 950000 + ai * 10000
        groups = [("cal", cal_wins, lambda i: base + i),
                  ("real", test_wins, lambda i: base + 100 + i)]
        for fi, fam in enumerate(FAM_BUILD):
            groups.append(("fam", (fam, forged[fam]),
                           lambda i, fi=fi: base + 200 + fi * 100 + i))
        for fj, tag in enumerate(fams_npz):
            groups.append(("fam", (tag, npz_fam[tag]),
                           lambda i, fj=fj: base + 700 + fj * 100 + i))

        for kind, payload, seedf in groups:
            if kind == "fam":
                fam, wins = payload
                per = {"ProdE": [], "ProdEm": []}
                for i, w in enumerate(wins):
                    Ws = prod_e_variants(w, K_EVAL,
                                         np.random.default_rng(seedf(i)),
                                         mu, inv, orient)
                    per["ProdE"].append(Ws["o"])
                    perProdEm = Ws["m"]
                    per["ProdEm"].append(perProdEm)
                key = fam[:2] if fam in FAM_BUILD else fam
                anom["ProdE"]["fam"][fam].append(-np.array(per["ProdE"]))
                anom["ProdEm"]["fam"][fam].append(-np.array(per["ProdEm"]))
                Wraw["ProdE"]["fam"][fam].append(np.array(per["ProdE"]))
                Wraw["ProdEm"]["fam"][fam].append(np.array(per["ProdEm"]))
            else:
                grp_wins = payload
                per = {"ProdE": [], "ProdEm": []}
                for i, w in enumerate(grp_wins):
                    Ws = prod_e_variants(w, K_EVAL,
                                         np.random.default_rng(seedf(i)),
                                         mu, inv, orient)
                    per["ProdE"].append(Ws["o"])
                    per["ProdEm"].append(Ws["m"])
                anom["ProdE"][kind].append(-np.array(per["ProdE"]))
                anom["ProdEm"][kind].append(-np.array(per["ProdEm"]))
                if kind == "real":
                    Wraw["ProdE"]["real"].append(np.array(per["ProdE"]))
                    Wraw["ProdEm"]["real"].append(np.array(per["ProdEm"]))
        print(f"    [{asset}] 完成 ({time.time() - t0:.0f}s)", flush=True)

    # ---- 池化（逐资产 cal q95 阈值，与 e2/v1 同协议）----
    res = {"protocol": {
        "prod_e": "3 readings (leverage_asym, vol_clust, -mahal8d) "
                  "self-calibrated over K+1 orbit members, oriented by "
                  "anchor-layer mean signs; W=(K+1)*softmax(sum o_j z_j); "
                  "exact E[W]=1 via Lemma 1",
        "prod_em": "same construction without anchor orientation "
                   "(mirror variant, mechanism note)",
        "gen2cal": "copied from e12_product_evalue.json (unchanged run)",
        "K_eval": K_EVAL, "alpha": ALPHA, "seed": SEED,
        "in_null": IN_NULL, "out_union": OUT_UNION,
        "orients": orients,
    }, "methods": {}}

    for tag_m in ("ProdE", "ProdEm"):
        recalls, flags_real, real_all, fam_all = {}, [], [], {}
        fam_auc, fam_wil = {}, {}
        for seg, _asset in enumerate(returns):
            cal = np.asarray(anom[tag_m]["cal"][seg])
            thr = float(np.quantile(cal, 1 - ALPHA))
            real = np.asarray(anom[tag_m]["real"][seg])
            flags_real.append(real > thr)
            real_all.append(real)
        for fam in fam_tags:
            hit, tot, f_scores = 0, 0, []
            for seg, _asset in enumerate(returns):
                cal = np.asarray(anom[tag_m]["cal"][seg])
                thr = float(np.quantile(cal, 1 - ALPHA))
                f = np.asarray(anom[tag_m]["fam"][fam][seg])
                fl = f > thr
                hit += int(np.sum(fl))
                tot += len(fl)
                f_scores.append(f)
            fam_all[fam] = np.concatenate(f_scores)
            key = fam[:2] if fam in FAM_BUILD else fam
            recalls[key] = hit / max(tot, 1)
            fam_wil[key] = list(wilson(hit, tot))
        real_sc = np.concatenate(real_all)
        for fam in fam_tags:
            key = fam[:2] if fam in FAM_BUILD else fam
            fam_auc[key] = round(auc_pair(fam_all[fam], real_sc), 3)
        all_f = np.concatenate([fam_all[f] for f in fam_tags])
        s = {"fpr": float(np.mean(np.concatenate(flags_real))),
             "recall": recalls, "recall_wilson": fam_wil,
             "auc_family": fam_auc,
             "auc_all": round(auc_pair(all_f, real_sc), 3)}
        s["auc_in_null"] = round(float(np.mean(
            [v for k, v in fam_auc.items() if k in IN_NULL])), 3)
        out_v = [v for k, v in fam_auc.items() if k in OUT_UNION]
        s["auc_out_union"] = round(float(np.mean(out_v)), 3)
        res["methods"][tag_m] = s
        print(f"\n[{tag_m}] FPR={s['fpr']:.3f} AUC_all={s['auc_all']} "
              f"in-null={s['auc_in_null']} out-union={s['auc_out_union']}")
        for k in fam_auc:
            print(f"  {k}: AUC={fam_auc[k]} recall={recalls[k]:.3f}"
                  f" Wilson[{fam_wil[k][0]:.2f},{fam_wil[k][1]:.2f}]")

    for tag_m in ("ProdE", "ProdEm"):
        Wr = np.concatenate(Wraw[tag_m]["real"])
        e_rule = {"real_certify": float(np.mean(Wr >= 1 / ALPHA))}
        for fam in fam_tags:
            Wf = np.concatenate(Wraw[tag_m]["fam"][fam])
            key = fam[:2] if fam in FAM_BUILD else fam
            e_rule[f"{key}_certify"] = float(np.mean(Wf >= 1 / ALPHA))
        res[f"e_rule_{tag_m}"] = e_rule
        print(f"\n{tag_m} e-rule:", json.dumps(e_rule, indent=1), flush=True)

    res["methods"]["Gen2cal"] = v1["methods"]["Gen2cal"]
    res["e_rule_ProdE_v1_unoriented"] = v1["e_rule_ProdE"]
    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved: e12_oriented.json ({res['elapsed_s']}s)")


if __name__ == "__main__":
    main()
