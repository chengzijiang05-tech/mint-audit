"""E12b merge：汇总 e12b_parts 的五资产状态，产出 e12_oriented.json。

池化逻辑逐字复刻 e12b_oriented.py 的 main 尾段；anom = -W。
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

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

from e12_product_evalue import (  # noqa: E402
    RETURNS_NPZ, RESULTS_DIR, FAM_BUILD, FAM_FILES,
    IN_NULL, OUT_UNION, K_EVAL, ALPHA, SEED,
    auc_pair, wilson,
)

V1_JSON = os.path.join(RESULTS_DIR, "e12_product_evalue.json")
OUT_JSON = os.path.join(RESULTS_DIR, "e12_oriented.json")
PARTS_DIR = os.path.join(RESULTS_DIR, "e12b_parts")


def main() -> None:
    t0 = time.time()
    with open(V1_JSON, encoding="utf-8") as fh:
        v1 = json.load(fh)
    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        assets = list(d.files)

    parts = []
    for a in assets:
        path = os.path.join(PARTS_DIR, f"{a}.pkl")
        with open(path, "rb") as fh:
            parts.append(pickle.load(fh))
    orients = {a: p["orient"] for a, p in zip(assets, parts)}

    fam_tags = FAM_BUILD + list(FAM_FILES)
    anom = {"ProdE": {"cal": [], "real": [], "fam": {f: [] for f in fam_tags}},
            "ProdEm": {"cal": [], "real": [], "fam": {f: [] for f in fam_tags}}}
    Wraw = {m: {"real": [], "fam": {f: [] for f in fam_tags}}
            for m in ("ProdE", "ProdEm")}
    for p in parts:
        for m in ("ProdE", "ProdEm"):
            anom[m]["cal"].append(-np.asarray(p["W"][m]["cal"][0]))
            anom[m]["real"].append(-np.asarray(p["W"][m]["real"][0]))
            Wraw[m]["real"].append(np.asarray(p["W"][m]["real"][0]))
            for f in fam_tags:
                anom[m]["fam"][f].append(-np.asarray(p["W"][m]["fam"][f][0]))
                Wraw[m]["fam"][f].append(np.asarray(p["W"][m]["fam"][f][0]))

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
        for seg, _asset in enumerate(assets):
            cal = np.asarray(anom[tag_m]["cal"][seg])
            thr = float(np.quantile(cal, 1 - ALPHA))
            real = np.asarray(anom[tag_m]["real"][seg])
            flags_real.append(real > thr)
            real_all.append(real)
        for fam in fam_tags:
            hit, tot, f_scores = 0, 0, []
            for seg, _asset in enumerate(assets):
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
