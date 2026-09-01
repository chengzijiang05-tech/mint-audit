"""E12b worker：单资产并行版（与 e12b_oriented.py 逐字同种子同计算）。

用法：python e12b_worker.py <asset>
计算一个资产的 ProdE/ProdEm 全部读数，存 results/e12b_parts/<asset>.pkl，
由 e12b_merge.py 汇总出与串行版逐位一致的 e12_oriented.json。
"""
from __future__ import annotations

import os
import pickle
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

from features import extract_features  # noqa: E402
from mint.operators import generate_orbit  # noqa: E402
from e12_product_evalue import (  # noqa: E402
    RETURNS_NPZ, RESULTS_DIR, SPLIT_SPEC, FAM_BUILD, FAM_FILES,
    WINDOW, STEP, K_EVAL,
    span_windows, forge, readings,
)

PARTS_DIR = os.path.join(RESULTS_DIR, "e12b_parts")


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
    asset = sys.argv[1]
    t0 = time.time()
    with np.load(RETURNS_NPZ, allow_pickle=True) as d:
        assets = list(d.files)
        r = d[asset]
    ai = assets.index(asset)

    fams_npz = {}
    for tag, fn in FAM_FILES.items():
        path = os.path.join(RESULTS_DIR, fn)
        with np.load(path, allow_pickle=True) as d:
            by_asset = defaultdict(list)
            for key in d.files:
                a, rest = key.split("/", 1)
                by_asset[a].append(d[key])
        fams_npz[tag] = by_asset
    npz_fam = {tag: list(fams_npz[tag].get(asset, [])) for tag in fams_npz}
    fam_tags = FAM_BUILD + list(FAM_FILES)

    sp = SPLIT_SPEC[asset]
    wins = [r[i:i + WINDOW] for i in range(0, len(r) - WINDOW + 1, STEP)]
    anchors = wins[:sp["n_anchor"]]
    cal_wins = span_windows(wins, *sp["cal"])
    test_wins = span_windows(wins, *sp["test"], cap=sp["cap"])
    forged = {fam: [forge(w, fam, seed=1000 + wi * 10 + fi)
                    for wi, w in enumerate(test_wins)]
              for fi, fam in enumerate(FAM_BUILD)}
    print(f"[{asset}] cal{len(cal_wins)} test{len(test_wins)} "
          f"| npz 族 {({t: len(v) for t, v in npz_fam.items()})}",
          flush=True)

    A = np.array([extract_features(w, full=False) for w in anchors])
    A = A[np.isfinite(A).all(axis=1)]
    mu = A.mean(axis=0)
    inv = np.linalg.pinv(np.cov(A, rowvar=False)
                         + 1e-6 * np.eye(A.shape[1]))

    RA = np.array([readings(w, mu, inv) for w in anchors])
    orient = np.where(RA.mean(axis=0) > 0, 1.0, -1.0)
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

    state = {"orient": orient.tolist(),
             "W": {m: {"cal": [], "real": [], "fam": {f: [] for f in fam_tags}}
                   for m in ("ProdE", "ProdEm")}}
    for kind, payload, seedf in groups:
        if kind == "fam":
            fam, wins = payload
            per = {"ProdE": [], "ProdEm": []}
            for i, w in enumerate(wins):
                Ws = prod_e_variants(w, K_EVAL,
                                     np.random.default_rng(seedf(i)),
                                     mu, inv, orient)
                per["ProdE"].append(Ws["o"])
                per["ProdEm"].append(Ws["m"])
            state["W"]["ProdE"]["fam"][fam].append(np.array(per["ProdE"]))
            state["W"]["ProdEm"]["fam"][fam].append(np.array(per["ProdEm"]))
        else:
            grp_wins = payload
            per = {"ProdE": [], "ProdEm": []}
            for i, w in enumerate(grp_wins):
                Ws = prod_e_variants(w, K_EVAL,
                                     np.random.default_rng(seedf(i)),
                                     mu, inv, orient)
                per["ProdE"].append(Ws["o"])
                per["ProdEm"].append(Ws["m"])
            state["W"]["ProdE"][kind].append(np.array(per["ProdE"]))
            state["W"]["ProdEm"][kind].append(np.array(per["ProdEm"]))

    os.makedirs(PARTS_DIR, exist_ok=True)
    out = os.path.join(PARTS_DIR, f"{asset}.pkl")
    with open(out, "wb") as fh:
        pickle.dump(state, fh)
    print(f"[{asset}] saved {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
