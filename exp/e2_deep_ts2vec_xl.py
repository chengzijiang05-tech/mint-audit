"""S3 充分训练 TS2Vec 对照：D1 在 8 倍算力预算下的收敛检验。

实验目的：深度基线以「与 MINT 相同的 250-epoch CPU
预算」训练，本实验以更大预算复测 TS2Vec，排除预算因素导致的低估。

本脚本给 D1（TS2Vec+马氏）一个明确超出常规的预算：
  epochs 250 → 1000（4×），每 epoch 切片 64 → 128（2×），
  总算力 8×，并记录每 100 epoch 的训练损失轨迹以证明收敛性。

协议与 e2_deep.py 逐位一致（SPLIT_SPEC / 种子 / q95 冻结 / 评分），
仅预算不同；结果与 e2_deep.json 的 D1 行直接可比。

产出：results/e2_deep_ts2vec_xl.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

MINT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(MINT_ROOT, "exp"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import torch  # noqa: E402

torch.set_num_threads(4)

import e2_deep as base  # noqa: E402

EPOCHS_XL = 1000
SLICES_XL = 128
CKPT_EVERY = 100
RESULTS_DIR = os.path.join(MINT_ROOT, "results")


def train_d1_xl(r, train_end, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = base.TS2VecEncoder()
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR_D,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_XL)
    traj = []
    t0 = time.time()
    for ep in range(EPOCHS_XL):
        X = torch.tensor(base.zscore(
            base.slices_batch(r, train_end, SLICES_XL, rng)),
            dtype=torch.float32)
        opt.zero_grad()
        loss = base.ts2vec_losses(model, X, rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if (ep + 1) % CKPT_EVERY == 0:
            traj.append({"epoch": ep + 1,
                         "loss": round(float(loss.item()), 4)})
    model.eval()
    return model, {"loss_traj": traj,
                   "final_loss": traj[-1]["loss"],
                   "s": round(time.time() - t0, 1)}


def main():
    with np.load(base.RETURNS_NPZ, allow_pickle=True) as d:
        returns = {k: d[k] for k in d.files}

    anom = {"cal": [], "real": [], "fam": {f: [] for f in base.FAMILIES}}
    per_asset, train_info = {}, {}

    for ai, (asset, r) in enumerate(returns.items()):
        sp = base.SPLIT_SPEC[asset]
        wins = [r[i:i + base.WINDOW]
                for i in range(0, len(r) - base.WINDOW + 1, base.STEP)]
        anchors = wins[:sp["n_anchor"]]
        cal_wins = base.span_windows(wins, *sp["cal"])
        test_wins = base.span_windows(wins, *sp["test"], cap=sp["cap"])
        train_end = sp["cal"][0] * base.STEP
        forged = {fam: [base.forge(w, fam, seed=1000 + wi * 10 + fi)
                        for wi, w in enumerate(test_wins)]
                  for fi, fam in enumerate(base.FAMILIES)}
        print(f"[{asset}] anchor{sp['n_anchor']} cal{len(cal_wins)} "
              f"test{len(test_wins)} | XL 1000ep x {SLICES_XL} slices",
              flush=True)

        # 与 e2_deep D1 同种子（mseed = SEED + 10000 + ai*1000 + 0）
        mseed = base.SEED + 10000 + ai * 1000
        model, info = train_d1_xl(r, train_end, mseed)
        train_info[asset] = info
        ctx_mu, ctx_inv = base.d1_fit_mahal(model, anchors)
        anom["cal"].append(base.d1_score(model, cal_wins, ctx_mu, ctx_inv))
        anom["real"].append(base.d1_score(model, test_wins, ctx_mu, ctx_inv))
        for fam in base.FAMILIES:
            anom["fam"][fam].append(
                base.d1_score(model, forged[fam], ctx_mu, ctx_inv))
        print(f"    D1-XL {info['s']}s final_loss={info['final_loss']} "
              f"traj[0]={info['loss_traj'][0]['loss']}", flush=True)

        cal = np.asarray(anom["cal"][-1])
        cal = cal[np.isfinite(cal)]
        thr = float(np.quantile(cal, 1 - base.ALPHA))
        real = np.asarray(anom["real"][-1])
        entry = {"n_test": len(test_wins),
                 "fpr": float(np.nanmean(real > thr))}
        for fam, key in zip(base.FAMILIES, base.FAM_SHORT):
            f = np.asarray(anom["fam"][fam][-1])
            entry[key] = float(np.nanmean(f > thr))
        per_asset[asset] = entry

    real_all = np.concatenate(anom["real"])
    ok_r = np.isfinite(real_all)
    fam_auc = {}
    for fam, key in zip(base.FAMILIES, base.FAM_SHORT):
        f = np.concatenate(anom["fam"][fam])
        f = f[np.isfinite(f)]
        fam_auc[key] = base.auc_pair(f, real_all[ok_r])
    all_f = np.concatenate([np.concatenate(anom["fam"][f])
                            for f in base.FAMILIES])
    all_f = all_f[np.isfinite(all_f)]

    res = {
        "variant": "TS2Vec+Mahalanobis, extended budget (S3 control)",
        "epochs": EPOCHS_XL, "n_slices": SLICES_XL,
        "compute_multiple_vs_paper": (EPOCHS_XL * SLICES_XL)
        / (base.EPOCHS_D * base.N_SLICES_D),
        "seed_scheme": "same as e2_deep D1 default seed",
        "input_norm": "per-window z-score (instance)",
        "auc_all": base.auc_pair(all_f, real_all[ok_r]),
        "auc_family": fam_auc,
        "per_asset": per_asset,
        "train_info": train_info,
    }
    out = os.path.join(RESULTS_DIR, "e2_deep_ts2vec_xl.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("\npooled AUC(all families) = %.3f" % res["auc_all"])
    print("per-family AUC:", {k: round(v, 3) for k, v in fam_auc.items()})
    print("FPR:", {k: round(v["fpr"], 3) for k, v in per_asset.items()})
    print("saved:", out)


if __name__ == "__main__":
    main()
