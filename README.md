# MINT

Manufactured-null invariant testing for auditing machine-generated financial time series.

This repository contains the code, frozen experimental results, and figure-generation scripts for the paper *Manufactured nulls: orbit-contrastive learning with anytime-valid evidence for auditing machine-generated financial series*.

## Overview

An unsourced number sequence leaves no reference to check against. MINT builds the audit for exactly that case.

Five surrogate operators with published mechanisms, among them permutation, time reversal, amplitude-adjusted Fourier transforms, iterated AAFT, and block shuffling, turn each real reference window into an orbit of forged twins. A contrastive encoder trained on real windows alone learns the statistic that separates them from their orbits. An orbit-normalized e-value converts the score gap into evidence that is exact by exchangeability, valid under any stopping rule, and portable across markets.

On 168 machine-generated paths produced by language models, a QuantGAN-style deep generator, a Chronos foundation-model resampler, and maximum-likelihood-fitted GJR/EGARCH recursions, the certification rule certifies none of them. The strongest forgery accumulates W = 19.45 against a certification line of 20, and false positives stay at zero across 120 transfer windows spanning CN, US, and HK markets.

## Repository layout

| Path | Content |
|---|---|
| `mint/` | core library, surrogate operators and the contrastive encoder |
| `shared_infra/` | feature extractors, benchmark builders, and frozen market data |
| `exp/` | experiment scripts, one file per experiment |
| `results/` | frozen results in JSON and NPZ, with a written report per experiment |
| `paper-figures/` | figure-generation scripts, shared figure data, and rendered outputs |

## Requirements

Python 3.10 with PyTorch, NumPy, and SciPy. Every experiment runs on a single CPU workstation; no GPU is required.

## Data

Reference data are daily returns of five assets retrieved through the open-source `akshare` interface: CSI 300, the SSE Treasury Bond Index, USD/CNY, and SHFE gold and copper. The US and HK transfer streams are SPDR S&P 500 ETF daily returns together with Hang Seng Index and Hang Seng China Enterprises Index daily returns. Frozen copies ship inside `shared_infra/fractal_consistency/data/`.

## Protocol

All experiments share one frozen split. Reference windows cover 1000 trading days with a stride of 50. For each asset the first layer of windows fits baselines and the encoder, the middle layer calibrates thresholds, and the final layer is held out for test. Layers are strictly time-ordered, so test data never enter training or calibration. Thresholds, seeds, and compute budgets were fixed before any evaluation was run, and the split specification is registered in `shared_infra/fractal_consistency/experiments/results/`.

## Experiments

Each experiment is a standalone script that writes its frozen result into `results/`. The written reports document protocol, tables, and reading for every experiment.

| ID | Script | Subject |
|---|---|---|
| E1b | `exp/e1b_collapse.py` | collapse of the classical fingerprint kit under time reversal |
| G1 | `exp/g1_prototype.py` | prototype gate, three seeds |
| E2 | `exp/e2_main.py`, `exp/e2_seed_agg.py` | main audit, five markets, nine forgery families, ten seeds |
| E2 deep | `exp/e2_deep.py`, `exp/e2_deep_agg.py`, `exp/e2_deep_ts2vec_xl.py` | deep baselines TS2Vec, Anomaly Transformer, DCdetector |
| N6 | `exp/e2_n6_garch.py` | GARCH-forged adversary family |
| E3 | `exp/e3_power.py` | power at small reference-set sizes |
| E4 | `exp/e4_transfer.py` | CN/US/HK market transfer, 3x3 matrix |
| E5 | `exp/e5_drift.py`, `exp/e5_sens.py` | dual-channel drift monitor, 2015 crash and COVID |
| E6 | `exp/e6_bench480.py` | 480 real LLM outputs, typed routing |
| E7 | `exp/e7_generators.py` | audit of real machine generators |
| E9 | `exp/e9_gen2_surrogate.py`, `exp/e9b_gen2_directional.py`, `exp/e9c_gen2_gjrnull.py` | second-generation surrogate-calibrated baselines |
| E10 | `exp/e10_dict_readout.py` | dictionary readout fidelity |
| E11 | `exp/e11_per_asset_cluster.py` | per-asset mechanism and cluster-robust inference |
| E12 | `exp/e12_product_evalue.py`, `exp/e12b_oriented.py` | product e-value baselines |
| E13 | `exp/e13_laundering.py` | laundering adversaries |
| E14 | `exp/e14_eg_encoder.py`, `exp/e15_eg_conditional.py` | encoder variants |
| E16 | `exp/e16_cluster_inf.py` | cluster-robust pairwise inference |
| E17 | `exp/e17_genpaths_w.py` | generator paths under Wilson bounds |
| N7a | `exp/n7a_llm_forge.py`, `exp/merge_n7a.py` | LLM-forged thousand-day paths |
| N7b | `exp/n7b_quantgan.py` | QuantGAN-style deep generator |
| N7c | `exp/n7c_chronos_forge.py`, `exp/n7c_score.py` | Chronos resampler adversary |
| N8 | `exp/n8_gjr_egarch.py` | GJR/EGARCH fitted recursions |
| R1 | `exp/r1_checkpoint_audit.py` | registered split audit |

A typical run:

```bash
python exp/e2_main.py
```

Forgery generators that require external models, among them N7a and N7c, load local checkpoints through the paths declared at the top of each script.

## Figures

The eight paper figures regenerate from the frozen results:

```bash
python paper-figures/figsrc/fig4_main_results.py
python paper-figures/figsrc/fig5_genaudit.py
python paper-figures/scripts/fig2_evidence.py
```

Each script reads its numbers from `paper-figures/scripts/figdata.json` and the hardcoded rows that mirror the frozen JSON, then writes PDF, PNG, and SVG into `paper-figures/figures/`. The framework diagram (Fig. 1) is edited as `paper-figures/figsrc/fig1_framework.drawio` and exported with draw.io desktop; the data-driven evidence figure (Fig. 2) lives in `paper-figures/scripts/`; the remaining six figures (Figs. 3–8) regenerate from Python scripts in `paper-figures/figsrc/`.

## Integrity

Results in `results/` are frozen. Regenerating a figure re-reads the same JSON, so a rerun cannot silently change a published number. The per-experiment reports state the seed, budget, and split for every table entry.
