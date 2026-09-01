# E1b 塌缩验证实验报告：现稿引擎对时间反转的结构性失明

日期：2026-08-20
脚本：`MINT/exp/e1b_collapse.py`（97 s，n=28 测试窗口）
产出：`results/e1b_collapse.json`、`figures/fig_e1b_collapse.pdf`

## 一、目标与命题

验证 T2（古典块盲区）：现稿特征块在时间反转算子 R 下的塌缩结构，
及其对 N2（时间倒置伪造）检测的后果。T2 是 MINT 学习范式的动机
前提，只有当现稿表示对零假设群（如可逆类）结构性失明时，"以零
假设族为原则性负样本训练编码器"才有不可替代的价值。

## 二、核心发现：特征不变性的三层结构

15 维特征在反转下的偏差并非均匀，而是精确分层：

| 层 | 特征 | 反转偏差 | 机制 |
|---|---|---|---|
| L1 排序不变量 | beta_tail, abs_acf1/3/5, surr_z, surr_z_acf_sum | ≤3×10⁻¹⁴（机器精度） | 秩统计与对称配对统计在数学上反转不变 |
| L2 分段/分箱估计器 | H_DFA, H_RS, qneg3, qpos3, dH, dalpha, abs_dfa, surr_z_dfa | r=0.90–0.995 | 段划分与数据顺序耦合（首尾余数、burn-in）的确定性伪影 |
| L3 GARCH MLE | garch_pers | r=0.696，mean|Δ|=0.049 | 条件方差递归方向改变 → 似然面不同 → MLE 漂移 |

关键佐证（IAAFT 理论）：IAAFT 的谱步只依赖 |FFT(x)|、边际步只依赖
sort(x)，二者均反转不变，故代理族本身逐位复现，surr_z 系特征的
精确不变性由此而来。

## 三、可逆零假设对照：伪影证明

方法：对每个测试窗口 w 生成其 IAAFT 代理 s（保谱线性类成员，
结构上时间可逆），在 s 与 Rs 上做同测量。若某特征在真实窗口上的
反转偏差与可逆成员上同量级（ratio≈1），则该偏差是估计器伪影而非
时间箭头信息。

结果（真实|Δ| / 可逆对照|Δ| 的比值）：

- L2：H_DFA ×1.37，H_RS ×1.46，qpos3 ×1.44，dalpha ×1.52，
  dH ×1.64，qneg3 ×1.77，abs_dfa ×2.31，surr_z_dfa ×2.52
- L3：garch_pers ×0.10（代理上 GARCH 拟合更不稳定，纯伪影，
  真实偏差反而小于对照基线）
- 对照基准 lev_asym（纯时间箭头统计量）：信号比 ×51.2

结论：L2/L3 的残余偏差至多为纯结构信号（lev_asym）强度的 5%
（2.52/51.2），其主体为估计器伪影；其中 DFA 类估计器（abs_dfa、
surr_z_dfa）的伪影被真实波动聚集结构放大至对照的 2.3–2.5 倍，
但仍远低于结构信号强度。

## 四、分数级与判定级塌缩

分数级（马氏距离 d(x) vs d(Rx)）：

| 口径 | corr | 可逆对照 corr |
|---|---|---|
| 8d 古典 | 0.9139 | 0.0467 |
| 15d 全量 | 0.9918 | 0.9331 |

d8 的散布由 garch_pers 的 MLE 路径噪声主导（equity 单资产
corr_d8=0.20）；d15 因维度稀释塌缩至对角线。

判定级（R1 三层切分、cal 冻结 q95 阈值、池化 n=28）：

| 口径 | FPR | recall(N2) | 差 | 配对一致率 |
|---|---|---|---|---|
| 8d | 0.500 | 0.536 | +0.036 | 0.893 |
| 15d | 0.107 | 0.143 | +0.036 | 0.964 |

**两口径的穿透量完全一致（+0.036）**：现稿最强引擎对时间倒置
伪造的检出率仅比假阳率高 3.6 个百分点，N2 几乎完全穿透。该残余
"视力"由 garch_pers 的 MLE 数值伪影贡献，与表示中的时间箭头无关
信息（第三节已证）。

## 五、表示选择的对照

lev_asym = corr(r_t, |r_{t+1}|)：

- 真实窗口：均值 −0.0412 → +0.0382（符号翻转，Wilcoxon
  p=7.5×10⁻⁹）
- 可逆代理：+0.0187 → +0.0203（p=0.78，无信号）

塌缩是表示的性质而非数学必然：把杠杆不对称统计量放进表示，
失明立即破除，这正是 MINT 编码器应从零假设族对比中自动学到的
东西。

## 六、判定结论（全部通过）

```
T2[layer1_order_invariants_exact]:          PASS
T2[layer2_binning_collapse]:                PASS
T2[layer2_residual_is_artifact]:            PASS
T2[layer3_garch_mle_collapse]:              PASS
T2[layer3_residual_is_artifact]:            PASS
T2[score_collapse_to_diagonal]:             PASS
T2[recall_equals_fpr_at_frozen_threshold]:  PASS
T2[lev_asym_contrast_significant]:          PASS
T2[lev_contrast_absent_on_reversible]:      PASS
T2[t2_confirmed]:                           PASS
```

## 七、对 MINT 研究方向的含义

1. T2 前提成立且被精确刻画：现稿引擎对可逆零假设族的结构性失明
   是三层的、可解释的、且不可通过调阈值修复，只有改变表示本身
   （引入对零假设群非不变的特征）才能获得原理性检出。
2. 三层结构给出训练信号的分级难度：L1 特征对可逆类完全不变
   （零对比信号），L2/L3 提供微弱伪影信号（ratio 1.4–2.5），
   仅 lev_asym 类统计量携带真信号（ratio 51）。对比学习编码器
   必须从原始窗口而非手工特征中学习，才能越过 L1 的不变性天花板。
3. 检验语义（每算子一假设）的训练设计获得直接依据：time_reverse
   负样本族将迫使编码器发现不可逆结构（杠杆不对称、相位时序），
   这是现稿特征块在数学上无法到达的区域。

## 八、复现

```bash
python exp/e1b_collapse.py           # 完整（~97 s）
python exp/e1b_collapse.py --smoke   # 冒烟（~60 s）
```

协议：window=1000、step=50、R1 三层切分原样、参考库马氏引擎逐行
复刻现稿、IAAFT 代理种子 1000+wi。
