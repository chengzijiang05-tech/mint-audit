# N7b 实验报告：QuantGAN 风格深度生成器敌手

日期：2026-08-21
脚本：`exp/n7b_quantgan.py`
数据冻结：`shared_infra/fractal_consistency/data/market_cache/returns.npz`
产出：`results/n7b_quantgan.npz`（20 条，5 资产 × 4 条）+ `results/n7b_quantgan.json`（协议头 + 训练史）

---

## 一、实验定位

N7b 是 C5（标题级主张"auditing machine-generated series"）与 C4
（泛化超越算子并集）的签名正交敌手。深度生成器具备拟合杠杆不对称
的既定能力（QuantGAN 类文献共识），因此它同时检验两个
问题：真实深度生成序列是否进入结构通道（C5），以及"编码器只是放
大的杠杆不对称读取器"这一替代解释是否成立（C4 的机制分离）。

## 二、生成器协议

- 生成器：TCN，5 个因果膨胀块，感受野 rf=251 日（覆盖短程波动
  聚集；
- 判别器：4 层 TCN + spectral norm，hinge loss；
- 训练数据：fit-layer 窗口（train_end = cal 层起点，与编码器 A0 训练
  数据同源，保证测试层零泄漏）；
- 训练步数：400 步 G/D 交替（与既有深度基线等 CPU 预算）；
- 尺度对齐：与该资产测试层真实窗平均标准差对齐，消除平凡量纲线索；
- 产物：每资产 4 条 T=1000 路径，`asset/quantgan/{j}` 键。

训练史判别器损失收敛至 d≈2.0（hinge 均衡点），生成器损失在零附近
震荡，五资产均收敛正常，无模式坍缩迹象。

## 三、产物与下游

20 条序列已冻结为 `n7b_quantgan.npz`。下游 E7（真实生成器审计）
将其作为 N7b 族读入；E9（第二代 TR+surrogate 基线）逐族逐资产
计算 p 值；E10（字典读出）将其纳入知识层评估。N7b 同时是 E11
聚类推断的敌手族之一。

## 四、局限与披露

1. 感受野 251 vs 511+、spectral norm vs gradient penalty、固定窗口
   vs 滚动窗口条件，三处工程简化换取 CPU 预算内可复现，均已写入
   协议头；
2. 每资产 4 条为点估计下限，主结果由 E7 的聚类稳健 CI 定量；
3. 生成器不追求拟合尾部极值（未加厚尾创新项），其相对真实窗的
   可检测签名由 E7/E10 实证，不在此处预设。