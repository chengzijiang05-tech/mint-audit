# N8 实验报告：带杠杆的参数化敌手（GJR / EGARCH）

日期：2026-08-21
脚本：`exp/n8_gjr_egarch.py`
数据冻结：`shared_infra/fractal_consistency/data/market_cache/returns.npz`
产出：`results/n8_gjr_egarch.npz`（80 条，5 资产 × 4 配置 × 4 条）
        + `results/n8_gjr_egarch.json`（协议头 + 逐资产参数）

---

## 一、实验定位

N8 是 C4（泛化超越算子并集）的机制分离敌手，与 N6（对称 t 创新
GARCH）形成对照：GJR(1,1) 与 EGARCH(1,1) 显式建模杠杆不对称
（负冲击放大波动），是"带时间箭头"的并集外敌手。若 MINT 仍检出，
则"杠杆读取器"替代解释被排除；若检出下降，如实报告并收缩 C4 主张。

## 二、生成协议

- 模型：GJR(1,1) 与 EGARCH(1,1)，在 fit 层做 MLE，逐资产估计参数；
- 创新分布两版：`skewt`（偏态-0.2 与 t(5) 混合，单位方差）与
  `boot`（fit 层残差重采样），四配置 = {gjr, egarch} × {skewt, boot}；
- 每资产每配置 4 条，共 80 条 T=1000 路径；
- 尺度对齐：该资产测试层真实窗平均标准差；
- **杠杆由构造成立**（协议头 `leverage_by_construction: true`）。

逐资产参数已冻结：equity GJR 持久性 0.96、EGARCH 0.986（强杠杆，
典型股指）；fx EGARCH 持久性 0.641、b 项 -4.40（弱杠杆/无杠杆，
符合管理汇率资产）。这一资产异质性正是 E11 机制归因的对照面。

## 三、产物与下游

80 条序列冻结为 `n8_gjr_egarch.npz`，键 `asset/{gjr_skewt|gjr_boot|
egarch_skewt|egarch_boot}/{j}`。下游 E7/E9/E10/E11 均将其作为
N8 族读入。认证率一并进入 E7 的 e-rule 报告（模仿型
敌手的近真窗认证率上界）。

## 四、局限与披露

1. GJR/EGARCH 是带杠杆但不带真实生成器纹理（无语义先验、无
   跨资产耦合）的参数化递归，其拟真上限受模型族限制；
2. MLE 在 fit 层的参数估计不保证收敛到全局最优（逐资产已记录
   参数供复核）；
3. 80 条规模在 grep 层面的稳健性由 E11 聚类重抽定量，此处不做
   点估计过度推断。