# PRD: 用小批高质量 Dimer 能量校正 PhyNEO-AutoParam

## 1. 背景

当前模型已经可以直接监督单分子原子参数：

- `charge`
- `dipole (local frame)`
- `quadrupole (local frame)`
- `polarizability`
- `C6 / C8 / C10`

这一步能让模型学到稳定、低成本、可批量生成的物理标签，但仍然存在一个明显缺口：

- 单分子标签拟合得好，不等于导出到 DMFF 后的二聚体相互作用能就准确。
- 尤其是 `electrostatics / polarization / dispersion` 的耦合项，最后是在 pair level 上体现误差。

因此需要引入一个第二阶段校正流程：在保留大规模低成本单分子数据预训练的前提下，用一小批高质量 dimer 参考能量对模型进行定向校正。

## 2. 目标

建立一个两阶段训练流程：

1. 阶段 A：用大规模单分子数据训练基础参数预测器。
2. 阶段 B：用少量高质量 dimer 参考能量做校正，使导出的 DMFF 参数在 pair interaction 上更接近高精度参考。

成功标准：

- 不显著破坏单分子参数回归质量。
- 显著降低 dimer 长程项与总相互作用能误差。
- 在有限高质量样本下稳定训练，不出现明显过拟合或参数漂移。

## 3. 非目标

本阶段不做以下事情：

- 不追求完全替代全量 CamCASP/SAPT 数据集。
- 不把所有训练都切换成纯能量监督。
- 不在第一版里引入复杂的端到端可微 MD 轨迹训练。
- 不强行覆盖 bonded force field 或全体系 condensed-phase observable 校正。

## 4. 用户与场景

主要用户：

- 需要快速为新电解液分子生成 DMFF 参数的研究人员。
- 需要在有限高质量量化预算下提升 pair interaction 可信度的力场开发者。

典型场景：

- 全量分子先用低成本 pipeline 生成标签并预训练。
- 挑选一批代表性 dimer 构型，用高质量方法得到参考能量。
- 用这批样本对模型做小步校正。
- 输出更接近高质量参考的 XML。

## 5. 数据设计

### 5.1 基础预训练数据

来源：

- `examples/production_results`

监督目标：

- `charge`
- `dipole local`
- `quadrupole local`
- `polarizability`
- `C6 / C8 / C10`

作用：

- 学到稳定的 atom-wise parameter prior
- 给第二阶段提供好的初始化

### 5.2 高质量 dimer 校正数据

建议规模：

- 第一版：20 到 100 个 dimer case
- 每个 case 含 10 到 100 个扫描点或代表性构型

每条样本最少需要：

- `monomer A` 结构
- `monomer B` 结构
- `dimer geometry`
- `distance / orientation metadata`
- 高质量参考能量

优先级从高到低：

1. `E_tot`
2. `E_es / E_pol / E_disp`
3. `E_ex / E_dhf`

建议覆盖：

- 常见溶剂-溶剂
- 阳离子-溶剂
- 阴离子-溶剂
- 阳离子-阴离子
- 强极性与弱极性体系

## 6. 方案设计

### 6.1 两阶段训练

阶段 A：单分子预训练

- 输入：分子图
- 输出：每原子参数
- loss：`charge + dipole_local + quadrupole_local + polarizability + C6 + C8 + C10`
- 所有目标先按训练集标准差做归一化

阶段 B：dimer 能量校正

- 输入：两分子结构和 dimer 构型
- 流程：
  - 模型预测 monomer 参数
  - 导出或直接映射到 DMFF 参数树
  - 计算 dimer interaction energy
  - 与高质量参考比较

### 6.2 推荐的参数更新策略

第一版建议采用保守策略：

- 冻结大部分 GNN encoder
- 只微调：
  - multipole readout
  - polarizability / dispersion readout
  - 可选 residual calibration head

原因：

- 小样本高质量数据容易过拟合
- 直接全模型微调更容易破坏单分子参数分布

### 6.3 推荐的 loss

总 loss：

`L = lambda_mono * L_mono + lambda_dimer * L_dimer + lambda_reg * L_reg`

其中：

- `L_mono`
  - 保留一小部分单分子 batch 继续混合训练
  - 防止校正阶段遗忘基础标签

- `L_dimer`
  - `w_tot * MSE(E_tot_pred, E_tot_ref)`
  - `w_es * MSE(E_es_pred, E_es_ref)`
  - `w_pol * MSE(E_pol_pred, E_pol_ref)`
  - `w_disp * MSE(E_disp_pred, E_disp_ref)`

- `L_reg`
  - 限制参数偏离阶段 A checkpoint 太远
  - 可选 L2 regularization 或 output drift penalty

第一版建议权重：

- `lambda_mono = 1.0`
- `lambda_dimer = 1.0`
- `lambda_reg = 0.1`

如果没有分解能量，先用：

- `L_dimer = MSE(E_tot_pred, E_tot_ref)`

### 6.4 校正对象的优先级

第一阶段优先校正：

- `charge`
- `dipole / quadrupole`
- `polarizability`
- `C6 / C8 / C10`

不建议第一版优先校正：

- bonded 项
- 所有 Slater A 系数一起自由漂移

更稳妥的方式是：

- 对 `dispersion` 和 `electrostatics/polarization` 相关 readout 做显式微调
- 短程排斥项先只做小范围 residual

## 7. 模型与工程落地

### 7.1 训练入口

新增或扩展一个训练入口，例如：

- `training/finetune_with_dimers.py`

职责：

- 读取高质量 dimer 数据
- 加载阶段 A checkpoint
- 混合单分子 batch 和 dimer batch
- 输出新 checkpoint 与评估报告

### 7.2 数据接口

建议 dimer 数据统一格式，至少包含：

- `name`
- `monomer_a`
- `monomer_b`
- `dimer_pdb` 或坐标数组
- `ref_tot`
- 可选 `ref_es`
- 可选 `ref_pol`
- 可选 `ref_disp`

### 7.3 评估产物

每次训练输出：

- train / val loss 曲线
- dimer parity plot
- 分能量分量 parity plot
- 每类体系误差表
- 与校正前 checkpoint 的对比报告

## 8. 成功指标

必须指标：

- dimer `E_tot` RMSE 相比阶段 A 至少下降 20%
- 如果有分解能量，`E_es / E_pol / E_disp` 中至少两个分量显著改善
- 单分子参数回归误差恶化不超过 10%

加分指标：

- 对未参与校正的 dimer 类型仍有泛化提升
- 导出 XML 后的 DMFF quick-check 数值更稳定

## 9. 风险与缓解

风险 1：小样本过拟合

- 缓解：
  - 冻结 encoder
  - 混合单分子 batch
  - early stopping

风险 2：DMFF 反传链不稳定

- 缓解：
  - 第一版先离线计算 energy target
  - 必要时先从不可微导出链切到近似 differentiable surrogate

风险 3：高质量 dimer 覆盖不足

- 缓解：
  - 先挑代表性最强的体系
  - 优先覆盖离子-溶剂和强极性体系

风险 4：总能量改善但分项劣化

- 缓解：
  - 尽量保留 `es/pol/disp` 分项监督
  - 训练后强制产出 component-wise report

## 10. 里程碑

M1：打通直接监督的单分子预训练

- 完成 `charge/dipole_local/quadrupole_local/polarizability/C6/C8/C10` 训练

M2：整理小批高质量 dimer 数据

- 明确数据格式
- 形成 train / val split

M3：实现 dimer fine-tuning

- checkpoint 加载
- dimer loss
- mixed training

M4：完成评估与决策

- 输出对比图
- 判断是否值得扩大高质量数据规模

## 11. 决策建议

建议先以一个非常小的验证集启动：

- `EC-EC`
- `Li-EC`
- `PF6-EC`
- `FEC-FEC`

先确认以下三件事：

- dimer 能量 loss 是否稳定下降
- 单分子标签是否没有明显漂移
- 导出的 XML 在 DMFF quick-check 下是否更接近参考

如果这一步成立，再扩大到更多 dimer 体系。这个顺序更稳，也更省高质量量化成本。
