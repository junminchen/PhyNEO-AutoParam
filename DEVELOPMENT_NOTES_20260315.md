# Development Notes: 2D Multi-Task Training, Multipoles, and Next Steps

## 1. 背景

这轮开发的核心目标是验证：

- `alpha`
- `C6 / C8 / C10`
- `dipole`
- `quadrupole`

是否应该直接监督进入训练，以及当前 2D GNN 架构对这些目标的适配程度。

## 2. 本轮完成的改动

### 2.1 训练目标扩展

当前训练已经支持直接监督以下目标：

- `charge`
- `polarizability`
- `dipole`
- `quadrupole`
- `C6`
- `C8`
- `C10`

并且训练时会按训练集标准差做归一化，避免不同量纲的目标互相压制。

相关文件：

- [`training/joint_trainer.py`](/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/training/joint_trainer.py)
- [`training/retrain_from_production.py`](/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/training/retrain_from_production.py)

### 2.2 模型结构改成 multi-head

当前模型已经从“共享 readout”改成“共享 encoder + 多独立 head”：

- `QEq/charge head`
- `polarizability head`
- `dipole head`
- `quadrupole head`
- `dispersion head`

相关文件：

- [`models/gnn_jax.py`](/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/models/gnn_jax.py)

### 2.3 支持两种 multipole 标签模式

当前训练和评估入口支持：

- `--multipole-frame local`
- `--multipole-frame global`

这样可以直接比较 local-frame 与 global Cartesian multipole 哪种更适合当前模型。

## 3. 本轮实验记录

### 3.1 V5: multi-head + local-frame multipoles

checkpoint:

- [`models/phyneo_production_results_v5_multihead_direct_alpha_disp_localframe_e120.flax`](/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/models/phyneo_production_results_v5_multihead_direct_alpha_disp_localframe_e120.flax)

summary:

- [`examples/results_pdb_bank_inference/retrain_diagnostics_v5/summary.json`](/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/examples/results_pdb_bank_inference/retrain_diagnostics_v5/summary.json)

关键结果：

- `best val loss = 1.606827`
- `charge RMSE = 0.03558`
- `dipole_local RMSE = 0.002872`
- `quadrupole_local RMSE = 0.001918`
- `polarizability RMSE = 3.30e-05`
- `C6 RMSE = 1.22e-04`
- `C8 RMSE = 1.10e-05`
- `C10 RMSE = 1.22e-06`

结论：

- `alpha + dispersion` 拟合得不错
- `charge` 不理想
- local-frame multipole 指标看起来可接受，但未必是当前 2D 模型的最佳监督方式

### 3.2 V6: multi-head + global multipoles

checkpoint:

- [`models/phyneo_production_results_v6_multihead_globalmultipole_e120.flax`](/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/models/phyneo_production_results_v6_multihead_globalmultipole_e120.flax)

summary:

- [`examples/results_pdb_bank_inference/retrain_diagnostics_v6_global/summary.json`](/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/examples/results_pdb_bank_inference/retrain_diagnostics_v6_global/summary.json)

关键结果：

- `best val loss = 1.157379`
- `charge RMSE = 0.03878`
- `dipole_global RMSE = 0.03390`
- `quadrupole_global RMSE = 0.02335`
- `polarizability RMSE = 2.93e-05`
- `C6 RMSE = 1.14e-04`
- `C8 RMSE = 1.32e-05`
- `C10 RMSE = 1.43e-06`

结论：

- 相比 `V5`，整体联合 loss 明显更好
- `global multipole` 对当前 2D 模型更容易优化
- `charge` 仍未改善

### 3.3 Dipole 数值分布核查

对 `V6` 做了训练标签与预测值范围检查，确认：

- 训练时和画图时用的是同一套 `global dipole` 标签
- parity 图上“预测值缩在 0 附近”不是画图 bug，而是模型真实输出偏小

核查结果：

- `dipole ref std = 0.03348`
- `dipole pred std = 0.00524`

而：

- `quadrupole ref std = 0.53144`
- `quadrupole pred std = 0.53296`

说明：

- `quadrupole` 的尺度基本学到了
- `dipole` 明显存在振幅收缩问题

### 3.4 V7: global multipoles + dipole weight x2

checkpoint:

- [`models/phyneo_production_results_v7_global_dipolex2_e120.flax`](/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/models/phyneo_production_results_v7_global_dipolex2_e120.flax)

metrics:

- [`models/phyneo_production_results_v7_global_dipolex2_e120.metrics.json`](/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/models/phyneo_production_results_v7_global_dipolex2_e120.metrics.json)

处理结果：

- 该训练已停止，不作为推荐版本

中间结果显示：

- 提高 `dipole loss weight` 到 `2.0` 会让联合训练明显变差
- `epoch 25 val = 3.1455`
- `epoch 50 val = 3.9861`

结论：

- 单纯增大 `dipole` 权重不是有效方案

## 4. 当前结论

### 4.1 2D GNN 对标量更友好

当前 2D GNN 更适合学习：

- `charge`
- `polarizability`
- `C6 / C8 / C10`

这些量要么是标量，要么更接近局部环境统计量。

### 4.2 Dipole / Quadrupole 本质上依赖 3D 几何

`dipole` 是 3D 向量，`quadrupole` 是 3D 二阶张量。它们与以下因素强相关：

- 原子坐标
- 局部空间方向
- 分子构型
- 参考坐标系 / local frame

因此，直接让纯 2D 图模型去回归它们，存在天然上限。

### 4.3 对当前 2D 模型，global multipole 比 local-frame multipole 更容易优化

这是一个经验结论，不代表 `global` 在物理上一定更合理，只表示：

- 在当前输入是 2D 图
- 当前模型不是 3D equivariant network

这种条件下，`global multipole` 对训练更友好。

### 4.4 当前最好的可用版本是 V6

当前建议保留的单分子预训练版本：

- [`models/phyneo_production_results_v6_multihead_globalmultipole_e120.flax`](/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/models/phyneo_production_results_v6_multihead_globalmultipole_e120.flax)

原因：

- 联合 loss 最低
- `alpha + dispersion` 继续保持较好
- `global multipole` 训练稳定性优于 local-frame 版本

## 5. 明确的下一步计划

### Plan A: 把 2D 模型收敛成“标量参数预测器”

目标：

- 让 2D 模型重点负责它真正擅长的目标

建议：

- 主推 `charge`
- 主推 `polarizability`
- 主推 `C6 / C8 / C10`
- 对 `dipole / quadrupole` 降级为辅助监督，或者暂时不作为核心 KPI

落地动作：

1. 基于 `V6` 继续调 `charge` 分支
2. 测试 `charge warmup -> joint fine-tune`
3. 测试更轻的 `dipole weight`，例如 `1.1 ~ 1.3`

### Plan B: 为 multipole 单独引入 3D 模型

目标：

- 把 `dipole / quadrupole` 从 2D-only 主干里拆出来

建议路线：

- 输入：`XYZ`
- 模型：3D-aware / equivariant network
- 输出：
  - `dipole`
  - `quadrupole`
  - 可选 `local-frame multipole`

推荐的系统拆分：

- `2D model`: 负责 `charge / alpha / C6 / C8 / C10`
- `3D model`: 负责 `dipole / quadrupole`

这样可以避免当前 2D 架构硬扛空间向量问题。

### Plan C: Dimer 校正阶段继续保留

即便未来引入 3D multipole 模型，仍然建议继续推进：

- 小批高质量 dimer 能量校正

原因：

- 单分子标签拟合好，不代表 pair interaction 就准确
- dimer 能量仍然是最终 DMFF 表现的重要闭环

相关文档：

- [`PRD_dimer_energy_calibration.md`](/Users/jeremychen/Desktop/Project/repo/PhyNEO-AutoParam/PRD_dimer_energy_calibration.md)

## 6. 推荐执行顺序

推荐按下面顺序推进，不要同时开太多线：

1. 固化 `V6` 为当前 2D baseline
2. 专门优化 `charge` 分支
3. 不再强推 2D 模型硬学高质量 `dipole`
4. 新开一条 3D multipole 分支
5. 完成后再接小批高质量 dimer 校正

## 7. 当前决策

当前开发决策是：

- 保留 `V6` 作为当前最优 2D baseline
- 停止继续在 `V7` 这种“单纯放大 dipole loss”的方向上投入
- 后续重点转向：
  - `charge` 改善
  - `3D multipole` 路线
  - `dimer calibration`
