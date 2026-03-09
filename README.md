# PhyNEO-AutoParam: 基于 JAX 的物理增强型力场自动化平台

PhyNEO-AutoParam 是下一代力场开发引擎，致力于通过 **JAX** 统一化学信息学（参考 bytemol）、图神经网络（参考 byteff2）与高精度物理模型（参考 Auto-Multipol）。

## 核心架构
- **Core**: 基于 RDKit 的分子语义处理，支持 SMILES 到 JAX 图的转换。
- **Graph Block**: 全可微的 EGT/GTConv 算子，提取高阶化学环境特征。
- **FF Block**: 注入 $V^{10/3}$ 等物理缩放律的参数读出层。
- **Engine**: 原生集成 **DMFF**，支持端到端能量拟合与力场生成。

## 运行流程
1. **数据生产**: 运行 `auto_multipol.py` 提取物理锚点。
2. **数据对齐**: 使用 `scripts/data_distill.py` 准备训练集。
3. **JAX 训练**: 结合单分子物理 Loss 与二聚体能量 Loss 训练 GNN。
4. **推理生成**: 输入 SMILES，瞬间获得 `FF.xml`。

---
详情请见 `PRD.md`。
