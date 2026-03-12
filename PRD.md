# 产品需求文档 (PRD): PhyNEO-AutoParam 2.0

## 1. 愿景与定位
PhyNEO-AutoParam 是一个高度模块化的、基于 JAX 的物理增强型力场参数化平台。它借鉴了 `bytemol` 的化学信息学底座和 `byteff2` 的混合力场建模思路，通过 JAX 的全可微特性与 DMFF 引擎无缝耦合，实现“SMILES/XYZ -> 高精度极化力场参数”的端到端自动化。

---

## 2. 模块化架构设计 (Architectural Layers)

项目分为四个核心层，严格遵循“数据-图-特征-参数-能量”的流转逻辑。

### 2.1 基础核心层 (Core Layer) - 参考 `bytemol/core/`
**目标**：构建分子的“物理语义”对象。
- **Molecule 类**：封装 RDKit，支持从 SMILES (mapped) 和 XYZ 构造。
- **拓扑引擎**：自动识别键连、环结构、共振态及对称性。
- **JAX 适配器**：将 RDKit 分子图转换为 `jraph.GraphsTuple`，包括原子序数、键属性（类型、共轭性）以及初始坐标。

### 2.2 特征提取层 (Graph Block) - 参考 `byteff2/model/gnn.py`
**目标**：从图结构中提取原子的局部化学环境特征。
- **算子选择**：利用 JAX/Jraph 实现 **EGT (Edge-augmented Graph Transformer)** 或带有注意力机制的 **GTConv**。
- **特征演化**：
  - 节点特征：学习原子局部拥挤度、电负性梯度。
  - 边特征：学习键级、电荷转移路径。
- **技术优势**：全 JAX 实现，支持 XLA 编译下的超大规模批处理。

### 2.3 物理生成层 (FF Block) - 参考 `byteff2/ff_layers/`
**目标**：将隐藏层 Embedding 映射为具有物理约束的力场参数。
- **ChargeVolume 模块**：预测原子电荷及有效体积因子 $\kappa_i$。
- **Dispersion 模块**：利用 `auto_multipol.py` 中的逻辑，根据 $\kappa_i$ 自动推导 $C_6, C_8, C_{10}$。
- **ShortRange 模块**：
  - 预测 Slater 斥力强度 $A$。
  - 按照 $B \propto \kappa^{-1/3}$ 物理缩放定律生成衰减常数 $B$。
- **Bonded 模块**：预测键、角、扭转角的力常数。

### 2.4 物理计算层 (Engine Layer) - 集成 `DMFF`
**目标**：利用预测的参数计算势能，驱动参数优化。
- **集成方式**：将 FF Block 的输出直接映射到 DMFF 的 `Hamiltonian` 参数树。
- **能量评估**：调用 `train_dimer_backend.py` 中的 `PairKernel` 逻辑进行 JAX 加速的能量计算。

### 2.5 动态修正层 (Delta-Learning Block) - 构型感知增强
**目标**：在 MD 过程中，根据实时 3D 构型对 2D 拓扑产生的“基础参数”进行微调。
- **逻辑核心**：$Param(R) = Param_{Base}(Topology) + \Delta Param(R)$
- **输入**：实时原子坐标 $R$ 及 2D 隐层 Embedding。
- **修正项**：
  - $\Delta q$: 处理构型变化导致的诱导电荷/极性变化。
  - $\Delta A$: 修正由于空间拥挤或排斥导致的非键强度漂移。
- **技术优势**：保证了势能面的大框架由物理规律稳定的 2D-GNN 支撑，而 3D 模块只学习高阶修正，降低了过拟合风险。

---

## 3. 端到端工作流 (End-to-End Workflow)

### Step 1: 数据准备 (Distillation)
- 使用 `/home/jmchen/project/PhyNEO/workflow/lr_param_multiwfn/auto_multipol.py` 生成物理锚点标签（$\alpha, C_n, V_{eff}$）。
- 使用 SAPT 计算二聚体能量拆解标签。

### Step 2: 模型训练 (JAX Training)
- **阶段 A (物理预训练)**：GNN 拟合单分子的物理锚点。
- **阶段 B (联合微调)**：GNN 接入 DMFF，通过拟合二聚体相互作用能（Ex, Es, Pol, Disp, DHF）对 Slater $A$ 参数进行微调。
- **损失函数**：
  $Loss = \sum (\lambda_{phys} \cdot Loss_{Phys} + \lambda_{energy} \cdot Loss_{Energy})$

### Step 3: 一键参数化 (Deployment)
- 用户输入：`SMILES`
- 输出：`FF.xml` (DMFF/OpenMM 格式)

---

## 4. 关键代码参考 (Source Map)
- **化学底座**：`/home/jmchen/project/polff/bytemol/core/`
- **GNN 算子**：`/home/jmchen/project/polff/byteff2/model/gnn.py`
- **物理层逻辑**：`/home/jmchen/project/PhyNEO/workflow/lr_param_multiwfn/auto_multipol.py`
- **JAX 训练底座**：`/home/jmchen/project_water_ethanol/phyneo-water-ethanol/train_dimer_backend.py`

---

## 5. 开发路线图
1. **[Core]** 构建 JAX 兼容的 `Molecule` 数据结构。
2. **[Model]** 移植 EGT/GTConv 到 JAX/Flax。
3. **[Physics]** 实现全物理约束的 Readout 层。
4. **[Training]** 打通 GNN -> DMFF -> Loss 的全求导链条。
