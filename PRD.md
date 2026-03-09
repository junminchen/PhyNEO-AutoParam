# 产品需求文档 (PRD): 物理增强型力场参数化平台 (Physics-Informed GNN FF)

## 1. 项目概述 (Overview)

### 1.1 背景
传统的极化力场（如 AMOEBA, DMFF）开发存在一个巨大的瓶颈：**可移植性差与参数化成本高**。每引入一个新分子，都需要进行昂贵的量子化学计算和极不稳定的能量拟合。
我们目前拥有两套成熟的局部方案：
1. **纯物理推导 (`Auto-Multipol`)**：利用 PySCF+MBIS，基于有效原子体积 ($V_{eff}$) 和物理缩放律（如 $V^{10/3}$）快速得到各向同性的极化率和色散系数 ($C_6, C_8, C_{10}$)。
2. **纯数据拟合 (`train_dimer_backend`)**：基于 JAX 和 DMFF，通过二聚体相互作用能（SAPT），反向拟合短程 Slater 斥力参数 ($A, B$)。

### 1.2 核心目标
本项目旨在创建一个**全新的代码仓库**，将上述两套逻辑通过**图神经网络 (GNN)** 统一起来。
最关键的技术跨越是：**将原先基于 PyTorch 的 GNN (`byteff2/model/gnn.py`) 彻底重写为 JAX 架构**。这样可以直接与 JAX 原生的 `DMFF` (Differentiable Molecular Force Field) 无缝打通，实现从“图结构输入”到“总能量梯度”的**端到端全可微训练**。

---

## 2. 核心系统架构 (System Architecture)

整个系统分为三大模块，形成一个闭环的“数据生产 -> 模型预测 -> 能量求导”流水线。

### 2.1 模块一：自动化数据对齐与特征工程 (Data Distillation)
**目标**：整合现有脚本产出的分散数据，生成标准的 JAX-GNN 训练集。
- **输入**：分子坐标、拓扑结构。
- **物理锚点 (Targets_Phys)**：从单分子计算中提取 $V_{eff}, \alpha, C_n$。
- **能量标签 (Targets_Energy)**：从二聚体扫描中提取 SAPT 能量分解（Ex, Es, Pol, Disp, DHF）。
- **输出**：结构化的 `.npz` 或 `.hdf5` 数据集，包含原子级的图特征和节点目标值。

### 2.2 模块二：JAX-GNN 物理参数预测器 (Param-Predictor GNN)
**目标**：用 JAX (如 `Jraph` 或 `Flax`) 重写原有的 PyTorch GNN，负责从化学环境预测物理参数。
- **输入层**：原子的 One-hot 编码、配位数、键属性等。
- **消息传递层 (Message Passing)**：学习原子的局部拥挤度与电荷分布环境。
- **物理约束读出层 (Physics-Informed Readout)**：
  - **网络预测**：网络**不**直接预测所有的参数。它主要预测两个核心变量：
    1. 体积缩放因子 $\kappa_i \approx V_i^{eff} / V_i^{free}$
    2. Slater 斥力强度预选值 $A_{raw}$
  - **硬编码物理层**：
    - $C_{10,i} = C_{10}^{free} \times \kappa_i^{10/3}$ (严格遵守物理定律)
    - $B_i = \text{const} \cdot (\kappa_i)^{-1/3}$ (衰减常数由体积决定)
- **输出**：完整的力场参数字典（$\alpha, C_n, A, B$ 等）。

### 2.3 模块三：DMFF 端到端联合训练 (E2E JAX Training)
**目标**：将 GNN 的输出直接喂给 DMFF，通过计算能量误差来更新 GNN 的权重。
- **联合损失函数 (Joint Loss)**：
  $Loss = \lambda_{E} \cdot MSE(E_{DMFF\_pred}, E_{SAPT}) + \lambda_{Phys} \cdot MSE(\kappa_{pred}, \kappa_{MBIS\_target})$
- **技术优势**：因为 GNN 和 DMFF 都在 JAX 下，我们可以直接调用 `jax.value_and_grad(Loss)`，计算力场能量关于 GNN 网络权重的梯度，实现真正的“端到端”。

---

## 3. 关键路径与参考代码 (References & Integration)

新 Repo 的开发将大量复用和改造以下现有代码资产：

1. **物理基准生成器** (提供 $V_{eff}, C_n$ 的 Ground Truth)：
   - **路径**：`/home/jmchen/project/PhyNEO/workflow/lr_param_multiwfn/auto_multipol.py`
   - **作用**：在新框架中作为离线的数据生产工具。

2. **GNN 网络架构原型** (需由 PyTorch 翻译为 JAX)：
   - **路径**：`/home/jmchen/project/polff/byteff2/model/gnn.py`
   - **作用**：参考其 `GTConv` 和 `EGTConv` 的注意力机制，利用 JAX/Jraph 重新实现图卷积。

3. **DMFF 能量评估与优化器底座** (提供物理引擎)：
   - **路径**：`/home/jmchen/project_water_ethanol/phyneo-water-ethanol/train_dimer_backend.py`
   - **作用**：提取其 `PairKernel` 类的计算逻辑。原本这部分是用 Optax 直接更新离散的参数字典，新框架中将改为：`GNN(Graph) -> Params -> PairKernel -> Energy -> Loss -> Optax(Update GNN Weights)`。

---

## 4. 实施阶段 (Development Roadmap)

### Phase 1: 数据流建设 (Data Pipeline)
- 编写脚本，批量运行 `auto_multipol.py`，并将结果与现有的 Dimer SAPT 扫描数据（如 `data_sr.pickle`）进行对齐。
- 构建图数据集构建器（将 XYZ 转换为 Jraph 的 `GraphsTuple`）。

### Phase 2: GNN 的 JAX 移植与单分子预训练 (JAX-GNN Translation)
- 使用 `Flax` 和 `Jraph` 复现 `byteff2/gnn.py` 中的网络结构。
- 进行**单分子预训练 (Pre-training)**：仅使用物理 Loss（拟合 $V_{eff}$ 和 $\alpha$），验证 GNN 能够学到 MBIS 的空间划分逻辑。

### Phase 3: DMFF 耦合与二聚体微调 (End-to-End Fine-tuning)
- 将训练好的 JAX-GNN 插入到 `train_dimer_backend.py` 的流程中。
- 替换原来基于 `random` 初始化的参数树，改为调用 GNN 推理得到参数树。
- 开启联合训练，利用 SAPT 能量微调 GNN 最后一层输出的 $A$（Slater 斥力强度）。

### Phase 4: 泛化测试 (Validation)
- 挑选一个训练集中未出现过的新分子（如全新的溶剂或阴离子）。
- 送入 GNN 获得参数，直接生成 `ec_dmff_ff.xml`，评估其二聚体能量误差。如果成功，标志着平台具备了“Zero-shot 泛化”的力场生成能力。

---

## 5. 预期收益 (Expected Outcomes)
- **极速参数化**：新分子的参数生成从**数天/周**缩短至**几毫秒**（一次 GNN 推理）。
- **物理鲁棒性**：告别“纯能量拟合”带来的病态解，所有长程极化/色散参数均受 $V_{eff}$ 物理锚点保护。
- **全生态闭环**：彻底统一技术栈至 JAX，享受 XLA 编译在 GPU 上的极致加速。
