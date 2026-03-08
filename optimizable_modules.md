# PI-GANO 可优化模块分析文档

## 场景定义

| 场景ID | 描述 | 训练模式 |
|--------|------|---------|
| S1 | Darcy 流 · 物理信息训练 | PINO |
| S2 | Darcy 流 · 有监督训练 | NO |
| S3 | 平板应力 · 物理信息训练 | PINO |
| S4 | 平板应力 · 有监督训练 | NO |

## 指标定义

| 指标ID | 描述 | 公式 |
|--------|------|------|
| M1 | PDE 残差损失 | MSE(∇²u+10, 0) / MSE(rx,0)+MSE(ry,0) |
| M2 | 边界条件损失（Darcy BC） | MSE(u_pred·flag, u_gt·flag) |
| M3 | 加载边界损失（Plate load） | MSE(u_load,u_gt)+MSE(v_load,v_gt) |
| M4 | 固定边界损失（Plate fix） | mean((u_BCxy·flag)²+(v_BCxy·flag)²) |
| M5 | 自由边界损失（Plate free） | mean((σ_yy·flag)²+(σ_xy·flag)²) |
| M6 | 总加权训练损失 | 各子损失加权求和 |
| M7 | 验证集相对 L2 误差 | ‖pred−gt‖₂/‖gt‖₂ |
| M8 | 测试集相对 L2 误差 | ‖pred−gt‖₂/‖gt‖₂ |

---

## 可优化模块清单

### A · 数据预处理模块

| # | 可优化模块 | 可优化模块说明 | 对应代码库位置 | 对应场景 | 对应指标 |
|---|-----------|--------------|--------------|---------|---------|
| A1 | 坐标归一化 | 几何边界点坐标与训练时的 PDE 查询坐标均未做归一化，值域不统一导致梯度尺度差异影响收敛。两者在数据流中均来源于同一 `coors` 张量，在数据预处理阶段统一归一化后，训练循环中的采样坐标（`pde_sampled_coors = coors[:, ss_index, :]`）同步受益，无需在训练侧额外处理 | `utils_data.py` 无实现；`utils_darcy_train.py:234-235`；`utils_plate_train.py:305-306` | S1 S2 S3 S4 | M6 M7 M8 |
| A2 | 物理参数输入归一化 | 边界条件参数（坐标、位移值等）直接输入模型，无归一化。Plate 中 Young's 模量的缩放因子 `scalar_factor=1e-4` 硬编码，无法根据数据分布自适应；各边界参数量纲差异导致网络偏向大数值特征。引入基于数据统计量的自适应归一化可同时解决硬编码缩放与一般参数量纲问题 | `utils_data.py:117-118`；`utils_data.py` 无通用参数归一化实现 | S1 S2 S3 S4 | M2 M3 M4 M5 M6 M7 M8 |
| A3 | Padding 策略 | 无效节点位置填充为零值（坐标和解值均为0），模型可能将 padding 区域误学习为有效零解，引入偏置 | `utils_data.py:51,55`；`utils_data.py:178-201` | S1 S2 S3 S4 | M7 M8 |
| A4 | 数据集划分比例 | 训练/验证/测试固定为 70/10/20，硬编码无法调整，小数据集下验证集过小导致模型选择不稳定 | `utils_data.py:83-85`；`utils_data.py:242-244` | S1 S2 S3 S4 | M7 M8 |
| A5 | 数据增强 | 训练过程中完全没有数据增强。对于物理问题，坐标旋转、反射、尺度变换等几何等变增强均为合法变换，可显著扩充有效样本量 | `utils_data.py` 无实现 | S1 S2 S3 S4 | M6 M7 M8 |

---

### B · 几何编码模块（Domain Geometry Encoder）

| # | 可优化模块 | 可优化模块说明 | 对应代码库位置 | 对应场景 | 对应指标 |
|---|-----------|--------------|--------------|---------|---------|
| B1 | 几何点特征提取网络结构 | MLP 层数由 N_layer 控制，但宽度固定为 fc_dim，各层等宽，无残差连接。可引入逐渐收缩/扩张结构或残差块提升表达能力 | `model_darcy.py:160-165`；`model_plate.py:185-190` | S1 S2 S3 S4 | M7 M8 |
| B2 | 几何编码器最后一层激活 | 最后一层 Linear 无激活函数，输出无界，可能与下游融合层产生数值不稳定 | `model_darcy.py:164`；`model_plate.py:189` | S1 S2 S3 S4 | M7 M8 |
| B3 | 集合聚合方式 | 当前使用 masked 均值池化：`sum(enc*flag)/sum(flag)`。均值对几何局部结构（尖角、凹陷）不敏感，且对点密度不均有偏 | `model_darcy.py:181`；`model_plate.py:203` | S1 S2 S3 S4 | M7 M8 |
| B4 | 几何点间交互 | 各边界点独立经过 MLP 后直接聚合，点与点之间无任何交互（无注意力、无图结构）。相邻点的几何曲率信息无法被利用 | `model_darcy.py:179-183`；`model_plate.py:200-205` | S1 S2 S3 S4 | M7 M8 |
| B5 | 几何编码器正则化 | 无 Dropout、BatchNorm、LayerNorm 等任何正则化手段，小数据集下几何编码器易过拟合 | `model_darcy.py:154-183`；`model_plate.py:179-205` | S1 S2 S3 S4 | M7 M8 |
| B6 | 几何点位置编码 | 几何坐标直接输入，无 Fourier 特征或正弦位置编码。对于具有周期性边界的几何，位置编码可显著改善表达能力 | `model_darcy.py:160`；`model_plate.py:185` | S1 S2 S3 S4 | M7 M8 |

---

### C · 参数编码模块（Branch / Parameter Encoder）

| # | 可优化模块 | 可优化模块说明 | 对应代码库位置 | 对应场景 | 对应指标 |
|---|-----------|--------------|--------------|---------|---------|
| C1 | 参数特征提取网络结构 | 等宽 MLP，无残差连接，无正则化，与几何编码器面临相同结构性问题。参数编码器输出直接调制预测头，结构缺陷传导至所有下游指标 | `model_darcy.py:194-199`；`model_plate.py:216-221` | S1 S2 S3 S4 | M2 M3 M6 M7 M8 |
| C2 | 全局聚合方式 | 使用 `torch.amax(enc_masked, dim=1)` 最大池化。amax 逐维度独立取最大，不同维度的最大值来自不同参数点，破坏特征协同关系 | `model_darcy.py:238`；`model_plate.py:280` | S1 S2 S3 S4 | M2 M3 M6 M7 M8 |
| C3 | 参数点间交互 | 各参数点独立编码后直接池化，参数点之间无注意力或图交互。多参数点的联合分布信息丢失 | `model_darcy.py:236-238`；`model_plate.py:278-280` | S1 S2 S3 S4 | M2 M3 M6 M7 M8 |
| C4 | 参数编码器最后一层激活 | 最后一层 Linear 无激活，输出无界，与几何编码器 B2 同类问题，影响下游调制层数值稳定性 | `model_darcy.py:198`；`model_plate.py:220` | S1 S2 S3 S4 | M6 M7 M8 |
| C5 | 几何与参数信息交互 | DG（几何）和 Branch（参数）完全独立编码，两者之间无任何交叉注意力或信息交换，最终只在融合层拼接 | `model_darcy.py:224-238`；`model_plate.py:264-280` | S1 S2 S3 S4 | M7 M8 |

---

### D · 坐标编码模块（Coordinate / Trunk Encoder）

| # | 可优化模块 | 可优化模块说明 | 对应代码库位置 | 对应场景 | 对应指标 |
|---|-----------|--------------|--------------|---------|---------|
| D1 | 坐标编码表达能力 | `xy_lift` 为单层 `Linear(2→F)` 无激活，等价于线性变换，对高频空间变化（边界层、应力集中）拟合能力严重不足。可替换为多层 MLP 或 Random Fourier Features，后者已被证明对 PINN/PINO 类方法的高频解拟合有显著提升；两种方案均作用于同一模块，实施其一即可 | `model_darcy.py:202`；`model_plate.py:224-225` | S1 S2 S3 S4 | M1 M7 M8 |
| D2 | Plate u/v 坐标编码共享 | `xy_lift1` 和 `xy_lift2` 为独立权重，增加参数量。可实验共享坐标编码，仅在预测头分叉，降低小样本场景下的过拟合风险 | `model_plate.py:224-225` | S3 S4 | M7 M8 |
| D3 | 域全局信息注入方式 | 当前仅使用 Concatenation 将局部坐标特征和全局域特征拼接。项目已实现 Add、Mul 变体但仅作实验类，未系统评估，注入方式对特征融合效果有直接影响 | `model_darcy.py:233`；`model_plate.py:274-275` | S1 S2 S3 S4 | M7 M8 |

---

### E · 算子逼近模块（Operator Approximation / Decoder）

| # | 可优化模块 | 可优化模块说明 | 对应代码库位置 | 对应场景 | 对应指标 |
|---|-----------|--------------|--------------|---------|---------|
| E1 | Branch 调制机制 | 当前仅做逐元素乘法 `u = u * enc`，为 FiLM 的退化版（只有缩放无偏置）。完整 FiLM 为 `γ*u + β`，表达能力更强 | `model_darcy.py:242,244`；`model_plate.py:284,287` | S1 S2 S3 S4 | M7 M8 |
| E2 | 调制层数不一致 | Plate 模型第3层调制 `u = u * enc` 被注释掉，Darcy 保留。各层是否调制对最终精度有影响，当前为未收敛的设计决策 | `model_plate.py:291,305` | S3 S4 | M7 M8 |
| E3 | 残差连接缺失 | 预测头 FC1~FC4 之间无任何残差/跳层连接，深层网络梯度回传困难 | `model_darcy.py:241-248`；`model_plate.py:283-308` | S1 S2 S3 S4 | M7 M8 |
| E4 | 输出层设计缺陷 | 最终输出用 `torch.mean(u*enc, dim=-1)` 在特征维取均值，本质是与 enc 的内积聚合。FC5u/FC5v 已定义但从未被调用，参数完全浪费，已有模块未被充分利用 | `model_darcy.py:248`；`model_plate.py:232,240,294,308` | S1 S2 S3 S4 | M7 M8 |
| E5 | 全网络激活函数 | 几何编码器 MLP 中间层（`model_darcy.py:161,163`）与预测头 FC 层（`model_darcy.py:207`）统一使用同一 Tanh，深层 Tanh 易饱和导致梯度消失，对非平滑解的拟合能力弱于 GELU/SiLU；替换时需覆盖全网络两处组件 | `model_darcy.py:161,163,207`；`model_plate.py:186,188,233` | S1 S2 S3 S4 | M7 M8 |
| E6 | 预测头正则化缺失 | 预测头 FC1~FC4 无 Dropout、LayerNorm、BatchNorm，在小样本场景下泛化能力受限 | `model_darcy.py:185-248`；`model_plate.py:207-308` | S1 S2 S3 S4 | M7 M8 |
| E7 | 网络深度硬编码 | N_layer 仅控制编码器层数，预测头层数（Darcy 3层、Plate 4层）硬编码，无法通过配置文件统一调整 | `model_darcy.py:203-206`；`model_plate.py:228-240`；`configs/*.yaml:4` | S1 S2 S3 S4 | M7 M8 |
| E8 | Plate u/v 预测头完全解耦 | u 和 v 各有独立的 FC1~FC5，参数量翻倍。u/v 物理上通过应力张量强耦合，完全解耦可能丢失物理关联 | `model_plate.py:228-240,283-308` | S3 S4 | M7 M8 |

---

### F · 物理约束模块（Physics-Informed Loss）

| # | 可优化模块 | 可优化模块说明 | 对应代码库位置 | 对应场景 | 对应指标 |
|---|-----------|--------------|--------------|---------|---------|
| F1 | 剪切应变定义错误 | `eps_xy = u_y + v_x`，缺少 ÷2。连续介质力学中张量剪切应变定义为 `(∂u/∂y + ∂v/∂x)/2`，当前定义为工程剪切应变，与应力计算中的 `G` 因子不一致 | `utils_losses.py:32` | S3 | M1 M5 M6 M7 M8 |
| F2 | 自由边界条件不完整 | `bc_edgeY_loss` 只约束 y 方向边界上的 σ_yy=0, σ_xy=0，缺少对应 x 方向自由边界（σ_xx=0, σ_xy=0）的约束函数 | `utils_losses.py:47-64` | S3 | M5 M6 M7 M8 |
| F3 | PDE 源项硬编码 | Darcy 方程源项 `+10` 硬编码在损失函数内，不同物理问题需要直接修改源文件 | `utils_losses.py:15` | S1 | M1 M6 M7 M8 |
| F4 | 损失权重失衡与固定 | 当前 `weight_bc=1000, weight_pde=1`，量级差3个数量级，根本原因是各子损失残差未归一化；权重训练全程固定，无自适应调整机制（NTK 权重、Residual-based 自适应等）。修复需同时解决归一化与自适应两个层面 | `utils_darcy_train.py:203-204`；`utils_plate_train.py:264-267`；`configs/GANO_Darcy_DG.yaml:13-14`；`configs/GANO_plate_stress_DG.yaml:13-16` | S1 S3 | M1 M2 M3 M4 M5 M6 |
| F5 | 梯度裁剪缺失 | 二阶自动微分链路（计算 PDE 残差）在训练初期易产生爆炸梯度，但全程无 `clip_grad_norm` | `utils_darcy_train.py:279-281`；`utils_plate_train.py:376-378` | S1 S3 | M1 M6 |

---

### G · 采样与优化模块（Sampling & Optimization）

| # | 可优化模块 | 可优化模块说明 | 对应代码库位置 | 对应场景 | 对应指标 |
|---|-----------|--------------|--------------|---------|---------|
| G1 | PDE 点自适应采样 | 当前使用均匀随机采样 `np.random.choice(max_pde_nodes, sampling_size)`，高残差区域（边界附近、奇异点）与内部低残差区域采样概率相同。可引入 RAR、RAD 等基于残差大小的重要性采样策略，使采样分布向高残差区域倾斜 | `utils_darcy_train.py:233-237`；`utils_plate_train.py:304-308` | S1 S3 | M1 M6 M7 M8 |
| G2 | 学习率调度 | 固定学习率 1e-4，训练全程无任何调度策略（无 CosineAnnealing、ReduceLROnPlateau、warmup） | `utils_darcy_train.py:187`；`utils_plate_train.py:248`；`configs/GANO_Darcy_DG.yaml:11` | S1 S2 S3 S4 | M6 M7 M8 |
| G3 | 优化器配置 | Adam 使用默认参数，`weight_decay=0`，无权重衰减正则化；β1、β2 未调整 | `utils_darcy_train.py:187`；`utils_plate_train.py:248` | S1 S2 S3 S4 | M6 M7 M8 |
| G4 | Early stopping 缺失 | 固定训练 200 epochs，无 early stopping。验证误差不再下降时仍继续训练，可能过拟合 | `configs/GANO_Darcy_DG.yaml:7`；`utils_darcy_train.py:211` | S1 S2 S3 S4 | M7 M8 |
| G5 | geo_node 条件判断 bug | `if args.geo_node == 'vary_bound' or 'vary_bound_sup':` 因非空字符串恒为 True，'vary_bound' 与 'vary_bound_sup' 两个条件分支被错误合并，导致几何输入提取逻辑在特定参数组合下出错 | `utils_darcy_train.py:253-258`；`utils_plate_train.py:331-339` | S1 S2 S3 S4 | M7 M8 |
| G6 | 课程学习缺失 | 无从简单到复杂几何的课程式训练策略，所有几何复杂度的样本从训练开始即混合训练 | `utils_darcy_train.py:228-281`；`utils_plate_train.py:299-378` | S1 S3 | M1 M6 M7 M8 |

---

## 模块 × 场景 × 指标 索引表

| 模块编号 | 模块名称 | S1 Darcy-PINO | S2 Darcy-NO | S3 Plate-PINO | S4 Plate-NO | 主要影响指标 |
|---------|---------|:---:|:---:|:---:|:---:|------------|
| A1-A5 | 数据预处理 | ✅ | ✅ | ✅ | ✅ | M2 M3 M4 M5 M6 M7 M8 |
| B1-B6 | 几何编码 | ✅ | ✅ | ✅ | ✅ | M7 M8 |
| C1-C5 | 参数编码 | ✅ | ✅ | ✅ | ✅ | M2 M3 M6 M7 M8 |
| D1-D3 | 坐标编码 | ✅ | ✅ | ✅ | ✅ | M1 M7 M8 |
| E1-E8 | 算子逼近 | ✅ | ✅ | ✅ | ✅ | M7 M8 |
| F1-F5 | 物理约束 | ✅ | ❌ | ✅ | ❌ | M1 M2 M3 M4 M5 M6 |
| G1-G6 | 采样与优化 | ✅ | △ | ✅ | △ | M1 M6 M7 M8 |

> △ = 有监督训练无PDE采样，但优化器和训练策略仍可改进

---

## 优先级评估

| 优先级 | 模块编号 | 理由 |
|--------|---------|------|
| P0 · 正确性修复 | F1 G5 | 剪切应变错误影响Plate物理约束正确性；geo_node条件bug导致分支逻辑错误 |
| P1 · 高收益优化 | F4 G2 D1 C2 | 损失权重归一化与自适应、学习率调度、坐标编码增强、聚合方式改进，成本低收益高 |
| P2 · 中等收益 | B3 B4 E1 E3 E4 A5 | 几何聚合改进、调制机制完善、残差连接、数据增强 |
| P3 · 结构性重构 | E7 E8 F2 G1 G6 | 网络结构配置化、u/v耦合设计、完整自由边界约束、自适应采样、课程学习 |
