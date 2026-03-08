"""
utils_plate_train_optimizable.py

本文件集中存放 utils_plate_train.py 中所有可优化模块的当前实现。
每个函数封装一个独立的优化项（G1-G6），保持与原代码等价的行为，
以便后续直接替换函数实现来改进对应指标，而无需修改调用方代码。

优化项与 utils_darcy_train_optimizable.py 中各项同源，差异仅在于
Plate 问题使用四元组节点列表 (max_pde_nodes, max_bcxy_nodes, max_bcy_nodes, max_par_nodes)。

优化项对应关系：
  G1 - sample_pde_indices      : PDE 点采样策略（当前为均匀随机采样）
  G2 - create_lr_scheduler     : 学习率调度器（当前无调度，固定学习率）
  G3 - create_optimizer        : 优化器配置（当前 Adam + weight_decay=0）
  G4 - check_early_stopping    : Early stopping 判断（当前始终返回 False）
  G5 - get_geo_node_index      : 几何节点索引提取（当前含条件判断 bug）
  G6 - get_curriculum_loader   : 课程学习数据加载（当前直接返回原始 loader）
"""

from typing import List, Optional, Any, Tuple
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


# ─────────────────────────── G1 ───────────────────────────
def sample_pde_indices(
    max_pde_nodes: int,
    sampling_size: int,
    residuals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """[G1] 从 Plate 问题 PDE 内部节点中采样用于残差计算的点的索引。

    当前使用均匀随机采样，高残差区域（孔洞边界附近、应力集中点）
    与内部低残差区域采样概率相同，导致 PDE 残差在关键区域收敛缓慢。

    优化方向：引入基于 PDE 残差 (rx, ry) 大小的重要性采样策略（RAR / RAD），
              使高残差节点被更高频率地采样，集中计算资源于困难区域。

    Args:
        max_pde_nodes (int): PDE 内部节点总数（填充后的上限值）。
        sampling_size (int): 每次采样的点数（由 config['train']['coor_sampling_size'] 决定）。
        residuals (np.ndarray | None): 各节点当前 PDE 残差范数，形状 (max_pde_nodes,)。
                                        当前实现忽略此参数，采用均匀采样。

    Returns:
        np.ndarray: 采样得到的节点索引数组，形状 (sampling_size,)，
                    元素范围 [0, max_pde_nodes)。
    """
    return np.random.choice(np.arange(max_pde_nodes), sampling_size)


# ─────────────────────────── G2 ───────────────────────────
def create_lr_scheduler(
    optimizer: optim.Optimizer,
    config: dict,
) -> Optional[Any]:
    """[G2] 为 Plate 训练优化器创建学习率调度器。

    当前训练全程使用固定学习率（1e-4），无任何衰减或预热策略，
    导致训练后期学习率偏大，难以精细收敛到低损失区域。

    优化方向：引入 CosineAnnealingLR、ReduceLROnPlateau 或 warmup 策略，
              使学习率随训练进行自适应调整。

    Args:
        optimizer (optim.Optimizer): 需要调度的优化器实例。
        config    (dict): 训练配置字典，包含 epochs、base_lr 等键值。

    Returns:
        Optional[Any]: 学习率调度器实例，或 None（当前实现返回 None，即不调度）。
    """
    return None


# ─────────────────────────── G3 ───────────────────────────
def create_optimizer(
    model: nn.Module,
    config: dict,
) -> optim.Optimizer:
    """[G3] 创建并配置 Plate 模型训练优化器。

    当前使用 Adam 优化器，weight_decay=0（无权重衰减正则化），
    β1、β2 均为 PyTorch 默认值（0.9, 0.999）。Plate 问题
    涉及 u/v 双输出以及多类边界条件损失，优化器配置的合理性
    对训练稳定性影响更显著。

    优化方向：加入 weight_decay；可尝试为不同参数组（编码器 vs 预测头）
              设置不同学习率（per-parameter learning rate）。

    Args:
        model  (nn.Module): 需要优化的模型，提供 model.parameters()。
        config (dict): 训练配置字典，需包含 config['train']['base_lr']。

    Returns:
        optim.Optimizer: 配置好的优化器实例。
                         当前实现返回 Adam(lr=base_lr, weight_decay=0)。
    """
    return optim.Adam(model.parameters(), lr=config['train']['base_lr'])


# ─────────────────────────── G4 ───────────────────────────
def check_early_stopping(
    val_errors: List[float],
    patience: int = 20,
) -> bool:
    """[G4] 根据验证误差历史判断 Plate 训练是否触发 Early stopping。

    当前训练循环固定运行 config['train']['epochs'] 个 epoch，
    即使验证误差已停止下降也不会提前终止，浪费计算资源并可能过拟合。
    Plate 问题训练成本更高（双输出 + 多类损失），早停收益更大。

    优化方向：当最近 patience 个 epoch 的验证误差均不优于历史最优时，
              返回 True 以触发提前停止。

    Args:
        val_errors (List[float]): 历史验证相对 L2 误差列表（按 epoch 顺序）。
        patience   (int): 容忍的无改善 epoch 数，默认 20。

    Returns:
        bool: True 表示应停止训练，False 表示继续训练。
              当前实现始终返回 False（不触发早停）。
    """
    return False


# ─────────────────────────── G5 ───────────────────────────
def get_geo_node_index(
    geo_node: str,
    max_pde_nodes: int,
    max_par_nodes: int,
    max_bcy_nodes: int,
    max_bcxy_nodes: int,
) -> np.ndarray:
    """[G5] 根据 geo_node 参数确定 Plate 几何编码器所使用的节点索引。

    原始代码中条件判断存在 bug：
        if args.geo_node == 'vary_bound' or 'vary_bound_sup':
    由于 'vary_bound_sup' 是非空字符串，Python 将其视为 True，
    导致该 if 分支始终执行。Plate 问题节点顺序为
    [pde | par | bcy | bcxy]，'vary_bound'/'vary_bound_sup' 应
    仅取 bcxy 节点（固定边界节点），但 bug 使所有模式均走此分支。

    当前实现：保留 bug 行为（始终返回 bcxy 节点索引），以维持向后兼容。
    优化方向：修正条件判断，分别正确处理三种 geo_node 取值。

    Args:
        geo_node       (str): 几何节点模式，取值为 'vary_bound'、
                               'vary_bound_sup'、'all_bound' 或 'all_domain'。
        max_pde_nodes  (int): PDE 内部节点数上限。
        max_par_nodes  (int): 加载边界节点数上限。
        max_bcy_nodes  (int): 自由边界节点数上限。
        max_bcxy_nodes (int): 固定边界节点数上限。

    Returns:
        np.ndarray: 几何节点在 coors 张量中的索引，形状 (n_geo_nodes,)。
                    当前实现（bug 行为）：始终返回固定边界节点（bcxy）索引，
                    范围 [max_pde_nodes+max_par_nodes+max_bcy_nodes,
                           max_pde_nodes+max_par_nodes+max_bcy_nodes+max_bcxy_nodes)。
    """
    bcxy_start = max_pde_nodes + max_par_nodes + max_bcy_nodes
    bcxy_end = bcxy_start + max_bcxy_nodes

    # BUG: 'vary_bound_sup' 作为独立表达式恒为 True，此分支始终被执行
    if geo_node == 'vary_bound' or 'vary_bound_sup':
        return np.arange(bcxy_start, bcxy_end)
    if geo_node == 'all_bound':
        return np.arange(max_pde_nodes, bcxy_end)
    if geo_node == 'all_domain':
        return np.arange(0, bcxy_end)
    return np.arange(bcxy_start, bcxy_end)


# ─────────────────────────── G6 ───────────────────────────
def get_curriculum_loader(
    epoch: int,
    train_loader: Any,
    config: dict,
) -> Any:
    """[G6] 根据当前训练阶段（epoch）返回适合课程学习的 Plate 数据加载器。

    当前训练从第 0 epoch 起即混合所有几何复杂度的样本（孔洞数量、
    孔洞形状、孔洞位置等变化范围覆盖全集），网络在初期需同时应对
    简单和复杂几何，收敛较慢。

    优化方向：按几何复杂度对样本排序（例如按孔洞数量或边界节点密度），
              训练初期仅使用简单子集，随训练进度逐步引入复杂样本。

    Args:
        epoch        (int): 当前 epoch 编号（从 0 开始）。
        train_loader (Any): 原始训练数据加载器（DataLoader 实例）。
        config       (dict): 训练配置字典，包含 epochs 等信息。

    Returns:
        Any: 当前 epoch 应使用的数据加载器。
             当前实现直接返回原始 train_loader（无课程学习）。
    """
    return train_loader
