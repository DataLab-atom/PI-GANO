"""
utils_darcy_train_optimizable.py

本文件集中存放 utils_darcy_train.py 中所有可优化模块的当前实现。
每个函数封装一个独立的优化项（G1-G6），保持与原代码等价的行为，
以便后续直接替换函数实现来改进对应指标，而无需修改调用方代码。

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
    """[G1] 从 PDE 内部节点中采样用于残差计算的点的索引。

    当前使用均匀随机采样，高残差区域（边界层、应力集中）与内部低残差区域
    采样概率相同，导致模型在难点区域训练不足。

    优化方向：引入基于残差大小的重要性采样（RAR / RAD），
              使 high-residual 区域被更频繁地采样，加速收敛并降低 PDE 损失。

    Args:
        max_pde_nodes (int): PDE 内部节点总数（填充后的上限值）。
        sampling_size (int): 每次采样的点数（由 config['train']['coor_sampling_size'] 决定）。
        residuals (np.ndarray | None): 各节点当前 PDE 残差绝对值，形状 (max_pde_nodes,)。
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
    """[G2] 为训练优化器创建学习率调度器。

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
    """[G3] 创建并配置模型训练优化器。

    当前使用 Adam 优化器，weight_decay=0（无权重衰减正则化），
    β1、β2 均为 PyTorch 默认值（0.9, 0.999），未针对 PINO 任务调优。
    缺乏权重衰减使得模型在小样本场景下更易过拟合。

    优化方向：加入 weight_decay（1e-4 ~ 1e-5 量级）以增强隐式正则化；
              可尝试 AdamW 或 Lion 优化器；同时调整 β1, β2 超参数。

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
    """[G4] 根据验证误差历史判断是否触发 Early stopping。

    当前训练循环固定运行 config['train']['epochs'] 个 epoch，
    即使验证误差已停止下降也不会提前终止，浪费计算资源并可能过拟合。

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
    max_bc_nodes: int,
) -> np.ndarray:
    """[G5] 根据 geo_node 参数确定几何编码器所使用的节点索引。

    原始代码中条件判断存在 bug：
        if args.geo_node == 'vary_bound' or 'vary_bound_sup':
    由于 'vary_bound_sup' 是非空字符串，Python 将其视为 True，
    导致该 if 分支始终执行，'vary_bound' 与 'vary_bound_sup' 的节点选取逻辑
    被错误地合并，'all_domain' 分支也可能被覆盖（变量被上方覆盖）。

    当前实现：保留 bug 行为（等同于始终走 vary_bound 分支），以维持向后兼容。
    优化方向：修正为正确的逻辑，分别处理三种 geo_node 取值。

    Args:
        geo_node      (str): 几何节点模式，取值为 'vary_bound'、
                              'vary_bound_sup' 或 'all_domain'。
        max_pde_nodes (int): PDE 内部节点数上限（填充后）。
        max_bc_nodes  (int): 边界节点数上限（填充后）。

    Returns:
        np.ndarray: 几何节点在 coors 张量中的索引，形状 (n_geo_nodes,)。
                    当前实现（bug 行为）：始终返回边界节点索引
                    [max_pde_nodes, max_pde_nodes + max_bc_nodes)。
    """
    # BUG: 'vary_bound_sup' 作为独立表达式恒为 True，此分支始终被执行
    if geo_node == 'vary_bound' or 'vary_bound_sup':
        return np.arange(max_pde_nodes, max_pde_nodes + max_bc_nodes)
    if geo_node == 'all_domain':
        return np.arange(0, max_pde_nodes + max_bc_nodes)
    return np.arange(max_pde_nodes, max_pde_nodes + max_bc_nodes)


# ─────────────────────────── G6 ───────────────────────────
def get_curriculum_loader(
    epoch: int,
    train_loader: Any,
    config: dict,
) -> Any:
    """[G6] 根据当前训练阶段（epoch）返回适合课程学习的数据加载器。

    当前训练从第 0 epoch 起即混合所有几何复杂度的样本，
    网络在初期需同时应对简单和复杂几何，收敛速度较慢。

    优化方向：按几何复杂度（边界节点数、形状复杂度等）对样本排序，
              训练初期仅使用简单样本子集，随 epoch 增加逐步引入复杂样本，
              实现从易到难的课程式训练。

    Args:
        epoch        (int): 当前 epoch 编号（从 0 开始）。
        train_loader (Any): 原始训练数据加载器（DataLoader 实例）。
        config       (dict): 训练配置字典，包含 epochs 等信息。

    Returns:
        Any: 当前 epoch 应使用的数据加载器。
             当前实现直接返回原始 train_loader（无课程学习）。
    """
    return train_loader
