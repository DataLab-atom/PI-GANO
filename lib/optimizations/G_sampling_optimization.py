"""
G_sampling_optimization.py — 采样与优化可优化模块 (G1-G9)

对应 optimizable_modules.md § G 节，共 9 个优化点。
本模块提供替换 utils_darcy_train.py / utils_plate_train.py 中
训练循环、采样逻辑、优化器配置和辅助工具的独立函数。

当前相关代码位置：
    PDE 采样:      utils_darcy_train.py:233, utils_plate_train.py:304
    学习率/优化器: utils_darcy_train.py:187, utils_plate_train.py:248
    Early stop:    无（固定 200 epochs）
    Geo node 判断: utils_darcy_train.py:253-258, utils_plate_train.py:331-339
    训练历史保存:  utils_darcy_train.py:193（声明但从未保存）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud


# ---------------------------------------------------------------------------
# G1 — PDE 点采样策略
# ---------------------------------------------------------------------------

def sample_pde_indices(
    max_pde_nodes: int,
    sampling_size: int,
    method: Literal["uniform", "stratified", "residual_weighted", "boundary_biased"] = "uniform",
    residuals: Optional[np.ndarray] = None,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """从 PDE 域节点中采样指定数量的点，支持多种非均匀采样策略。

    当前代码（utils_darcy_train.py:233, utils_plate_train.py:304）使用：
        ss_index = np.random.choice(np.arange(max_pde_nodes), sampling_size)
    均匀随机采样不考虑残差分布，高残差区域（边界附近、奇异点、应力集中）
    与低残差内部区域采样概率相同，训练初期大量计算浪费在已收敛区域。

    Args:
        max_pde_nodes (int): PDE 域节点总数（padding 前的最大数量）。
        sampling_size (int): 每次采样的节点数，与 config['train']['coor_sampling_size'] 对应。
        method (str): 采样策略：
            - "uniform"          : 均匀随机采样（原始行为）。
            - "stratified"       : 分层采样，将 [0, max_pde_nodes) 均分为
                                   sampling_size 个区间，每区间随机取一个点；
                                   保证覆盖均匀性，同时带随机性。
            - "residual_weighted": 按节点残差大小做重要性采样，高残差节点更易被选中；
                                   需提供 residuals 数组。
            - "boundary_biased"  : 对靠近边界的节点赋予更高采样权重（基于坐标估计）；
                                   适用于未掌握残差信息的训练初期。
        residuals (np.ndarray | None): 各 PDE 节点的当前残差值，形状 (max_pde_nodes,)；
            仅 method="residual_weighted" 时需要。
        temperature (float): 残差重要性采样的温度参数，越大越接近均匀采样；
            temperature=1.0 为标准重要性采样，>1 使分布更平坦。
        rng (np.random.Generator | None): 随机数生成器，None 时使用全局随机状态。
            传入固定 rng 可使单次测试可复现。

    Returns:
        ss_index (np.ndarray): 采样到的节点索引，形状 (sampling_size,)，
            值域 [0, max_pde_nodes)，无重复（除非 sampling_size > max_pde_nodes）。

    Raises:
        ValueError: 若 method="residual_weighted" 但 residuals 为 None。
        ValueError: 若 sampling_size > max_pde_nodes 且不允许重复。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> ss_index = sample_pde_indices(max_pde_nodes, config['train']['coor_sampling_size'],
        ...                               method="uniform")
        >>> # 替换 utils_darcy_train.py:233 中的 np.random.choice(...)
    """
    raise NotImplementedError(
        "G1: sample_pde_indices 尚未实现。"
        "'uniform' 即原始代码：np.random.choice(max_pde_nodes, sampling_size)，一行实现。"
        "'residual_weighted' 建议：probs = softmax(residuals / temperature)，"
        "再 np.random.choice(..., p=probs, replace=False)。"
    )


# ---------------------------------------------------------------------------
# G2 — 自适应采样（RAR / RAD）
# ---------------------------------------------------------------------------

def adaptive_residual_sampling(
    model: nn.Module,
    coors_all: torch.Tensor,
    par: torch.Tensor,
    par_flag: torch.Tensor,
    shape_coor: torch.Tensor,
    shape_flag: torch.Tensor,
    params: tuple,
    max_pde_nodes: int,
    sampling_size: int,
    method: Literal["RAR", "RAD", "uniform"] = "RAD",
    update_every: int = 100,
    current_step: int = 0,
    cached_indices: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """基于当前模型残差做自适应重要性采样，减少训练初期的无效计算。

    当前代码（utils_darcy_train.py:233-237, utils_plate_train.py:304-308）
    无任何自适应采样机制，均匀随机采样在训练早期大量计算浪费在残差已很小的区域。

    Residual-based Adaptive Refinement (RAR) 和
    Residual-based Adaptive Distribution (RAD) 是 PINN 领域常用的自适应采样方法：
        - RAR: 直接在残差最大的区域添加更多采样点（累积）。
        - RAD: 按残差分布做重要性采样，每轮动态更新采样概率（分布式）。

    Args:
        model (nn.Module): 当前模型（用于计算节点残差）。
        coors_all (torch.Tensor): 所有 PDE 节点坐标，形状 (B, max_pde_nodes, 2)。
        par (torch.Tensor): 参数张量，形状 (B, M', D)。
        par_flag (torch.Tensor): 参数有效性掩码，形状 (B, M')。
        shape_coor (torch.Tensor): 几何编码坐标，形状 (B, Mgeo, 2)。
        shape_flag (torch.Tensor): 几何有效性掩码，形状 (B, Mgeo)。
        params (tuple): 物理材料参数 (E, ν)（Plate 场景）或空 tuple（Darcy）。
        max_pde_nodes (int): PDE 域节点总数。
        sampling_size (int): 目标采样数量。
        method (str): 自适应采样方法：
            - "uniform" : 退化为均匀采样（基线，无自适应）。
            - "RAR"     : 残差自适应细化，选取残差最大的前 k 个点。
            - "RAD"     : 残差自适应分布，以残差大小为权重做重要性采样。
        update_every (int): 每隔多少步更新一次采样分布（计算残差代价较高）；
            0 表示每步都更新（精确但慢）。
        current_step (int): 当前训练步数，用于判断是否需要更新。
        cached_indices (np.ndarray | None): 上次计算的采样索引缓存；
            若当前步不需要更新，直接返回缓存以节省计算。
        device (torch.device | None): 计算设备，None 时从 model 推断。

    Returns:
        ss_index (np.ndarray): 自适应采样的节点索引，形状 (sampling_size,)。

    Raises:
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> ss_index = adaptive_residual_sampling(
        ...     model, coors_all, par, par_flag, shape_coor, shape_flag, params,
        ...     max_pde_nodes, config['train']['coor_sampling_size'],
        ...     method="RAD", update_every=50, current_step=step
        ... )
        >>> # 替换 utils_darcy_train.py:233 中的 np.random.choice(...)
    """
    raise NotImplementedError(
        "G2: adaptive_residual_sampling 尚未实现。"
        "实现前需为 PI_GANO 实现 compute_residuals(coors) 接口，"
        "用于在 no_grad 模式下快速评估所有 PDE 节点的残差大小。"
        "update_every 控制频率：每 100 步更新一次残差计算代价约为正常 forward 的 max_pde_nodes/sampling_size 倍。"
    )


# ---------------------------------------------------------------------------
# G3 — 学习率调度
# ---------------------------------------------------------------------------

def build_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: Literal[
        "none", "cosine", "cosine_warmup", "reduce_on_plateau",
        "step", "exponential", "cyclic"
    ] = "none",
    total_epochs: int = 200,
    warmup_epochs: int = 10,
    min_lr: float = 1e-6,
    step_size: int = 50,
    gamma: float = 0.5,
    patience: int = 10,
    factor: float = 0.5,
) -> Optional[object]:
    """构建学习率调度器，替换全程固定学习率 1e-4 的设置。

    当前代码（utils_darcy_train.py:187, utils_plate_train.py:248）使用
    固定学习率，训练全程无任何调度。物理信息神经网络的损失曲面通常具有
    多个局部最优，较大的初始学习率有助于跳出局部最优，训练后期需要更小的
    学习率以精细收敛，因此学习率调度对 PINN 尤为重要。

    Args:
        optimizer (optim.Optimizer): 待包装的优化器实例。
        scheduler_type (str): 调度策略：
            - "none"               : 无调度（原始行为，固定 LR）。
            - "cosine"             : 余弦退火，从 base_lr 衰减到 min_lr。
            - "cosine_warmup"      : 先线性预热 warmup_epochs 步，再余弦退火。
            - "reduce_on_plateau"  : 验证误差不改善 patience 个 epoch 后 LR × factor。
            - "step"               : 每 step_size 个 epoch 将 LR × gamma。
            - "exponential"        : 每 epoch LR × gamma（指数衰减）。
            - "cyclic"             : 循环学习率（适合短周期快速探索）。
        total_epochs (int): 总训练 epoch 数（cosine 类调度需要）。
        warmup_epochs (int): 预热 epoch 数（cosine_warmup 需要）。
        min_lr (float): 最小学习率下界（cosine 类调度需要）。
        step_size (int): StepLR 的步长（scheduler_type="step" 时有效）。
        gamma (float): LR 乘法因子（"step"/"exponential" 时有效）。
        patience (int): ReduceLROnPlateau 的观察窗口（epoch 数）。
        factor (float): ReduceLROnPlateau 的衰减因子（< 1.0）。

    Returns:
        scheduler: PyTorch lr_scheduler 实例，可直接在训练循环中调用
            scheduler.step() 或 scheduler.step(val_metric)；
            若 scheduler_type="none"，返回 None。

    Raises:
        ValueError: 若 scheduler_type 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> optimizer = build_optimizer(model, config)
        >>> scheduler = build_lr_scheduler(optimizer, scheduler_type="cosine_warmup",
        ...                                total_epochs=config['train']['epochs'])
        >>> # 在训练循环每个 epoch 结束时：
        >>> if scheduler is not None:
        ...     scheduler.step()  # 或 scheduler.step(val_err) for ReduceLROnPlateau
        >>> # 替换 utils_darcy_train.py:187 中的固定 lr Adam
    """
    raise NotImplementedError(
        "G3: build_lr_scheduler 尚未实现。"
        "'cosine_warmup' 可用 torch.optim.lr_scheduler.CosineAnnealingLR 实现退火部分，"
        "预热使用 LinearLR，通过 SequentialLR 拼接：SequentialLR([warmup_sched, cosine_sched], milestones=[warmup_epochs])。"
        "建议将 scheduler_type 和相关参数暴露到 config['train'] 中。"
    )


# ---------------------------------------------------------------------------
# G4 — 优化器配置
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    optimizer_type: Literal["adam", "adamw", "sgd", "rmsprop", "lbfgs"] = "adam",
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    param_groups: Optional[List[Dict]] = None,
) -> optim.Optimizer:
    """构建优化器，暴露 weight_decay 和 β 参数配置。

    当前代码（utils_darcy_train.py:187, utils_plate_train.py:248）使用：
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
    weight_decay=0（无权重衰减正则化），β1、β2 使用 PyTorch 默认值，
    未经调优。weight_decay > 0 是防止过拟合的有效手段，在小数据集上尤为重要。

    Args:
        model (nn.Module): 模型实例，其参数将被优化。
        optimizer_type (str): 优化器类型：
            - "adam"    : torch.optim.Adam（原始行为）。
            - "adamw"   : torch.optim.AdamW，Adam + 解耦 weight decay，推荐替代。
            - "sgd"     : SGD + momentum，可配合 cyclic LR 使用。
            - "rmsprop" : RMSprop，适合非平稳目标函数。
            - "lbfgs"   : L-BFGS，二阶方法，适合小数据精细微调，
                          内存占用大且每步需多次 forward（闭包函数调用）。
        lr (float): 基础学习率，对应 config['train']['base_lr']。
        weight_decay (float): L2 正则化系数，0.0 = 无正则化（原始行为）；
            建议范围 [1e-5, 1e-3]，AdamW 中含义与 Adam 略有不同（解耦）。
        betas (tuple): Adam/AdamW 的动量系数 (β1, β2)，
            默认 (0.9, 0.999)（PyTorch 默认值）。
        eps (float): Adam/AdamW 的数值稳定性小量，默认 1e-8。
        momentum (float): SGD 的动量系数（optimizer_type="sgd" 时有效）。
        param_groups (list[dict] | None): 自定义参数组，允许对不同层设置不同 LR；
            None 时所有参数使用相同配置（原始行为）。

    Returns:
        optimizer (optim.Optimizer): 配置好的优化器实例。

    Raises:
        ValueError: 若 optimizer_type 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> optimizer = build_optimizer(model, optimizer_type="adamw",
        ...                             lr=config['train']['base_lr'],
        ...                             weight_decay=1e-4)
        >>> # 替换 utils_darcy_train.py:187 中的 optim.Adam(...)
    """
    raise NotImplementedError(
        "G4: build_optimizer 尚未实现。"
        "AdamW 被证明在大多数任务中优于 Adam（weight decay 解耦更合理），"
        "建议将其作为默认优化器并添加 weight_decay: 1e-4 到 yaml 配置。"
    )


# ---------------------------------------------------------------------------
# G5 — Early Stopping
# ---------------------------------------------------------------------------

def check_early_stopping(
    val_errors: List[float],
    patience: int = 20,
    min_delta: float = 1e-5,
    mode: Literal["min", "max"] = "min",
    restore_best: bool = True,
) -> Tuple[bool, int]:
    """检查是否满足 early stopping 条件，避免在验证误差平台期浪费计算资源。

    当前代码（configs/GANO_Darcy_DG.yaml:7, utils_darcy_train.py:211）固定训练
    200 epochs，无 early stopping。当验证误差在 epoch 50 就已收敛时，
    后续 150 个 epoch 完全浪费算力，甚至因过拟合导致最终模型质量下降。

    Args:
        val_errors (list[float]): 历史验证误差列表，最后一个元素为当前 epoch 的误差。
            建议在训练循环中维护此列表，每个 visual_freq 间隔追加一次。
        patience (int): 允许验证误差无改善的最大观察轮数；
            即最近 patience 次验证均未比历史最优改善 min_delta，则停止。
        min_delta (float): 视为"改善"的最小变化量，防止数值噪声触发停止。
        mode (str): 优化方向：
            - "min" : 验证误差越小越好（回归/L2 误差，适用于本项目）。
            - "max" : 验证指标越大越好（准确率等场景）。
        restore_best (bool): 若为 True，停止时应从外部恢复最优模型权重；
            本函数仅返回建议，恢复操作由调用方负责。

    Returns:
        should_stop (bool): True 表示满足 early stopping 条件，建议终止训练。
        best_epoch (int): 历史最优验证误差对应的 epoch 索引（在 val_errors 中的下标）。

    Raises:
        ValueError: 若 patience < 1 或 val_errors 为空。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> should_stop, best_epoch = check_early_stopping(err_hist, patience=20)
        >>> if should_stop:
        ...     print(f"Early stopping at epoch {e}, best at val_errors[{best_epoch}]")
        ...     break
        >>> # 在 utils_darcy_train.py 的 epoch 循环中，每次 val() 后调用
    """
    raise NotImplementedError(
        "G5: check_early_stopping 尚未实现。"
        "实现逻辑：best_val = min(val_errors)；best_epoch = argmin(val_errors)；"
        "若 val_errors[-1] > best_val - min_delta 且 len(val_errors) - best_epoch > patience，"
        "则 should_stop = True。"
        "建议将 patience 添加到 config['train']['early_stop_patience'] 中。"
    )


# ---------------------------------------------------------------------------
# G6 — geo_node 条件判断 bug 修复
# ---------------------------------------------------------------------------

def select_geo_node_indices(
    geo_node_mode: str,
    max_pde_nodes: int,
    max_bc_nodes: int,
    max_par_nodes: int = 0,
    max_bcy_nodes: int = 0,
    max_bcxy_nodes: int = 0,
    problem_type: Literal["darcy", "plate"] = "darcy",
) -> np.ndarray:
    """根据 geo_node 模式正确选取几何描述坐标的节点索引，修复原始代码的逻辑 bug。

    当前代码（utils_darcy_train.py:253-258, utils_plate_train.py:331-339）存在 bug：
        if args.geo_node == 'vary_bound' or 'vary_bound_sup':    # 第一个 if
            ss_index = np.arange(max_pde_nodes, ...)
        if args.geo_node == 'all_domain':                        # 第二个 if（非 elif）
            ss_index = np.arange(0, ...)
    由于 Python 中 `'vary_bound_sup'` 为非空字符串，恒为 True，
    第一个 if 条件永远成立（无论 args.geo_node 取何值），
    导致第二个 if 即使条件成立也会被第一个 if 覆盖，
    `all_domain` 分支实际上是死代码，永远不会执行。

    Args:
        geo_node_mode (str): 几何节点模式，对应 args.geo_node 的取值：
            - "vary_bound"     : 仅使用变化边界（BC 节点）作为几何描述。
            - "vary_bound_sup" : 有监督训练时的变化边界（功能同上）。
            - "all_bound"      : 使用所有边界节点（Plate 特有）。
            - "all_domain"     : 使用全部节点（PDE + BC，当前为死代码，修复后生效）。
        max_pde_nodes (int): PDE 节点最大数量。
        max_bc_nodes (int): BC 节点最大数量（Darcy）。
        max_par_nodes (int): 参数/加载 BC 节点最大数量（Plate，Darcy 传 0）。
        max_bcy_nodes (int): 自由边界节点最大数量（Plate，Darcy 传 0）。
        max_bcxy_nodes (int): 固定边界节点最大数量（Plate，Darcy 传 0）。
        problem_type (str): 问题类型，"darcy" 或 "plate"，影响节点布局解释。

    Returns:
        ss_index (np.ndarray): 应选取的节点索引数组，形状 (n_geo,)，
            用于 coors[:, ss_index, :] 提取几何描述坐标。

    Raises:
        ValueError: 若 geo_node_mode 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> ss_index = select_geo_node_indices(
        ...     args.geo_node, max_pde_nodes, max_bc_nodes, problem_type="darcy"
        ... )
        >>> shape_coor = coors[:, ss_index, :].float().to(device)
        >>> # 替换 utils_darcy_train.py:253-258 中的条件判断（修复 bug）
    """
    raise NotImplementedError(
        "G6: select_geo_node_indices 尚未实现。"
        "修复 bug 极为简单：将两个 if 改为 if/elif/elif/else 结构，"
        "并修正 or 条件：改为 `geo_node_mode in ('vary_bound', 'vary_bound_sup')`。"
        "本函数封装后，utils_darcy_train.py 和 utils_plate_train.py 均可调用同一函数，"
        "消除重复代码并统一 bug 修复。"
    )


# ---------------------------------------------------------------------------
# G7 — 训练损失历史持久化
# ---------------------------------------------------------------------------

def save_training_history(
    err_hist: List[float],
    loss_hist: Optional[Dict[str, List[float]]],
    save_path: Union[str, Path],
    format: Literal["npy", "csv", "json"] = "json",
    append: bool = True,
) -> None:
    """将训练误差和损失历史保存到文件，支持训练过程复盘。

    当前代码（utils_darcy_train.py:193, utils_plate_train.py:254）声明了
    err_hist = [] 用于记录验证误差历史，但训练结束后从未将其保存到文件。
    一旦训练完成（或意外中断），所有历史数据丢失，无法复盘训练曲线、
    判断过拟合时间点，也无法比较不同超参数配置的训练动态。

    Args:
        err_hist (list[float]): 验证集相对 L2 误差历史，每个元素对应一个
            visual_freq 间隔的误差值。
        loss_hist (dict | None): 各损失项的历史，键为损失名（如 "pde_loss",
            "bc_loss" 等），值为对应历史列表。None 表示仅保存误差。
        save_path (str | Path): 保存文件路径（含文件名但不含扩展名，
            函数自动添加对应格式的扩展名）。
            建议路径：'./res/training_history/{geo_node}_{model}_{data}'。
        format (str): 保存格式：
            - "npy"  : numpy 格式（.npy），加载快但不可读。
            - "csv"  : 逗号分隔文本（.csv），可直接用 Excel / pandas 查看。
            - "json" : JSON 格式（.json），人类可读且保留键名，推荐。
        append (bool): 若文件已存在，是否追加而非覆盖。
            True 时将本次历史追加到已有文件（适合断点续训）；
            False 时覆盖（适合全新训练）。

    Returns:
        None

    Raises:
        IOError: 若目录不存在且无法创建。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> save_training_history(
        ...     err_hist, loss_hist={'pde': avg_pde_hist, 'bc': avg_bc_hist},
        ...     save_path=f'./res/training_history/{args.geo_node}_{args.model}_{args.data}',
        ...     format="json"
        ... )
        >>> # 在 utils_darcy_train.py 训练结束前调用，替换当前的 err_hist 死代码
    """
    raise NotImplementedError(
        "G7: save_training_history 尚未实现。"
        "json 格式最简：json.dump({'val_err': err_hist, **loss_hist}, f)。"
        "建议同时保存训练配置（config 字典）以便完整复现实验设置。"
    )


# ---------------------------------------------------------------------------
# G8 — n_head 参数使用验证
# ---------------------------------------------------------------------------

def validate_config_params(
    config: Dict,
    model: nn.Module,
) -> List[str]:
    """检查配置文件中的参数是否被模型实际使用，报告无效配置项。

    当前代码（configs/*.yaml:4 / model_darcy.py / model_plate.py）中配置文件
    定义了 n_head: 1，但所有模型类均未引用该参数，注意力头数的设计意图
    从未实现。该参数在整个代码库中是无效的死配置，容易误导后续开发者。

    本函数通过检查 config 中各参数是否在 model.__init__ 签名或实际使用中被引用，
    报告潜在的无效配置，辅助代码维护。

    Args:
        config (dict): 完整的配置字典，包含 'model'、'train' 等子字典。
        model (nn.Module): 已初始化的模型实例，用于检查实际使用的参数。

    Returns:
        warnings (list[str]): 警告信息列表，每条描述一个可能无效的配置项。
            空列表表示未发现明显问题。

    Raises:
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Note:
        正确的修复方案不只是删除 n_head 字段，而是决定：
        (1) 实现注意力机制并使用 n_head（见 B5/C4 优化点），或
        (2) 从 yaml 和代码中删除该未使用的配置项。
        本函数帮助系统性地发现此类问题。

    Example:
        >>> warnings = validate_config_params(config, model)
        >>> for w in warnings:
        ...     print(f"[CONFIG WARNING] {w}")
        >>> # 在训练开始前调用，用于早期发现配置错误
    """
    raise NotImplementedError(
        "G8: validate_config_params 尚未实现。"
        "可通过对 model 做 forward hook 或分析 model.__init__ 源码中的 config 访问，"
        "与 config.keys() 做差集检测未使用的参数。"
        "简单实现：维护一个已知有效键名的白名单集合，报告 config 中不在白名单的键。"
    )


# ---------------------------------------------------------------------------
# G9 — 课程学习采样器
# ---------------------------------------------------------------------------

def build_curriculum_dataloader(
    dataset: tud.Dataset,
    difficulty_scores: np.ndarray,
    schedule: Literal["linear", "step", "exponential"] = "linear",
    total_epochs: int = 200,
    initial_easy_fraction: float = 0.3,
    batch_size: int = 4,
    shuffle: bool = True,
) -> Callable[[int], tud.DataLoader]:
    """构建课程学习 DataLoader 工厂函数，按训练进度从简单到复杂几何采样。

    当前代码（utils_darcy_train.py:228-281, utils_plate_train.py:299-378）
    在训练开始即将所有几何复杂度的样本混合训练，无任何课程策略。
    对于几何差异较大的数据集（简单矩形域 vs 复杂异形域），
    模型在训练初期面对所有样本，可能在简单几何上产生的梯度信号
    被复杂几何的高残差淹没，收敛变慢。

    课程学习（Curriculum Learning, Bengio et al. 2009）建议从"简单"样本开始，
    逐步引入复杂样本，使模型获得更稳定的初始化后再处理困难样本。

    Args:
        dataset (tud.Dataset): 完整训练数据集。
        difficulty_scores (np.ndarray): 各样本的"难度"分数，形状 (N,)，
            值越大表示几何越复杂（如边界点数量、凹陷程度等度量）。
        schedule (str): 课程推进策略：
            - "linear"      : 随 epoch 线性增加采样的样本复杂度上界。
            - "step"        : 阶段式，每隔 total_epochs/5 将复杂度上界提高一级。
            - "exponential" : 指数式推进，初期保守，后期快速扩展样本范围。
        total_epochs (int): 总训练 epoch 数，用于计算调度曲线。
        initial_easy_fraction (float): 训练初期仅使用最简单的 initial_easy_fraction
            比例的样本（按 difficulty_scores 排序后取前 k%）；
            1.0 表示无课程（全部样本，退化为原始行为）。
        batch_size (int): 每个 batch 的样本数。
        shuffle (bool): 是否在已选样本范围内随机打乱。

    Returns:
        get_loader (callable): 一个接受 epoch (int) 并返回对应 DataLoader 的函数；
            调用方在每个 epoch 开始时调用 get_loader(epoch) 获取本轮的 DataLoader：
                for epoch in pbar:
                    loader = get_loader(epoch)
                    for (par, coors, ...) in loader: ...

    Raises:
        ValueError: 若 initial_easy_fraction 不在 (0, 1]。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> difficulty_scores = compute_difficulty_scores(coors_all)  # 用户自定义
        >>> get_loader = build_curriculum_dataloader(
        ...     train_dataset, difficulty_scores, schedule="linear",
        ...     total_epochs=config['train']['epochs'], initial_easy_fraction=0.3,
        ...     batch_size=config['train']['batchsize']
        ... )
        >>> for e in pbar:
        ...     train_loader = get_loader(e)
        ...     for batch in train_loader: ...
    """
    raise NotImplementedError(
        "G9: build_curriculum_dataloader 尚未实现。"
        "实现思路：对 difficulty_scores argsort 得到从易到难的样本顺序；"
        "在 get_loader(epoch) 中计算当前应使用前 k% 样本（k 由 schedule 和 epoch 决定）；"
        "使用 tud.Subset(dataset, sorted_indices[:k]) 创建子集后封装为 DataLoader。"
        "难度度量可简单定义为边界点数量（max_bc_nodes 减去 padding 数量）。"
    )
