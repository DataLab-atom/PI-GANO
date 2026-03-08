"""
A_data_preprocessing.py — 数据预处理可优化模块 (A1-A7)

对应 optimizable_modules.md § A 节，共 7 个优化点。
所有函数均为纯函数或工厂函数，不依赖具体模型类，可直接嵌入
generate_darcy_data_loader / generate_plate_stress_data_loader 中替换对应逻辑。
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data as tud


# ---------------------------------------------------------------------------
# A1 — 输入坐标归一化
# ---------------------------------------------------------------------------

def normalize_coordinates(
    coors: np.ndarray,
    method: Literal["minmax", "zscore", "unit_sphere"] = "minmax",
    stats: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """对节点坐标做归一化，消除量纲差异并统一值域，提升梯度尺度一致性。

    当前代码（utils_data.py）中原始坐标直接送入模型，未做任何归一化。
    不同样本的几何域可能跨越 [0,0.1] 到 [0,10] 等不同量级，导致网络
    对不同坐标维度的梯度尺度差异悬殊，影响收敛速度和最终精度。

    本函数在数据加载阶段统一完成坐标归一化，并返回统计量 stats，
    供推理时对新样本坐标做相同变换（或反变换还原物理坐标）。

    Args:
        coors (np.ndarray): 原始节点坐标，形状 (N, 2) 或 (M, N, 2)，
            最后一维为 [x, y]。
        method (str): 归一化策略。
            - "minmax"      : 将每个维度线性映射到 [0, 1]。
            - "zscore"      : 减均值除标准差，输出近似 N(0,1)。
            - "unit_sphere" : 先中心化再除以最大半径，输出落入单位球。
        stats (dict | None): 若提供，直接用该统计量做变换（用于验证/测试集），
            保持与训练集变换一致；若为 None，则从 coors 中估计统计量。

    Returns:
        coors_normalized (np.ndarray): 归一化后的坐标，与 coors 同形状。
        stats (dict): 包含归一化所需统计量，键名因 method 而异：
            - "minmax"      : {"x_min": ..., "x_max": ...}
            - "zscore"      : {"mean": ..., "std": ...}
            - "unit_sphere" : {"center": ..., "radius": ...}

    Raises:
        ValueError: 若 method 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> coors_train_norm, stats = normalize_coordinates(coors_train, method="minmax")
        >>> coors_val_norm, _       = normalize_coordinates(coors_val,   method="minmax", stats=stats)
    """
    raise NotImplementedError(
        "A1: normalize_coordinates 尚未实现。"
        "请参照 method 参数选择 minmax / zscore / unit_sphere 逻辑填充函数体，"
        "并在 generate_darcy_data_loader / generate_plate_stress_data_loader 的"
        "坐标拼接步骤前调用本函数。"
    )


# ---------------------------------------------------------------------------
# A2 — Young's 模量缩放因子自适应
# ---------------------------------------------------------------------------

def compute_young_scalar_factor(
    young_raw: float,
    method: Literal["fixed", "order_of_magnitude", "data_driven"] = "order_of_magnitude",
    target_order: float = 1.0,
    reference_values: Optional[np.ndarray] = None,
) -> float:
    """自适应计算 Young's 模量缩放因子，替代硬编码的 scalar_factor=1e-4。

    当前代码（utils_data.py:117-118）使用固定 scalar_factor=1e-4，
    对不同数据集需要手动修改源码。本函数根据 young_raw 的数量级或
    数据分布自动确定合适的缩放比例，使缩放后的 young 落在目标数量级。

    Args:
        young_raw (float): 从 .mat 文件读取的原始 Young's 模量值（Pa）。
        method (str): 缩放策略。
            - "fixed"               : 返回固定值 1e-4（保持原行为，作为基线）。
            - "order_of_magnitude"  : 将 young_raw 缩放到 target_order 数量级，
                                      即 scalar = target_order / young_raw。
            - "data_driven"         : 根据 reference_values（训练集中所有
                                      young 值的数组）统计均值，使均值归一到
                                      target_order；需提供 reference_values。
        target_order (float): 目标数量级（仅 "order_of_magnitude" 和
            "data_driven" 模式下有效），默认 1.0。
        reference_values (np.ndarray | None): 形状 (N,) 的训练集 young 值数组，
            仅 "data_driven" 模式需要。

    Returns:
        scalar_factor (float): 建议的缩放因子，用法：young = young_raw * scalar_factor。

    Raises:
        ValueError: 若 method="data_driven" 但未提供 reference_values。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> sf = compute_young_scalar_factor(young_raw=2.1e11, method="order_of_magnitude")
        >>> young = young_raw * sf   # 替换 utils_data.py:117-118
    """
    raise NotImplementedError(
        "A2: compute_young_scalar_factor 尚未实现。"
        "请在 generate_plate_stress_data_loader 中，将 scalar_factor=1e-4 的硬编码"
        "替换为本函数的返回值。"
    )


# ---------------------------------------------------------------------------
# A3 — Padding 掩码策略
# ---------------------------------------------------------------------------

def pad_array_with_mask(
    arr: np.ndarray,
    target_size: int,
    pad_value: float = 0.0,
    mask_value: float = -1.0,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """对变长数组做 padding 并同步生成布尔掩码，避免零值歧义。

    当前代码（utils_data.py:51,55,178-201）对无效节点位置用零值填充，
    包括坐标（0,0）和解值（0）。模型在训练时无法区分"填充零"与"物理零解"，
    可能将 padding 区域误学习为有效零解，引入系统性偏置。

    本函数将 padding 值与掩码分离：padding 区域使用 pad_value 填充数组，
    同时返回独立的 mask 数组标记有效/无效区域，调用方用 mask 过滤梯度或损失。

    Args:
        arr (np.ndarray): 待 padding 的原始数组，形状任意，沿 axis 维度扩充。
        target_size (int): padding 后沿 axis 方向的目标长度。
        pad_value (float): 用于填充无效位置的数值，建议与物理零解有显著区分，
            如 np.nan 或远离数据分布的常数。默认 0.0（保持原行为）。
        mask_value (float): 在 flag 数组中标记无效 padding 区域的值，
            当前代码约定为 -1，此处保持一致。默认 -1.0。
        axis (int): 沿哪个轴做 padding，默认 0（节点维度）。

    Returns:
        arr_padded (np.ndarray): padding 后的数组，沿 axis 的长度 == target_size。
        mask (np.ndarray): 与 arr_padded 同形状的 float32 掩码；
            有效位置为 1.0，padding 位置为 mask_value（-1.0）。

    Raises:
        ValueError: 若 arr.shape[axis] > target_size。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> up_padded, mask = pad_array_with_mask(up, max_pde_nodes, pad_value=-999.0)
        >>> # 替换 utils_data.py:51 中的 np.zeros((1, max_pde_nodes-num_pde))
    """
    raise NotImplementedError(
        "A3: pad_array_with_mask 尚未实现。"
        "建议使用非零 pad_value（如 NaN 或极大值）并配合返回的 mask 在损失计算中屏蔽。"
        "需同步修改 darcy_loss / plate_stress_loss 中的 flag 判断逻辑。"
    )


# ---------------------------------------------------------------------------
# A4 — 数据集划分比例可配置
# ---------------------------------------------------------------------------

def split_dataset_indices(
    dataset_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按可配置比例划分数据集索引，支持随机打乱和种子固定。

    当前代码（utils_data.py:83-85, 242-244）将训练/验证/测试划分比例硬编码为
    70/10/20，小数据集下验证集过小（仅10%）导致模型选择不稳定，无法通过配置
    文件调整。本函数将比例参数化，并支持随机打乱索引以减小顺序偏差。

    Args:
        dataset_size (int): 数据集总样本数 M。
        train_ratio (float): 训练集占比，默认 0.7。
        val_ratio (float): 验证集占比，默认 0.1。
        test_ratio (float): 测试集占比，默认 0.2。三者之和应为 1.0。
        shuffle (bool): 是否在划分前随机打乱索引，默认 True。
            当前代码按顺序切分，未打乱，可能导致某些物理参数分布不均匀。
        seed (int | None): 随机种子，保证可复现；None 表示不固定。

    Returns:
        train_indices (np.ndarray): 训练集样本索引，形状 (n_train,)。
        val_indices   (np.ndarray): 验证集样本索引，形状 (n_val,)。
        test_indices  (np.ndarray): 测试集样本索引，形状 (n_test,)。

    Raises:
        ValueError: 若三比例之和不等于 1.0（允许 1e-6 误差）。
        ValueError: 若 dataset_size 过小导致任一子集为空。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> train_idx, val_idx, test_idx = split_dataset_indices(
        ...     len(u), train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
        ... )
        >>> # 替换 utils_data.py:83-85 中的 bar1/bar2/bar3 硬编码
    """
    raise NotImplementedError(
        "A4: split_dataset_indices 尚未实现。"
        "建议将 train_ratio/val_ratio/test_ratio 暴露到 config['data'] 中，"
        "通过 yaml 文件统一配置，并在 generate_*_data_loader 中调用本函数。"
    )


# ---------------------------------------------------------------------------
# A5 — 数据增强
# ---------------------------------------------------------------------------

def apply_geometric_augmentation(
    coors: np.ndarray,
    u: np.ndarray,
    flag: np.ndarray,
    transforms: List[Literal["flip_x", "flip_y", "rotate_90", "rotate_180", "rotate_270"]],
    v: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """对物理样本施加几何增强变换，等效扩充训练集。

    当前代码（utils_data.py）训练过程中完全没有数据增强。对于线性弹性、
    Darcy 渗流等物理问题，若边界条件具有对称性，则坐标翻转、90°/180°/270°
    旋转等几何变换对应合法的等价样本，可在不增加数据采集成本的前提下
    显著扩充有效训练样本量。

    注意：旋转变换会同时旋转位移场方向（对 plate 问题需旋转 u/v 分量），
    翻转变换同理需对向量场做符号修正，调用方需根据具体物理方程确认变换合法性。

    Args:
        coors (np.ndarray): 节点坐标，形状 (N, 2)，列顺序为 [x, y]。
        u (np.ndarray): 标量/矢量场的 x 分量，形状 (N,) 或 (N, 1)。
        flag (np.ndarray): 节点有效性 flag，形状 (N,)，-1 为 padding，其余为有效。
        transforms (list[str]): 要施加的变换列表，每个元素触发一种增强；
            最终返回所有增强样本拼接后的批次。
        v (np.ndarray | None): 矢量场的 y 分量（plate 问题），形状 (N,) 或 (N, 1)；
            标量场（Darcy）传 None。

    Returns:
        coors_aug (np.ndarray): 增强后的坐标，形状 (N * (1+len(transforms)), 2)；
            第一段为原始样本，后续各段为各变换结果。
        u_aug (np.ndarray): 增强后的 u 场，形状与 coors_aug 对齐。
        flag_aug (np.ndarray): 增强后的 flag，形状 (N * (1+len(transforms)),)。
        v_aug (np.ndarray | None): 增强后的 v 场；输入 v=None 时返回 None。

    Raises:
        ValueError: 若 transforms 包含不支持的变换名称。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> coors_aug, u_aug, flag_aug, _ = apply_geometric_augmentation(
        ...     coors, u, flag, transforms=["flip_x", "flip_y"]
        ... )
    """
    raise NotImplementedError(
        "A5: apply_geometric_augmentation 尚未实现。"
        "实现时注意：(1) flip_x 对 plate 的 u 分量取负；"
        "(2) 旋转变换需对坐标和位移向量同时旋转；"
        "(3) 仅对训练集做增强，验证/测试集保持原始坐标系。"
    )


# ---------------------------------------------------------------------------
# A6 — val/test DataLoader batch size
# ---------------------------------------------------------------------------

def create_eval_dataloader(
    dataset: tud.Dataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tud.DataLoader:
    """为验证集/测试集创建高效 DataLoader，支持较大 batch_size 以加速评估。

    当前代码（utils_data.py:258-259）将 Plate 验证集和测试集的 batch_size
    固定为 1，每个样本单独前向传播，评估耗时随样本数线性增长，epoch 内
    验证开销极大。批量评估不改变结果，只影响效率。

    Args:
        dataset (torch.utils.data.Dataset): 验证集或测试集 Dataset 对象。
        batch_size (int): 评估时的批大小。建议设置与训练 batch_size 相同或
            更大（评估无反向传播，显存占用更低）。典型值：16-64。
        num_workers (int): DataLoader 的 worker 数量，默认 0（主进程加载）。
            在 Linux 多核机器上设置为 4-8 可加速 IO。
        pin_memory (bool): 是否使用锁页内存（CUDA 加速 H2D 传输），
            仅在使用 GPU 且数据量大时建议开启，默认 False。

    Returns:
        loader (torch.utils.data.DataLoader): shuffle=False、drop_last=False
            的评估专用 DataLoader。

    Raises:
        ValueError: 若 batch_size <= 0。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> val_loader = create_eval_dataloader(val_dataset, batch_size=config['train']['batchsize'])
        >>> # 替换 utils_data.py:258-259 中的 batch_size=1
    """
    raise NotImplementedError(
        "A6: create_eval_dataloader 尚未实现。"
        "核心改动仅一行：DataLoader(..., batch_size=batch_size, shuffle=False)。"
        "建议将 eval_batch_size 作为独立字段暴露到 config['train'] 中。"
    )


# ---------------------------------------------------------------------------
# A7 — par_flag 维度截取
# ---------------------------------------------------------------------------

def extract_par_flag(
    par_flag_raw: np.ndarray,
    strategy: Literal["first_col", "all_cols_and", "all_cols_or", "any_nonzero"] = "first_col",
) -> np.ndarray:
    """从多维 par_flag 张量中提取一维有效性掩码，支持多种聚合策略。

    当前代码（utils_data.py:75, 231）仅取 flag 张量第一列：
        par_flagT[:,:,0]
    若各参数维度的有效性标记不完全一致（如某些维度因数据缺失而局部置零），
    则仅看第一列会产生不准确的掩码，导致有效参数被错误屏蔽或无效 padding
    参与聚合。

    Args:
        par_flag_raw (np.ndarray): 原始多维 flag 张量，形状 (M, max_par, D)，
            其中 D 为参数特征维度（Darcy D=3，Plate D=4）。
        strategy (str): 从 D 维中提取一维掩码的策略：
            - "first_col"     : 仅取第 0 列（原始行为，兼容基线）。
            - "all_cols_and"  : 所有列均为 1 才认为有效（逻辑与，最严格）。
            - "all_cols_or"   : 任意列为 1 即认为有效（逻辑或，最宽松）。
            - "any_nonzero"   : 任意列非零即有效（处理 flag 非严格 0/1 情形）。

    Returns:
        par_flag (np.ndarray): 提取后的一维 flag，形状 (M, max_par)，
            元素为 0.0（无效 padding）或 1.0（有效参数点）。

    Raises:
        ValueError: 若 strategy 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> par_flag = extract_par_flag(par_flag_raw, strategy="all_cols_and")
        >>> # 替换 utils_data.py:75 中的 par_flagT[:,:,0]
    """
    raise NotImplementedError(
        "A7: extract_par_flag 尚未实现。"
        "建议先打印原始 par_flag_raw 各列的一致性统计，确认实际数据中各列是否完全一致，"
        "再选择合适的 strategy。若各列始终一致，'first_col' 即为最优解，无需修改。"
    )
