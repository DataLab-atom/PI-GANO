"""
B_geometry_encoder.py — 几何编码可优化模块 (B1-B7)

对应 optimizable_modules.md § B 节，共 7 个优化点。
本模块提供替换 model_darcy.py / model_plate.py 中 DG 类各组件的工厂函数。
所有函数返回 nn.Module，可直接赋给 self.branch 或 self.DG 等属性。

当前 DG 类核心结构（两个文件共用）：
    branch: Linear(2→F) + [Tanh + Linear(F→F)] * N_layer + Linear(F→F)
    聚合:   sum(enc*flag) / sum(flag)   （masked mean pooling）
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# B1 — 几何点特征提取网络结构
# ---------------------------------------------------------------------------

def build_geometry_feature_extractor(
    in_dim: int,
    fc_dim: int,
    n_layer: int,
    activation: nn.Module,
    architecture: Literal["uniform", "funnel", "pyramid", "residual"] = "uniform",
) -> nn.Module:
    """构建几何点特征提取网络（替换 DG.branch）。

    当前代码（model_darcy.py:160-165, model_plate.py:185-190）中 MLP 各层
    宽度固定为 fc_dim（等宽），层数由 N_layer 控制。等宽网络表达能力受限，
    且不具备跳层梯度路径。本函数支持多种非等宽或含残差的结构。

    Args:
        in_dim (int): 输入维度，即几何坐标维度，通常为 2（x, y）。
            若已过 Fourier 编码（见 B7），则 in_dim = 2 * n_freq * 2。
        fc_dim (int): 隐层基础宽度，与 config['model']['fc_dim'] 对应。
        n_layer (int): 隐层数量，与 config['model']['N_layer'] 对应。
        activation (nn.Module): 隐层激活函数实例（如 nn.Tanh()、nn.GELU()）。
            由 B2 的 get_geometry_activation() 获取。
        architecture (str): 网络结构类型：
            - "uniform"  : 等宽 MLP，与原始代码等效（基线）。
            - "funnel"   : 宽度从 fc_dim*2 逐层收缩到 fc_dim（漏斗型）。
            - "pyramid"  : 宽度先扩张再收缩（U 型）。
            - "residual" : 等宽 MLP + 每层加残差连接（PreNorm ResBlock）。

    Returns:
        extractor (nn.Module): 特征提取网络，接受形状 (B, M, in_dim) 的输入，
            输出形状 (B, M, fc_dim)。注意：最后一层输出维度固定为 fc_dim，
            与聚合层 (B4) 的输入维度保持一致。

    Raises:
        ValueError: 若 architecture 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> act = get_geometry_activation("gelu")
        >>> self.branch = build_geometry_feature_extractor(2, config['model']['fc_dim'],
        ...                                                config['model']['N_layer'], act,
        ...                                                architecture="residual")
        >>> # 替换 model_darcy.py:160-165 中的 trunk_layers 构建逻辑
    """
    raise NotImplementedError(
        "B1: build_geometry_feature_extractor 尚未实现。"
        "'residual' 结构建议每层使用 LayerNorm + Linear + activation + skip，"
        "注意第一层 in_dim→fc_dim 无法直接加残差，需用投影层或先做 embedding。"
    )


# ---------------------------------------------------------------------------
# B2 — 几何编码器激活函数
# ---------------------------------------------------------------------------

def get_geometry_activation(
    name: Literal["tanh", "gelu", "silu", "relu", "leaky_relu", "elu"] = "tanh",
) -> nn.Module:
    """按名称返回几何编码器激活函数实例，替换全程硬编码的 nn.Tanh()。

    当前代码（model_darcy.py:161,163, model_plate.py:186,188）全程使用 Tanh，
    深层网络中 Tanh 易饱和导致梯度消失，且对非平滑解的拟合能力弱于现代激活
    函数。GELU/SiLU 在 Transformer 类模型中已被证明更优。

    Args:
        name (str): 激活函数名称（大小写不敏感）：
            - "tanh"       : nn.Tanh()（原始行为，基线）。
            - "gelu"       : nn.GELU()，推荐替代方案之一。
            - "silu"       : nn.SiLU()（即 Swish），另一推荐方案。
            - "relu"       : nn.ReLU()（不建议用于 PINN，可能导致解不光滑）。
            - "leaky_relu" : nn.LeakyReLU(0.01)。
            - "elu"        : nn.ELU()。

    Returns:
        activation (nn.Module): 对应激活函数的实例，可直接传入 nn.Sequential。

    Raises:
        ValueError: 若 name 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> act = get_geometry_activation("gelu")
        >>> # 替换 model_darcy.py:161 中的 nn.Tanh()
    """
    raise NotImplementedError(
        "B2: get_geometry_activation 尚未实现。"
        "实现极为简单：dict 映射 name → nn.Module 实例即可。"
        "同一函数可共享给 C/D/E 模块的激活函数选择。"
    )


# ---------------------------------------------------------------------------
# B3 — 几何编码器最后一层激活
# ---------------------------------------------------------------------------

def build_geometry_encoder_output_layer(
    fc_dim: int,
    out_dim: int,
    output_activation: Optional[nn.Module] = None,
) -> nn.Module:
    """构建几何编码器的最后输出层，支持可选的输出激活函数。

    当前代码（model_darcy.py:164, model_plate.py:189）最后一层为裸 Linear，
    无任何激活，输出值域无界（(-∞, +∞)）。当下游聚合层（mean pooling）
    或融合层（Concat/Add）对输入值域敏感时，无界输出可能导致数值不稳定。

    Args:
        fc_dim (int): 倒数第二层的宽度（即本层的输入维度）。
        out_dim (int): 输出维度，通常等于 fc_dim（几何嵌入维度）。
        output_activation (nn.Module | None): 输出激活函数。
            - None    : 裸线性输出（原始行为，基线）。
            - nn.Tanh(): 将输出限制到 (-1, 1)，与后续 Tanh 编码器输出对齐。
            - nn.Sigmoid(): 将输出限制到 (0, 1)（适用于某些注意力权重场景）。

    Returns:
        output_layer (nn.Module): 包含 Linear + 可选激活的顺序模块，
            接受形状 (..., fc_dim) 的输入，输出形状 (..., out_dim)。

    Raises:
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> out_layer = build_geometry_encoder_output_layer(fc_dim, fc_dim, nn.Tanh())
        >>> # 替换 model_darcy.py:164 中的 nn.Linear(fc_dim, fc_dim)
    """
    raise NotImplementedError(
        "B3: build_geometry_encoder_output_layer 尚未实现。"
        "若选择 Tanh 输出激活，需确认与聚合层（B4）的数值范围假设一致。"
        "建议消融实验：分别对比 None / Tanh / LayerNorm 三种输出处理。"
    )


# ---------------------------------------------------------------------------
# B4 — 集合聚合方式
# ---------------------------------------------------------------------------

def aggregate_geometry_features(
    enc: torch.Tensor,
    shape_flag: torch.Tensor,
    method: Literal["mean", "max", "sum", "attention", "multi_head_attention"] = "mean",
    attn_module: Optional[nn.Module] = None,
) -> torch.Tensor:
    """对几何点特征集合做全局聚合，输出单一域嵌入向量。

    当前代码（model_darcy.py:181, model_plate.py:203）使用 masked mean pooling：
        Domain_enc = sum(enc * flag) / sum(flag)
    均值池化对几何局部结构（尖角、凹陷等高曲率区域）不敏感，且当各边界点密度
    不均匀时存在偏差。本函数支持多种聚合策略供对比实验。

    Args:
        enc (torch.Tensor): 几何点特征张量，形状 (B, M, F)。
        shape_flag (torch.Tensor): 有效性掩码，形状 (B, M)，1.0 为有效，0.0 为 padding。
        method (str): 聚合策略：
            - "mean"               : Masked mean pooling（原始行为，基线）。
            - "max"                : Masked max pooling（对尖角/极值敏感）。
            - "sum"                : Masked sum pooling（保留总量信息）。
            - "attention"          : 单头自注意力全局池化（可学习聚合权重）；
                                     需提供 attn_module。
            - "multi_head_attention": 多头注意力池化；需提供 attn_module。
        attn_module (nn.Module | None): 注意力聚合模块（method 为 attention 类时必填），
            接受 (B, M, F) 输入，输出 (B, 1, F) 的全局表示。

    Returns:
        Domain_enc (torch.Tensor): 全局几何嵌入，形状 (B, 1, F)，
            与原始代码输出形状一致，可直接替换。

    Raises:
        ValueError: 若 method 需要 attn_module 但未提供。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> Domain_enc = aggregate_geometry_features(enc, shape_flag, method="max")
        >>> # 替换 model_darcy.py:181 中的 torch.sum(enc_masked,...)/torch.sum(flag,...)
    """
    raise NotImplementedError(
        "B4: aggregate_geometry_features 尚未实现。"
        "'attention' 聚合建议实现为：学习一个 query 向量 q∈R^F，"
        "计算 softmax(q·enc_masked^T / sqrt(F)) 作为权重，再加权求和。"
        "需将无效位置（flag=0）在 softmax 前置为 -inf。"
    )


# ---------------------------------------------------------------------------
# B5 — 几何点间交互
# ---------------------------------------------------------------------------

def build_geometry_interaction_layer(
    fc_dim: int,
    interaction_type: Literal["none", "self_attention", "graph_conv", "pointnet_local"] = "none",
    n_heads: int = 4,
    k_neighbors: int = 8,
) -> nn.Module:
    """构建几何边界点之间的交互层，捕获相邻点的曲率和上下文信息。

    当前代码（model_darcy.py:179-183, model_plate.py:200-205）中各边界点
    独立经过 MLP 后直接聚合，点与点之间无任何交互。相邻点的几何曲率信息
    （如拐角附近的曲率变化）无法被网络利用，限制了几何表达能力。

    本函数返回一个交互模块，插入在 self.branch（MLP 编码）之后、
    aggregate_geometry_features（全局聚合）之前。

    Args:
        fc_dim (int): 点特征维度（与 MLP 输出维度对齐）。
        interaction_type (str): 交互机制：
            - "none"           : 恒等映射，无交互（原始行为，基线）。
            - "self_attention" : 标准多头自注意力（Transformer Encoder Layer），
                                 计算量为 O(M²F)，适合边界点数 M 较小时使用。
            - "graph_conv"     : 基于 k 近邻的图卷积（EdgeConv），
                                 在坐标空间寻找 k_neighbors 个最近点做消息传递。
            - "pointnet_local" : PointNet++ 风格的局部特征聚合，
                                 分组→PointNet→上采样。
        n_heads (int): 多头注意力的头数（interaction_type="self_attention" 时有效）。
        k_neighbors (int): k 近邻图的邻居数（interaction_type="graph_conv" 时有效）。

    Returns:
        interaction_layer (nn.Module): 接受 (B, M, fc_dim) 输入，
            输出 (B, M, fc_dim) 的交互特征，形状不变。

    Raises:
        ValueError: 若 interaction_type 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> self.geo_interaction = build_geometry_interaction_layer(fc_dim, "self_attention", n_heads=4)
        >>> # 在 DG.forward 中：enc = self.branch(shape_coor); enc = self.geo_interaction(enc)
        >>> # 替换 model_darcy.py:179-183 的直接聚合逻辑
    """
    raise NotImplementedError(
        "B5: build_geometry_interaction_layer 尚未实现。"
        "'self_attention' 可直接使用 nn.TransformerEncoderLayer(d_model=fc_dim, nhead=n_heads)。"
        "注意需将 padding 位置（shape_flag=0）对应的 key_padding_mask 传入，"
        "避免 padding 点参与注意力计算。"
    )


# ---------------------------------------------------------------------------
# B6 — 几何编码器正则化
# ---------------------------------------------------------------------------

def build_geometry_encoder_with_regularization(
    in_dim: int,
    fc_dim: int,
    n_layer: int,
    activation: nn.Module,
    dropout_rate: float = 0.0,
    norm_type: Literal["none", "batch_norm", "layer_norm", "instance_norm"] = "none",
) -> nn.Module:
    """构建带正则化的几何编码器，缓解小数据集过拟合。

    当前代码（model_darcy.py:154-183, model_plate.py:179-205）中几何编码器
    无任何正则化手段（无 Dropout、无 BatchNorm、无 LayerNorm），在小样本
    数据集（Darcy ~200 samples，Plate ~200 samples）下几何编码器易过拟合，
    特别是当几何形状多样性较大时泛化能力受限。

    Args:
        in_dim (int): 输入特征维度，通常为 2（坐标）或 Fourier 编码后的维度。
        fc_dim (int): 隐层宽度。
        n_layer (int): 隐层数量。
        activation (nn.Module): 激活函数实例。
        dropout_rate (float): Dropout 比例，0.0 表示不使用（原始行为）；
            建议范围 [0.05, 0.2]，在 eval 模式下自动关闭。
        norm_type (str): 归一化类型：
            - "none"          : 无归一化（原始行为，基线）。
            - "batch_norm"    : BatchNorm1d，对 batch 做归一化；
                                注意点云数据中 batch_size 可能为 1，此时效果差。
            - "layer_norm"    : LayerNorm，对每个点的特征维度归一化，
                                不受 batch_size 影响，推荐。
            - "instance_norm" : InstanceNorm，对每个样本的每个通道归一化。

    Returns:
        encoder (nn.Module): 带正则化的完整几何编码器，
            接受 (B, M, in_dim)，输出 (B, M, fc_dim)。

    Raises:
        ValueError: 若 norm_type 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> self.branch = build_geometry_encoder_with_regularization(
        ...     in_dim=2, fc_dim=config['model']['fc_dim'],
        ...     n_layer=config['model']['N_layer'],
        ...     activation=nn.GELU(), dropout_rate=0.1, norm_type="layer_norm"
        ... )
        >>> # 替换 model_darcy.py:160-165 的 trunk_layers 构建
    """
    raise NotImplementedError(
        "B6: build_geometry_encoder_with_regularization 尚未实现。"
        "LayerNorm 在点云场景最为稳定，建议作为首选。"
        "Dropout 应插在每层激活函数之后，eval() 模式下 PyTorch 自动关闭。"
    )


# ---------------------------------------------------------------------------
# B7 — 输入位置编码（Fourier 特征）
# ---------------------------------------------------------------------------

def fourier_encode_geometry_coords(
    shape_coor: torch.Tensor,
    n_freq: int = 16,
    scale: float = 1.0,
    learnable: bool = False,
    include_original: bool = True,
) -> Tuple[torch.Tensor, int]:
    """对几何坐标应用 Random Fourier Feature 编码，提升高频空间表达能力。

    当前代码（model_darcy.py:160, model_plate.py:185）将原始 (x, y) 坐标
    直接输入几何编码器 MLP。MLP 对高频信号（如尖角边界的快速变化）拟合能力
    有限（spectral bias 现象）。Random Fourier Features 通过正弦/余弦映射
    将坐标提升到高维频率空间，已被证明对 PINN/PINO 类方法显著改善。

    编码公式（Rahimi & Recht 2007）：
        γ(x) = [cos(Bx), sin(Bx)]，其中 B ∈ R^{n_freq × d}，B ~ N(0, scale²)

    Args:
        shape_coor (torch.Tensor): 原始几何坐标，形状 (B, M, 2)，范围任意。
        n_freq (int): 随机频率数量，编码后特征维度为 2*n_freq（+2 若保留原始）。
            较大的 n_freq 捕获更高频细节，但增加后续 MLP 输入维度；建议 [8, 64]。
        scale (float): 随机矩阵 B 的标准差，控制频率分布范围。
            值越大频率越高；需与坐标值域匹配，建议从 1.0 开始网格搜索。
        learnable (bool): 是否将频率矩阵 B 设为可学习参数。
            True → nn.Parameter；False → 固定随机矩阵（原始 RFF 定义）。
        include_original (bool): 编码后是否拼接原始坐标 (x, y)，
            True → 输出维度 = 2*n_freq + 2；False → 输出维度 = 2*n_freq。

    Returns:
        encoded (torch.Tensor): Fourier 编码后的特征，
            形状 (B, M, 2*n_freq + (2 if include_original else 0))。
        out_dim (int): 编码后的特征维度，用于设置后续 MLP 的 in_dim（B1 的入参）。

    Raises:
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> encoded, in_dim = fourier_encode_geometry_coords(shape_coor, n_freq=16, scale=1.0)
        >>> # encoded 作为 DG.branch 的输入，替换 model_darcy.py:160 的直接坐标输入
        >>> # in_dim 传给 build_geometry_feature_extractor 的 in_dim 参数
    """
    raise NotImplementedError(
        "B7: fourier_encode_geometry_coords 尚未实现。"
        "实现要点：B 矩阵在 __init__ 阶段生成并注册（register_buffer 或 nn.Parameter），"
        "forward 时计算 cos(shape_coor @ B.T) 和 sin(shape_coor @ B.T) 后拼接。"
        "scale 参数对性能影响显著，建议作为超参数纳入网格搜索。"
    )
