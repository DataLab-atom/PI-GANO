"""
C_parameter_encoder.py — 参数编码可优化模块 (C1-C6)

对应 optimizable_modules.md § C 节，共 6 个优化点。
本模块提供替换 model_darcy.py / model_plate.py 中 PI_GANO.branch（参数编码器）
及相关聚合逻辑的工厂函数。

当前参数编码器结构：
    branch: Linear(3→2F) + [Tanh + Linear(2F→2F)] * N_layer + Linear(2F→2F)
    聚合:   torch.amax(enc * par_flag, dim=1)   （masked max pooling，逐维独立）

Darcy 参数维度 D=3（x, y, u_BC），Plate 参数维度 D=4（x, y, u_load, v_load）。
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# C1 — 参数特征提取网络结构
# ---------------------------------------------------------------------------

def build_parameter_feature_extractor(
    in_dim: int,
    fc_dim: int,
    n_layer: int,
    activation: nn.Module,
    dropout_rate: float = 0.0,
    norm_type: Literal["none", "layer_norm", "batch_norm"] = "none",
    architecture: Literal["uniform", "residual"] = "uniform",
) -> nn.Module:
    """构建参数特征提取网络（替换 PI_GANO.branch），支持残差和正则化。

    当前代码（model_darcy.py:194-199, model_plate.py:216-221）与几何编码器
    存在完全相同的问题：等宽 MLP、无残差、无正则化（见 B1/B6 的分析）。
    本函数将同样的改进方案应用到参数编码器，接口设计与 B1/B6 保持一致
    以降低实验切换成本。

    Args:
        in_dim (int): 输入维度，Darcy 为 3，Plate 为 4；
            若已对参数做 Fourier 编码，则为编码后维度。
        fc_dim (int): 基础隐层宽度；注意原始代码参数编码器宽度为 2*fc_dim，
            建议此处 fc_dim 传入 2*config['model']['fc_dim'] 以保持等效。
        n_layer (int): 隐层数量，与 config['model']['N_layer'] 对应。
        activation (nn.Module): 激活函数实例。
        dropout_rate (float): Dropout 比例，0.0 = 无正则化（原始行为）。
        norm_type (str): 归一化类型，"none"/"layer_norm"/"batch_norm"。
            点云/集合数据建议 "layer_norm"。
        architecture (str): 网络拓扑：
            - "uniform"  : 等宽 MLP（原始行为）。
            - "residual" : 每层加残差连接。

    Returns:
        extractor (nn.Module): 参数特征提取网络，接受 (B, M', in_dim)，
            输出 (B, M', fc_dim)。

    Raises:
        ValueError: 若 architecture / norm_type 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> self.branch = build_parameter_feature_extractor(
        ...     in_dim=3, fc_dim=2*config['model']['fc_dim'],
        ...     n_layer=config['model']['N_layer'],
        ...     activation=nn.GELU(), dropout_rate=0.1, norm_type="layer_norm"
        ... )
        >>> # 替换 model_darcy.py:194-199 的 trunk_layers 构建
    """
    raise NotImplementedError(
        "C1: build_parameter_feature_extractor 尚未实现。"
        "实现与 B1/B6 几乎完全相同，建议提取公共函数 build_mlp() 供两者复用。"
    )


# ---------------------------------------------------------------------------
# C2 — 全局聚合方式（替换 torch.amax）
# ---------------------------------------------------------------------------

def aggregate_parameter_features(
    enc: torch.Tensor,
    par_flag: torch.Tensor,
    method: Literal["amax", "mean", "attention", "mean_max_concat"] = "amax",
    attn_module: Optional[nn.Module] = None,
) -> torch.Tensor:
    """对参数点特征做全局聚合，输出单一参数嵌入向量。

    当前代码（model_darcy.py:238, model_plate.py:280）使用：
        enc = torch.amax(enc_masked, dim=1, keepdim=True)
    torch.amax 在各维度独立取最大值，不同维度的最大来自不同参数点，
    破坏了特征的协同关系（同一参数点在不同维度应联合表示该点的完整信息）。
    例如，维度 0 的最大值可能来自参数点 A，维度 1 的最大值来自参数点 B，
    导致拼接后的"全局特征"实际上是多个点的混合，丧失了物理语义。

    Args:
        enc (torch.Tensor): 参数点特征，形状 (B, M', F)，
            M' 为 max_par_nodes（含 padding），F 为特征维度。
        par_flag (torch.Tensor): 有效性掩码，形状 (B, M')，
            1.0 为有效参数点，0.0 为 padding。
        method (str): 聚合策略：
            - "amax"           : 逐维独立最大值（原始行为，存在协同破坏问题）。
            - "mean"           : Masked mean pooling（保持特征协同性）。
            - "attention"      : 可学习注意力权重池化；需提供 attn_module，
                                 权重对各参数点计算，同一组权重作用于所有维度，
                                 从根本上保证特征协同性。
            - "mean_max_concat": 均值 + 最大值双路拼接（PointNet 经典做法），
                                 输出维度为 2F，需下游层适配。
        attn_module (nn.Module | None): 注意力权重生成模块（method="attention" 时必填），
            接受 (B, M', F) 输入，输出 (B, M', 1) 的归一化注意力分数。

    Returns:
        enc_global (torch.Tensor): 全局参数嵌入，形状 (B, 1, F)（或 (B, 1, 2F)
            当 method="mean_max_concat" 时），可直接替换原始 enc 变量。

    Raises:
        ValueError: 若 method 需要 attn_module 但未提供。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> enc_global = aggregate_parameter_features(enc, par_flag, method="attention",
        ...                                           attn_module=self.par_attn)
        >>> # 替换 model_darcy.py:237-238 中的 enc_masked + amax 逻辑
    """
    raise NotImplementedError(
        "C2: aggregate_parameter_features 尚未实现。"
        "'attention' 模式：定义 query q∈R^F，对每个参数点 i 计算 score_i = sigmoid(q·enc_i)，"
        "将 padding 位置的 score 置 0，归一化后加权求和。"
        "注意区分 sigmoid（非排他）与 softmax（排他）的语义差异。"
    )


# ---------------------------------------------------------------------------
# C3 — 参数输入归一化
# ---------------------------------------------------------------------------

def normalize_parameter_inputs(
    par: torch.Tensor,
    par_flag: torch.Tensor,
    stats: Optional[Dict[str, torch.Tensor]] = None,
    method: Literal["minmax", "zscore", "none"] = "zscore",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """对边界条件参数做归一化，消除不同边界量纲差异。

    当前代码（utils_data.py，无对应实现）中边界条件参数（坐标 + BC 值）
    直接输入参数编码器，未做任何归一化。不同参数维度的值域差异可能导致网络
    偏向大数值特征：坐标 (x,y) 通常在 [0,1]，而 BC 位移值可能在 [0, 0.001]，
    数量级差异达 3 个数量级，网络很难同等学习两类信息。

    Args:
        par (torch.Tensor): 原始参数张量，形状 (B, M', D)；
            Darcy: D=3 ([x, y, u_BC])，Plate: D=4 ([x, y, u_load, v_load])。
        par_flag (torch.Tensor): 有效性掩码，形状 (B, M')，用于仅统计有效点。
        stats (dict | None): 预计算的统计量（用于验证/测试集保持一致变换）。
            若为 None，从当前 par 中估计（仅限训练时调用）。
        method (str): 归一化策略：
            - "none"    : 不做归一化（原始行为，基线）。
            - "minmax"  : 各维度映射到 [0, 1]；仅基于有效点计算 min/max。
            - "zscore"  : 各维度减均值除标准差；仅基于有效点计算均值/标准差。

    Returns:
        par_normalized (torch.Tensor): 归一化后的参数，与 par 同形状。
            padding 位置（par_flag=0）的值保持为 0（或原始填充值），不影响结果。
        stats (dict): 归一化统计量，供验证/测试时使用，键名因 method 而异。

    Raises:
        ValueError: 若 method 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> par_norm, stats = normalize_parameter_inputs(par_train, par_flag_train, method="zscore")
        >>> par_val_norm, _  = normalize_parameter_inputs(par_val,   par_flag_val,   stats=stats)
        >>> # 在 generate_*_data_loader 的参数组织步骤后调用，替换直接使用 parpv
    """
    raise NotImplementedError(
        "C3: normalize_parameter_inputs 尚未实现。"
        "注意仅对有效点（par_flag=1）计算统计量，padding 位置不应影响均值/方差估计。"
        "建议用 par[par_flag.bool()] 提取有效点后计算统计量。"
    )


# ---------------------------------------------------------------------------
# C4 — 参数点间交互
# ---------------------------------------------------------------------------

def build_parameter_interaction_layer(
    fc_dim: int,
    interaction_type: Literal["none", "self_attention", "cross_attention"] = "none",
    n_heads: int = 4,
) -> nn.Module:
    """构建参数点之间的交互层，捕获多参数点的联合分布信息。

    当前代码（model_darcy.py:236-238, model_plate.py:278-280）中各参数点
    独立编码后直接池化，参数点之间无任何交互。当边界条件由多个点共同描述时
    （如 Plate 的加载边界有多个施力点），点与点之间的空间关系和联合分布
    信息在独立编码阶段完全丢失。

    Args:
        fc_dim (int): 参数点特征维度（与 branch 输出维度对应）。
        interaction_type (str): 交互机制：
            - "none"             : 恒等映射（原始行为，基线）。
            - "self_attention"   : 多头自注意力，参数点互相关注；
                                   适用于参数点数量较少（M' < 100）的场景。
            - "cross_attention"  : 参数点与几何点间的交叉注意力；
                                   需要在 forward 时同时传入几何编码（见 C6）。
        n_heads (int): 注意力头数（interaction_type 非 "none" 时有效）。

    Returns:
        interaction_layer (nn.Module): 接受 (B, M', fc_dim) 输入，
            输出 (B, M', fc_dim) 的交互特征，形状不变。

    Raises:
        ValueError: 若 interaction_type 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> self.par_interaction = build_parameter_interaction_layer(
        ...     fc_dim=2*config['model']['fc_dim'], interaction_type="self_attention", n_heads=4
        ... )
        >>> # 在 PI_GANO.forward 中：enc = self.branch(par); enc = self.par_interaction(enc)
        >>> # 再调用 aggregate_parameter_features
    """
    raise NotImplementedError(
        "C4: build_parameter_interaction_layer 尚未实现。"
        "self_attention 可直接复用 nn.TransformerEncoderLayer，"
        "需将 par_flag=0 的位置传入 key_padding_mask。"
    )


# ---------------------------------------------------------------------------
# C5 — 参数编码器最后一层激活
# ---------------------------------------------------------------------------

def build_parameter_encoder_output_layer(
    fc_dim: int,
    out_dim: int,
    output_activation: Optional[nn.Module] = None,
) -> nn.Module:
    """构建参数编码器的最后输出层，支持可选的输出激活函数。

    当前代码（model_darcy.py:198, model_plate.py:220）最后一层为裸 Linear，
    无激活，与几何编码器（B3）存在完全相同的问题：输出无界可能导致后续
    调制操作（E1 中的 u*enc）出现数值不稳定。

    Args:
        fc_dim (int): 输入维度（倒数第二层的输出宽度）。
        out_dim (int): 输出维度，通常为 2*fc_dim（参数编码器宽度）。
        output_activation (nn.Module | None): 输出激活函数。
            - None     : 裸线性输出（原始行为）。
            - nn.Tanh(): 限制到 (-1, 1)，与解码器调制层（E1）的数值尺度匹配。

    Returns:
        output_layer (nn.Module): Linear + 可选激活，
            接受 (..., fc_dim)，输出 (..., out_dim)。

    Raises:
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> out_layer = build_parameter_encoder_output_layer(2*fc_dim, 2*fc_dim, nn.Tanh())
        >>> # 替换 model_darcy.py:198 中的 nn.Linear(2*fc_dim, 2*fc_dim)
    """
    raise NotImplementedError(
        "C5: build_parameter_encoder_output_layer 尚未实现。"
        "与 B3 接口完全对称，实现后可考虑合并为单一通用函数。"
    )


# ---------------------------------------------------------------------------
# C6 — 几何与参数信息交互
# ---------------------------------------------------------------------------

def build_geometry_parameter_cross_encoder(
    geo_dim: int,
    par_dim: int,
    out_dim: int,
    n_heads: int = 4,
    n_cross_layers: int = 1,
) -> nn.Module:
    """构建几何与参数编码之间的交叉注意力层，实现两路信息的早期融合。

    当前代码（model_darcy.py:224-238, model_plate.py:264-280）中几何编码（DG）
    和参数编码（branch）完全独立，两者之间无任何信息交换，最终仅在坐标提升
    后通过 Concatenation 做晚期融合。

    在物理上，几何形状与边界条件密切相关（不同几何域上相同 BC 值的影响完全
    不同）。早期交互使参数编码能感知几何上下文，几何编码也能感知参数分布，
    有望提升两路特征的语义对齐程度。

    Args:
        geo_dim (int): 几何全局嵌入的维度（DG 输出，通常为 fc_dim）。
        par_dim (int): 参数全局嵌入的维度（branch 输出，通常为 2*fc_dim）。
        out_dim (int): 交叉融合后的输出维度。建议 = max(geo_dim, par_dim)。
        n_heads (int): 交叉注意力头数，geo_dim 和 par_dim 需能被 n_heads 整除。
        n_cross_layers (int): 堆叠的交叉注意力层数，默认 1 层。

    Returns:
        cross_encoder (nn.Module): 接受
            - geo_enc (torch.Tensor): 几何全局嵌入，形状 (B, 1, geo_dim)。
            - par_enc (torch.Tensor): 参数全局嵌入，形状 (B, 1, par_dim)。
            输出 fused_enc (torch.Tensor): 融合后嵌入，形状 (B, 1, out_dim)，
            替换原始代码中拼接前的 enc 变量。

    Raises:
        ValueError: 若 n_heads 不能整除 geo_dim 或 par_dim。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> self.cross_enc = build_geometry_parameter_cross_encoder(
        ...     geo_dim=fc_dim, par_dim=2*fc_dim, out_dim=2*fc_dim, n_heads=4
        ... )
        >>> # 在 PI_GANO.forward 中，得到 Domain_enc 和 enc（参数全局）后：
        >>> # enc = self.cross_enc(Domain_enc, enc)
        >>> # 替换 model_darcy.py:233 的 xy_global = torch.cat((xy_local, Domain_enc...), -1)
    """
    raise NotImplementedError(
        "C6: build_geometry_parameter_cross_encoder 尚未实现。"
        "可基于 nn.MultiheadAttention 实现：以 geo_enc 为 query，par_enc 为 key/value，"
        "输出更新后的 geo_enc；同理反向用 par_enc 查询 geo_enc。"
        "双向交叉注意力后拼接/相加，再经一层 FFN 投影到 out_dim。"
    )
