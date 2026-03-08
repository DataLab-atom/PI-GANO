"""
model_darcy_optimizable.py

本文件集中存放 model_darcy.py 中 DG 和 PI_GANO 两个类的所有可优化模块的当前实现。
每个函数封装一个独立的优化项（B1-B6, C1-C5, D1-D3, E1-E8），
保持与原代码等价的行为，以便后续直接替换函数实现来改进对应指标。

注意：本文件仅覆盖 PI-GANO 主体架构（DG + PI_GANO）。
      baseline 类（PI_DCON, PI_PN, PI_GANO_add 等）不在重构范围内。

优化项对应关系：
  B1 - build_geometry_mlp              : 几何编码器 MLP 结构（等宽，无残差）
  B2 - (内嵌于 build_geometry_mlp)     : 几何编码器最后一层无激活
  B3 - aggregate_geometry_features     : 几何特征集合聚合（均值池化）
  B4 - apply_geometry_interaction      : 几何点间交互（当前无操作）
  B5 - (内嵌于 build_geometry_mlp)     : 几何编码器无正则化
  B6 - encode_geometry_positions       : 几何点位置编码（当前直接透传）
  C1 - build_parameter_mlp             : 参数编码器 MLP 结构（等宽，无残差，无正则化）
  C2 - aggregate_parameter_features    : 参数特征集合聚合（amax）
  C3 - apply_parameter_interaction     : 参数点间交互（当前无操作）
  C4 - (内嵌于 build_parameter_mlp)    : 参数编码器最后一层无激活
  C5 - cross_attend_geo_param          : 几何与参数信息交叉注意力（当前无操作）
  D1 - build_coordinate_encoder        : 坐标编码层（当前为单层线性）
  D3 - inject_global_info              : 域全局信息注入方式（当前为 Concat）
  E1 - modulate_features               : Branch 调制机制（当前为乘法，FiLM 退化版）
  E3 - apply_residual                  : 残差连接（当前无操作）
  E4 - aggregate_output                : 输出聚合方式（当前为内积 mean）
  E5 - create_activation               : 全网络激活函数（当前为 Tanh）
  E6 - build_prediction_head_block     : 预测头单个 FC 块（当前无正则化）
  E7 - build_prediction_head_layers    : 预测头所有层（当前层数硬编码）
"""

from typing import Tuple, List
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════
#  B 模块 — 几何编码器（DG 类）
# ═══════════════════════════════════════════════════════════

def build_geometry_mlp(
    in_dim: int,
    fc_dim: int,
    n_layer: int,
) -> nn.Sequential:
    """[B1/B2/B5] 构建几何编码器（DG）的 MLP 主体网络。

    覆盖三个优化项：
      B1: 各隐层宽度固定为 fc_dim（等宽结构），无逐渐收缩/扩张，无残差连接。
      B2: 最后一层为纯 Linear，无激活函数，输出无界，与下游融合层
          可能产生数值不稳定。
      B5: 无任何正则化手段（无 Dropout、无 LayerNorm），小数据集下易过拟合。

    优化方向（B1）：改为逐渐收缩或扩张的宽度，或引入残差块。
    优化方向（B2）：在最后一层 Linear 后追加激活函数（如 Tanh/GELU）。
    优化方向（B5）：在隐层中加入 Dropout 或 LayerNorm。

    注意：若 B6 (encode_geometry_positions) 改变输入维度，
          需同步修改此函数的 in_dim 参数。

    Args:
        in_dim  (int): 输入特征维度，等于几何坐标编码后的维度
                       （B6 透传时为 2，即原始 [x, y]）。
        fc_dim  (int): 隐层宽度（来自 config['model']['fc_dim']）。
        n_layer (int): 隐层数量（来自 config['model']['N_layer']）。

    Returns:
        nn.Sequential: 几何编码器 MLP，
                       输入形状 (B, M'', in_dim)，输出形状 (B, M'', fc_dim)。
                       当前实现：等宽 MLP + 中间层 Tanh + 最后层无激活。
    """
    layers: List[nn.Module] = [nn.Linear(in_dim, fc_dim), nn.Tanh()]
    for _ in range(n_layer - 1):
        layers.append(nn.Linear(fc_dim, fc_dim))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(fc_dim, fc_dim))   # B2: 最后一层无激活
    return nn.Sequential(*layers)


def encode_geometry_positions(shape_coor: torch.Tensor) -> torch.Tensor:
    """[B6] 对几何边界点坐标施加位置编码，再输入 MLP。

    直接使用原始坐标 [x, y] 作为 MLP 输入，缺少位置编码，
    对具有周期性边界或复杂边界曲率的几何的表达能力不足。

    优化方向：用 Random Fourier Features 或正弦/余弦位置编码将 [x, y]
              映射到高维空间，改善 MLP 对空间频率的感知能力。
              注意：修改此函数后输出维度会改变，需同步修改
              build_geometry_mlp 的 in_dim 参数。

    Args:
        shape_coor (torch.Tensor): 几何边界点坐标，形状 (B, M'', 2)，
                                   每个点为 [x, y]。

    Returns:
        torch.Tensor: 位置编码后的特征，形状 (B, M'', D_enc)。
                      当前实现直接透传，D_enc = 2，无任何变换。
    """
    return shape_coor


def apply_geometry_interaction(
    enc: torch.Tensor,
    shape_flag: torch.Tensor,
) -> torch.Tensor:
    """[B4] 在几何点特征序列上执行点间交互操作。

    各边界点独立经过 MLP 后直接聚合，点与点之间无任何交互，
    相邻点的几何曲率信息（方向变化、曲率符号）无法被捕获。

    优化方向：在 MLP 编码和聚合之间插入多头自注意力层或图卷积层，
              允许各点感知邻域几何结构。

    Args:
        enc        (torch.Tensor): MLP 编码后的逐点特征，形状 (B, M'', F)。
        shape_flag (torch.Tensor): 几何点有效性掩码，形状 (B, M'')，
                                   1 为有效节点，0 为 padding。

    Returns:
        torch.Tensor: 经点间交互后的逐点特征，形状 (B, M'', F)。
                      当前实现无操作，直接返回 enc。
    """
    return enc


def aggregate_geometry_features(
    enc: torch.Tensor,
    shape_flag: torch.Tensor,
) -> torch.Tensor:
    """[B3] 将逐点几何特征聚合为单一域嵌入向量。

    当前使用 masked 均值池化：sum(enc * flag) / sum(flag)。
    均值对几何局部结构（尖角、凹陷）不敏感，且对点密度不均匀有偏。

    优化方向：改用最大池化、注意力加权池化或多尺度聚合，
              以更好地保留局部几何特征。

    Args:
        enc        (torch.Tensor): 逐点几何特征，形状 (B, M'', F)。
        shape_flag (torch.Tensor): 几何点有效性掩码，形状 (B, M'')，
                                   1 为有效节点，0 为 padding。

    Returns:
        torch.Tensor: 聚合后的域嵌入，形状 (B, 1, F)。
                      当前实现：masked 均值池化。
    """
    enc_masked = enc * shape_flag.unsqueeze(-1)                                         # (B, M'', F)
    domain_enc = (
        torch.sum(enc_masked, dim=1, keepdim=True)
        / torch.sum(shape_flag.unsqueeze(-1), dim=1, keepdim=True)
    )                                                                                   # (B, 1, F)
    return domain_enc


# ═══════════════════════════════════════════════════════════
#  C 模块 — 参数编码器（PI_GANO.branch）
# ═══════════════════════════════════════════════════════════

def build_parameter_mlp(
    in_dim: int,
    fc_dim: int,
    n_layer: int,
) -> nn.Sequential:
    """[C1/C4] 构建参数编码器（Branch）的 MLP 主体网络。

    覆盖两个优化项：
      C1: 等宽 MLP，无残差连接，无正则化（Dropout/LayerNorm）。
          参数编码器输出直接调制预测头，结构缺陷传导至所有下游指标。
      C4: 最后一层为纯 Linear，无激活函数，输出无界，
          与几何编码器 B2 同类问题，影响调制层数值稳定性。

    优化方向（C1）：引入残差块；添加 Dropout 或 LayerNorm 正则化。
    优化方向（C4）：在最后 Linear 后追加激活函数。

    Args:
        in_dim  (int): 输入特征维度，Darcy 中为 3（[x, y, u_BC]），
                       Plate 中为 4（[x, y, u_load, v_load]）。
        fc_dim  (int): 隐层宽度，当前为 2 * config['model']['fc_dim']。
        n_layer (int): 隐层数量（来自 config['model']['N_layer']）。

    Returns:
        nn.Sequential: 参数编码器 MLP，
                       输入形状 (B, M', in_dim)，输出形状 (B, M', fc_dim)。
                       当前实现：等宽 MLP + 中间层 Tanh + 最后层无激活。
    """
    layers: List[nn.Module] = [nn.Linear(in_dim, fc_dim), nn.Tanh()]
    for _ in range(n_layer - 1):
        layers.append(nn.Linear(fc_dim, fc_dim))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(fc_dim, fc_dim))   # C4: 最后一层无激活
    return nn.Sequential(*layers)


def apply_parameter_interaction(
    enc: torch.Tensor,
    par_flag: torch.Tensor,
) -> torch.Tensor:
    """[C3] 在参数点特征序列上执行点间交互操作。

    各参数点（边界条件节点）独立经过 MLP 后直接聚合，点间无交互，
    多参数点的联合分布信息（例如加载分布的整体形状）无法被捕获。

    优化方向：插入多头自注意力或图卷积，使各参数点能感知其在
              边界上的相对位置和全局加载分布。

    Args:
        enc      (torch.Tensor): MLP 编码后的逐点参数特征，形状 (B, M', F)。
        par_flag (torch.Tensor): 参数点有效性掩码，形状 (B, M')，
                                  1 为有效节点，0 为 padding。

    Returns:
        torch.Tensor: 经点间交互后的逐点参数特征，形状 (B, M', F)。
                      当前实现无操作，直接返回 enc。
    """
    return enc


def aggregate_parameter_features(
    enc: torch.Tensor,
    par_flag: torch.Tensor,
) -> torch.Tensor:
    """[C2] 将逐点参数特征聚合为单一参数嵌入向量（Branch 全局特征）。

    当前使用 torch.amax(enc_masked, dim=1)（逐维度独立取最大）。
    不同维度的最大值可能来自不同参数点，破坏特征维度间的协同关系，
    且对加载分布的全局统计（均值、分布形状）不敏感。

    优化方向：改用注意力加权池化、均值池化或 DeepSets 式聚合，
              保留参数点间的联合分布信息。

    Args:
        enc      (torch.Tensor): 逐点参数特征（交互后），形状 (B, M', F)。
        par_flag (torch.Tensor): 参数点有效性掩码，形状 (B, M')，
                                  1 为有效节点，0 为 padding。

    Returns:
        torch.Tensor: 聚合后的参数嵌入，形状 (B, 1, F)。
                      当前实现：masked amax（逐维度取最大），keepdim=True。
    """
    enc_masked = enc * par_flag.unsqueeze(-1)                   # (B, M', F)
    return torch.amax(enc_masked, dim=1, keepdim=True)          # (B, 1, F)


def cross_attend_geo_param(
    domain_enc: torch.Tensor,
    param_enc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """[C5] 在几何嵌入与参数嵌入之间执行交叉注意力（信息交互）。

    DG（几何）和 Branch（参数）完全独立编码，两者在融合层之前
    没有任何信息交换，无法让几何感知参数分布、参数感知几何边界。

    优化方向：引入交叉注意力机制，使几何嵌入能查询参数分布信息，
              参数嵌入能查询几何结构信息，提升联合表达能力。

    Args:
        domain_enc (torch.Tensor): 几何域嵌入，形状 (B, 1, F)。
        param_enc  (torch.Tensor): 参数聚合嵌入，形状 (B, 1, F)。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (domain_enc_updated, param_enc_updated)，形状均为 (B, 1, F)。
            当前实现无操作，直接返回输入（无交叉注意力）。
    """
    return domain_enc, param_enc


# ═══════════════════════════════════════════════════════════
#  D 模块 — 坐标编码（PI_GANO.xy_lift）
# ═══════════════════════════════════════════════════════════

def build_coordinate_encoder(fc_dim: int) -> nn.Module:
    """[D1] 构建查询坐标编码层（xy_lift），将 (x, y) 映射到特征空间。

    当前实现为单层线性变换 Linear(2, fc_dim)，无激活函数，
    等价于线性投影，对高频空间变化（边界层、应力集中区）
    表达能力严重不足。

    优化方向：
      (1) 替换为多层 MLP + 激活函数，增强非线性表达能力；
      (2) 使用 Random Fourier Features 将坐标映射到高维三角函数空间，
          已被证明对 PINN/PINO 高频解拟合有显著提升。
    注意：修改此函数后输出维度可能改变，需同步修改 inject_global_info
          和下游 FC 层的输入维度。

    Args:
        fc_dim (int): 输出特征维度（来自 config['model']['fc_dim']）。

    Returns:
        nn.Module: 坐标编码模块，
                   输入形状 (B, M, 2)，输出形状 (B, M, fc_dim)。
                   当前实现：单层 Linear(2, fc_dim)，无激活。
    """
    return nn.Linear(2, fc_dim)


def inject_global_info(
    xy_local: torch.Tensor,
    domain_enc: torch.Tensor,
    n_points: int,
) -> torch.Tensor:
    """[D3] 将域全局嵌入（几何信息）注入到局部坐标特征中。

    当前使用 Concatenation：将 xy_local (B, M, F) 与 domain_enc (B, 1, F)
    在特征维度拼接，得到 (B, M, 2F)。
    Concatenation 保留局部和全局特征的独立性，但参数量翻倍（下游 FC 输入增大）。
    项目中已实现 Add（PI_GANO_add）和 Mul（PI_GANO_mul）变体，但未系统评估。

    优化方向：系统评估 Add / Mul / Concat 三种注入方式对 M7/M8 的影响；
              可进一步探索门控注入（Gating）或 FiLM-style 注入。

    Args:
        xy_local   (torch.Tensor): 局部坐标特征，形状 (B, M, F)，
                                    由 coordinate_encoder 输出。
        domain_enc (torch.Tensor): 域全局嵌入，形状 (B, 1, F)，
                                    由几何编码器聚合输出。
        n_points   (int): 查询点数 M，用于广播 domain_enc。

    Returns:
        torch.Tensor: 注入全局信息后的坐标特征，形状 (B, M, 2F)。
                      当前实现：Concat(xy_local, domain_enc.expand(B, M, F))。
    """
    return torch.cat((xy_local, domain_enc.repeat(1, n_points, 1)), dim=-1)    # (B, M, 2F)


# ═══════════════════════════════════════════════════════════
#  E 模块 — 算子逼近（PI_GANO 预测头）
# ═══════════════════════════════════════════════════════════

def create_activation() -> nn.Module:
    """[E5] 创建全网络共用的激活函数模块。

    整个网络（几何编码器 MLP 中间层 + 参数编码器中间层 + 预测头 FC 层）
    统一使用 Tanh 激活。深层 Tanh 在输入较大时易饱和，导致梯度消失，
    对非平滑解（大梯度区域）的拟合能力弱于 GELU/SiLU。

    优化方向：将 Tanh 替换为 GELU 或 SiLU，或按模块分别配置激活函数。

    Returns:
        nn.Module: 激活函数模块实例。
                   当前实现返回 nn.Tanh()。
    """
    return nn.Tanh()


def build_prediction_head_block(
    in_dim: int,
    out_dim: int,
) -> nn.Linear:
    """[E6] 构建预测头中的单个全连接层（不含激活）。

    预测头 FC1~FC4 之间无 Dropout、LayerNorm、BatchNorm 等正则化手段，
    在小样本场景（训练集 < 100 样本）下泛化能力受限。

    优化方向：返回 nn.Sequential(nn.Linear(...), nn.LayerNorm(...)) 或
              在层后追加 Dropout，增强正则化效果。

    Args:
        in_dim  (int): 输入特征维度。
        out_dim (int): 输出特征维度。

    Returns:
        nn.Linear: 纯线性层，无正则化。
                   当前实现：nn.Linear(in_dim, out_dim)，无任何正则化。
    """
    return nn.Linear(in_dim, out_dim)


def build_prediction_head_layers(
    fc_dim: int,
) -> Tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]:
    """[E7] 构建 Darcy PI_GANO 预测头的全部 FC 层（FC1u ~ FC4u）。

    预测头层数（Darcy: 3 个隐层 + 1 个输出层）硬编码在 PI_GANO.__init__ 中，
    无法通过配置文件调整；fc_dim 也硬编码为 2*fc_dim。

    优化方向：将层数和宽度提升为可配置参数（通过 config['model'] 读取），
              支持更灵活的网络深度搜索。

    Args:
        fc_dim (int): 基础特征维度（来自 config['model']['fc_dim']）。
                      预测头实际维度为 2 * fc_dim（已与域嵌入拼接）。

    Returns:
        Tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]:
            (FC1u, FC2u, FC3u, FC4u)，维度均为 (2*fc_dim → 2*fc_dim)，
            最后一层 FC4u 为 (2*fc_dim → 1)（注意原代码 FC4u 实际为该尺寸）。
    """
    d = 2 * fc_dim
    FC1u = build_prediction_head_block(d, d)
    FC2u = build_prediction_head_block(d, d)
    FC3u = build_prediction_head_block(d, d)
    FC4u = build_prediction_head_block(d, 1)
    return FC1u, FC2u, FC3u, FC4u


def modulate_features(
    u: torch.Tensor,
    enc: torch.Tensor,
) -> torch.Tensor:
    """[E1] 用参数嵌入（Branch 输出）对坐标特征施加调制（FiLM 退化版）。

    当前实现仅做逐元素乘法 u = u * enc，为完整 FiLM（Feature-wise Linear Modulation）
    的退化版本（γ * u，缺少偏置项 β）。完整 FiLM 为 γ * u + β，
    可为每个特征维度独立学习缩放和平移，表达能力更强。

    优化方向：引入额外的 bias 线性层，将 enc 映射为 (γ, β)，
              执行 γ * u + β，实现完整 FiLM 调制。

    Args:
        u   (torch.Tensor): 当前预测特征，形状 (B, M, F)。
        enc (torch.Tensor): 参数嵌入（聚合后），形状 (B, 1, F)，
                             会自动广播到 (B, M, F)。

    Returns:
        torch.Tensor: 调制后的特征，形状 (B, M, F)。
                      当前实现：u * enc（仅缩放，无偏置）。
    """
    return u * enc


def apply_residual(
    u_new: torch.Tensor,
    u_skip: torch.Tensor,
) -> torch.Tensor:
    """[E3] 在预测头 FC 层之间添加残差（跳跃）连接。

    预测头 FC1~FC4 之间当前无任何残差连接，深层网络梯度回传路径
    全部通过非线性层，梯度消失风险随深度增加。

    优化方向：在每两个 FC 层之间加入 u_new + u_skip 的残差连接，
              为梯度提供直接回传通道，改善深层网络训练稳定性。

    Args:
        u_new  (torch.Tensor): 当前 FC 层输出，形状 (B, M, F)。
        u_skip (torch.Tensor): 跳跃连接来源（上一个残差块输入），形状 (B, M, F)。

    Returns:
        torch.Tensor: 残差连接后的特征，形状 (B, M, F)。
                      当前实现无操作，直接返回 u_new（忽略 u_skip）。
    """
    return u_new


def aggregate_output(
    u: torch.Tensor,
    enc: torch.Tensor,
) -> torch.Tensor:
    """[E4] 将最终隐层特征聚合为标量输出（预测值）。

    当前使用 torch.mean(u * enc, dim=-1)，即 u 与 enc 的逐元素乘积
    在特征维度上取均值，本质上是一种内积聚合。
    模型中 FC5u 已定义但从未被调用（模型参数浪费），
    当前的内积聚合也未充分利用已有的线性映射能力。

    优化方向：改用 FC5u（nn.Linear(2*fc_dim, 1)）输出最终预测值，
              使输出层有可学习参数（而非固定的均值内积操作）。

    Args:
        u   (torch.Tensor): 最终隐层特征，形状 (B, M, F)。
        enc (torch.Tensor): 参数嵌入（聚合后），形状 (B, 1, F)，
                             会自动广播到 (B, M, F)。

    Returns:
        torch.Tensor: 逐节点标量预测值，形状 (B, M)。
                      当前实现：mean(u * enc, dim=-1)（内积均值）。
    """
    return torch.mean(u * enc, dim=-1)
