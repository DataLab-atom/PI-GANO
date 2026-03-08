"""
D_coordinate_encoder.py — 坐标编码可优化模块 (D1-D5)

对应 optimizable_modules.md § D 节，共 5 个优化点。
本模块提供替换 model_darcy.py / model_plate.py 中 xy_lift / xy_lift1 / xy_lift2
及特征融合逻辑的工厂函数。

当前坐标处理流程：
    xy_lift  = Linear(2 → F)              # 单层线性提升，无激活
    xy_local = xy_lift(xy)               # (B, M, F)
    xy_global = cat(xy_local, Domain_enc.repeat(...)) → (B, M, 2F)
    u = FC1u(xy_global) → ...            # 后续解码器以 xy_global 为输入
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# D1 — 坐标提升层表达能力
# ---------------------------------------------------------------------------

def build_coordinate_lift_network(
    in_dim: int,
    out_dim: int,
    n_layer: int = 1,
    activation: Optional[nn.Module] = None,
    architecture: Literal["linear", "mlp", "mlp_residual"] = "linear",
) -> nn.Module:
    """构建查询坐标的特征提升网络，替换当前单层无激活的 Linear。

    当前代码（model_darcy.py:202, model_plate.py:224-225）中 xy_lift 为单层
    Linear(2→F)，等价于线性变换，对高频空间变化（边界层、应力集中区域）的
    拟合能力严重不足。对于物理上存在高梯度区域的 PDE 解，线性坐标提升无法
    为解码器提供足够的空间表达能力。

    Args:
        in_dim (int): 输入维度，原始坐标为 2；若已过 Fourier 编码（D2）
            则为编码后维度（2*n_freq + 2 等）。
        out_dim (int): 输出维度，需与后续融合层（D5）的维度要求匹配：
            - 若使用 Concat 融合，out_dim = fc_dim（下游 cat 后为 2*fc_dim）。
            - 若使用 Add/Mul 融合，out_dim = 2*fc_dim（与 Domain_enc 同维度）。
        n_layer (int): 网络层数，1 表示单层（原始行为）。
        activation (nn.Module | None): 层间激活函数。None = 无激活（原始行为，
            即线性变换）；nn.GELU() 等激活函数可显著提升非线性表达能力。
        architecture (str): 网络拓扑：
            - "linear"      : 单层 Linear，无激活（原始行为）。
            - "mlp"         : 多层 MLP，层间有激活。
            - "mlp_residual": 多层 MLP + 残差连接（需 in_dim == out_dim 或添加投影）。

    Returns:
        lift_net (nn.Module): 坐标提升网络，接受 (B, M, in_dim)，
            输出 (B, M, out_dim)。

    Raises:
        ValueError: 若 architecture="mlp_residual" 且 in_dim != out_dim 且未能自动添加投影。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> self.xy_lift = build_coordinate_lift_network(
        ...     in_dim=2, out_dim=config['model']['fc_dim'],
        ...     n_layer=2, activation=nn.GELU(), architecture="mlp"
        ... )
        >>> # 替换 model_darcy.py:202 中的 self.xy_lift = nn.Linear(2, fc_dim)
    """
    raise NotImplementedError(
        "D1: build_coordinate_lift_network 尚未实现。"
        "建议首先对比 architecture='linear'（基线）和 'mlp'（n_layer=2, GELU）。"
        "若使用 D2 的 Fourier 编码，本函数的 in_dim 需改为 fourier_encode_coordinates 的 out_dim。"
    )


# ---------------------------------------------------------------------------
# D2 — 坐标 Fourier 特征
# ---------------------------------------------------------------------------

def fourier_encode_coordinates(
    xy: torch.Tensor,
    n_freq: int = 16,
    scale: float = 1.0,
    freq_matrix: Optional[torch.Tensor] = None,
    learnable: bool = False,
    include_original: bool = True,
) -> Tuple[torch.Tensor, int]:
    """对查询坐标 (x, y) 应用 Fourier 特征编码，提升对高频 PDE 解的拟合能力。

    当前代码（model_darcy.py:202, model_plate.py:224-225）坐标直接输入线性提升层，
    未使用任何位置编码。Fourier 特征（Random Fourier Features / Positional Encoding）
    已在多篇 PINN/PINO 论文中被证明可显著提升对高频解（如薄边界层、应力集中）的
    拟合能力，原因是消除了 MLP 的 spectral bias（低频偏向性）。

    与 B7（几何坐标编码）的区别：
        - B7 作用于几何编码器的输入（边界点坐标，仅影响 Domain_enc）。
        - D2 作用于解码器的查询坐标（所有 PDE 点 + BC 点，影响最终预测）。

    Args:
        xy (torch.Tensor): 原始查询坐标，形状 (B, M, 2)。
        n_freq (int): 随机频率数，编码后维度 = 2*n_freq（+ 2 若 include_original）。
            建议范围 [8, 64]，过大增加计算量且无额外收益。
        scale (float): 频率随机矩阵的标准差，控制采样频率范围。
            需与坐标值域和 PDE 解的特征频率匹配，建议从 1.0 网格搜索。
        freq_matrix (torch.Tensor | None): 预定义的频率矩阵，形状 (n_freq, 2)。
            若为 None，按 N(0, scale²) 随机初始化。训练集和验证集必须共用同一矩阵。
        learnable (bool): 是否将 freq_matrix 设为可学习参数（nn.Parameter）。
            True = 端对端学习频率；False = 固定随机矩阵（标准 RFF 设定）。
        include_original (bool): 是否在编码结果中拼接原始坐标 (x, y)，
            有助于保留绝对位置信息，默认 True。

    Returns:
        xy_encoded (torch.Tensor): Fourier 编码后的坐标特征，
            形状 (B, M, out_dim)，其中 out_dim = 2*n_freq + (2 if include_original else 0)。
        out_dim (int): 输出维度，用于设置 xy_lift（D1）的 in_dim。

    Raises:
        ValueError: 若 freq_matrix 形状与 n_freq 不匹配。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Note:
        freq_matrix 应在模型 __init__ 中生成并通过 register_buffer 注册，
        避免每次 forward 重新随机初始化导致训练不稳定。

    Example:
        >>> # 在 PI_GANO.__init__ 中：
        >>> self.register_buffer('freq_matrix', torch.randn(16, 2) * scale)
        >>> # 在 PI_GANO.forward 中：
        >>> xy_enc, _ = fourier_encode_coordinates(xy, freq_matrix=self.freq_matrix)
        >>> xy_local = self.xy_lift(xy_enc)   # xy_lift 的 in_dim 需用 D2 的 out_dim
        >>> # 替换 model_darcy.py:230 中的 xy_local = self.xy_lift(xy)
    """
    raise NotImplementedError(
        "D2: fourier_encode_coordinates 尚未实现。"
        "核心公式：xy_encoded = cat([cos(xy @ B.T), sin(xy @ B.T)], dim=-1)。"
        "注意 B 需按列广播：xy 形状 (B,M,2)，B 形状 (n_freq,2)，"
        "torch.einsum('bmd,fd->bmf', xy, B) 可高效实现。"
    )


# ---------------------------------------------------------------------------
# D3 — 坐标归一化
# ---------------------------------------------------------------------------

def normalize_query_coordinates(
    xy: torch.Tensor,
    method: Literal["minmax", "zscore", "unit_interval", "none"] = "none",
    stats: Optional[dict] = None,
) -> Tuple[torch.Tensor, dict]:
    """将查询坐标归一化到统一值域，提升不同几何尺度下的训练稳定性。

    当前代码（utils_darcy_train.py:234-235, utils_plate_train.py:305-306）中
    查询坐标未归一化，直接传入 Variable 后做自动微分。在不同几何域尺度下
    （如单位正方形 vs 2×3 矩形），坐标值域不一致导致网络对不同样本的梯度
    尺度不均匀，训练过程不稳定。

    注意：坐标归一化会影响 PDE 残差计算中的偏导数链式法则，自动微分无需手动
    修正（PyTorch 的 autograd 会正确处理变量变换），但解释 PDE 结果时需
    反归一化坐标。

    Args:
        xy (torch.Tensor): 查询坐标张量，形状 (B, M, 2) 或 (B, M)（单轴）。
            通常在 training loop 中对 pde_sampled_coors 和 bc_coors 分别调用。
        method (str): 归一化策略：
            - "none"          : 不做归一化（原始行为）。
            - "minmax"        : 按批次内各样本分别映射到 [0, 1]。
            - "zscore"        : 减均值除标准差（跨样本统计）。
            - "unit_interval" : 按几何域的已知边界（如 [0,1]²）做固定归一化。
        stats (dict | None): 预计算统计量（如使用 "unit_interval"，则包含
            {"x_min": ..., "x_max": ..., "y_min": ..., "y_max": ...}）；
            None 时从输入估计（仅适合 "minmax"/"zscore"）。

    Returns:
        xy_normalized (torch.Tensor): 归一化后的坐标，与 xy 同形状。
        stats (dict): 本次归一化使用的统计量，供对应 BC 坐标做相同变换。

    Raises:
        ValueError: 若 method 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> xy_pde = torch.stack([x_pde, y_pde], dim=-1)
        >>> xy_pde_norm, stats = normalize_query_coordinates(xy_pde, method="unit_interval",
        ...                                                   stats=domain_stats)
        >>> x_pde_norm = xy_pde_norm[..., 0]
        >>> y_pde_norm = xy_pde_norm[..., 1]
        >>> # 替换 utils_darcy_train.py:234-235 的原始坐标使用
    """
    raise NotImplementedError(
        "D3: normalize_query_coordinates 尚未实现。"
        "关键点：若使用 Variable(requires_grad=True) 做自动微分，"
        "归一化必须在 Variable 创建之前完成，否则计算图断裂。"
        "即先归一化 numpy 坐标 → 再转 tensor → 再 Variable(requires_grad=True)。"
    )


# ---------------------------------------------------------------------------
# D4 — Plate u/v 坐标编码共享
# ---------------------------------------------------------------------------

def build_shared_or_split_coordinate_lift(
    in_dim: int,
    out_dim: int,
    share_weights: bool = False,
    activation: Optional[nn.Module] = None,
) -> Tuple[nn.Module, nn.Module]:
    """为 Plate 模型的 u、v 分量创建坐标提升层，支持权重共享实验。

    当前代码（model_plate.py:224-225）中 xy_lift1（用于 u）和 xy_lift2（用于 v）
    为独立权重的两个 Linear 层，参数量翻倍。u 和 v 在物理上通过应力-应变
    关系强耦合，独立编码引入了不必要的自由度，且无法利用两者的物理对称性。

    本函数支持共享权重（share_weights=True）和独立权重（False）两种模式，
    便于通过消融实验确定最优方案。

    Args:
        in_dim (int): 输入坐标维度，通常为 2 或 Fourier 编码后维度。
        out_dim (int): 提升后的特征维度（= fc_dim）。
        share_weights (bool): 是否共享 u/v 的坐标编码权重。
            - True  : xy_lift1 和 xy_lift2 为同一 nn.Module 实例（参数量减半）。
            - False : 两者为独立 nn.Module 实例（原始行为）。
        activation (nn.Module | None): 提升层后的激活函数，None = 无激活（原始行为）。

    Returns:
        xy_lift_u (nn.Module): u 分量的坐标提升层。
        xy_lift_v (nn.Module): v 分量的坐标提升层。
            当 share_weights=True 时，xy_lift_u is xy_lift_v（同一对象）。

    Raises:
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> self.xy_lift1, self.xy_lift2 = build_shared_or_split_coordinate_lift(
        ...     in_dim=2, out_dim=config['model']['fc_dim'], share_weights=True
        ... )
        >>> # 替换 model_plate.py:224-225 中的独立 nn.Linear 定义
    """
    raise NotImplementedError(
        "D4: build_shared_or_split_coordinate_lift 尚未实现。"
        "share_weights=True 时只需创建一个 lift 层并返回两次相同引用：return lift, lift。"
        "share_weights=False 时各自独立初始化：return lift_u, lift_v。"
        "消融建议：先在小实验中对比两种模式的验证误差，若无显著差异则共享以节省参数。"
    )


# ---------------------------------------------------------------------------
# D5 — 域全局信息注入方式
# ---------------------------------------------------------------------------

def fuse_local_global_features(
    xy_local: torch.Tensor,
    domain_enc: torch.Tensor,
    mode: Literal["concat", "add", "mul", "film"] = "concat",
    film_gamma: Optional[torch.Tensor] = None,
    film_beta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """将局部坐标特征与全局域嵌入融合，替换当前固定的 Concatenation。

    当前代码（model_darcy.py:233, model_plate.py:274-275）固定使用 Concat：
        xy_global = cat(xy_local, Domain_enc.repeat(1,mD,1))    # (B,M,2F)
    Concat 只是最简单的融合方式，未充分探索其他融合机制的效果。
    项目中已实现 Add（PI_GANO_add）和 Mul（PI_GANO_mul）变体，但仅作实验类，
    未系统评估。本函数将所有融合变体统一接口，便于系统对比实验。

    Args:
        xy_local (torch.Tensor): 局部坐标特征，形状 (B, M, F_local)。
            来自 xy_lift 的输出。
        domain_enc (torch.Tensor): 全局域嵌入，形状 (B, 1, F_global)。
            来自 DG（几何编码器）的输出，需在使用前 repeat 到 (B, M, F_global)。
        mode (str): 融合方式：
            - "concat" : 拼接，输出 (B, M, F_local + F_global)（原始行为）。
                         要求后续 FC 层输入维度为 F_local + F_global。
            - "add"    : 逐元素加法，要求 F_local == F_global，输出 (B, M, F)。
            - "mul"    : 逐元素乘法，要求 F_local == F_global，输出 (B, M, F)。
            - "film"   : FiLM 调制（γ * xy_local + β），γ/β 来自 domain_enc
                         的线性投影；输出 (B, M, F_local)，不改变维度。
                         需提供 film_gamma 和 film_beta（(B, M, F_local) 形状）。
        film_gamma (torch.Tensor | None): FiLM 缩放因子，形状 (B, M, F_local)，
            仅 mode="film" 时需要，由 domain_enc 经线性投影生成。
        film_beta (torch.Tensor | None): FiLM 偏置因子，形状 (B, M, F_local)，
            仅 mode="film" 时需要。

    Returns:
        xy_global (torch.Tensor): 融合后的特征，形状因 mode 而异（见上方说明）。

    Raises:
        ValueError: 若 mode 要求维度匹配但 F_local != F_global。
        ValueError: 若 mode="film" 但未提供 film_gamma 或 film_beta。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> xy_global = fuse_local_global_features(xy_local, domain_enc, mode="add")
        >>> # 替换 model_darcy.py:233 中的 torch.cat((xy_local, Domain_enc...), -1)
        >>> # 注意 mode="add"/"mul" 时需同步修改 xy_lift 的 out_dim 为 2*fc_dim
    """
    raise NotImplementedError(
        "D5: fuse_local_global_features 尚未实现。"
        "FiLM 模式建议在 PI_GANO.__init__ 中额外定义 film_proj 层："
        "nn.Linear(fc_dim, 2*fc_dim)，forward 时将其输出拆分为 gamma 和 beta。"
        "消融实验顺序建议：concat（基线）→ add → mul → film。"
    )
