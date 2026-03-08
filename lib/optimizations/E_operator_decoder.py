"""
E_operator_decoder.py — 算子逼近可优化模块 (E1-E8)

对应 optimizable_modules.md § E 节，共 8 个优化点。
本模块提供替换 model_darcy.py / model_plate.py 中预测头（FC1u~FC5u/v）
及调制机制的工厂函数。

当前预测头结构（Darcy）：
    u = FC1u(xy_global);  act;  u = u * enc
    u = FC2u(u);          act;  u = u * enc
    u = FC3u(u);          act
    u = mean(FC4u(u) * enc, dim=-1)      ← FC5u 已定义但从未调用

当前预测头结构（Plate，u/v 各自独立）：
    u = FC1u(xy_global);  act;  u = u * enc
    u = FC2u(u);          act;  u = u * enc
    u = FC3u(u);          act   # 第3层 u*enc 被注释掉
    u = FC4u(u)                 # 无激活
    u = mean(u * enc, dim=-1)   ← FC5u 已定义但从未调用
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# E1 — Branch 调制机制（FiLM）
# ---------------------------------------------------------------------------

def film_modulate(
    features: torch.Tensor,
    enc: torch.Tensor,
    gamma_proj: Optional[nn.Linear] = None,
    beta_proj: Optional[nn.Linear] = None,
) -> torch.Tensor:
    """对特征施加完整 FiLM 调制（γ * features + β），替换当前只有缩放的退化版。

    当前代码（model_darcy.py:242,244, model_plate.py:284,287）中调制为：
        u = u * enc
    这是 FiLM（Feature-wise Linear Modulation）的退化版：只有逐元素缩放（γ），
    没有偏置（β）。完整 FiLM 为 γ * u + β，额外的 β 允许网络对特征做平移，
    显著扩展了调制的表达能力，在条件生成和元学习任务中已被广泛证明更优。

    Args:
        features (torch.Tensor): 待调制的特征，形状 (B, M, F)。
            来自解码器中间层（如 FC1u 的输出经激活后）。
        enc (torch.Tensor): 条件编码（参数全局嵌入），形状 (B, 1, F)。
            来自 aggregate_parameter_features（C2）的输出。
        gamma_proj (nn.Linear | None): 将 enc 投影为 γ 的线性层，形状 (F → F)。
            若为 None，直接用 enc 作为 γ（原始行为，即 γ = enc，β = 0）。
        beta_proj (nn.Linear | None): 将 enc 投影为 β 的线性层，形状 (F → F)。
            若为 None，β = 0（原始行为）。

    Returns:
        features_modulated (torch.Tensor): 调制后的特征，形状 (B, M, F)。

    Raises:
        ValueError: 若 gamma_proj 或 beta_proj 的输出维度与 features 的 F 不匹配。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> # 在 PI_GANO.__init__ 中添加：
        >>> self.film_gamma = nn.Linear(2*fc_dim, 2*fc_dim)
        >>> self.film_beta  = nn.Linear(2*fc_dim, 2*fc_dim)
        >>> # 在 PI_GANO.forward 中替换 u = u * enc：
        >>> u = film_modulate(u, enc, self.film_gamma, self.film_beta)
        >>> # 替换 model_darcy.py:243,245 和 model_plate.py:285,288 的 u*enc
    """
    raise NotImplementedError(
        "E1: film_modulate 尚未实现。"
        "完整公式：gamma = gamma_proj(enc); beta = beta_proj(enc);"
        "return gamma * features + beta。"
        "注意 enc 形状为 (B,1,F)，需广播到 (B,M,F)。"
        "gamma_proj 和 beta_proj 应在模型 __init__ 中定义，按调制层数各需一组。"
    )


# ---------------------------------------------------------------------------
# E2 — 调制层数一致性
# ---------------------------------------------------------------------------

def apply_modulation_schedule(
    layer_outputs: List[torch.Tensor],
    enc: torch.Tensor,
    modulate_flags: List[bool],
    gamma_projs: Optional[List[nn.Linear]] = None,
    beta_projs: Optional[List[nn.Linear]] = None,
) -> List[torch.Tensor]:
    """对解码器各层按指定 schedule 施加（或跳过）FiLM 调制。

    当前代码（model_plate.py:291,305）中第 3 层的调制 u = u * enc 被注释掉，
    而 Darcy 模型保留了对应位置的调制。各层是否调制对最终精度有影响，
    但当前状态是"未完成的实验决策"而非经过验证的最优设计。

    本函数将调制 schedule 参数化，通过 modulate_flags 列表明确控制哪些层
    施加调制，消除代码中的注释残留，使实验配置清晰可复现。

    Args:
        layer_outputs (list[torch.Tensor]): 各解码层的输出特征，每个形状 (B, M, F)。
            列表长度 = 解码器层数（Darcy 3 层，Plate 4 层）。
        enc (torch.Tensor): 参数全局嵌入，形状 (B, 1, F)。
        modulate_flags (list[bool]): 与 layer_outputs 等长的布尔列表。
            True = 对该层输出施加调制；False = 跳过（恒等）。
            示例（Darcy）: [True, True, True]；
            示例（Plate，复现当前注释状态）: [True, True, False, False]。
        gamma_projs (list[nn.Linear] | None): 各调制层的 gamma 投影，长度等于
            sum(modulate_flags)。None 时退化为原始 u*enc 行为。
        beta_projs (list[nn.Linear] | None): 各调制层的 beta 投影，同上。

    Returns:
        modulated_outputs (list[torch.Tensor]): 与 layer_outputs 等长，
            已调制层的输出经过 FiLM，未调制层保持原始值。

    Raises:
        ValueError: 若 modulate_flags 长度与 layer_outputs 不匹配。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> # 在配置文件中添加：modulate_schedule: [true, true, false, false]
        >>> modulated = apply_modulation_schedule(
        ...     [u1, u2, u3, u4], enc,
        ...     modulate_flags=config['model']['modulate_schedule'],
        ...     gamma_projs=self.gammas, beta_projs=self.betas
        ... )
    """
    raise NotImplementedError(
        "E2: apply_modulation_schedule 尚未实现。"
        "实现即遍历 zip(layer_outputs, modulate_flags)，"
        "对 flag=True 的层调用 film_modulate（E1），flag=False 的层直接 pass-through。"
        "建议将 modulate_schedule 暴露到 config['model'] 中以便 yaml 配置。"
    )


# ---------------------------------------------------------------------------
# E3 — 残差连接
# ---------------------------------------------------------------------------

def build_decoder_with_residuals(
    fc_dim: int,
    n_layer: int,
    activation: nn.Module,
    use_residual: bool = True,
    norm_type: Literal["none", "layer_norm"] = "none",
) -> nn.ModuleList:
    """构建带残差连接的解码器 FC 层列表，缓解深层网络梯度消失。

    当前代码（model_darcy.py:241-248, model_plate.py:283-308）中预测头 FC1~FC4
    之间无任何残差或跳层连接，深层网络中梯度需经过多个线性变换才能回传，
    梯度消失风险较高，尤其当使用 Tanh 激活（易饱和）时。

    Args:
        fc_dim (int): 每层的特征维度，残差连接要求输入输出维度相同，
            故所有层宽度统一为 fc_dim。
        n_layer (int): FC 层数（不含最终输出层），Darcy 为 3，Plate 为 4。
        activation (nn.Module): 层间激活函数实例。
        use_residual (bool): 是否添加残差连接：
            - True  : 每层输出 = layer(x) + x（PreNorm 风格）。
            - False : 每层输出 = layer(x)（原始行为）。
        norm_type (str): 残差块内的归一化类型：
            - "none"       : 无归一化（原始行为）。
            - "layer_norm" : 在残差加法前对 layer(x) 做 LayerNorm（推荐）。

    Returns:
        layers (nn.ModuleList): 长度为 n_layer 的模块列表，每个元素为一个
            残差块或普通线性层；在 forward 中按顺序调用，调制操作在层间插入。

    Raises:
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> self.decoder_layers = build_decoder_with_residuals(
        ...     fc_dim=2*config['model']['fc_dim'], n_layer=3,
        ...     activation=nn.GELU(), use_residual=True, norm_type="layer_norm"
        ... )
        >>> # 替换 model_darcy.py:203-206 中的 FC1u/FC2u/FC3u/FC4u 定义
        >>> # forward 时：for layer in self.decoder_layers: u = layer(u); u = film_modulate(u, enc)
    """
    raise NotImplementedError(
        "E3: build_decoder_with_residuals 尚未实现。"
        "可定义 ResidualBlock(nn.Module) 内部类，包含 Linear + LayerNorm + activation + skip。"
        "注意第一层需要将 xy_global（2*fc_dim）投影到 fc_dim，不能加残差（维度不同），"
        "可在第一层前添加一个投影线性层，或使用 D1 的 build_coordinate_lift_network 替代。"
    )


# ---------------------------------------------------------------------------
# E4 — 输出层设计（修复 FC5u/FC5v 从未调用的缺陷）
# ---------------------------------------------------------------------------

def build_decoder_output_head(
    fc_dim: int,
    out_dim: int = 1,
    output_mode: Literal["mean_dot", "linear", "mean_dot_linear"] = "linear",
    enc_dim: Optional[int] = None,
) -> nn.Module:
    """构建解码器输出头，修复 FC5u/FC5v 已定义但从未调用的设计缺陷。

    当前代码（model_darcy.py:248, model_plate.py:294,308）的输出层为：
        u = torch.mean(u * enc, dim=-1)    # (B, M)
    这等价于将 u（FC4u 输出，形状 (B,M,F)）与 enc（参数嵌入，形状 (B,1,F)）
    做内积后对特征维取均值，是一种"软内积"输出方式。
    FC5u = nn.Linear(2*fc_dim, 1) 已在 __init__ 中定义（model_plate.py:232,240），
    但在 forward 中从未被调用，其参数完全浪费。

    Args:
        fc_dim (int): 解码器最后一个 FC 层的输出维度（即输出头的输入维度）。
        out_dim (int): 最终输出维度，通常为 1（标量场 Darcy u；Plate u 或 v 各为 1）。
        output_mode (str): 输出聚合方式：
            - "mean_dot"       : torch.mean(u * enc, dim=-1)（原始行为，内积均值）。
            - "linear"         : FC5u(u).squeeze(-1)（使用已定义但未调用的 FC5u）。
            - "mean_dot_linear": 先做 mean_dot，再经过一层 Linear(1→1)（缩放+偏置）。
        enc_dim (int | None): enc 的特征维度，仅 output_mode="mean_dot" 或
            "mean_dot_linear" 时需要，用于确认 u 与 enc 的维度匹配。

    Returns:
        output_head (nn.Module): 接受
            - u (torch.Tensor): 形状 (B, M, fc_dim)
            - enc (torch.Tensor): 形状 (B, 1, enc_dim)（"mean_dot" 模式需要）
            输出 pred (torch.Tensor): 形状 (B, M)，即最终预测场。

    Raises:
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Note:
        修复步骤：在 model_plate.py forward 中，将
            u = torch.mean(u * enc, dim=-1)
        替换为
            u = self.FC5u(u).squeeze(-1)
        或调用本函数返回的 output_head。
        同步删除或注释 FC5u/FC5v 在 __init__ 的冗余定义（或保留供 linear 模式使用）。

    Example:
        >>> self.output_head_u = build_decoder_output_head(2*fc_dim, 1, output_mode="linear")
        >>> # 在 forward 中：u = self.output_head_u(u)
        >>> # 替换 model_plate.py:294 中的 u = torch.mean(u * enc, -1)
    """
    raise NotImplementedError(
        "E4: build_decoder_output_head 尚未实现。"
        "'linear' 模式最简单：return nn.Sequential(nn.Linear(fc_dim, out_dim), Squeeze(-1))。"
        "需确认 FC5u = nn.Linear(2*fc_dim, 1) 与本函数的 fc_dim 参数对应，"
        "避免维度不匹配导致报错。"
    )


# ---------------------------------------------------------------------------
# E5 — 激活函数统一配置（整网络）
# ---------------------------------------------------------------------------

def get_decoder_activation(
    name: Literal["tanh", "gelu", "silu", "relu", "leaky_relu"] = "tanh",
) -> nn.Module:
    """按名称返回解码器激活函数实例，统一全网络的激活函数配置。

    当前代码（model_darcy.py:207, model_plate.py:233）中 self.act = nn.Tanh()
    硬编码，整个预测头统一使用 Tanh。Tanh 在深层网络中易饱和（输出趋向 ±1，
    梯度趋向 0），对非平滑解（如应力集中点附近的高梯度区域）拟合能力弱于
    现代激活函数 GELU/SiLU。

    Args:
        name (str): 激活函数名称（大小写不敏感）：
            - "tanh"       : nn.Tanh()（原始行为）。
            - "gelu"       : nn.GELU()，光滑且不饱和，推荐首选替代。
            - "silu"       : nn.SiLU()（Swish），与 GELU 性能相近。
            - "relu"       : nn.ReLU()，不连续梯度，不推荐用于 PINN。
            - "leaky_relu" : nn.LeakyReLU(negative_slope=0.01)。

    Returns:
        activation (nn.Module): 激活函数实例。

    Raises:
        ValueError: 若 name 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Note:
        与 B2 get_geometry_activation 接口相同，可合并为单一通用函数
        get_activation(name) 供 B/C/D/E 所有模块共用。

    Example:
        >>> self.act = get_decoder_activation("gelu")
        >>> # 替换 model_darcy.py:207 中的 self.act = nn.Tanh()
    """
    raise NotImplementedError(
        "E5: get_decoder_activation 尚未实现。"
        "建议合并 B2 / E5 为单一函数 get_activation(name)，"
        "放在 lib/optimizations/utils.py 中供所有模块 import 使用。"
    )


# ---------------------------------------------------------------------------
# E6 — 解码器正则化
# ---------------------------------------------------------------------------

def build_decoder_layer_with_norm(
    in_dim: int,
    out_dim: int,
    activation: nn.Module,
    dropout_rate: float = 0.0,
    norm_type: Literal["none", "layer_norm", "batch_norm"] = "none",
) -> nn.Module:
    """构建带归一化和 Dropout 的单个解码器层，缓解小样本过拟合。

    当前代码（model_darcy.py:185-248, model_plate.py:207-308）预测头无任何
    Dropout 或归一化，在训练样本数仅 ~140（70% of 200）的情况下，高容量解码器
    极易过拟合。

    Args:
        in_dim (int): 本层输入维度。
        out_dim (int): 本层输出维度（通常与 in_dim 相同，用于等宽层）。
        activation (nn.Module): 激活函数实例，在 LayerNorm 之后应用。
        dropout_rate (float): Dropout 概率，0.0 = 无正则化（原始行为）；
            建议范围 [0.05, 0.2]。
        norm_type (str): 归一化方式：
            - "none"       : 无归一化（原始行为）。
            - "layer_norm" : LayerNorm(out_dim)，位于 Linear 之后、activation 之前。
            - "batch_norm" : BatchNorm1d(out_dim)，适用于 batch 较大的场景。

    Returns:
        layer (nn.Module): 组合层，执行顺序：
            Linear → [LayerNorm/BatchNorm] → Activation → [Dropout]
            接受 (..., in_dim)，输出 (..., out_dim)。

    Raises:
        ValueError: 若 norm_type 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> self.FC1u = build_decoder_layer_with_norm(
        ...     2*fc_dim, 2*fc_dim, nn.GELU(), dropout_rate=0.1, norm_type="layer_norm"
        ... )
        >>> # 替换 model_darcy.py:203 中的 self.FC1u = nn.Linear(2*fc_dim, 2*fc_dim)
    """
    raise NotImplementedError(
        "E6: build_decoder_layer_with_norm 尚未实现。"
        "可基于 nn.Sequential 组合：Linear、可选 Norm、Activation、可选 Dropout。"
        "与 B6、C1 的正则化逻辑高度相似，建议提取为公共工具函数。"
    )


# ---------------------------------------------------------------------------
# E7 — 网络深度配置化
# ---------------------------------------------------------------------------

def build_configurable_decoder(
    input_dim: int,
    fc_dim: int,
    n_decoder_layers: int,
    out_dim: int,
    activation: nn.Module,
    use_residual: bool = False,
    dropout_rate: float = 0.0,
    norm_type: Literal["none", "layer_norm"] = "none",
) -> nn.Module:
    """构建完全可配置深度和宽度的解码器，替换硬编码的层数。

    当前代码（model_darcy.py:203-206, model_plate.py:228-240）中预测头层数
    硬编码：Darcy 为 3 层（FC1u~FC4u，但 FC4u 实为输出层），Plate 为 4 层。
    N_layer 仅控制编码器（DG/branch）层数，无法通过配置文件统一调整解码器深度。

    本函数将解码器深度 n_decoder_layers 完全参数化，并支持所有其他正则化选项，
    是 E3/E5/E6 等多个优化点的综合实现入口。

    Args:
        input_dim (int): 解码器输入维度，即 xy_global 的维度（通常 2*fc_dim）。
        fc_dim (int): 隐层宽度（各中间层统一宽度）。
        n_decoder_layers (int): 解码器隐层数量（不含输出层）；
            对应 config['model']['N_decoder_layer']（新增配置项）。
            Darcy 原始行为等效于 3，Plate 等效于 4。
        out_dim (int): 最终输出维度，通常为 1。
        activation (nn.Module): 隐层激活函数实例。
        use_residual (bool): 是否在隐层添加残差连接（见 E3）。
        dropout_rate (float): Dropout 比例（见 E6）。
        norm_type (str): 归一化类型（见 E6）。

    Returns:
        decoder (nn.Module): 完整解码器，接受 (B, M, input_dim) 和可选的
            enc (B, 1, fc_dim)（调制信号），输出 (B, M) 的预测场。
            注意：调制逻辑（E1/E2）需在外部 forward 中配合使用，
            或将 enc 作为 decoder.forward 的第二个参数。

    Raises:
        ValueError: 若 n_decoder_layers < 1。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> self.decoder_u = build_configurable_decoder(
        ...     input_dim=2*fc_dim, fc_dim=2*fc_dim,
        ...     n_decoder_layers=config['model'].get('N_decoder_layer', 3),
        ...     out_dim=1, activation=nn.GELU(),
        ...     use_residual=True, dropout_rate=0.1, norm_type="layer_norm"
        ... )
        >>> # 替换 model_darcy.py:203-207 中的 FC1u~FC4u 及 self.act 定义
    """
    raise NotImplementedError(
        "E7: build_configurable_decoder 尚未实现。"
        "建议在 configs/GANO_Darcy_DG.yaml 中新增 N_decoder_layer: 3 字段，"
        "并在 model_darcy.py 的 __init__ 中用 config['model']['N_decoder_layer'] 读取。"
        "本函数综合了 E3/E5/E6 的功能，是这三个优化点的统一入口。"
    )


# ---------------------------------------------------------------------------
# E8 — Plate u/v 预测头耦合
# ---------------------------------------------------------------------------

def build_coupled_uv_decoder(
    input_dim: int,
    fc_dim: int,
    n_layer: int,
    activation: nn.Module,
    coupling_mode: Literal["decoupled", "shared_trunk", "joint_output"] = "decoupled",
    use_residual: bool = False,
) -> Tuple[nn.Module, nn.Module]:
    """构建 Plate 模型的 u/v 联合预测头，探索物理耦合的架构实现。

    当前代码（model_plate.py:228-240,283-308）中 u 和 v 各有独立的 FC1~FC5，
    参数量翻倍。u 和 v 在物理上通过应力-应变张量强耦合（σ_xx 同时依赖 ε_xx 和 ε_yy，
    即同时依赖 ∂u/∂x 和 ∂v/∂y），完全解耦的预测头无法利用这种物理关联，
    可能导致 u/v 预测不满足物理一致性约束。

    Args:
        input_dim (int): 解码器输入维度（xy_global 维度，通常 2*fc_dim）。
        fc_dim (int): 隐层宽度。
        n_layer (int): 解码器层数。
        activation (nn.Module): 激活函数实例。
        coupling_mode (str): u/v 耦合方式：
            - "decoupled"     : u/v 完全独立（原始行为，参数量最大）。
            - "shared_trunk"  : 共享前 (n_layer-1) 层，最后一层分叉输出 u 和 v；
                                参数量约减半，迫使网络在共享表示上同时学习两个分量。
            - "joint_output"  : 共享所有层，最终输出层直接输出 2 维（u 和 v），
                                最大化参数共享，但可能限制两分量的个体差异表达。
        use_residual (bool): 是否在共享层中使用残差连接。

    Returns:
        decoder_u (nn.Module): u 分量预测模块，接受 (B, M, input_dim)，
            输出 (B, M)。
        decoder_v (nn.Module): v 分量预测模块，接受 (B, M, input_dim)，
            输出 (B, M)。
        当 coupling_mode="shared_trunk" 或 "joint_output" 时，
        decoder_u 和 decoder_v 共享部分参数（应使用同一个 nn.Module 对象
        的不同分支，或分别是共享 trunk 的两个头）。

    Raises:
        ValueError: 若 coupling_mode 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> self.decoder_u, self.decoder_v = build_coupled_uv_decoder(
        ...     input_dim=2*fc_dim, fc_dim=2*fc_dim, n_layer=4,
        ...     activation=nn.GELU(), coupling_mode="shared_trunk"
        ... )
        >>> # 替换 model_plate.py:228-240 中的独立 FC1u~FC5u / FC1v~FC5v 定义
    """
    raise NotImplementedError(
        "E8: build_coupled_uv_decoder 尚未实现。"
        "'shared_trunk' 建议实现为一个 SharedTrunkDecoder(nn.Module) 类，"
        "包含共享 trunk（n_layer-1 个 FC）和两个独立的 head（各 1 个 FC→1）。"
        "forward 时 trunk 只跑一次，两个 head 各自接收 trunk 输出。"
    )
