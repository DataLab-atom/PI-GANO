"""
F_physics_loss.py — 物理约束可优化模块 (F1-F7)

对应 optimizable_modules.md § F 节，共 7 个优化点。
本模块提供替换 utils_losses.py 和 utils_darcy_train.py / utils_plate_train.py
中物理约束损失相关代码的独立函数。

当前物理约束结构（Plate PINO）：
    - plate_stress_loss: 计算 PDE 残差 (rx, ry)
    - bc_edgeY_loss: 计算 y 向自由边界应力 (σ_yy, σ_xy)
    - 权重固定: weight_pde, weight_load, weight_fix, weight_free
    - 每 batch 内三次独立 forward: BCxy 点、load 点、PDE 点（可合并）

当前 Darcy PINO：
    - darcy_loss: PDE 残差 (∇²u + 10 = 0)
    - 源项 +10 硬编码，权重固定
"""

from __future__ import annotations

from typing import Callable, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# F1 — 剪切应变定义修正
# ---------------------------------------------------------------------------

def compute_plate_strains(
    u: torch.Tensor,
    v: torch.Tensor,
    x_coor: torch.Tensor,
    y_coor: torch.Tensor,
    shear_strain_convention: Literal["tensor", "engineering"] = "tensor",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算平板问题的应变分量，修正剪切应变的因子 1/2 缺失错误。

    当前代码（utils_losses.py:32）定义：
        eps_xy = u_y + v_x          # 工程剪切应变 γ_xy
    连续介质力学中张量剪切应变为：
        ε_xy = (∂u/∂y + ∂v/∂x) / 2  # 张量剪切应变
    当前定义等价于工程剪切应变 γ_xy = 2ε_xy，而应力计算中使用的剪切模量 G 对应
    张量剪切应变（σ_xy = G * ε_xy），两者混用导致剪切应力被高估 2 倍，
    物理约束不正确。

    Args:
        u (torch.Tensor): x 方向位移预测，形状 (B, M)，已启用 requires_grad。
        v (torch.Tensor): y 方向位移预测，形状 (B, M)，已启用 requires_grad。
        x_coor (torch.Tensor): x 坐标，形状 (B, M)，已启用 requires_grad。
        y_coor (torch.Tensor): y 坐标，形状 (B, M)，已启用 requires_grad。
        shear_strain_convention (str): 剪切应变约定：
            - "tensor"      : ε_xy = (u_y + v_x) / 2（连续介质力学张量定义，正确）。
            - "engineering" : γ_xy = u_y + v_x（工程定义，原始代码行为，存在错误）。
            注意：若同时修改此处约定，需确认 G 因子的定义（G = E / 2(1+ν)）
            对应张量剪切应变，否则需相应调整 G。

    Returns:
        eps_xx (torch.Tensor): 正应变 ε_xx = ∂u/∂x，形状 (B, M)。
        eps_yy (torch.Tensor): 正应变 ε_yy = ∂v/∂y，形状 (B, M)。
        eps_xy (torch.Tensor): 剪切应变（张量或工程，取决于 convention），形状 (B, M)。

    Raises:
        ValueError: 若 shear_strain_convention 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Note:
        修复后需同步检查 bc_edgeY_loss（utils_losses.py:47-64）中的 eps_xy 定义，
        保持两处约定一致（F2 中亦涉及）。

    Example:
        >>> eps_xx, eps_yy, eps_xy = compute_plate_strains(
        ...     u_pde_pred, v_pde_pred, x_pde, y_pde, shear_strain_convention="tensor"
        ... )
        >>> # 替换 utils_losses.py:28-32 中的应变计算段落
    """
    raise NotImplementedError(
        "F1: compute_plate_strains 尚未实现。"
        "修复极为简单：在 eps_xy = u_y + v_x 后添加 / 2 即可。"
        "注意同步修改 bc_edgeY_loss 中的 eps_xy（utils_losses.py:58），"
        "确保 PDE 残差和边界条件损失使用相同的应变约定。"
    )


# ---------------------------------------------------------------------------
# F2 — 自由边界条件（补全 x 方向）
# ---------------------------------------------------------------------------

def compute_free_boundary_stress(
    u: torch.Tensor,
    v: torch.Tensor,
    x_coor: torch.Tensor,
    y_coor: torch.Tensor,
    params: Tuple[float, float],
    boundary_normal: Literal["y", "x", "both"] = "y",
    shear_strain_convention: Literal["tensor", "engineering"] = "tensor",
) -> Dict[str, torch.Tensor]:
    """计算自由边界上的应力分量，补全缺失的 x 方向自由边界约束。

    当前代码（utils_losses.py:47-64）中 bc_edgeY_loss 只约束 y 方向边界
    (法向量为 y) 上的 σ_yy=0 和 σ_xy=0，完全缺少对应 x 方向自由边界
    (法向量为 x) 上 σ_xx=0 和 σ_xy=0 的约束函数。
    若几何域存在 x 方向的自由边界（如左右两侧无约束），则物理约束不完整，
    训练时这些边界上的应力可以任意取值，导致解不满足真实自由边界条件。

    Args:
        u (torch.Tensor): x 方向位移，形状 (B, M)，已启用 requires_grad。
        v (torch.Tensor): y 方向位移，形状 (B, M)，已启用 requires_grad。
        x_coor (torch.Tensor): x 坐标，形状 (B, M)，已启用 requires_grad。
        y_coor (torch.Tensor): y 坐标，形状 (B, M)，已启用 requires_grad。
        params (tuple): 材料参数 (E, ν)，E 为 Young's 模量，ν 为 Poisson 比。
        boundary_normal (str): 自由边界的法向量方向：
            - "y"    : 计算 σ_yy, σ_xy（原始 bc_edgeY_loss，基线）。
            - "x"    : 计算 σ_xx, σ_xy（补全缺失的 x 方向约束）。
            - "both" : 同时计算两组，返回所有 4 个应力分量。
        shear_strain_convention (str): 应变约定，与 F1 保持一致（默认 "tensor"）。

    Returns:
        stresses (dict): 字典，包含计算的应力分量，键名因 boundary_normal 而异：
            - "y"    : {"sigma_yy": ..., "sigma_xy": ...}，各形状 (B, M)。
            - "x"    : {"sigma_xx": ..., "sigma_xy": ...}，各形状 (B, M)。
            - "both" : {"sigma_xx": ..., "sigma_yy": ..., "sigma_xy": ...}。

    Raises:
        ValueError: 若 boundary_normal 不在支持列表中。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> stresses_y = compute_free_boundary_stress(
        ...     u_BCy_pred, v_BCy_pred, x_pde_bcy, y_pde_bcy, params, boundary_normal="y"
        ... )
        >>> sigma_yy, sigma_xy = stresses_y["sigma_yy"], stresses_y["sigma_xy"]
        >>> # 替换 utils_losses.py:47-64 中的 bc_edgeY_loss
        >>>
        >>> # 若存在 x 方向自由边界（bcx 节点），还需：
        >>> stresses_x = compute_free_boundary_stress(
        ...     u_BCx_pred, v_BCx_pred, x_pde_bcx, y_pde_bcx, params, boundary_normal="x"
        ... )
    """
    raise NotImplementedError(
        "F2: compute_free_boundary_stress 尚未实现。"
        "实现时需先调用 compute_plate_strains（F1）获取 eps_xx, eps_yy, eps_xy，"
        "再按平面应力本构关系计算 sigma_xx, sigma_yy, sigma_xy。"
        "boundary_normal='x' 时，需额外加载 bcx 节点数据（目前 utils_data.py 未存储），"
        "可能需要同步修改数据预处理管线（A 模块）。"
    )


# ---------------------------------------------------------------------------
# F3 — PDE 源项参数化
# ---------------------------------------------------------------------------

def darcy_pde_residual(
    u: torch.Tensor,
    x_coor: torch.Tensor,
    y_coor: torch.Tensor,
    flag_pde: torch.Tensor,
    source_term: float = 10.0,
    source_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """计算 Darcy 方程 PDE 残差，将源项从硬编码改为可配置参数。

    当前代码（utils_losses.py:15）中源项 +10 硬编码：
        pde_residual = u_xx + u_yy + 10
    不同 Darcy 问题的源项（外加体力或渗流源）可能不同，硬编码要求每次换
    问题都需修改源文件，不利于代码复用。

    Args:
        u (torch.Tensor): 预测解场，形状 (B, M)，已启用 requires_grad。
        x_coor (torch.Tensor): x 坐标，形状 (B, M)，已启用 requires_grad。
        y_coor (torch.Tensor): y 坐标，形状 (B, M)，已启用 requires_grad。
        flag_pde (torch.Tensor): PDE 节点有效性掩码，形状 (B, M)。
        source_term (float): 常数源项值，默认 10.0（原始行为）。
            当 source_fn 不为 None 时，本参数被忽略。
        source_fn (callable | None): 空间变化的源项函数 f(x, y) → source，
            接受 (B, M) 形状的 x_coor 和 y_coor，返回 (B, M) 形状的源项值。
            None 时使用常数 source_term。

    Returns:
        pde_loss (torch.Tensor): PDE 残差 MSE 损失，标量。
            等价于 mse(residual * flag_pde, zeros_like(residual))，
            其中 residual = u_xx + u_yy + source。

    Raises:
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> pde_loss = darcy_pde_residual(u_pde_pred, x_pde, y_pde, pde_flag,
        ...                               source_term=config['pde']['source'])
        >>> # 替换 utils_losses.py:5-17 中的 darcy_loss 函数
        >>> # 并在 configs/GANO_Darcy_DG.yaml 中添加 pde: {source: 10.0}
    """
    raise NotImplementedError(
        "F3: darcy_pde_residual 尚未实现。"
        "实现极为简单，核心仅需将 utils_losses.py:15 的 +10 改为 +source_term。"
        "同时建议在 GANO_Darcy_DG.yaml 新增 source_term: 10.0 字段以完成配置化。"
    )


# ---------------------------------------------------------------------------
# F4 — 损失权重自适应调整
# ---------------------------------------------------------------------------

def compute_adaptive_loss_weights(
    current_losses: Dict[str, torch.Tensor],
    method: Literal["fixed", "ntk", "residual_based", "softmax_norm", "grad_norm"] = "fixed",
    fixed_weights: Optional[Dict[str, float]] = None,
    step: int = 0,
    model: Optional[nn.Module] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """自适应计算各损失项的权重，替换全程固定权重的设置。

    当前代码（utils_darcy_train.py:203-204, utils_plate_train.py:264-267）中
    所有权重（weight_pde, weight_bc, weight_load 等）在训练全程固定。
    PINN 类方法中，不同损失项在训练初期和后期的量级差异悬殊，固定权重会导致
    某些损失项长期主导优化方向，而另一些长期被忽略。
    自适应权重方法（NTK权重、RAD、GradNorm等）已被广泛证明可显著改善收敛。

    Args:
        current_losses (dict): 当前 step 各损失项的值，键为损失名称，
            值为标量 tensor（如 {'pde': tensor(0.1), 'bc': tensor(0.05)}）。
        method (str): 权重更新策略：
            - "fixed"          : 使用 fixed_weights 中的固定值（原始行为）。
            - "ntk"            : 基于神经正切核（NTK）谱分析自适应调整；
                                 需要计算每个损失项对参数的梯度，计算较昂贵。
            - "residual_based" : 权重正比于各损失当前值（高残差损失获得更高权重）；
                                 w_i = loss_i / sum(losses)。
            - "softmax_norm"   : 对各损失取 softmax 作为归一化权重，
                                 防止某项主导。
            - "grad_norm"      : GradNorm 方法（Chen et al. 2018），
                                 通过均衡各损失的梯度范数来自适应调整权重；
                                 需要 model 参数。
        fixed_weights (dict | None): method="fixed" 时使用的权重字典，
            键名与 current_losses 对应。None 时所有权重默认为 1.0。
        step (int): 当前训练步数，部分方法（如带预热的 residual_based）需要。
        model (nn.Module | None): 模型实例，method="grad_norm" 时必填。
        eps (float): 数值稳定性小量，防止除零。

    Returns:
        weights (dict): 更新后的权重字典，键名与 current_losses 对应，
            值为 Python float（已 detach，不参与反向传播图）。

    Raises:
        ValueError: 若 method 需要 model 但未提供。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> weights = compute_adaptive_loss_weights(
        ...     {'pde': pde_loss, 'bc': bc_loss},
        ...     method="residual_based"
        ... )
        >>> total_loss = weights['pde'] * pde_loss + weights['bc'] * bc_loss
        >>> # 替换 utils_darcy_train.py:272 中的 weight_pde*pde_loss + weight_bc*bc_loss
    """
    raise NotImplementedError(
        "F4: compute_adaptive_loss_weights 尚未实现。"
        "'residual_based' 最容易实现且通常有效：计算 total = sum(losses.values())，"
        "各权重 w_i = loss_i.item() / total.item()，不需要梯度信息。"
        "'grad_norm' 需要 torch.autograd.grad 对每个损失分别求梯度范数，"
        "计算量为 loss 数量倍。建议从 'residual_based' 开始实验。"
    )


# ---------------------------------------------------------------------------
# F5 — BC 与 PDE 损失量纲均衡
# ---------------------------------------------------------------------------

def normalize_pde_bc_losses(
    pde_loss: torch.Tensor,
    bc_loss: torch.Tensor,
    pde_normalization: Literal["none", "mean_residual", "std_residual"] = "none",
    bc_normalization: Literal["none", "mean_bc", "std_bc"] = "none",
    pde_ref_value: Optional[float] = None,
    bc_ref_value: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 PDE 残差损失和 BC 损失做量纲归一化，消除需要 3 个数量级权重差异的根本原因。

    当前配置（configs/GANO_Darcy_DG.yaml:13-14）中 weight_bc=1000, weight_pde=1，
    权重相差 3 个数量级，表明两项损失的量纲严重不匹配。这是因为 PDE 残差
    （∇²u + 10 的均方误差）和 BC 损失（解值均方误差）的数量级完全不同。
    正确做法是先对两项损失做量纲归一化，使其在同一量级后再使用接近 1:1 的权重。

    Args:
        pde_loss (torch.Tensor): 原始 PDE 残差损失，标量。
        bc_loss (torch.Tensor): 原始边界条件损失，标量。
        pde_normalization (str): PDE 损失归一化策略：
            - "none"           : 不归一化（原始行为）。
            - "mean_residual"  : 除以 PDE 残差的参考量级（pde_ref_value）。
            - "std_residual"   : 除以残差的标准差估计。
        bc_normalization (str): BC 损失归一化策略：
            - "none"           : 不归一化（原始行为）。
            - "mean_bc"        : 除以 BC 解值的参考量级（bc_ref_value）。
            - "std_bc"         : 除以 BC 解值的标准差估计。
        pde_ref_value (float | None): PDE 损失的参考归一化量级；
            可在训练初始几步统计后固定，或动态跟踪（滑动平均）。
        bc_ref_value (float | None): BC 损失的参考归一化量级。

    Returns:
        pde_loss_normalized (torch.Tensor): 归一化后的 PDE 损失。
        bc_loss_normalized (torch.Tensor): 归一化后的 BC 损失。
        归一化后的两项损失理想状态下数量级相近，可将 weight_pde/weight_bc 调整至 1:1。

    Raises:
        ValueError: 若归一化策略需要参考值但未提供。
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> pde_norm, bc_norm = normalize_pde_bc_losses(
        ...     pde_loss, bc_loss,
        ...     pde_normalization="mean_residual", pde_ref_value=pde_scale,
        ...     bc_normalization="mean_bc", bc_ref_value=bc_scale
        ... )
        >>> total_loss = weight_pde * pde_norm + weight_bc * bc_norm
        >>> # 替换 utils_darcy_train.py:272 中的直接相加
    """
    raise NotImplementedError(
        "F5: normalize_pde_bc_losses 尚未实现。"
        "最简单的实现：在训练前几步（warmup）计算各损失的滑动平均作为 ref_value，"
        "之后除以 ref_value 做归一化。ref_value 用 exponential moving average 跟踪。"
        "该方法与 F4 的 'residual_based' 自适应权重等价，可择一实现。"
    )


# ---------------------------------------------------------------------------
# F6 — 梯度裁剪
# ---------------------------------------------------------------------------

def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """对模型参数梯度做范数裁剪，防止二阶自动微分链路产生爆炸梯度。

    当前代码（utils_darcy_train.py:279-281, utils_plate_train.py:376-378）中
    在 total_loss.backward() 之后直接 optimizer.step()，没有任何梯度裁剪。
    计算 PDE 残差需要二阶自动微分（对 u 求关于 x, y 的二阶偏导），
    训练初期网络参数随机初始化，二阶梯度链路极易产生数值溢出，导致训练不稳定
    乃至 NaN 损失。

    Args:
        model (nn.Module): 模型实例，其参数的梯度将被原地裁剪。
        max_norm (float): 梯度 L_p 范数的最大允许值，超过则按比例缩小。
            建议范围 [0.1, 10.0]；初始可设 1.0，若训练仍不稳定则调小至 0.5。
        norm_type (float): 梯度范数类型：
            - 2.0 : L2 范数（Euclidean 范数），最常用。
            - 1.0 : L1 范数（所有梯度绝对值之和）。
            - inf : 最大梯度绝对值（Chebyshev 范数）。

    Returns:
        grad_norm (float): 裁剪前的梯度全局范数，用于监控训练稳定性。
            若该值持续 > max_norm，说明网络仍不稳定，需调小 max_norm 或学习率。

    Raises:
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Example:
        >>> # 在 utils_darcy_train.py:279-281 的 backward 和 step 之间插入：
        >>> optimizer.zero_grad()
        >>> total_loss.backward()
        >>> grad_norm = clip_gradients(model, max_norm=1.0)
        >>> optimizer.step()
    """
    raise NotImplementedError(
        "F6: clip_gradients 尚未实现。"
        "实现极为简单：return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type).item()。"
        "需在 config['train'] 中添加 max_grad_norm: 1.0 字段，"
        "并在 utils_darcy_train.py / utils_plate_train.py 的 backward 后调用。"
    )


# ---------------------------------------------------------------------------
# F7 — PDE 点与 BC 点合并前向传播
# ---------------------------------------------------------------------------

def merged_forward_pass(
    model: nn.Module,
    pde_coors: torch.Tensor,
    bc_coors: torch.Tensor,
    par: torch.Tensor,
    par_flag: torch.Tensor,
    shape_coor: torch.Tensor,
    shape_flag: torch.Tensor,
    free_bc_coors: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """将多次独立 forward 合并为单次 forward，消除冗余计算。

    当前代码（utils_darcy_train.py:262-267, utils_plate_train.py:346-361）
    在每个 mini-step 内做多次独立 forward：
        1. model(bc_coors, ...)      → u_BC_pred
        2. model(x_pde, y_pde, ...) → u_pde_pred（需要 autograd）
        3. model(bcy_coors, ...)     → u_BCy_pred（自由边界，需要 autograd）
    三次 forward 分别经过编码器（DG + branch），计算了 3 次几何编码和参数编码，
    而这两者在同一 batch 内的结果完全相同，存在大量冗余计算。

    优化思路：先一次性通过编码器得到全局嵌入 (Domain_enc, enc_global)，
    再将所有类型的坐标拼接后一次性做解码，最后按类型切分结果。

    Args:
        model (nn.Module): PI-GANO 模型实例，支持访问 DG 和 branch 子模块。
        pde_coors (torch.Tensor): PDE 采样坐标，形状 (B, Ms, 2)，
            需启用 requires_grad=True（用于自动微分）。
        bc_coors (torch.Tensor): BC 坐标，形状 (B, Mbc, 2)，不需要 autograd。
        par (torch.Tensor): 参数张量，形状 (B, M', D)。
        par_flag (torch.Tensor): 参数有效性掩码，形状 (B, M')。
        shape_coor (torch.Tensor): 几何描述坐标，形状 (B, Mgeo, 2)。
        shape_flag (torch.Tensor): 几何有效性掩码，形状 (B, Mgeo)。
        free_bc_coors (torch.Tensor | None): 自由边界坐标，形状 (B, Mfree, 2)，
            需启用 requires_grad=True；None 表示无自由边界（Darcy 场景）。

    Returns:
        results (dict): 包含各类型节点的预测结果：
            - "u_pde"    : PDE 节点预测，形状 (B, Ms)，叶节点保留 autograd。
            - "u_bc"     : BC 节点预测，形状 (B, Mbc)。
            - "u_free_bc": 自由边界预测，形状 (B, Mfree)（若 free_bc_coors 非 None）。
            Plate 场景中每项对应 (u, v) 元组；Darcy 场景仅为 u 标量场。

    Raises:
        NotImplementedError: 占位，待实现具体逻辑后移除。

    Note:
        实现时需修改 PI_GANO.forward，将编码器（DG + branch）和解码器分离，
        暴露 encode() 和 decode(xy, enc_global, domain_enc) 接口；
        或使用 torch.cat 将所有坐标拼接后一次 forward，之后用 split 切分。

    Example:
        >>> results = merged_forward_pass(
        ...     model, pde_coors, bc_coors, par, par_flag, shape_coor, shape_flag,
        ...     free_bc_coors=bcy_coors
        ... )
        >>> u_pde = results["u_pde"]
        >>> u_bc  = results["u_bc"]
        >>> u_free_bc = results["u_free_bc"]
        >>> # 替换 utils_darcy_train.py:262-267 中的三次独立 forward 调用
    """
    raise NotImplementedError(
        "F7: merged_forward_pass 尚未实现。"
        "最简单的实现：将 pde_coors + bc_coors + free_bc_coors 沿 dim=1 拼接，"
        "一次 forward 后用 torch.split 切分预测结果。"
        "注意 pde 和 free_bc 坐标需要 requires_grad=True，"
        "拼接时使用 torch.cat([pde_coors.requires_grad_(True), ...])，"
        "切分后的子张量仍持有原始计算图。"
    )
