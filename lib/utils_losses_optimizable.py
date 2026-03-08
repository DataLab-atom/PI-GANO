"""
utils_losses_optimizable.py

本文件集中存放 utils_losses.py 中所有可优化模块的当前实现。
每个函数封装一个独立的优化项（F1-F5），保持与原代码等价的行为，
以便后续直接替换函数实现来改进对应指标，而无需修改调用方代码。

优化项对应关系：
  F1 - compute_shear_strain         : 剪切应变定义（当前为工程应变，存在错误）
  F2 - compute_free_boundary_stress : 自由边界应力计算（当前仅约束 y 方向）
  F3 - darcy_source_term            : Darcy 方程源项（当前硬编码为 10）
  F4 - compute_total_loss           : 损失加权求和（当前固定权重，无自适应）
  F5 - apply_gradient_clipping      : 梯度裁剪（当前无操作）
"""

from typing import Tuple
import torch
import torch.nn as nn


# ─────────────────────────── F1 ───────────────────────────
def compute_shear_strain(
    u_y: torch.Tensor,
    v_x: torch.Tensor,
) -> torch.Tensor:
    """[F1] 计算剪切应变分量 eps_xy。

    连续介质力学中张量剪切应变定义为 eps_xy = (∂u/∂y + ∂v/∂x) / 2。
    当前实现缺少除以 2，实际返回的是工程剪切应变（gamma_xy = ∂u/∂y + ∂v/∂x），
    与后续应力计算中使用的剪切模量 G = E / (2(1+ν)) 不一致，
    导致 sigma_xy = G * eps_xy 的量级偏大约 2 倍。

    优化方向：将返回值改为 (u_y + v_x) / 2，恢复正确的张量剪切应变定义。

    Args:
        u_y (torch.Tensor): ∂u/∂y，x 方向位移对 y 的偏导，形状 (B, M)。
        v_x (torch.Tensor): ∂v/∂x，y 方向位移对 x 的偏导，形状 (B, M)。

    Returns:
        torch.Tensor: 剪切应变 eps_xy，形状 (B, M)。
                      当前实现返回 u_y + v_x（工程剪切应变，含 F1 错误）。
    """
    return u_y + v_x


# ─────────────────────────── F2 ───────────────────────────
def compute_free_boundary_stress(
    u: torch.Tensor,
    v: torch.Tensor,
    x_coor: torch.Tensor,
    y_coor: torch.Tensor,
    params: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """[F2] 计算自由边界上的应力分量，用于构造自由边界条件损失。

    完整的自由边界条件需要对 x 方向和 y 方向边界分别约束：
      - y 方向自由边界 (法向量 n=[0,1]): σ_yy = 0, σ_xy = 0
      - x 方向自由边界 (法向量 n=[1,0]): σ_xx = 0, σ_xy = 0

    当前实现仅计算 y 方向边界所需的 (σ_yy, σ_xy)，缺少 x 方向的约束
    （σ_xx 和对应的 σ_xy），导致 bc_edgeY_loss 不完整（F2 问题）。

    优化方向：增加返回 sigma_xx，并在调用方增加 x 方向自由边界损失项。

    Args:
        u       (torch.Tensor): x 方向位移预测，形状 (B, M_free)，需开启梯度。
        v       (torch.Tensor): y 方向位移预测，形状 (B, M_free)，需开启梯度。
        x_coor  (torch.Tensor): 自由边界节点 x 坐标，形状 (B, M_free)，
                                 requires_grad=True。
        y_coor  (torch.Tensor): 自由边界节点 y 坐标，形状 (B, M_free)，
                                 requires_grad=True。
        params  (Tuple[float, float]): 材料参数 (E, mu)，E 为 Young's 模量，
                                        mu 为 Poisson 比。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (sigma_yy, sigma_xy)，各形状 (B, M_free)。
            当前实现仅覆盖 y 方向自由边界所需应力（F2 缺失 sigma_xx）。
    """
    E, mu = params
    G = E / 2 / (1 + mu)

    eps_xx = torch.autograd.grad(
        outputs=u, inputs=x_coor,
        grad_outputs=torch.ones_like(u), create_graph=True,
    )[0]
    eps_yy = torch.autograd.grad(
        outputs=v, inputs=y_coor,
        grad_outputs=torch.ones_like(v), create_graph=True,
    )[0]
    u_y = torch.autograd.grad(
        outputs=u, inputs=y_coor,
        grad_outputs=torch.ones_like(u), create_graph=True,
    )[0]
    v_x = torch.autograd.grad(
        outputs=v, inputs=x_coor,
        grad_outputs=torch.ones_like(v), create_graph=True,
    )[0]

    eps_xy = compute_shear_strain(u_y, v_x)

    sigma_yy = (E / (1 - mu ** 2)) * (eps_yy + mu * eps_xx)
    sigma_xy = G * eps_xy

    return sigma_yy, sigma_xy


# ─────────────────────────── F3 ───────────────────────────
def darcy_source_term(reference: torch.Tensor) -> torch.Tensor:
    """[F3] 返回 Darcy 方程右端源项 f，满足 -Δu = f。

    当前实现将源项硬编码为标量 10，不支持不同问题设置（例如变系数源项
    或由配置文件指定的源项），修改时需直接改动损失函数源文件。

    优化方向：将源项值提升为可配置参数（yaml 或函数参数），
              支持空间变化源项 f(x, y)。

    Args:
        reference (torch.Tensor): 与源项同形状的参考张量，用于生成
                                   与计算图兼容的常数张量，形状 (B, M)。

    Returns:
        torch.Tensor: 源项张量，形状与 reference 相同 (B, M)。
                      当前实现返回全为 10 的常数张量。
    """
    return torch.full_like(reference, 10.0)


# ─────────────────────────── F4 ───────────────────────────
def compute_total_loss(
    pde_loss: torch.Tensor,
    bc_loss: torch.Tensor,
    weight_pde: float,
    weight_bc: float,
) -> torch.Tensor:
    """[F4] 按固定权重对 PDE 残差损失和边界条件损失加权求和。

    当前权重配置为 weight_bc=1000, weight_pde=1，量级相差 3 个数量级，
    根本原因是 PDE 残差（∇²u + f）和 BC 残差（u_pred - u_gt）未经过
    量级归一化；且权重在训练全程保持不变，无法自适应调整。

    优化方向：
      (1) 对各子损失先做量级归一化（除以其初始值或滑动平均），
          消除 3 个数量级的差异；
      (2) 引入自适应权重机制（NTK 权重、Residual-based adaptive 等），
          根据梯度信息或残差大小动态调整权重。

    Args:
        pde_loss  (torch.Tensor): PDE 残差均方误差损失，标量。
        bc_loss   (torch.Tensor): 边界条件均方误差损失，标量。
        weight_pde      (float): PDE 损失权重（当前默认 1）。
        weight_bc       (float): 边界条件损失权重（当前默认 1000）。

    Returns:
        torch.Tensor: 总加权损失 = weight_pde * pde_loss + weight_bc * bc_loss，
                      标量。
    """
    return weight_pde * pde_loss + weight_bc * bc_loss


def compute_total_loss_plate(
    pde_loss: torch.Tensor,
    load_loss: torch.Tensor,
    fix_loss: torch.Tensor,
    free_loss: torch.Tensor,
    weight_pde: float,
    weight_load: float,
    weight_fix: float,
    weight_free: float,
) -> torch.Tensor:
    """[F4] Plate 问题：按固定权重对四类损失加权求和。

    Plate 问题包含 PDE 残差、加载边界、固定边界、自由边界四类损失，
    各损失在量级上存在差异但权重固定。与 darcy 版本面临相同的
    固定权重 + 无归一化问题（F4）。

    优化方向同 compute_total_loss，需对各损失归一化并引入自适应权重。

    Args:
        pde_loss   (torch.Tensor): PDE 动量平衡残差损失，标量。
        load_loss  (torch.Tensor): 加载边界位移监督损失，标量。
        fix_loss   (torch.Tensor): 固定边界位移约束损失，标量。
        free_loss  (torch.Tensor): 自由边界应力约束损失，标量。
        weight_pde  (float): PDE 损失权重。
        weight_load (float): 加载边界损失权重。
        weight_fix  (float): 固定边界损失权重。
        weight_free (float): 自由边界损失权重。

    Returns:
        torch.Tensor: 总加权损失，标量。
    """
    return weight_pde * pde_loss + weight_load * load_loss + weight_fix * fix_loss + weight_free * free_loss


# ─────────────────────────── F5 ───────────────────────────
def apply_gradient_clipping(model: nn.Module, max_norm: float = None) -> None:
    """[F5] 在反向传播后、参数更新前对模型梯度施加裁剪。

    PI-GANO 使用二阶自动微分计算 PDE 残差，在训练初期梯度值可能
    极大（梯度爆炸），但当前训练循环中无任何 clip_grad_norm 调用，
    导致早期训练不稳定、损失震荡。

    当前实现：无操作（基线行为）。
    优化方向：调用 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
              在每次 backward 后、optimizer.step() 前裁剪梯度范数。

    Args:
        model    (nn.Module): 需要裁剪梯度的模型。
        max_norm (float | None): 允许的最大梯度 L2 范数。
                                  当前为 None，表示不裁剪。

    Returns:
        None
    """
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
