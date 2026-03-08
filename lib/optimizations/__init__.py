"""
lib/optimizations — PI-GANO 可优化模块插件包

本子包将 optimizable_modules.md 中列举的全部 44 个优化点各自封装为独立 Python 函数。
每个函数拥有完整的入参/出参签名和详细文档字符串，便于逐项实验与替换。

子模块划分：
    A_data_preprocessing    A1-A7   数据预处理
    B_geometry_encoder      B1-B7   几何编码（Domain Geometry Encoder）
    C_parameter_encoder     C1-C6   参数编码（Branch / Parameter Encoder）
    D_coordinate_encoder    D1-D5   坐标编码（Coordinate / Trunk Encoder）
    E_operator_decoder      E1-E8   算子逼近（Operator Approximation / Decoder）
    F_physics_loss          F1-F7   物理约束（Physics-Informed Loss）
    G_sampling_optimization G1-G9   采样与优化（Sampling & Optimization）
"""

from . import A_data_preprocessing
from . import B_geometry_encoder
from . import C_parameter_encoder
from . import D_coordinate_encoder
from . import E_operator_decoder
from . import F_physics_loss
from . import G_sampling_optimization

__all__ = [
    "A_data_preprocessing",
    "B_geometry_encoder",
    "C_parameter_encoder",
    "D_coordinate_encoder",
    "E_operator_decoder",
    "F_physics_loss",
    "G_sampling_optimization",
]
