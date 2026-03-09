"""
u2e_api.py

为 open-agents-U2E 提供两个标准输入函数：

    get_funcs()       → list[dict]   返回所有可优化函数的元数据 + 源码
    evaluate_funcs()  → list[dict]   注入修改后的函数并运行训练，返回评估指标

用法（在 U2E 侧）：
    import sys
    sys.path.insert(0, "/path/to/PI-GANO")
    from u2e_api import get_funcs, evaluate_funcs

    agent = ReEvo2D(
        cfg=cfg,
        get_funcs=get_funcs,
        evaluate_funcs=evaluate_funcs,
    )
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Optional

# ── 路径 ──────────────────────────────────────────────────────────────────────

_ROOT    = Path(__file__).parent
_LIB_DIR = _ROOT / "lib"

# ── 内部工具 ──────────────────────────────────────────────────────────────────

def _extract_func_source(path: Path, func_name: str) -> str:
    """从文件中精确提取指定函数的完整源码字符串。"""
    source = path.read_text(encoding="utf-8")
    tree   = ast.parse(source, filename=str(path))
    lines  = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return "".join(lines[node.lineno - 1 : node.end_lineno])
    raise ValueError(f"函数 '{func_name}' 未在 {path} 中找到")


def _parse_file_metadata(path: Path) -> list[dict]:
    """解析单个 *_optimizable.py，返回函数元数据列表（不含源码）。"""
    source = path.read_text(encoding="utf-8")
    tree   = ast.parse(source, filename=str(path))
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        docstring = ast.get_docstring(node) or ""
        results.append({
            "func_name": node.name,
            "file":      path.name,
            "docstring": docstring,
        })
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  公开 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_funcs() -> list[dict]:
    """
    扫描 lib/*_optimizable.py，返回所有可优化函数的列表。

    每个元素格式（符合 U2E reevo2d.py 的约定）：
        {
            "func_name":        str,   # 函数名
            "func_source":      str,   # 完整函数定义源码
            "func_description": str,   # 一行简短描述（取 docstring 第一行）
            "doc":              str,   # 完整 docstring（供 LLM 理解背景）
            "file":             str,   # 所属文件名（供 evaluate_funcs 路由使用）
        }

    Returns:
        list[dict]: 所有可优化函数的信息列表，按文件名 + 函数在文件中的出现顺序排列。
    """
    items = []
    for py_file in sorted(_LIB_DIR.glob("*_optimizable.py")):
        for meta in _parse_file_metadata(py_file):
            func_source = _extract_func_source(py_file, meta["func_name"])
            doc         = meta["docstring"]
            # 取 docstring 第一个非空行作为简短描述
            first_line  = next(
                (l.strip() for l in doc.splitlines() if l.strip()), meta["func_name"]
            )
            items.append({
                "func_name":        meta["func_name"],
                "func_source":      func_source,
                "func_description": first_line,
                "doc":              doc,
                "file":             meta["file"],
            })
    return items


def evaluate_funcs(
    func_list: list[dict],
    epochs: Optional[int] = None,
) -> list[dict]:
    """
    接收 U2E 传入的函数列表，将其中被修改的函数注入 PI-GANO 并运行训练评估。

    Args:
        func_list: U2E 提供的函数列表，每个元素至少包含：
            {
                "func_name":   str,   # 函数名
                "func_source": str,   # 当前（可能已被 LLM 修改）的函数源码
                "file":        str,   # 所属文件名（由 get_funcs() 附带）
                "is_modified": bool,  # True 表示 LLM 修改了该函数
            }
        epochs: 覆盖 YAML 配置中的训练轮数；None 表示使用配置文件默认值。

    Returns:
        list[dict]: 评估指标列表，每个元素格式：
            {
                "name":      str,     # 指标名称，如 "darcy_test_L2"
                "value":     float,   # 指标数值
                "direction": "min",   # 均为最小化目标
                "weight":    float,   # 权重（默认 1.0）
            }

    Raises:
        ValueError: 若没有任何函数被修改（is_modified=True），则无需评估，
                    抛出 ValueError 提示调用方。
        RuntimeError: run_experiment 内部失败（文件已自动还原）。
    """
    # 只处理被修改的函数
    replacements = []
    for entry in func_list:
        if not entry.get("is_modified", False):
            continue
        file_name = entry.get("file")
        if file_name is None:
            raise ValueError(
                f"函数 '{entry['func_name']}' 缺少 'file' 字段，"
                "请确保 get_funcs() 返回的字典已被原样传入。"
            )
        replacements.append({
            "function": entry["func_name"],
            "file":     file_name,
            "code":     entry["func_source"],
        })

    if not replacements:
        raise ValueError(
            "func_list 中没有任何 is_modified=True 的函数，无需运行实验。"
        )

    # 延迟导入，避免在仅调用 get_funcs() 时加载 torch 等重量级依赖
    from scripts.run_experiment import run_experiment

    results = run_experiment(replacements, epochs=epochs)

    # 将 PI-GANO 的结果格式转换为 U2E 的 evaluate_funcs 返回格式
    metrics: list[dict] = []
    for problem, scores in results.items():
        if scores.get("returncode", 1) != 0:
            # 训练失败：返回极大值作为惩罚
            metrics.append({
                "name":      f"{problem}_test_L2",
                "value":     float("inf"),
                "direction": "min",
                "weight":    1.0,
            })
            continue

        if "test_relative_L2" in scores:
            metrics.append({
                "name":      f"{problem}_test_L2",
                "value":     float(scores["test_relative_L2"]),
                "direction": "min",
                "weight":    1.0,
            })

        if "val_relative_L2_last" in scores:
            metrics.append({
                "name":      f"{problem}_val_L2",
                "value":     float(scores["val_relative_L2_last"]),
                "direction": "min",
                "weight":    0.5,
            })

    return metrics
