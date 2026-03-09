"""
u2e_api.py

为 open-agents-U2E 提供两个标准输入函数：

    get_funcs()       → list[dict]   返回所有可优化函数的元数据 + 源码
    evaluate_funcs()  → list[dict]   注入修改后的函数并运行训练，返回评估指标

并发安全性
----------
evaluate_funcs() 使用隔离临时目录（每次调用独立复制 lib/ + configs/ + 训练脚本），
通过子进程运行 run_experiment.py，多次并发调用互不影响，可配合
ThreadPoolExecutor 等并发机制使用。

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
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

# ── 路径 ──────────────────────────────────────────────────────────────────────

_ROOT    = Path(__file__).parent
_LIB_DIR = _ROOT / "lib"

# 需要复制到隔离目录的子目录和文件（均为纯代码，无大型数据文件）
_COPY_DIRS  = ["lib", "configs", "scripts"]
_COPY_GLOBS = ["*.py"]          # 根目录下的训练脚本

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


def _patch_source(source: str, func_name: str, new_code: str) -> str:
    """将 source 中 func_name 函数的实现替换为 new_code，返回新 source。"""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            start, end = node.lineno, node.end_lineno
            new_code = textwrap.dedent(new_code).strip() + "\n"
            lines = source.splitlines(keepends=True)
            return "".join(lines[: start - 1]) + new_code + "\n" + "".join(lines[end:])
    raise ValueError(f"函数 '{func_name}' 未找到")


def _build_isolated_workdir(replacements: list[dict]) -> Path:
    """
    创建一个隔离的临时工程目录，其中已应用 replacements 补丁。

    复制内容（轻量，纯代码）：
        lib/        – 含 *_optimizable.py（被补丁的目标）
        configs/    – YAML 训练配置
        scripts/    – run_experiment.py 等工具脚本
        *.py        – 根目录训练入口脚本

    Returns:
        Path: 临时目录路径（调用方负责在使用后删除）
    """
    tmp = Path(tempfile.mkdtemp(prefix="pi_gano_eval_"))

    # 复制代码目录
    for d in _COPY_DIRS:
        src = _ROOT / d
        if src.exists():
            shutil.copytree(src, tmp / d)

    # 复制根目录 *.py
    for pattern in _COPY_GLOBS:
        for f in _ROOT.glob(pattern):
            shutil.copy2(f, tmp / f.name)

    # 应用补丁（在临时副本的 lib/ 中修改）
    tmp_lib = tmp / "lib"
    for rep in replacements:
        fpath   = tmp_lib / rep["file"]
        current = fpath.read_text(encoding="utf-8")
        patched = _patch_source(current, rep["function"], rep["code"])
        fpath.write_text(patched, encoding="utf-8")

    return tmp


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
        list[dict]: 所有可优化函数的信息列表，按文件名 + 出现顺序排列。
    """
    items = []
    for py_file in sorted(_LIB_DIR.glob("*_optimizable.py")):
        for meta in _parse_file_metadata(py_file):
            func_source = _extract_func_source(py_file, meta["func_name"])
            doc         = meta["docstring"]
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
    接收 U2E 传入的函数列表，在隔离的临时目录中运行 PI-GANO 训练评估。

    并发安全：每次调用均在独立的临时工程副本中执行，多个并发调用互不干扰，
    可配合 concurrent.futures.ThreadPoolExecutor 使用。

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
        ValueError: 若 func_list 中没有任何 is_modified=True 的函数。
        RuntimeError: 临时目录创建或补丁应用失败。
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

    # 创建隔离临时工程目录（含补丁），用完即删
    tmp_dir = _build_isolated_workdir(replacements)
    try:
        # 将替换规格写入临时目录内的 spec.json
        spec_path    = tmp_dir / "spec.json"
        results_path = tmp_dir / "results.json"
        spec_path.write_text(
            json.dumps(replacements, ensure_ascii=False), encoding="utf-8"
        )

        # 通过子进程运行临时目录内的 run_experiment.py
        # 每个并发调用拥有独立的 cwd 和独立的 lib/，完全隔离
        cmd = [
            sys.executable,
            str(tmp_dir / "scripts" / "run_experiment.py"),
            "--spec",   str(spec_path),
            "--out",    str(results_path),
            "--no-stdout",
        ]
        if epochs is not None:
            cmd += ["--epochs", str(epochs)]

        subprocess.run(cmd, cwd=str(tmp_dir), check=False)

        # 读取结果
        if not results_path.exists():
            raise RuntimeError("run_experiment.py 未生成结果文件，训练可能崩溃。")
        raw: dict = json.loads(results_path.read_text(encoding="utf-8"))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 转换为 U2E evaluate_funcs 返回格式
    metrics: list[dict] = []
    for problem, scores in raw.items():
        if scores.get("returncode", 1) != 0:
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
