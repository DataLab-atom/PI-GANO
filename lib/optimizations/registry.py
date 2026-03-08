"""
registry.py — 可优化函数注册表与热替换工具

提供两个核心功能：
  1. get_optimizable_functions_info()  — 扫描所有优化存根，返回结构化 JSON 列表
  2. replace_and_evaluate()           — 传入新实现字符串，原地替换并返回性能指标 JSON
"""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import inspect
import json
import re
import shutil
import sys
import textwrap
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── 路径常量 ─────────────────────────────────────────────────────────────────
_OPT_DIR = Path(__file__).parent          # lib/optimizations/
_LIB_DIR = _OPT_DIR.parent               # lib/
_ROOT_DIR = _LIB_DIR.parent              # 项目根目录
_BACKUP_DIR = _OPT_DIR / "_backups"

# 模块文件前缀 → 类别全名
_CATEGORY_NAMES: Dict[str, str] = {
    "A": "data_preprocessing",
    "B": "geometry_encoder",
    "C": "parameter_encoder",
    "D": "coordinate_encoder",
    "E": "operator_decoder",
    "F": "physics_loss",
    "G": "sampling_optimization",
}

# 优化文件列表（有序）
_OPT_FILES: List[str] = [
    "A_data_preprocessing",
    "B_geometry_encoder",
    "C_parameter_encoder",
    "D_coordinate_encoder",
    "E_operator_decoder",
    "F_physics_loss",
    "G_sampling_optimization",
]


# ─────────────────────────────────────────────────────────────────────────────
# 内部：AST 解析工具
# ─────────────────────────────────────────────────────────────────────────────

def _read_source(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def _find_opt_id_for_func(source: str, func_lineno: int) -> Optional[str]:
    """在函数定义行之前，向上扫描注释行，提取形如 'A1' 的优化点 ID。"""
    lines = source.splitlines()
    # 函数定义在 func_lineno（1-based），向上最多扫描 8 行
    start = max(0, func_lineno - 9)
    end = func_lineno - 1
    pattern = re.compile(r"#\s+([A-G]\d+)\s+[—–-]")
    for i in range(end, start - 1, -1):
        m = pattern.search(lines[i])
        if m:
            return m.group(1)
    return None


def _find_opt_title_for_func(source: str, func_lineno: int) -> Optional[str]:
    """提取优化点标题（# A1 — 这里的文字）。"""
    lines = source.splitlines()
    start = max(0, func_lineno - 9)
    end = func_lineno - 1
    pattern = re.compile(r"#\s+[A-G]\d+\s+[—–-]\s+(.+)")
    for i in range(end, start - 1, -1):
        m = pattern.search(lines[i])
        if m:
            return m.group(1).strip()
    return None


def _is_stub(func_node: ast.FunctionDef) -> bool:
    """判断函数体是否只包含 docstring + raise NotImplementedError。"""
    body = func_node.body
    # 过滤掉 docstring（Expr(Constant)）
    non_doc = [n for n in body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    if len(non_doc) == 1 and isinstance(non_doc[0], ast.Raise):
        exc = non_doc[0].exc
        if exc is None:
            return True
        # raise NotImplementedError(...) 或 raise NotImplementedError
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            return exc.func.id == "NotImplementedError"
        if isinstance(exc, ast.Name):
            return exc.id == "NotImplementedError"
    return False


def _parse_docstring_sections(docstring: str) -> Dict[str, Any]:
    """
    解析 Google-style docstring，提取：
      - short_description (str)
      - long_description  (str)
      - parameters        (list[dict])
      - returns           (dict)
      - raises            (list[dict])
      - notes             (str)
      - example           (str)
    """
    if not docstring:
        return {
            "short_description": "",
            "long_description": "",
            "parameters": [],
            "returns": {"type": "", "description": ""},
            "raises": [],
            "notes": "",
            "example": "",
        }

    lines = docstring.expandtabs(4).splitlines()
    # 去掉首尾空行
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    # 第一行 = short description
    short_desc = lines[0].strip() if lines else ""

    # 分 section：按 Args:/Returns:/Raises:/Note:/Example: 等
    section_re = re.compile(r"^(\s*)(Args|Returns|Raises?|Note|Notes|Example|Examples)\s*:\s*$", re.IGNORECASE)
    sections: Dict[str, List[str]] = {}
    current_section = "body"
    sections["body"] = []
    for line in lines[1:]:
        m = section_re.match(line)
        if m:
            current_section = m.group(2).lower().rstrip("s")  # normalize
            sections[current_section] = []
        else:
            sections.setdefault(current_section, []).append(line)

    long_desc = textwrap.dedent("\n".join(sections.get("body", []))).strip()

    # 解析 Args section
    parameters: List[Dict[str, str]] = []
    arg_lines = sections.get("arg", [])
    param_re = re.compile(r"^\s{4}(\w+)\s*\(([^)]+)\)\s*:\s*(.*)")
    cont_re  = re.compile(r"^\s{8,}(.*)")
    cur_param: Optional[Dict[str, str]] = None
    for line in arg_lines:
        m = param_re.match(line)
        if m:
            if cur_param:
                parameters.append(cur_param)
            cur_param = {
                "name": m.group(1),
                "type": m.group(2).strip(),
                "default": None,
                "description": m.group(3).strip(),
            }
        elif cur_param:
            mc = cont_re.match(line)
            if mc:
                cur_param["description"] += " " + mc.group(1).strip()
    if cur_param:
        parameters.append(cur_param)

    # 从参数类型中提取 default（形如 "str, optional" 不太标准；但 signature 里有）
    # 此处留空，由 signature 解析补充

    # 解析 Returns section
    ret_lines = sections.get("return", [])
    ret_type, ret_desc = "", ""
    ret_re = re.compile(r"^\s{4}(\w[\w\[\], |]*?)\s*\(([^)]+)\)\s*:\s*(.*)")
    # 更简单：合并所有行
    ret_text = "\n".join(l for l in ret_lines if l.strip())
    # 尝试匹配 "name (type): desc" 格式
    rm = re.search(r"(\w[\w\[\], |]*?)\s*\(([^)]+)\)\s*:\s*(.*)", ret_text)
    if rm:
        ret_type = rm.group(2).strip()
        ret_desc = rm.group(3).strip()
    else:
        ret_desc = ret_text.strip()

    # 解析 Raises section
    raises_list: List[Dict[str, str]] = []
    raise_lines = sections.get("raise", [])
    raise_re = re.compile(r"^\s{4}(\w+)\s*:\s*(.*)")
    cur_raise: Optional[Dict[str, str]] = None
    for line in raise_lines:
        mr = raise_re.match(line)
        if mr:
            if cur_raise:
                raises_list.append(cur_raise)
            cur_raise = {"exception": mr.group(1), "description": mr.group(2).strip()}
        elif cur_raise:
            mc = cont_re.match(line)
            if mc:
                cur_raise["description"] += " " + mc.group(1).strip()
    if cur_raise:
        raises_list.append(cur_raise)

    note_text = "\n".join(sections.get("note", [])).strip()
    example_text = "\n".join(sections.get("example", [])).strip()

    return {
        "short_description": short_desc,
        "long_description": long_desc,
        "parameters": parameters,
        "returns": {"type": ret_type, "description": ret_desc},
        "raises": raises_list,
        "notes": note_text,
        "example": example_text,
    }


def _parse_func_signature(func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
    """从 AST 节点提取函数参数的名称、类型注解、默认值。"""
    args = func_node.args
    all_args = args.posonlyargs + args.args + (([args.vararg] if args.vararg else []) +
               args.kwonlyargs + ([args.kwarg] if args.kwarg else []))

    # 默认值对齐（位置参数尾部）
    pos_args = args.posonlyargs + args.args
    n_defaults = len(args.defaults)
    defaults_map: Dict[str, Any] = {}
    for i, d in enumerate(args.defaults):
        idx = len(pos_args) - n_defaults + i
        arg_name = pos_args[idx].arg
        if isinstance(d, ast.Constant):
            defaults_map[arg_name] = d.value
        else:
            defaults_map[arg_name] = ast.unparse(d)

    for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
        if kw_default is not None:
            if isinstance(kw_default, ast.Constant):
                defaults_map[kw_arg.arg] = kw_default.value
            else:
                defaults_map[kw_arg.arg] = ast.unparse(kw_default)

    params = []
    for arg in all_args:
        ann = ast.unparse(arg.annotation) if arg.annotation else None
        params.append({
            "name": arg.arg,
            "type": ann,
            "default": defaults_map.get(arg.arg, None),
            "description": "",   # 由 docstring 解析补充
        })

    return params


def _get_return_annotation(func_node: ast.FunctionDef) -> str:
    if func_node.returns:
        return ast.unparse(func_node.returns)
    return ""


def _get_func_source_lines(source: str, func_node: ast.FunctionDef) -> tuple[int, int]:
    """返回函数定义的起止行号（1-based，inclusive）。"""
    lines = source.splitlines()
    start = func_node.lineno - 1  # 转为 0-based
    # 找到函数的最后一行
    end = func_node.end_lineno - 1  # 0-based，inclusive
    return start + 1, end + 1  # 还原为 1-based


# ─────────────────────────────────────────────────────────────────────────────
# 函数 1：获取所有可优化函数的详细信息
# ─────────────────────────────────────────────────────────────────────────────

def get_optimizable_functions_info() -> List[Dict[str, Any]]:
    """扫描 lib/optimizations/ 下的全部优化文件，返回所有可优化函数的结构化信息。

    Returns:
        list[dict]: 每个元素对应一个可优化函数，包含：
            id              (str)   优化点编号，如 "A1"
            category        (str)   类别字母，如 "A"
            category_name   (str)   类别全名，如 "data_preprocessing"
            function_name   (str)   Python 函数名
            module          (str)   所在模块名（不含 .py）
            file_path       (str)   相对于项目根目录的路径
            title           (str)   优化点中文标题
            short_description (str) docstring 首行简介
            long_description  (str) docstring 详细说明
            parameters      (list)  参数列表，每项含 name/type/default/description
            returns         (dict)  返回值信息 {type, description}
            raises          (list)  可能抛出的异常列表
            return_annotation (str) 函数签名中的返回注解
            notes           (str)   docstring Notes 节
            example         (str)   docstring Example 节
            is_implemented  (bool)  False = 仍是 NotImplementedError 存根
            source_line_start (int) 函数定义起始行（1-based）
            source_line_end   (int) 函数定义结束行（1-based）

    Example:
        >>> info = get_optimizable_functions_info()
        >>> print(json.dumps(info, ensure_ascii=False, indent=2))
        >>> # 获取所有未实现的函数
        >>> stubs = [f for f in info if not f["is_implemented"]]
    """
    results: List[Dict[str, Any]] = []

    for module_name in _OPT_FILES:
        file_path = _OPT_DIR / f"{module_name}.py"
        if not file_path.exists():
            continue

        source = _read_source(file_path)
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            results.append({
                "id": None,
                "module": module_name,
                "file_path": str(file_path.relative_to(_ROOT_DIR)),
                "error": f"SyntaxError: {e}",
            })
            continue

        category = module_name[0]

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            # 跳过私有辅助函数（以 _ 开头）
            if node.name.startswith("_"):
                continue

            opt_id = _find_opt_id_for_func(source, node.lineno)
            title  = _find_opt_title_for_func(source, node.lineno)

            raw_doc = ast.get_docstring(node) or ""
            doc_sections = _parse_docstring_sections(raw_doc)

            sig_params = _parse_func_signature(node)
            # 将 docstring 里解析到的 description 补充到 sig_params
            doc_param_map = {p["name"]: p["description"] for p in doc_sections["parameters"]}
            for p in sig_params:
                p["description"] = doc_param_map.get(p["name"], "")

            line_start, line_end = _get_func_source_lines(source, node)

            results.append({
                "id": opt_id,
                "category": category,
                "category_name": _CATEGORY_NAMES.get(category, ""),
                "function_name": node.name,
                "module": module_name,
                "file_path": str(file_path.relative_to(_ROOT_DIR)),
                "title": title or "",
                "short_description": doc_sections["short_description"],
                "long_description": doc_sections["long_description"],
                "parameters": sig_params,
                "returns": {
                    "annotation": _get_return_annotation(node),
                    "type": doc_sections["returns"]["type"],
                    "description": doc_sections["returns"]["description"],
                },
                "raises": doc_sections["raises"],
                "notes": doc_sections["notes"],
                "example": doc_sections["example"],
                "is_implemented": not _is_stub(node),
                "source_line_start": line_start,
                "source_line_end": line_end,
            })

    # 按 ID 排序（None 排最后）
    results.sort(key=lambda x: (x["id"] is None, x["id"] or ""))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 内部：函数替换工具
# ─────────────────────────────────────────────────────────────────────────────

def _find_function_in_module(
    module_name: str,
    function_name: str,
) -> Optional[Dict[str, Any]]:
    """在指定模块文件中定位目标函数，返回其位置信息。"""
    file_path = _OPT_DIR / f"{module_name}.py"
    if not file_path.exists():
        return None
    source = _read_source(file_path)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            line_start, line_end = _get_func_source_lines(source, node)
            return {
                "file_path": file_path,
                "source": source,
                "line_start": line_start,
                "line_end": line_end,
                "node": node,
            }
    return None


def _resolve_target(target: str) -> Optional[Dict[str, Any]]:
    """
    根据 target 字符串（可为优化点 ID 如 'A1'，或函数名如 'normalize_coordinates'）
    定位对应的函数，返回位置信息字典。
    """
    all_info = get_optimizable_functions_info()

    # 尝试按 ID 匹配
    by_id = [f for f in all_info if f.get("id") == target.upper()]
    if by_id:
        hit = by_id[0]
        return _find_function_in_module(hit["module"], hit["function_name"])

    # 尝试按函数名匹配
    by_name = [f for f in all_info if f["function_name"] == target]
    if by_name:
        hit = by_name[0]
        return _find_function_in_module(hit["module"], hit["function_name"])

    return None


def _validate_new_impl(new_impl_str: str, expected_func_name: str) -> Dict[str, Any]:
    """
    校验新实现字符串：
      - 是否为合法 Python
      - 是否包含同名函数定义
      - 新函数是否仍然是 NotImplementedError 存根
    """
    try:
        tree = ast.parse(textwrap.dedent(new_impl_str))
    except SyntaxError as e:
        return {"valid": False, "error": f"SyntaxError: {e}", "is_stub": True}

    func_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    matching = [n for n in func_nodes if n.name == expected_func_name]

    if not matching:
        # 如果只有一个函数定义，允许函数名不同（自动识别）
        if len(func_nodes) == 1:
            return {
                "valid": True,
                "error": None,
                "detected_name": func_nodes[0].name,
                "is_stub": _is_stub(func_nodes[0]),
            }
        return {
            "valid": False,
            "error": f"新实现中未找到函数 '{expected_func_name}'",
            "is_stub": True,
        }

    return {
        "valid": True,
        "error": None,
        "detected_name": expected_func_name,
        "is_stub": _is_stub(matching[0]),
    }


def _backup_file(file_path: Path) -> Path:
    """备份原文件，返回备份路径。"""
    _BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = _BACKUP_DIR / f"{file_path.stem}_{ts}.py.bak"
    shutil.copy2(file_path, backup_path)
    return backup_path


def _replace_function_in_source(
    source: str,
    line_start: int,
    line_end: int,
    new_impl_str: str,
) -> str:
    """
    将 source 中第 line_start~line_end 行（1-based, inclusive）
    替换为 new_impl_str，保留前后其余内容。
    """
    lines = source.splitlines(keepends=True)
    # 检测新实现的缩进（应为 0，即顶层函数）
    new_code = textwrap.dedent(new_impl_str)
    # 确保末尾有换行
    if not new_code.endswith("\n"):
        new_code += "\n"

    before = lines[:line_start - 1]
    after  = lines[line_end:]     # line_end 是 inclusive，所以跳过它

    return "".join(before) + new_code + "".join(after)


def _reload_module(module_name: str, file_path: Path) -> Dict[str, Any]:
    """尝试重新导入模块，返回 {success, error}。"""
    full_module = f"lib.optimizations.{module_name}"
    try:
        spec = importlib.util.spec_from_file_location(full_module, file_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # 更新 sys.modules 缓存
        sys.modules[full_module] = mod
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": traceback.format_exc(limit=5)}


# ─────────────────────────────────────────────────────────────────────────────
# 内部：性能评估（轻量版）
# ─────────────────────────────────────────────────────────────────────────────

def _quick_eval(
    function_name: str,
    module_name: str,
    eval_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    轻量性能评估：
      - 若 eval_config 为 None 或 eval_config['mode']=='import_only'，仅做导入测试。
      - 若 eval_config['mode']=='call_test'，调用函数并传入 eval_config['args']，
        记录耗时与返回值摘要。
      - 若 eval_config['mode']=='training_eval'，触发完整的 val() 评估，
        需要 eval_config 中提供 'loader'、'model'、'device'、'args'、'num_nodes_list'。
    """
    if eval_config is None or eval_config.get("mode") == "import_only":
        return {
            "mode": "import_only",
            "note": "仅校验语法与导入，未运行实际计算。",
            "val_L2_relative": None,
            "pde_loss": None,
            "bc_loss": None,
            "call_time_seconds": None,
            "return_summary": None,
            "error": None,
        }

    mode = eval_config.get("mode", "import_only")

    if mode == "call_test":
        args   = eval_config.get("args",   [])
        kwargs = eval_config.get("kwargs", {})
        full_module = f"lib.optimizations.{module_name}"
        try:
            mod  = sys.modules.get(full_module)
            if mod is None:
                file_path = _OPT_DIR / f"{module_name}.py"
                spec = importlib.util.spec_from_file_location(full_module, file_path)
                mod  = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            fn = getattr(mod, function_name)
            t0 = time.perf_counter()
            ret = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            # 构造返回值摘要
            try:
                import torch, numpy as np
                if isinstance(ret, (list, tuple)):
                    summary = f"{type(ret).__name__}[{len(ret)}]"
                elif isinstance(ret, torch.Tensor):
                    summary = f"Tensor{list(ret.shape)}"
                elif isinstance(ret, np.ndarray):
                    summary = f"ndarray{list(ret.shape)}"
                else:
                    summary = repr(ret)[:200]
            except Exception:
                summary = str(type(ret))
            return {
                "mode": "call_test",
                "val_L2_relative": None,
                "pde_loss": None,
                "bc_loss": None,
                "call_time_seconds": round(elapsed, 6),
                "return_summary": summary,
                "error": None,
            }
        except NotImplementedError:
            return {
                "mode": "call_test",
                "val_L2_relative": None,
                "pde_loss": None,
                "bc_loss": None,
                "call_time_seconds": None,
                "return_summary": None,
                "error": "函数仍为 NotImplementedError 存根，未能运行。",
            }
        except Exception:
            return {
                "mode": "call_test",
                "val_L2_relative": None,
                "pde_loss": None,
                "bc_loss": None,
                "call_time_seconds": None,
                "return_summary": None,
                "error": traceback.format_exc(limit=5),
            }

    if mode == "training_eval":
        # 完整 val() 评估，需调用方提供已初始化的 model/loader 等
        try:
            loader         = eval_config["loader"]
            model          = eval_config["model"]
            device         = eval_config["device"]
            args_ns        = eval_config["args"]
            num_nodes_list = eval_config["num_nodes_list"]
            problem        = eval_config.get("problem", "darcy")  # "darcy" | "plate"

            t0 = time.perf_counter()
            if problem == "darcy":
                # 动态导入，避免顶层循环依赖
                _train_mod = importlib.import_module("lib.utils_darcy_train")
                val_err = _train_mod.val(model, loader, device, args_ns, num_nodes_list)
            else:
                _train_mod = importlib.import_module("lib.utils_plate_train")
                val_err = _train_mod.val(model, loader, args_ns, device, num_nodes_list)
            elapsed = time.perf_counter() - t0

            val_err_scalar = float(val_err) if hasattr(val_err, "__float__") else None
            return {
                "mode": "training_eval",
                "val_L2_relative": val_err_scalar,
                "pde_loss": None,
                "bc_loss": None,
                "call_time_seconds": round(elapsed, 3),
                "return_summary": f"val_L2={val_err_scalar:.6f}" if val_err_scalar else None,
                "error": None,
            }
        except Exception:
            return {
                "mode": "training_eval",
                "val_L2_relative": None,
                "pde_loss": None,
                "bc_loss": None,
                "call_time_seconds": None,
                "return_summary": None,
                "error": traceback.format_exc(limit=5),
            }

    return {
        "mode": mode,
        "val_L2_relative": None,
        "pde_loss": None,
        "bc_loss": None,
        "call_time_seconds": None,
        "return_summary": None,
        "error": f"未知 eval mode: {mode}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 函数 2：替换可优化函数并返回性能指标
# ─────────────────────────────────────────────────────────────────────────────

def replace_and_evaluate(
    target: str,
    new_impl_str: str,
    eval_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """替换指定优化函数的实现，并返回结构化性能指标 JSON。

    Args:
        target (str): 目标函数标识，可以是：
            - 优化点 ID，如 "A1"、"B4"、"G6"（不区分大小写）。
            - Python 函数名，如 "normalize_coordinates"、"aggregate_geometry_features"。
        new_impl_str (str): 新的函数实现代码字符串（完整函数定义，含 def 行和 docstring）。
            函数名需与目标一致（或字符串中只有一个函数定义时自动对齐）。
            示例（字符串内容）::

                def normalize_coordinates(coors, method="minmax", stats=None):
                    # docstring 省略
                    x_min = coors.min(axis=0)
                    x_max = coors.max(axis=0)
                    return (coors - x_min) / (x_max - x_min + 1e-8), {"x_min": x_min, "x_max": x_max}
        eval_config (dict | None): 评估配置，控制性能测量方式。
            None 或省略 → 仅做语法 + 导入检查。
            可选结构：
              {
                "mode": "import_only"    # 仅导入检查（默认）
                        "call_test"      # 调用函数并计时
                        "training_eval"  # 使用已有 model/loader 跑完整 val()
                "args":   [...],         # call_test 时的位置参数
                "kwargs": {...},         # call_test 时的关键字参数
                "loader":  ...,          # training_eval 时的 DataLoader
                "model":   ...,          # training_eval 时的模型
                "device":  ...,          # training_eval 时的 device
                "args":    ...,          # training_eval 时的 argparse.Namespace
                "num_nodes_list": (...), # training_eval 时的节点数元组
                "problem": "darcy"       # training_eval 时的问题类型
              }

    Returns:
        dict: 替换结果与性能指标，结构如下：
            {
              "success": bool,
              "target_id": str,            # 解析到的优化点 ID，如 "A1"
              "function_name": str,
              "module": str,
              "file_path": str,
              "timestamp": str,            # ISO 格式时间戳
              "backup_path": str | null,   # 备份文件路径
              "replacement": {
                "status": "replaced" | "failed",
                "syntax_valid": bool,
                "import_valid": bool,
                "is_implemented": bool,    # 新实现是否已非存根
                "error": str | null
              },
              "performance": {
                "mode": str,               # 评估模式
                "val_L2_relative": float | null,
                "pde_loss": float | null,
                "bc_loss": float | null,
                "call_time_seconds": float | null,
                "return_summary": str | null,
                "error": str | null
              }
            }

    Raises:
        ValueError: 若 target 无法解析为已知优化函数。

    Example:
        >>> impl = (
        ...     "def normalize_coordinates(coors, method='minmax', stats=None):\\n"
        ...     "    x_min = coors.min(axis=0)\\n"
        ...     "    x_max = coors.max(axis=0)\\n"
        ...     "    return (coors - x_min) / (x_max - x_min + 1e-8), {'x_min': x_min, 'x_max': x_max}\\n"
        ... )
        >>> result = replace_and_evaluate("A1", impl, eval_config={"mode": "import_only"})
        >>> print(json.dumps(result, ensure_ascii=False, indent=2))
    """
    timestamp = datetime.now().isoformat()

    # ── 1. 解析 target → 定位文件 + 函数 ──────────────────────────────────────
    location = _resolve_target(target)
    if location is None:
        return {
            "success": False,
            "target_id": target,
            "function_name": None,
            "module": None,
            "file_path": None,
            "timestamp": timestamp,
            "backup_path": None,
            "replacement": {
                "status": "failed",
                "syntax_valid": False,
                "import_valid": False,
                "is_implemented": False,
                "error": f"无法解析 target='{target}'，请提供有效的优化点 ID（如 'A1'）或函数名。",
            },
            "performance": {
                "mode": "import_only",
                "val_L2_relative": None,
                "pde_loss": None,
                "bc_loss": None,
                "call_time_seconds": None,
                "return_summary": None,
                "error": "未执行，替换失败。",
            },
        }

    file_path: Path = location["file_path"]
    source: str     = location["source"]
    line_start: int = location["line_start"]
    line_end: int   = location["line_end"]
    func_node       = location["node"]
    func_name: str  = func_node.name
    module_name: str = file_path.stem

    # 从注册表取 ID
    all_info = get_optimizable_functions_info()
    matched  = [f for f in all_info if f["function_name"] == func_name and f["module"] == module_name]
    opt_id   = matched[0]["id"] if matched else None

    # ── 2. 校验新实现的语法 ────────────────────────────────────────────────────
    validation = _validate_new_impl(new_impl_str, func_name)
    if not validation["valid"]:
        return {
            "success": False,
            "target_id": opt_id or target,
            "function_name": func_name,
            "module": module_name,
            "file_path": str(file_path.relative_to(_ROOT_DIR)),
            "timestamp": timestamp,
            "backup_path": None,
            "replacement": {
                "status": "failed",
                "syntax_valid": False,
                "import_valid": False,
                "is_implemented": False,
                "error": validation["error"],
            },
            "performance": {
                "mode": "import_only",
                "val_L2_relative": None,
                "pde_loss": None,
                "bc_loss": None,
                "call_time_seconds": None,
                "return_summary": None,
                "error": "未执行，语法验证失败。",
            },
        }

    # ── 3. 备份原文件 ──────────────────────────────────────────────────────────
    backup_path = _backup_file(file_path)

    # ── 4. 替换函数 ────────────────────────────────────────────────────────────
    try:
        new_source = _replace_function_in_source(source, line_start, line_end, new_impl_str)
        file_path.write_text(new_source, encoding="utf-8")
    except Exception as e:
        # 恢复备份
        shutil.copy2(backup_path, file_path)
        return {
            "success": False,
            "target_id": opt_id or target,
            "function_name": func_name,
            "module": module_name,
            "file_path": str(file_path.relative_to(_ROOT_DIR)),
            "timestamp": timestamp,
            "backup_path": str(backup_path.relative_to(_ROOT_DIR)),
            "replacement": {
                "status": "failed",
                "syntax_valid": True,
                "import_valid": False,
                "is_implemented": False,
                "error": f"写文件失败: {e}",
            },
            "performance": {
                "mode": "import_only",
                "val_L2_relative": None,
                "pde_loss": None,
                "bc_loss": None,
                "call_time_seconds": None,
                "return_summary": None,
                "error": "未执行，替换失败。",
            },
        }

    # ── 5. 重新导入模块，检查 import 是否正常 ────────────────────────────────
    reload_result = _reload_module(module_name, file_path)
    if not reload_result["success"]:
        # 恢复备份
        shutil.copy2(backup_path, file_path)
        return {
            "success": False,
            "target_id": opt_id or target,
            "function_name": func_name,
            "module": module_name,
            "file_path": str(file_path.relative_to(_ROOT_DIR)),
            "timestamp": timestamp,
            "backup_path": str(backup_path.relative_to(_ROOT_DIR)),
            "replacement": {
                "status": "failed",
                "syntax_valid": True,
                "import_valid": False,
                "is_implemented": not validation.get("is_stub", True),
                "error": reload_result["error"],
            },
            "performance": {
                "mode": "import_only",
                "val_L2_relative": None,
                "pde_loss": None,
                "bc_loss": None,
                "call_time_seconds": None,
                "return_summary": None,
                "error": "未执行，模块导入失败，原文件已恢复。",
            },
        }

    # ── 6. 性能评估 ────────────────────────────────────────────────────────────
    perf = _quick_eval(func_name, module_name, eval_config)

    return {
        "success": True,
        "target_id": opt_id or target,
        "function_name": func_name,
        "module": module_name,
        "file_path": str(file_path.relative_to(_ROOT_DIR)),
        "timestamp": timestamp,
        "backup_path": str(backup_path.relative_to(_ROOT_DIR)),
        "replacement": {
            "status": "replaced",
            "syntax_valid": True,
            "import_valid": True,
            "is_implemented": not validation.get("is_stub", True),
            "error": None,
        },
        "performance": perf,
    }
