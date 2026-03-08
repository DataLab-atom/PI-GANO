"""
run_experiment.py

给定一批函数替换规格，将新实现注入对应的 *_optimizable.py 文件，
运行相关场景的训练+测试，收集指标，完成后自动还原所有文件。

━━━━━━━━━━━━━ 作为模块调用 ━━━━━━━━━━━━━
    from scripts.run_experiment import run_experiment

    results = run_experiment(
        replacements=[
            {
                "function": "build_geometry_mlp",
                "file": "model_darcy_optimizable.py",
                "code": '''
def build_geometry_mlp(in_dim: int, fc_dim: int, n_layer: int) -> nn.Sequential:
    \"\"\"改进版：加入 LayerNorm。\"\"\"
    layers = [nn.Linear(in_dim, fc_dim), nn.LayerNorm(fc_dim), nn.Tanh()]
    for _ in range(n_layer - 1):
        layers += [nn.Linear(fc_dim, fc_dim), nn.LayerNorm(fc_dim), nn.Tanh()]
    layers.append(nn.Linear(fc_dim, fc_dim))
    return nn.Sequential(*layers)
''',
            }
        ],
        epochs=10,          # 覆盖 config 中的训练轮数（None = 使用配置文件默认值）
    )
    # results: {"darcy": {"test_relative_L2": 0.048, ...}, ...}

━━━━━━━━━━━━━ 命令行调用 ━━━━━━━━━━━━━
    python scripts/run_experiment.py --spec my_exp.json --epochs 10 --pretty

替换规格 JSON 格式（数组）：
    [
      {
        "function": "build_geometry_mlp",
        "file": "model_darcy_optimizable.py",
        "code": "def build_geometry_mlp(...):\\n    ..."
      }
    ]

输出格式：
    {
      "darcy": {
        "val_relative_L2_last": 0.052,   // 最后一个 visual_freq 的验证 L2 误差
        "test_relative_L2":     0.048,   // 最优模型在测试集上的 L2 相对误差
        "pde_loss_last":        1.2e-4,  // 最后一个 visual_freq 的 PDE 残差损失
        "bc_loss_last":         3.1e-5,  // [darcy only] 边界条件损失
        "fix_bc_loss_last":     ...,     // [plate only]
        "free_bc_loss_last":    ...,     // [plate only]
        "load_bc_loss_last":    ...,     // [plate only]
        "returncode":           0,       // 子进程退出码（0=成功）
        "stdout":               "..."    // 完整 stdout（--no-stdout 可省略）
      },
      "plate": { ... }
    }
"""

from __future__ import annotations

import ast
import json
import re
import sys
import textwrap
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# ── 路径常量 ──────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent.parent
LIB_DIR    = ROOT / "lib"
CONFIG_DIR = ROOT / "configs"

# ── 场景 → 受影响问题的映射 ──────────────────────────────────────────────────

# 文件级别：如果某文件中的任意函数被替换，需要运行对应场景
_FILE_TO_PROBLEMS: dict[str, list[str]] = {
    "model_darcy_optimizable.py":          ["darcy"],
    "model_plate_optimizable.py":          ["plate"],
    "utils_darcy_train_optimizable.py":    ["darcy"],
    "utils_plate_train_optimizable.py":    ["plate"],
    "utils_data_optimizable.py":           ["darcy", "plate"],
    "utils_losses_optimizable.py":         [],   # 由下方函数级映射精确解析
}

# utils_losses_optimizable.py 中各函数的精确归属
_LOSSES_FUNC_TO_PROBLEMS: dict[str, list[str]] = {
    "darcy_source_term":            ["darcy"],
    "compute_total_loss":           ["darcy"],
    "compute_shear_strain":         ["plate"],
    "compute_free_boundary_stress": ["plate"],
    "compute_total_loss_plate":     ["plate"],
    "apply_gradient_clipping":      ["darcy", "plate"],
}

# ── 场景定义 ─────────────────────────────────────────────────────────────────

_SCENARIOS: dict[str, dict] = {
    "darcy": {
        "script": "PINO_darcy_training.py",
        "argv":   ["--model=GANO", "--geo_node=vary_bound", "--phase=train"],
        "config": "GANO_Darcy_DG.yaml",
    },
    "plate": {
        "script": "PINO_plate_training.py",
        "argv":   ["--model=GANO", "--geo_node=vary_bound", "--phase=train"],
        "config": "GANO_plate_stress_DG.yaml",
    },
}

# ── 指标正则表达式 ────────────────────────────────────────────────────────────

_FLOAT = r"([\deE+\-\.]+)"
_METRIC_RE: dict[str, re.Pattern] = {
    "val_relative_L2_last": re.compile(rf"Current epoch error:\s+{_FLOAT}"),
    "pde_loss_last":        re.compile(rf"current epochs pde loss:\s+{_FLOAT}"),
    "bc_loss_last":         re.compile(rf"bc loss:\s+{_FLOAT}"),
    "fix_bc_loss_last":     re.compile(rf"fix bc loss:\s+{_FLOAT}"),
    "free_bc_loss_last":    re.compile(rf"free bc loss:\s+{_FLOAT}"),
    "load_bc_loss_last":    re.compile(rf"load bc loss:\s+{_FLOAT}"),
    "test_relative_L2":     re.compile(rf"Best L2 relative error on test loader:\s+{_FLOAT}"),
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  核心工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_function_range(source: str, func_name: str) -> tuple[int, int]:
    """返回函数在源代码中的起止行号（1-indexed，含首尾）。"""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node.lineno, node.end_lineno
    raise ValueError(f"函数 '{func_name}' 未找到")


def _patch_source(source: str, func_name: str, new_code: str) -> str:
    """将 source 中 func_name 函数的实现替换为 new_code，返回新 source。"""
    start, end = _find_function_range(source, func_name)
    # 规范化缩进（允许调用方用任意缩进传入代码）
    new_code = textwrap.dedent(new_code).strip() + "\n"
    lines = source.splitlines(keepends=True)
    return "".join(lines[: start - 1]) + new_code + "\n" + "".join(lines[end:])


def _patch_config_epochs(config_path: Path, epochs: int) -> str:
    """临时将 YAML config 中的 epochs 覆盖为给定值，返回原始内容。"""
    original = config_path.read_text(encoding="utf-8")
    patched = re.sub(r"(epochs:\s*)\d+", rf"\g<1>{epochs}", original)
    config_path.write_text(patched, encoding="utf-8")
    return original


def _parse_metrics(stdout: str) -> dict:
    """从子进程 stdout 中解析所有已知指标，取每类最后一次出现的值。"""
    metrics: dict[str, float] = {}
    for name, pattern in _METRIC_RE.items():
        matches = pattern.findall(stdout)
        if matches:
            try:
                metrics[name] = float(matches[-1])
            except ValueError:
                pass
    return metrics


def _run_scenario(scenario_name: str, epochs: Optional[int]) -> dict:
    """运行单个场景，返回解析后的指标 + returncode + stdout。"""
    scene = _SCENARIOS[scenario_name]
    config_path = CONFIG_DIR / scene["config"]

    # 覆盖 epochs
    original_config: Optional[str] = None
    if epochs is not None:
        original_config = _patch_config_epochs(config_path, epochs)

    # 确保输出目录存在
    (ROOT / "res" / "saved_models").mkdir(parents=True, exist_ok=True)
    (ROOT / "res" / "plots").mkdir(parents=True, exist_ok=True)

    try:
        cmd = [sys.executable, scene["script"]] + scene["argv"]
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        stdout = proc.stdout + ("\n[STDERR]\n" + proc.stderr if proc.stderr.strip() else "")
        returncode = proc.returncode
    finally:
        if original_config is not None:
            config_path.write_text(original_config, encoding="utf-8")

    metrics = _parse_metrics(stdout)
    metrics["returncode"] = returncode
    metrics["stdout"] = stdout
    return metrics


def _infer_problems(replacements: list[dict]) -> list[str]:
    """根据被替换函数所在文件推断需要运行的场景列表。"""
    problems: set[str] = set()
    for rep in replacements:
        f = rep["file"]
        if f == "utils_losses_optimizable.py":
            problems.update(
                _LOSSES_FUNC_TO_PROBLEMS.get(rep["function"], ["darcy", "plate"])
            )
        else:
            problems.update(_FILE_TO_PROBLEMS.get(f, []))
    return sorted(problems)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  公开 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_experiment(
    replacements: list[dict],
    epochs: Optional[int] = None,
    scenarios: Optional[list[str]] = None,
) -> dict:
    """
    注入新函数实现，运行对应训练场景，返回指标字典，并还原所有文件。

    Args:
        replacements: 函数替换规格列表，每项格式：
            {
              "function": str,  # 函数名，须与 *_optimizable.py 中的定义精确匹配
              "file":     str,  # 所在文件名（仅文件名，不含路径）
              "code":     str,  # 完整函数定义字符串（def ... 到末尾，支持任意缩进）
            }
        epochs: 覆盖 YAML config 中的训练轮数；None 表示使用原始配置。
        scenarios: 指定运行场景 ["darcy", "plate"] 的子集；
                   None 表示根据 replacements 自动推断。

    Returns:
        dict，键为场景名，值为指标字典：
        {
          "darcy": {
            "val_relative_L2_last": float,   # 最后一个打印周期的验证误差
            "test_relative_L2":     float,   # 最优模型测试误差（主要指标）
            "pde_loss_last":        float,
            "bc_loss_last":         float,   # darcy 专有
            "fix_bc_loss_last":     float,   # plate 专有
            "free_bc_loss_last":    float,   # plate 专有
            "load_bc_loss_last":    float,   # plate 专有
            "returncode":           int,     # 0 = 成功
            "stdout":               str,     # 完整训练输出
          },
          "plate": { ... },
        }

    Raises:
        ValueError: 函数名不存在于目标文件中。
        RuntimeError: 文件补丁失败（文件已自动还原）。
    """
    # 1. 推断场景
    active_scenarios = scenarios if scenarios is not None else _infer_problems(replacements)
    if not active_scenarios:
        raise ValueError("未找到任何有效场景，请检查 replacements 中的 'file' 字段")

    # 2. 逐个补丁文件（同文件可有多个函数替换，顺序应用）
    backups: dict[Path, str] = {}
    try:
        for rep in replacements:
            func   = rep["function"]
            fname  = rep["file"]
            code   = rep["code"]
            fpath  = LIB_DIR / fname
            # 以当前（可能已部分补丁）内容为基础继续替换
            current = fpath.read_text(encoding="utf-8")
            if fpath not in backups:
                backups[fpath] = current   # 只保存原始版本
            patched = _patch_source(current, func, code)
            fpath.write_text(patched, encoding="utf-8")
            print(f"[patch] {fname}::{func}", file=sys.stderr)
    except Exception as exc:
        for path, orig in backups.items():
            path.write_text(orig, encoding="utf-8")
        raise RuntimeError(f"补丁失败，已还原所有文件。原因：{exc}") from exc

    # 3. 运行各场景
    results: dict[str, dict] = {}
    try:
        for sc in active_scenarios:
            print(f"[run]   scenario={sc}, epochs={epochs}", file=sys.stderr)
            results[sc] = _run_scenario(sc, epochs)
            rc = results[sc]["returncode"]
            print(f"[done]  scenario={sc}, returncode={rc}, "
                  f"test_L2={results[sc].get('test_relative_L2', 'N/A')}", file=sys.stderr)
    finally:
        # 4. 无论是否出错，还原所有文件
        for path, orig in backups.items():
            path.write_text(orig, encoding="utf-8")
        print("[restore] 所有文件已还原", file=sys.stderr)

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI 入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    parser = argparse.ArgumentParser(
        description="注入函数实现并运行对应训练场景，输出指标 JSON"
    )
    parser.add_argument(
        "--spec", required=True,
        help="替换规格 JSON 文件路径，格式见模块 docstring"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="覆盖 config 中的训练轮数（留空则使用配置文件默认值）"
    )
    parser.add_argument(
        "--scenarios", default=None,
        help="指定运行场景，逗号分隔，如 darcy,plate（留空则自动推断）"
    )
    parser.add_argument(
        "--out", default=None,
        help="将结果写入该 JSON 文件（留空则打印到 stdout）"
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="缩进格式输出"
    )
    parser.add_argument(
        "--no-stdout", dest="no_stdout", action="store_true",
        help="结果中省略 stdout 字段（减少输出体积）"
    )
    args = parser.parse_args()

    replacements = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    scenarios    = args.scenarios.split(",") if args.scenarios else None

    results = run_experiment(replacements, epochs=args.epochs, scenarios=scenarios)

    if args.no_stdout:
        for v in results.values():
            v.pop("stdout", None)

    output = json.dumps(results, ensure_ascii=False, indent=2 if args.pretty else None)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"结果已写入 {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
