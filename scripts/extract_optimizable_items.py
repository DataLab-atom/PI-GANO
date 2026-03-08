"""
extract_optimizable_items.py

解析 lib/*_optimizable.py 中的所有函数，提取：
  - 函数名
  - 所属文件
  - 参数列表（名称 + 类型注解）
  - 返回类型注解
  - docstring（完整原文）

用法：
    python scripts/extract_optimizable_items.py
    python scripts/extract_optimizable_items.py --out optimizable_items.json
    python scripts/extract_optimizable_items.py --pretty
"""

import ast
import json
import sys
import argparse
from pathlib import Path


# ── 工具函数 ─────────────────────────────────────────────────────────────

def annotation_to_str(node) -> str:
    """将 AST 注解节点转为可读字符串。"""
    if node is None:
        return None
    return ast.unparse(node)


def parse_file(path: Path) -> list[dict]:
    """解析单个 *_optimizable.py，返回函数列表。"""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    results = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        # 参数
        args_info = []
        func_args = node.args
        # 构建 annotation 查找表
        annotations = {arg.arg: arg.annotation for arg in func_args.args}
        for arg in func_args.args:
            args_info.append({
                "name": arg.arg,
                "type": annotation_to_str(annotations.get(arg.arg)),
            })

        # 返回类型
        return_type = annotation_to_str(node.returns)

        # docstring
        docstring = ast.get_docstring(node) or ""

        results.append({
            "function": node.name,
            "file": path.name,
            "args": args_info,
            "returns": return_type,
            "docstring": docstring,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract optimizable function metadata as JSON")
    parser.add_argument("--out", default=None, help="输出文件路径（默认 stdout）")
    parser.add_argument("--pretty", action="store_true", help="缩进格式输出")
    args = parser.parse_args()

    root = Path(__file__).parent.parent / "lib"
    files = sorted(root.glob("*_optimizable.py"))

    if not files:
        print("未找到任何 *_optimizable.py 文件", file=sys.stderr)
        sys.exit(1)

    all_items = []
    for f in files:
        all_items.extend(parse_file(f))

    indent = 2 if args.pretty else None
    output = json.dumps(all_items, ensure_ascii=False, indent=indent)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"已写入 {args.out}，共 {len(all_items)} 个函数", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
