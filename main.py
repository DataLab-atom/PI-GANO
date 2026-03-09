"""
main.py  ─  PI-GANO × ReEvo2D 一键启动入口

用法
----
# 最简启动（使用环境变量中的 API key）
python main.py --model deepseek-coder

# 完整参数
python main.py \
    --model      deepseek-coder \
    --api-key    $OPENAI_API_KEY \
    --base-url   https://api.example.com/v1 \
    --temperature 0.5 \
    --max-fe     500 \
    --pop-size   20 \
    --init-pop-size 50 \
    --mutation-rate 0.8 \
    --epochs     50 \
    --output-dir ./runs/exp1

环境变量（可替代命令行参数）
---------------------------
OPENAI_API_KEY   LLM API Key
OPENAI_BASE_URL  API Base URL（如使用第三方代理）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# ── 路径配置 ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# 把 optimizer/ 加入 sys.path，使 optimizer/reevo2d.py 内的
# "from utils.utils import *" 能找到 optimizer/utils/utils.py
sys.path.insert(0, str(ROOT_DIR / "optimizer"))
sys.path.insert(0, str(ROOT_DIR))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PI-GANO × ReEvo2D：用 LLM 进化优化 PI-GANO 的可优化函数",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── LLM 配置 ──────────────────────────────────────────────────────────────
    p.add_argument("--model", default="deepseek-coder",
                   help="LLM 模型名，如 gpt-4o-mini / deepseek-coder / GLM-4")
    p.add_argument("--api-key", default=None,
                   help="LLM API Key（也可通过 OPENAI_API_KEY 环境变量传入）")
    p.add_argument("--base-url", default=None,
                   help="API Base URL（也可通过 OPENAI_BASE_URL 环境变量传入）")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="LLM 采样温度")

    # ── 进化算法配置 ───────────────────────────────────────────────────────────
    p.add_argument("--max-fe", type=int, default=500,
                   help="最大函数评估次数（停止条件）")
    p.add_argument("--pop-size", type=int, default=20,
                   help="每代种群大小")
    p.add_argument("--init-pop-size", type=int, default=50,
                   help="初始种群大小")
    p.add_argument("--mutation-rate", type=float, default=0.8,
                   help="变异率")

    # ── PI-GANO 训练配置 ───────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=None,
                   help="每次评估的训练轮数（None=使用 configs/*.yaml 默认值）")

    # ── 输出配置 ──────────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="./runs/pi_gano_reevo",
                   help="运行输出目录（存放中间结果、日志）")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args()


def _build_cfg(args: argparse.Namespace) -> SimpleNamespace:
    """将命令行参数组装成 ReEv2d 期望的 cfg 结构（SimpleNamespace）。"""
    problem = SimpleNamespace(
        problem_name="pi_gano",
        description=(
            "Optimize the internal functions of PI-GANO (Physics-Informed Geometry-Aware "
            "Neural Operator) to minimize the relative L2 error on Darcy flow and thin-plate "
            "bending problems. The functions include geometry encoders, loss terms, and data "
            "preprocessing utilities defined in lib/*_optimizable.py."
        ),
        problem_size=1,
        obj_type="min",
        problem_type="default",
    )
    return SimpleNamespace(
        algorithm="reevo2d",
        problem=problem,
        reevo_func_index=[0],
        max_fe=args.max_fe,
        pop_size=args.pop_size,
        init_pop_size=args.init_pop_size,
        mutation_rate=args.mutation_rate,
        timeout=600,           # 单次评估超时（秒），训练需要时间
        model=args.model,
        temperature=args.temperature,
        diversify_init_pop=True,
    )


def _build_client(args: argparse.Namespace):
    """初始化 LLM 客户端。"""
    model       = args.model
    temperature = args.temperature
    api_key     = args.api_key or os.environ.get("OPENAI_API_KEY")
    base_url    = args.base_url or os.environ.get("OPENAI_BASE_URL")

    from utils.llm_client.openai import OpenAIClient
    from utils.llm_client.llama_api import LlamaAPIClient

    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return OpenAIClient(model, temperature, base_url=base_url, api_key=api_key)
    elif model.startswith("GLM"):
        from utils.llm_client.zhipuai import ZhipuAIClient
        return ZhipuAIClient(model, temperature, base_url=base_url, api_key=api_key)
    else:
        # deepseek / gemini / llama 等，走 OpenAI 兼容接口
        return OpenAIClient(model, temperature, base_url=base_url, api_key=api_key)


def main() -> None:
    args = parse_result = _parse_args()

    # ── 日志 ─────────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # ── 输出目录 ──────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir).resolve()
    # ReEv2d 会把中间文件写到 cwd/problems/pi_gano/
    work_dir = output_dir / "problems" / "pi_gano"
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)            # ReEv2d 把响应/stdout 写到 cwd
    logging.info(f"Output directory : {output_dir}")
    logging.info(f"PI-GANO root     : {ROOT_DIR}")
    logging.info(f"Model            : {args.model}")
    logging.info(f"Max FE           : {args.max_fe}")

    # ── 组装 cfg ─────────────────────────────────────────────────────────────
    cfg = _build_cfg(args)

    # ── LLM 客户端 ────────────────────────────────────────────────────────────
    client = _build_client(args)

    # ── PI-GANO 评估函数 ──────────────────────────────────────────────────────
    from u2e_api import get_funcs, evaluate_funcs
    import functools
    _evaluate_funcs = functools.partial(evaluate_funcs, epochs=args.epochs)

    # ── 启动 ReEvo2D ─────────────────────────────────────────────────────────
    from reevo2d import ReEv2d

    optimizer = ReEv2d(
        cfg=cfg,
        root_dir=str(ROOT_DIR),     # prompts/ 和 output 都相对于 ROOT_DIR
        generator_llm=client,
        get_funcs=get_funcs,
        evaluate_funcs=_evaluate_funcs,
    )

    best_code, best_code_path = optimizer.evolve()

    logging.info("=" * 60)
    logging.info("Evolution complete.")
    logging.info(f"Best code path : {best_code_path}")
    logging.info("Best code:\n" + str(best_code))


if __name__ == "__main__":
    main()
