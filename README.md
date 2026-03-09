# PI-GANO

This repository is the official implementation of the paper: [Physics-Informed Geometry-Aware Neural Operator](https://www.sciencedirect.com/science/article/pii/S0045782524007941?via%3Dihub), published in Journal of Computer Methods in Applied Mechanics and Engineering. The arxiv version of paper can also be found [here](https://arxiv.org/html/2408.01600v1).

Our research explores physics-informed machine learning methods for **variable domain geometry** where PDE solution domains are represented as a set of collocation point coordinates, making the approach **geometry-aware**. In this paper, we introduce the first neural operator model that is geometry-aware and can be trained without any FEM data needed. However, the model architecture we propose in this paper is intentionally kept straightforward, and we encourage researchers to explore and develop more advanced architectures to further enhance this approach. 

## Overview

If you're interested in using our well-trained model, please refer to the **"User mode"** section. For those with similar research interests looking to explore more advanced model architectures and training algorithms, please check the **"Developer mode"** section. This work is also one of our work for developing [**Neural Operators as Foundation Model for solving PDEs**](https://github.com/WeihengZ/Physics-informed-Neural-Foundation-Operator). Please feel free to check it out as well if you are interested! We are excited to see more and more interesting ideas coming out for this research goal!

## User mode

If you want to reproduce the results of our paper, please first download our [dataset](https://drive.google.com/drive/folders/1ZcKAMCESzhQZXNjbxItKRlpISjAhT2hI?usp=sharing) into the folder named "data", and download our [well-trained models](https://drive.google.com/drive/folders/1n9ens6nK_-QcidqLZq1Pq_wkLo-TPkzu?usp=sharing) into the folder named "res/saved_models". 

After preparing all the data and well-trained models, you need to first install all the required python package (with the Python>=3.8 is preferred) by
```
pip install -r requirements.txt
```

Then you can evaluate the prediction accuracy of our proposed GANO model in the testing dataset for the darcy problem and 2D plate stress problem with the following commands:
```
python PINO_darcy_training.py --model='GANO' --phase='test'
python PINO_plate_training.py --model='GANO' --phase='test'
```

Or implement the model training by replacing the "phase" argument:
```
python PINO_darcy_training.py --model='GANO' --phase='train'
python PINO_plate_training.py --model='GANO' --phase='train'
```

If you want to reproduce the results of the baseline model: Physics-informed Mesh-independent Deep Compositional Operator Network, you can simply replace the "model" argument:
```
python PINO_darcy_training.py --model='DCON' --phase='train'
python PINO_plate_training.py --model='DCON' --phase='train'
```

## Developer mode

This repository is user-friendly for developing new model architecture of the neural operator model. You can simply explore your self-designed model architecture by the following steps:
* open the script of the model_darcy.py or model_plate.py, input your self-defined model architectur into the function class of "New_model_darcy" and "New_model_plate".
* Add the hyper-parameters in the file "configs/self_defined_Darcy_star.yaml" and "configs/self_defined_plate_stress_DG.yaml". 
* Train your model by simply run the following command:
```
cd Main
python PINO_darcy_training.py --model='self_defined' --phase='train'
python PINO_plate_training.py --model='self_defined' --phase='train'
```

If you are interested in developing more advanced training algorithms, please check our the script "utils_darcy_train.py" and "utils_plate_train.py".

## LLM-Driven Architecture Optimization (ReEvo2D)

PI-GANO integrates [ReEvo2D](https://github.com/DataLab-atom/open-agents-U2E) — a large language model-based evolutionary optimizer — to automatically search for better implementations of the key internal functions defined in `lib/*_optimizable.py`. The optimizer iteratively proposes, evaluates, and reflects on function variants, guided by training metrics on the Darcy and plate problems.

### Repository Structure

```
PI-GANO/
├── main.py                    # One-command launch entry
├── u2e_api.py                 # get_funcs / evaluate_funcs (concurrency-safe)
├── optimizer/                 # ReEvo2D algorithm package
│   ├── reevo2d.py             # Evolutionary main loop
│   └── utils/
│       ├── utils.py           # Helper utilities
│       └── llm_client/        # OpenAI / LlamaAPI / ZhipuAI clients
├── prompts/common/            # LLM prompt templates
├── lib/                       # Optimizable function definitions (*_optimizable.py)
├── configs/                   # Training configuration YAML files
└── scripts/run_experiment.py  # Training experiment executor
```

### Quick Start

Install dependencies first:
```bash
pip install -r requirements.txt
```

Then launch the optimizer:
```bash
# Minimal launch (API key via environment variable)
export OPENAI_API_KEY=your_key_here
python main.py --model deepseek-coder

# Full options
python main.py \
    --model         deepseek-coder \
    --api-key       $OPENAI_API_KEY \
    --base-url      https://api.example.com/v1 \
    --temperature   0.5 \
    --max-fe        500 \
    --pop-size      20 \
    --init-pop-size 50 \
    --mutation-rate 0.8 \
    --epochs        50 \
    --output-dir    ./runs/exp1
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `deepseek-coder` | LLM model name |
| `--api-key` | env `OPENAI_API_KEY` | LLM API key |
| `--base-url` | env `OPENAI_BASE_URL` | API base URL (for proxies / alternative providers) |
| `--temperature` | `0.5` | LLM sampling temperature |
| `--max-fe` | `500` | Max function evaluations (stopping criterion) |
| `--pop-size` | `20` | Population size per generation |
| `--init-pop-size` | `50` | Initial population size |
| `--mutation-rate` | `0.8` | Mutation rate |
| `--epochs` | config default | Training epochs per evaluation (overrides `configs/*.yaml`) |
| `--output-dir` | `./runs/pi_gano_reevo` | Output directory for results and logs |

Results are written to `--output-dir/problems/pi_gano/`. The best-found function implementations are logged at the end of the run.

---

## Optimization & Experiment Scripts

Two utility scripts are provided under `scripts/` to support systematic architecture optimization.

### `scripts/extract_optimizable_items.py` — 查询可优化函数列表

解析所有 `lib/*_optimizable.py` 文件，以 JSON 格式输出每个可优化函数的名称、所在文件、参数类型、返回类型和功能描述。

```bash
# 输出到 stdout（紧凑格式）
python scripts/extract_optimizable_items.py

# 缩进格式，写入文件
python scripts/extract_optimizable_items.py --pretty --out optimizable_items.json
```

每条记录格式：
```json
{
  "function": "build_geometry_mlp",
  "file": "model_darcy_optimizable.py",
  "args": [{"name": "in_dim", "type": "int"}, ...],
  "returns": "nn.Sequential",
  "docstring": "[B1/B2/B5] 构建几何编码器的 MLP ..."
}
```

---

### `scripts/run_experiment.py` — 注入新实现并自动评测指标

给定一批函数的新实现，自动将其注入对应的 `*_optimizable.py` 文件，运行相关训练场景，收集验证/测试指标，完成后**自动还原所有文件**。

**作为模块调用：**
```python
from scripts.run_experiment import run_experiment

results = run_experiment(
    replacements=[
        {
            "function": "build_geometry_mlp",
            "file": "model_darcy_optimizable.py",
            "code": """
def build_geometry_mlp(in_dim: int, fc_dim: int, n_layer: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, fc_dim), nn.LayerNorm(fc_dim), nn.Tanh()]
    for _ in range(n_layer - 1):
        layers += [nn.Linear(fc_dim, fc_dim), nn.LayerNorm(fc_dim), nn.Tanh()]
    layers.append(nn.Linear(fc_dim, fc_dim))
    return nn.Sequential(*layers)
""",
        }
    ],
    epochs=10,      # 覆盖训练轮数（None = 使用配置文件默认值）
)
# results["darcy"]["test_relative_L2"]   → 测试集 L2 相对误差
# results["plate"]["test_relative_L2"]
```

**命令行调用：**
```bash
python scripts/run_experiment.py --spec my_exp.json --epochs 10 --pretty --no-stdout
```

替换规格 JSON 格式（`my_exp.json`）：
```json
[
  {
    "function": "build_geometry_mlp",
    "file": "model_darcy_optimizable.py",
    "code": "def build_geometry_mlp(...):\n    ..."
  }
]
```

输出指标字段：

| 字段 | 说明 |
|------|------|
| `test_relative_L2` | 最优模型在测试集上的 L2 相对误差（主要指标） |
| `val_relative_L2_last` | 最后一个打印周期的验证误差 |
| `pde_loss_last` | PDE 残差损失 |
| `bc_loss_last` | 边界条件损失（Darcy） |
| `fix/free/load_bc_loss_last` | 各类边界损失（Plate） |
| `returncode` | 子进程退出码（0 = 成功） |

场景自动推断：根据被替换函数所在文件决定运行 `darcy` / `plate` / 两者；`utils_losses_optimizable.py` 精确到函数级别。

---

**If you think that the work of the PI-GANO is useful in your research, please consider citing our paper in your manuscript:**
```
@article{zhong2025physics,
  title={Physics-Informed Geometry-Aware Neural Operator},
  author={Zhong, Weiheng and Meidani, Hadi},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={434},
  pages={117540},
  year={2025},
  publisher={Elsevier}
}
```
