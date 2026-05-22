# Spatial Understanding for Small VLMs

This repository contains experiments to improve the spatial reasoning capabilities of small Vision-Language Models (~1B parameters), specifically evaluating performance on `nyu-visionx/CV-Bench`.

## Objective
To enhance 3D spatial awareness (depth estimation, relative positioning, and counting) in small VLMs using parameter-efficient fine-tuning techniques.

## Baseline Results
The initial benchmark was conducted using an unmodified model to establish a performance floor.

* **Model:** `HuggingFaceTB/SmolVLM-500M-Instruct`
* **Dataset:** `nyu-visionx/CV-Bench` (Test Split - 2638 samples)
* **Hardware:** RTX 4060 (8GB VRAM)
* **Baseline Accuracy:** 43.18%

## Project Structure
* `configs/`: Hyperparameters and global settings.
* `datasets/`: Data loading and preprocessing pipelines.
* `evaluation/`: Benchmarking scripts and metric calculations.
* `experiments/`: Saved checkpoints and run logs.
* `models/`: VLM wrappers and vision encoders.
* `training/`: Fine-tuning and distillation logic.


```
spatial-vlm/
│
├── configs/
│   ├── smolvlm.yaml
│   ├── internvl.yaml
│   └── train.yaml
│
├── datasets/
│   ├── cvbench_loader.py
│   ├── preprocess.py
│   └── spatial_tasks.py
│
├── models/
│   ├── vlm/
│   │   ├── smolvlm.py
│   │   ├── internvl.py
│   │   └── paligemma.py
│   │
│   ├── encoders/
│   │   ├── dinov2.py
│   │   ├── siglip.py
│   │   └── clip.py
│   │
│   └── distillation/
│       ├── teacher.py
│       └── kd_loss.py
│
├── training/
│   ├── finetune.py
│   ├── distill.py
│   └── trainer.py
│
├── evaluation/
│   ├── benchmark.py
│   ├── metrics.py
│   └── visualize.py
│
├── experiments/
│   ├── baseline/
│   ├── encoder_swap/
│   └── depth_finetune/
│
├── results/
│   ├── tables/
│   ├── plots/
│   └── logs/
│
├── utils/
│   ├── logger.py
│   └── seed.py
│
├── README.md
├── requirements.txt
└── run.py
```