# Spatial Understanding for Small VLMs

This repository contains experiments to improve the spatial reasoning capabilities of small Vision-Language Models (~1B parameters), specifically evaluating performance on `nyu-visionx/CV-Bench`.

## Objective
To enhance 3D spatial awareness (depth estimation, relative positioning, and counting) in small VLMs using parameter-efficient fine-tuning techniques.

## Baseline Results
The initial benchmark was conducted using an unmodified model to establish a performance floor.

* **Model:** `HuggingFaceTB/SmolVLM-500M-Instruct`
* **Dataset:** `nyu-visionx/CV-Bench` (Test Split - 2638 samples)
* **Hardware:** RTX 5060 (8GB VRAM)
* **Baseline Accuracy:** 43.18%

## Experiments & Analysis

**Experiment 1: Standard LoRA Fine-Tuning**
* **Approach:** Fine-tuned the model on a subset of the LLaVA-instruct dataset using standard conversational formatting.
* **Result:** 42.04% (a slight drop from baseline). 
* **Analysis:** The model suffered from a formatting mismatch (free-text training vs. multiple-choice evaluation) and under-training (only 1 epoch on 460 samples). While spatial understanding may have improved, the model lost its ability to reliably output exact multiple-choice tokens (e.g., "(a)").

**Experiment 2: Task-Aligned Multiple-Choice Fine-Tuning**
* **Approach:** Addressed the format mismatch by dynamically injecting dummy choices during training, forcing the model to learn spatial reasoning while strictly adhering to a multiple-choice structure. Increased training to 4 epochs on 2000 samples.
* **Result:** 45.60% (+2.42% over baseline).
* **Analysis:** Aligning the training data format with the evaluation benchmark allowed the model's newly acquired spatial reasoning capabilities to map correctly to the CV-Bench metrics. This confirms that small VLMs require strict structural alignment alongside conceptual training.
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
├── setup.py
├── requirements.txt
└── run.py
```


