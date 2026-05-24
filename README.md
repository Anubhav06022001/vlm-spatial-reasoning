# Spatial Understanding for Small Vision-Language Models

<p align="center">

Parameter-Efficient Adaptation of Small Vision-Language Models for Spatial Reasoning

</p>

---

## Overview

This repository investigates methods to improve spatial reasoning capabilities of small Vision-Language Models (~500M–1B parameters) using parameter-efficient fine-tuning.

The focus is improving performance on spatial understanding tasks including:

- Relative positioning (left / right / behind)
- Depth reasoning (closest / furthest)
- Object counting

Evaluation is performed on:

```text
nyu-visionx/CV-Bench
```

---

## Objective

Improve spatial understanding in small VLMs while preserving parameter efficiency.

### Model

```text
HuggingFaceTB/SmolVLM-500M-Instruct
```

### Hardware

```text
RTX 5060 (8GB VRAM)
```

### Methodology

- LoRA Fine-Tuning
- Spatial data filtering
- Task-aligned instruction formatting
- Strict output constraint alignment

---

## Baseline Benchmark

### Evaluation Setup

Dataset:

```text
nyu-visionx/CV-Bench
```

Evaluation split:

```text
2638 samples
```

### Baseline Performance

| Model | Accuracy |
|:------|----------:|
| SmolVLM-500M-Instruct | 43.10% |

---

# Experiment 1 — Standard LoRA Fine-Tuning

## Approach

- LoRA adaptation
- LLaVA-instruct subset
- Standard conversational supervision

---

## Result

| Configuration | Accuracy |
|:--------------|----------:|
| Baseline | 43.10% |
| Standard LoRA | 42.04% |

---

## Analysis

Performance degraded slightly.

Observed failure modes:

- Format mismatch between training and evaluation
- Under-training
- Weak adherence to strict multiple-choice output constraints

The model learned conversational behavior but struggled to reliably output exact benchmark answer formatting.

---

# Experiment 2 — Task-Aligned Multiple Choice Fine-Tuning

## Approach

Spatial examples were filtered and reformatted into strict multiple-choice supervision.

Key modifications:

- Dynamic option generation
- Spatial task filtering
- LoRA fine-tuning
- Instruction alignment with benchmark evaluation format
- Increased training duration

---

## Final Results

| Configuration | Accuracy |
|:--------------|----------:|
| Baseline | 43.10% |
| Fine-Tuned (LoRA) | 46.02% |

### Improvement

```text
+2.92%
```

---

## Analysis

Evaluation was performed using highly restrictive answer formatting.

The baseline model frequently violated formatting constraints despite containing partial spatial understanding.

Task-aligned fine-tuning successfully improved instruction adherence while preserving underlying reasoning capabilities.

This suggests that small VLM performance on benchmark tasks depends not only on representation quality but also on structural alignment between supervision and evaluation.

---

## Repository Structure

```text
spatial-vlm/
│
├── datasets/
│   └── preprocess.py
│
├── models/
│   └── vlm/
│       └── smolvlm.py
│
├── training/
│   └── finetune.py
│
├── evaluation/
│   ├── benchmark.py
│   └── bechnamark_fine_tune.py
│
├── run.py
├── setup.py
│
├── README.md
│
└── requirements.txt
```

---

### datasets/

Dataset loading and preprocessing.

### models/

Model loading and LoRA adaptation.

### training/

Fine-tuning pipeline.

### evaluation/

Benchmark scripts and accuracy evaluation.

### experiments/

Saved adapters and experimental artifacts.

---

## Reproducibility

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python training/finetune.py
```

### Evaluation

```bash
python evaluation/benchmark.py
```

---

## Key Takeaway

Small Vision-Language Models benefit significantly from task-aligned supervision.

Instruction formatting and evaluation alignment can materially influence benchmark performance alongside representation learning.

---

<p align="center">

Research Prototype • Spatial Reasoning • Parameter Efficient Fine-Tuning • Small VLMs

</p>