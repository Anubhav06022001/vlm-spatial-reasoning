# Knowledge Distillation Components
This folder contains the logic required to distill spatial reasoning capabilities from a large VLM into a ~1B parameter student.

* **teacher.py**: Manages loading and generating soft targets/logits from a large teacher model (like InternVL-8B).
* **kd_loss.py**: Implements custom loss functions, combining Kullback-Leibler divergence and cross-entropy for distillation training.