# Training Directory
This folder isolates standard model optimization pipelines from specialized knowledge distillation workflows.

* **finetune.py**: Handles standard downstream fine-tuning loop configurations utilizing PEFT/QLoRA for hardware efficiency.
* **distill.py**: Coordinates the specialized training loop that optimizes the student model using teacher feedback.
* **trainer.py**: Contains shared utilities used by all loops, such as checkpoint saving, epoch logging, and evaluation intervals.