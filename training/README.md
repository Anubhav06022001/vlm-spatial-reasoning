# Training Directory
This folder isolates standard model optimization pipelines.

* **finetune.py**: Handles the downstream fine-tuning loop, specifically utilizing Parameter-Efficient Fine-Tuning (PEFT/LoRA) on the attention matrices to optimize spatial reasoning within strict hardware memory constraints.