# Evaluation Directory
This folder handles the benchmarking of models and computing accuracy metrics.

* **benchmark.py**: A unified evaluation script that runs inference across the CV-Bench dataset. It accepts CLI arguments to seamlessly swap between testing the baseline model and the LoRA fine-tuned model, as well as adjusting image resolution for test-time augmentation.
* **visualize.py**: Generates comparison plots and accuracy charts to document performance changes visually.