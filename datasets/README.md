# Datasets Directory
This folder manages data downloading, splitting, and preprocessing pipelines for spatial benchmarking.

* **cvbench_loader.py**: Downloads the nyu-visionx/CV-Bench dataset from Hugging Face and creates PyTorch DataLoaders.
* **preprocess.py**: Handles image resizing, normalization, and text tokenization to format raw data for VLM input.
* **spatial_tasks.py**: Organizes and splits dataset questions into specific categories like depth, counting, and left/right relations.