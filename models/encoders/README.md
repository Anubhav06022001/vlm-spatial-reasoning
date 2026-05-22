# Vision Encoders
This folder contains modular vision backbones used to evaluate the impact of different image features on spatial understanding[cite: 14].

* **dinov2.py**: Integrates Meta's DINOv2 self-supervised visual features into the VLM architecture.
* **siglip.py**: Integrates Google's SigLIP vision encoder to test contrastive language-image pre-training representations.
* **clip.py**: Integrates OpenAI's classic CLIP vision backbone as a baseline for visual representation comparison.