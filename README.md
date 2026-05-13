# Data-Centric VLA

Efficient Vision-Language-Action (VLA) Model Adaptation via Data Pruning and PEFT.

## 🌟 Overview

This repository implements a pipeline for adapting large-scale VLA models to specific downstream robotic tasks using limited compute resources.

### Key Features:
- **Smart Data Pruning**: Filters out redundant and low-quality trajectory data using kinematic and semantic heuristics.
- **PEFT (QLoRA)**: Fine-tunes 7B-parameter models on single GPUs (e.g., V100 32GB) using 4-bit quantization and LoRA.
- **Sim-to-Real Ready**: Modular design for easy integration with Isaac Sim and real-world robot controllers.

## 🛠️ Structure

- `data/`: Placeholder for datasets.
- `models/`: Model loading and architecture definitions.
- `scripts/`: Training, pruning, and inference scripts.
- `utils/`: Filtering logic and helper functions.
- `configs/`: YAML configuration files.

## 🚀 Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (In progress) Prune your dataset:
   ```bash
   python scripts/prune_dataset.py --input_dir ./raw_data --output_dir ./pruned_data
   ```
3. (In progress) Start PEFT training:
   ```bash
   python scripts/train_peft.py --config configs/train_config.yaml
   ```
