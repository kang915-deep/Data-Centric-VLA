# Data-Centric VLA

Efficient Vision-Language-Action (VLA) Model Adaptation via Data Pruning and PEFT.

Fine-tune 7B-scale VLA models (OpenVLA) on a **single V100 32GB GPU** using data-centric techniques and QLoRA.

## Overview

Large VLA models achieve strong zero-shot performance on robotic manipulation, but full fine-tuning is prohibitively expensive for most labs. This project shifts the focus from "scaling compute" to **optimizing data quality and training efficiency**.

### Pipeline

```
Raw Demonstrations (HDF5)
    │
    ▼
┌─────────────────────────────────┐
│  Kinematic Filter               │  Removes idle / low-motion frames
│  (variance-based idle detection) │  using joint-state velocity thresholds
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Semantic Filter (Optional)     │  Filters out frames with low
│  (CLIP image-text alignment)    │  image-instruction alignment
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  QLoRA Fine-Tuning              │  4-bit NF4 quantization +
│  (PEFT with LoRA adapters)      │  LoRA adapters on attention projections
└─────────────────────────────────┘
    │
    ▼
        Deploy (Sim / Real Robot)
```

### Key Features

- **Smart Data Pruning**: Two-stage filtering — kinematic (motion-based) + semantic (CLIP-based) — removes up to 40–60% of redundant frames.
- **QLoRA on a Single V100**: 4-bit NF4 quantization + LoRA adapters (rank 32) fits OpenVLA-7B in ~18 GB VRAM, leaving room for batch size 4.
- **Transformers v5 Compatible**: Includes compatibility shim for running OpenVLA's remote model code with ``transformers>=5.x``.
- **Modular Design**: Each stage (data generation, pruning, training, inference) is an independent script with YAML-driven configuration.

---

## Requirements

- Python 3.10
- NVIDIA GPU with **24 GB+ VRAM** (tested on Tesla V100 32GB)
- CUDA Driver >= 12.1 (PyTorch built with CUDA 12.1)

### Dependencies

All dependencies are managed through ``environment.yml`` (conda) or ``requirements.txt`` (pip):

| Package | Version | Purpose |
|---|---|---|
| ``pytorch`` | 2.5.1+cu121 | Deep learning framework |
| ``transformers`` | >=4.40.0 | HuggingFace model/tokenizer APIs |
| ``peft`` | latest | LoRA / QLoRA adapter injection |
| ``bitsandbytes`` | latest | 4-bit quantization (NF4) |
| ``accelerate`` | latest | Device placement utilities |
| ``open_clip_torch`` | latest | Semantic filtering (ViT-B-32) |
| ``h5py`` | latest | HDF5 dataset I/O |

---

## Setup

### 1. Create Conda Environment

```bash
# From the project root
conda env create -f environment.yml
conda activate vla_env
```

> **Note for multi-GPU servers:** If ``torch.cuda.is_available()`` reports ``False`` after setup, ensure there is no system CUDA toolkit path leaking into ``LD_LIBRARY_PATH``:
> ```bash
> unset LD_LIBRARY_PATH
> python -c "import torch; print(torch.cuda.is_available())"
> ```

### 2. Verify CUDA

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()} | Devices: {torch.cuda.device_count()}')"
```

Expected output: ``CUDA available: True | Devices: 1+``

---

## Data Pipeline

### 1. Generate Mock Data (or use your own HDF5)

```bash
mkdir -p data/raw data/pruned checkpoints logs

python scripts/generate_mock_data.py \
    --output_dir data/raw \
    --num_demos 5 \
    --seq_len 50
```

This creates an HDF5 file with:
- 7-DOF joint states (with movement + idle segments)
- 224×224 RGB images (encoded as PNG bytes)
- 7-DOF action vectors
- Instruction text attribute

### 2. Prune the Dataset

Two-stage filtering removes low-quality and redundant frames:

```bash
# Kinematic-only pruning (faster, no CLIP needed)
python scripts/prune_dataset.py \
    --input_dir data/raw \
    --output_dir data/pruned \
    --k_var_thresh 1e-4 \
    --window_size 5 \
    --skip_semantic

# Full pruning with CLIP semantic filtering
python scripts/prune_dataset.py \
    --input_dir data/raw \
    --output_dir data/pruned \
    --k_var_thresh 1e-4 \
    --window_size 5 \
    --device cuda \
    --s_thresh 0.2
```

**Filter details:**

| Filter | Method | Purpose |
|---|---|---|
| **Kinematic** | Windowed variance of joint positions | Removes idle frames (robot not moving) |
| **Semantic** | CLIP ViT-B-32 image-text cosine similarity | Removes frames with low alignment to instruction |

---

## Training

### Configuration

Edit ``configs/train_config.yaml`` to match your setup:

```yaml
model:
  model_id: "openvla/openvla-7b"
  load_in_4bit: true

peft:
  r: 32
  alpha: 64
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

dataset:
  train_path: "data/pruned_dataset.h5"
  batch_size: 4
  grad_accumulation_steps: 4
  num_workers: 4

training:
  epochs: 10
  learning_rate: 2.0e-4
  lr_scheduler: "cosine"
  warmup_steps: 500
  weight_decay: 0.01
  logging_steps: 10
  save_steps: 500
  output_dir: "checkpoints/vla_peft_v1"
  wandb_project: "data-centric-vla"

device: "cuda"
```

### Run Training

```bash
# Recommended: select target GPU and disable system CUDA env interference
unset LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=2 MKL_SERVICE_FORCE_INTEL=1 \
    python scripts/train_peft.py --config configs/train_config.yaml
```

**What happens during training:**

1. **Model Loading**: Downloads OpenVLA-7B from HuggingFace, applies 4-bit NF4 quantization (``BitsAndBytesConfig``)
2. **Compatibility Shim**: Injects missing ``tokenization_utils`` symbols for transformers v5 compatibility
3. **LoRA Injection**: Wraps attention projections (``q_proj``, ``v_proj``, etc.) with rank-32 LoRA adapters
4. **Training Loop**: Cosine LR schedule with 500-step warmup, gradient accumulation for effective batch size 16
5. **Checkpointing**: Saves LoRA adapters every 500 steps to ``checkpoints/``

### Memory Footprint (V100 32GB)

| Component | VRAM |
|---|---|
| Base model (4-bit) | ~8 GB |
| LoRA adapters + gradients | ~4 GB |
| Optimizer states | ~2 GB |
| Activations (batch 4) | ~4 GB |
| **Total** | **~18 GB** |

---

## Inference

Load a trained checkpoint and run predictions:

```python
from inference import VLAPredictor

predictor = VLAPredictor(
    model_id="openvla/openvla-7b",
    adapter_path="checkpoints/vla_peft_v1/epoch_9"
)

action = predictor.predict(image, instruction)
print(f"Predicted 7-DOF action: {action}")
```

---

## Project Structure

```
├── configs/
│   └── train_config.yaml          # Training hyperparameters
├── data/
│   ├── raw/                        # Generated / collected HDF5 data
│   └── pruned/                     # Post-filtering dataset
├── models/
│   └── vla_model.py                # VLA model wrapper with transformers v5 compat
├── scripts/
│   ├── generate_mock_data.py       # HDF5 data generator for testing
│   ├── prune_dataset.py            # Kinematic + semantic pruning pipeline
│   ├── train_peft.py               # PEFT / QLoRA training loop
│   └── inference.py                # Deployment / inference stub
├── utils/
│   ├── kinematic_filter.py         # Variance-based idle frame detection
│   └── semantic_filter.py          # CLIP-based image-text alignment scoring
├── environment.yml                 # Conda environment specification
└── requirements.txt                # Pip dependencies
```

---

## Troubleshooting

### "CUDA unknown error" or ``cuInit`` returns 999

This is a **system-level driver issue**, not a PyTorch problem. Likely causes:

1. **LD_LIBRARY_PATH conflict**: System CUDA toolkit paths leak into the conda environment
   ```bash
   unset LD_LIBRARY_PATH
   ```

2. **Persistence mode off**: GPU driver enters a low-power state after idle periods
   ```bash
   # Requires root
   sudo nvidia-smi -pm 1
   ```

3. **Driver initialization failed**: The CUDA driver API cannot create a GPU context
   ```bash
   # Requires root
   sudo nvidia-smi -r
   ```

### ``cannot import name 'PreTokenizedInput' from 'tokenization_utils'``

If using ``transformers>=5.x``, the ``tokenization_utils`` module was restructured. The compatibility shim in ``models/vla_model.py`` automatically injects the missing symbols. If the issue persists, pin transformers to 4.40.1:

```bash
pip install transformers==4.40.1 tokenizers==0.19.1
```

### Model loads on CPU instead of GPU

Check CUDA availability and ensure ``device: "cuda"`` in ``configs/train_config.yaml``:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If ``False``, verify your PyTorch build matches your CUDA driver:

```bash
python -c "import torch; print(torch.version.cuda)"
# Should be <= your nvidia-smi CUDA version
```

### ``Failed to load CPU gemm_4bit_forward from kernels-community``

This is a non-fatal warning from bitsandbytes. The 4-bit matrix operations will use a slightly slower fallback path. Install the optional kernel package to silence it:

```bash
pip install kernels
```

---

## Citation

If you use this project in your research, please consider citing:

```bibtex
@misc{datacentricvla2025,
  title = {Data-Centric VLA: Efficient VLA Model Adaptation via Data Pruning and PEFT},
  author = {Kang, et al.},
  year = {2025}
}
```
