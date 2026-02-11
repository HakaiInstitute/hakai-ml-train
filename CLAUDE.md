# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the training component for Kelp-o-Matic, a deep learning system for semantic segmentation of kelp and mussels from aerial/drone imagery. Part of the Hakai Institute marine ecology toolset.

## Common Commands

### Environment Setup
```bash
# Install dependencies with uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Training
```bash
# Train a model using Lightning CLI
python trainer.py fit --config configs/kelp-rgb/segformer_b3.yaml

# Resume training from checkpoint
python trainer.py fit --config configs/kelp-rgb/segformer_b3.yaml --ckpt_path checkpoints/last.ckpt

# Test a trained model
python trainer.py test --config configs/kelp-rgb/segformer_b3.yaml --ckpt_path checkpoints/best.ckpt
```

### Data Preparation
```bash
# Create chip dataset from GeoTIFF mosaics
python -m src.prepare.make_chip_dataset <data_dir> <output_dir> \
  --size 224 \
  --stride 224 \
  --num_bands 3 \
  --remap 0 -100 1 2

# Remove background-only tiles
python -m src.prepare.remove_bg_only_tiles <chip_dir>

# Remove tiles with nodata areas
python -m src.prepare.remove_tiles_with_nodata_areas <chip_dir>

# Calculate channel statistics for normalization
python -m src.prepare.channel_stats <chip_dir>
```

### Model Export
```bash
# Export to ONNX format
python -m src.deploy.kom_onnx <config_path> <ckpt_path> <output_path> [--opset 11]

# Export legacy kelp models
python -m src.deploy.kom_onnx_legacy_kelp_rgb <config_path> <ckpt_path> <output_path>
python -m src.deploy.kom_onnx_legacy_kelp_rgbi <config_path> <ckpt_path> <output_path>

# Export to TorchScript
python -m src.deploy.kom_torchscript <config_path> <ckpt_path> <output_path>
```

### Code Quality
```bash
# Format and lint with Ruff
ruff check --fix .
ruff format .

# Run pre-commit hooks
pre-commit run --all-files
```

## Architecture

### Core Components

**trainer.py**: Entry point using PyTorch Lightning CLI. Automatically discovers model and data classes from YAML configs.

**src/models/smp.py**: Lightning module wrappers for segmentation-models-pytorch:
- `SMPBinarySegmentationModel`: Binary segmentation (kelp vs background)
- `SMPMulticlassSegmentationModel`: Multi-class segmentation (background, macrocystis, nereocystis)
- Includes HuggingFace Hub integration for model sharing
- Configurable loss functions, optimizers, and learning rate schedulers

**src/data.py**: Lightning data modules for loading preprocessed NPZ chips:
- `DataModule`: Base class with Albumentations transform support
- `NpzSegmentationDataset`: Loads compressed image/label pairs from NPZ files
- `MAEDataModule`: Specialized for Masked Autoencoder pretraining
- `KOMBaselineRGBIDataModule`: Custom normalization for 4-channel RGBI data

**src/losses.py**: Loss function registry including Dice, Focal, Lovasz, Jaccard, and Tversky losses from segmentation-models-pytorch.

**src/transforms.py**: Albumentations augmentation pipelines:
- `get_train_transforms()`: Heavy augmentation for training (D4 symmetries, noise, blur, color jitter, geometric transforms)
- `get_test_transforms()`: Minimal normalization for validation/test

### Data Pipeline

1. **Preprocessing** (src/prepare/make_chip_dataset.py):
   - Uses TorchGeo RasterDataset to load GeoTIFF mosaics and labels
   - Grids images into fixed-size chips (default 224x224)
   - Remaps label values and handles nodata regions
   - Saves as compressed NPZ files (image + label)

2. **Training**:
   - NpzSegmentationDataset loads chips on-the-fly
   - Applies Albumentations transforms in __getitem__
   - DataModule handles train/val/test splitting
   - Lightning handles distributed training, logging, checkpointing

3. **Deployment** (src/deploy/):
   - Export trained models to ONNX or TorchScript
   - Strips Lightning wrapper to expose bare segmentation model
   - Supports dynamic batch/spatial dimensions

### Configuration System

YAML configs in `configs/` define complete training pipelines:
- Model architecture, backbone, hyperparameters
- Loss function and optimizer settings
- Data paths and augmentation pipelines (serialized Albumentations transforms)
- Lightning Trainer settings (devices, precision, callbacks, loggers)

Common config structure:
```yaml
seed_everything: 42
model:
  class_path: "src.models.smp.SMPMulticlassSegmentationModel"
  init_args:
    architecture: "Segformer"
    backbone: "mit_b3"
    num_classes: 3
    loss: "LovaszLoss"
data:
  class_path: "src.data.DataModule"
  init_args:
    train_chip_dir: "/path/to/train"
    batch_size: 3
    train_transforms: {...}  # Serialized Albumentations
trainer:
  accelerator: auto
  precision: bf16-mixed
  logger:
    - class_path: lightning.pytorch.loggers.WandbLogger
```

### Model Support

Primary architectures via segmentation-models-pytorch:
- **Segformer** (mit_b0-b5): Transformer-based, efficient for high-resolution
- **UperNet** (Swin/ConvNeXt backbones): Strong multi-scale feature fusion
- **Unet++** (various backbones): Skip connections for fine detail
- **DPT** (Vision Transformer): Dense prediction transformers

Backbones include ImageNet-pretrained ResNet, EfficientNet, MobileNet, Swin Transformer, SegFormer, etc.

## Project-Specific Conventions

### Ignore Index
Use `-100` as the ignore_index throughout (label remapping, loss functions, metrics). This is PyTorch's default for ignoring labels during training.

### Label Remapping
The preprocessing pipeline remaps raw label values to model-expected values. Common patterns:
- Binary: `0->0 (bg), 1->-100 (ignore), 2->1 (kelp)`
- Multiclass: `0->0 (bg), 1->-100 (ignore), 2->1 (macro), 3->2 (nereo)`

### Normalization
- **RGB images**: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Multispectral**: Compute channel stats with `src.prepare.channel_stats`
- **RGBI legacy models**: Min-max scaling per-image in KOMBaselineRGBIDataModule

### Experiment Tracking
Weights & Wandb is the primary logger. Configs specify:
- `entity: hakai`
- `project: kom-<dataset>` (e.g., kom-kelp-rgb)
- `group`: Dataset version or experiment batch
- `name`: Model architecture identifier
- `log_model: true` to save checkpoints to W&B

### Model Checkpointing
ModelCheckpoint callback monitors `val/iou_epoch` and saves:
- Top 2 checkpoints by validation IoU
- Last checkpoint for resuming
- Checkpoint filenames include epoch and validation metric

## Development Notes

- This project uses **uv** for dependency management (modern pip replacement)
- Python >=3.12 required
- Ruff enforces flake8-bugbear, flake8-simplify, and isort rules
- Pre-commit hooks run Ruff automatically
- The codebase follows PyTorch Lightning 2.x patterns (LightningModule, LightningDataModule, LightningCLI)
- All models inherit from `pl.LightningModule` and optionally `PyTorchModelHubMixin` for HuggingFace integration
