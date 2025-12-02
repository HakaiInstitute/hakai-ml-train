# Hakai ML Train

Deep learning models and training configuration for semantic segmentation in marine ecology applications, specifically kelp and mussel detection from aerial/drone imagery. This is the training component of the [Kelp-o-Matic](https://github.com/HakaiInstitute/kelp-o-matic) ecosystem.

## Project Overview

This repository provides PyTorch Lightning-based training infrastructure for computer vision models used in marine ecology research. The models perform semantic segmentation on aerial imagery to detect kelp forests and mussel beds.

**Key Features:**
- Configurable training pipelines via YAML configs
- Support for multiple architectures (UNet++, SegFormer, UperNet, etc.) via segmentation-models-pytorch
- Integration with Weights & Biases for experiment tracking
- Model export to ONNX and TorchScript for production deployment
- Built-in data augmentation and preprocessing with Albumentations
- Efficient preprocessing pipeline for GeoTIFF imagery

## Environment Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd hakai-ml-train
   ```

2. **Install dependencies with uv:**
   ```bash
   # Installs dependencies and creates virtual environment
   uv sync

   # For development (includes Jupyter, pre-commit, etc.)
   uv sync --all-groups
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

4. **Set up pre-commit hooks (optional but recommended):**
   ```bash
   pre-commit install
   ```

## Dataset Preparation

The training pipeline expects preprocessed image chips in NPZ format. Follow these steps to prepare your dataset from GeoTIFF imagery.

### 1. Organize Raw Data

Your raw data should follow this structure:

```
raw_data/
├── train/
│   ├── images/
│   │   ├── mosaic_01.tif
│   │   ├── mosaic_02.tif
│   │   └── ...
│   └── labels/
│       ├── mosaic_01.tif
│       ├── mosaic_02.tif
│       └── ...
├── val/
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
└── test/
    ├── images/
    │   └── ...
    └── labels/
        └── ...
```

**Requirements:**
- **Images**: GeoTIFF files (.tif) with proper georeferencing
- **Labels**: GeoTIFF files with the same spatial extent and resolution as images
- **Naming**: Image and label files must have matching names
- **Pixel values**:
  - Labels should use integer values (e.g., 0=background, 1=kelp, 2=nereo, etc.)
  - Images can be uint8 (0-255) for RGB or uint16 for multispectral

### 2. Create Chip Dataset

Use the `make_chip_dataset.py` script to tile your GeoTIFF mosaics into smaller chips:

```bash
python -m src.prepare.make_chip_dataset <raw_data_dir> <output_dir> \
  --size 224 \
  --stride 224 \
  --num_bands 3 \
  --remap 0 -100 1 2
```

**Parameters:**
- `<raw_data_dir>`: Path to directory containing train/val/test folders
- `<output_dir>`: Where to save preprocessed NPZ chips
- `--size`: Size of square chips (default: 224)
- `--stride`: Stride for chip extraction (default: 224, equals --size for no overlap)
- `--num_bands`: Number of image bands to keep (3 for RGB, 4 for RGBI, etc.)
- `--remap`: Label value remapping. Format: `old_0 new_0 old_1 new_1 ...`
  - Example: `0 0 1 -100 2 1 3 2` means: 0→0 (bg), 1→-100 (ignore), 2→1 (class 1), 3→2 (class 2)
  - Use `-100` for pixels to ignore during training
- `--dtype`: Data type for image values (default: uint8)

**Example (Binary Kelp Detection):**
```bash
python -m src.prepare.make_chip_dataset \
  /data/kelp_raw \
  /data/kelp_chips_224 \
  --size 224 \
  --stride 224 \
  --num_bands 3 \
  --remap 0 0 1 -100 2 1
```

**Example (Multi-class Kelp Species):**
```bash
python -m src.prepare.make_chip_dataset \
  /data/kelp_species_raw \
  /data/kelp_species_chips_1024 \
  --size 1024 \
  --stride 1024 \
  --num_bands 3 \
  --remap 0 0 1 -100 2 1 3 2
```

This creates NPZ files containing compressed image and label arrays in:
```
output_dir/
├── train/
│   ├── mosaic_01_0.npz
│   ├── mosaic_01_1.npz
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### 3. (Optional) Clean Up Dataset

Remove unwanted chips to balance your dataset:

**Remove background-only tiles:**
```bash
python -m src.prepare.remove_bg_only_tiles /data/kelp_chips_224/train
python -m src.prepare.remove_bg_only_tiles /data/kelp_chips_224/val
python -m src.prepare.remove_bg_only_tiles /data/kelp_chips_224/test
```

This removes chips where all pixels are background (label ≤ 0), which is useful to reduce class imbalance.

**Remove tiles with nodata areas:**
```bash
python -m src.prepare.remove_tiles_with_nodata_areas /data/kelp_chips_224/train --num_channels 3
python -m src.prepare.remove_tiles_with_nodata_areas /data/kelp_chips_224/val --num_channels 3
python -m src.prepare.remove_tiles_with_nodata_areas /data/kelp_chips_224/test --num_channels 3
```

This removes chips containing all-black pixels (assumed to be nodata areas from mosaicking).

### 4. (Optional) Compute Dataset Statistics

For custom normalization, compute channel statistics from your training data:

```bash
python -m src.prepare.channel_stats /data/kelp_chips_224/train --max_pixel_val 255.0
```

This outputs mean and std for each channel, saved to `/data/channel_stats.npz`. Use these values in your config's normalization transform.

## Training

### Quick Start

Once you have prepared your dataset, training is straightforward using the PyTorch Lightning CLI:

```bash
python trainer.py fit --config configs/kelp-rgb/segformer_b3.yaml
```

### Configuring a Training Run

Training is controlled via YAML configuration files in the `configs/` directory. The configs use PyTorch Lightning CLI format and are organized by dataset type:

- **`kelp-rgb/`**: RGB kelp detection models
- **`kelp-rgbi/`**: 4-channel RGBI kelp detection models
- **`kelp-ps8b/`**: 8-band PlanetScope multispectral kelp models
- **`mussels-rgb/`**: RGB mussel detection models
- **`mussels-goosenecks-rgb/`**: RGB mussel and gooseneck barnacle multi-class models

#### Configuration File Structure

Here's an annotated example configuration (`configs/kelp-rgb/segformer_b3.yaml`):

```yaml
seed_everything: 42

# Model configuration
model:
  class_path: "src.models.smp.SMPMulticlassSegmentationModel"
  init_args:
    architecture: "Segformer"        # Architecture: UnetPlusPlus, DeepLabV3Plus, FPN, MAnet, Segformer, etc.
    backbone: "mit_b3"               # Encoder backbone (see segmentation-models-pytorch docs)
    model_opts:
      encoder_weights: imagenet      # Pretrained weights
      in_channels: 3                 # Number of input channels
    num_classes: 3                   # Number of output classes (including background)
    ignore_index: &ignore_index -100 # Label value to ignore during training
    lr: 3e-4                         # Learning rate
    wd: 0.01                         # Weight decay
    b1: 0.9                          # Adam beta1
    b2: 0.95                         # Adam beta2
    loss: "LovaszLoss"               # Loss function: DiceLoss, LovaszLoss, FocalLoss, etc.
    loss_opts:
      mode: "multiclass"             # "binary" or "multiclass"
      ignore_index: *ignore_index
      from_logits: true

# Data configuration
data:
  class_path: "src.data.DataModule"
  init_args:
    # UPDATE THESE PATHS TO YOUR PREPROCESSED DATA
    train_chip_dir: "/path/to/your/chips/train"
    val_chip_dir: "/path/to/your/chips/val"
    test_chip_dir: "/path/to/your/chips/test"

    batch_size: 3
    num_workers: 8
    pin_memory: true
    persistent_workers: true

    # Training augmentations (serialized Albumentations pipeline)
    train_transforms:
      __version__: 2.0.9
      transform:
        __class_fullname__: Compose
        transforms:
          - __class_fullname__: D4
            p: 1.0
          - __class_fullname__: OneOf
            p: 0.5
            transforms:
              - __class_fullname__: RandomBrightnessContrast
                brightness_limit: [-0.1, 0.1]
                contrast_limit: [-0.1, 0.1]
                p: 1.0
              - __class_fullname__: HueSaturationValue
                hue_shift_limit: [-5.0, 5.0]
                sat_shift_limit: [-10.0, 10.0]
                val_shift_limit: [-15.0, 15.0]
                p: 1.0
          # ... more augmentations ...
          - __class_fullname__: Normalize
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            p: 1.0
          - __class_fullname__: ToTensorV2
            p: 1.0

    # Validation/test transforms (minimal, just normalization)
    test_transforms:
      __version__: 2.0.9
      transform:
        __class_fullname__: Compose
        transforms:
          - __class_fullname__: Normalize
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            p: 1.0
          - __class_fullname__: ToTensorV2
            p: 1.0

# Trainer configuration
trainer:
  accelerator: auto                    # "auto", "gpu", "cpu"
  devices: auto                        # Number of GPUs (auto = all available)
  precision: bf16-mixed                # "32", "16-mixed", "bf16-mixed"
  log_every_n_steps: 50
  max_epochs: 500
  accumulate_grad_batches: 8           # Effective batch size = batch_size * accumulate_grad_batches
  gradient_clip_val: 0.5
  default_root_dir: checkpoints

  # Weights & Biases logging
  logger:
    - class_path: lightning.pytorch.loggers.WandbLogger
      init_args:
        entity: hakai                  # W&B entity name
        project: kom-kelp-rgb          # W&B project name
        name: segformer_b3             # Run name
        group: Jul2025                 # Group related runs
        log_model: true                # Upload checkpoints to W&B
        tags:
          - kelp
          - Jul2025

  # Callbacks
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        filename: kelp_rgb_segformer_b3_epoch-{epoch:02d}_val-iou-{val/iou_epoch:.4f}
        monitor: val/iou_epoch         # Metric to monitor
        mode: max                      # "max" or "min"
        save_last: True
        save_top_k: 2                  # Save top 2 checkpoints
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: step
```

### Steps to Launch Training

1. **Choose or create a config file** in `configs/` directory based on your task

2. **Edit the config file** to update these key parameters:
   ```yaml
   data:
     init_args:
       train_chip_dir: "/path/to/your/preprocessed/chips/train"
       val_chip_dir: "/path/to/your/preprocessed/chips/val"
       test_chip_dir: "/path/to/your/preprocessed/chips/test"
       batch_size: 3  # Adjust based on GPU memory

   trainer:
     logger:
       - class_path: lightning.pytorch.loggers.WandbLogger
         init_args:
           project: "your-project-name"
           name: "your-run-name"
   ```

3. **Start training:**
   ```bash
   python trainer.py fit --config configs/kelp-rgb/segformer_b3.yaml
   ```

4. **Resume from checkpoint** (if training was interrupted):
   ```bash
   python trainer.py fit --config configs/kelp-rgb/segformer_b3.yaml \
     --ckpt_path checkpoints/last.ckpt
   ```

5. **Test a trained model:**
   ```bash
   python trainer.py test --config configs/kelp-rgb/segformer_b3.yaml \
     --ckpt_path checkpoints/best_checkpoint.ckpt
   ```

### Common Configuration Options

**Model Architectures** (via `architecture` parameter):
- `Unet`, `UnetPlusPlus` - Classic U-Net variants with skip connections
- `DeepLabV3`, `DeepLabV3Plus` - Atrous convolution-based
- `FPN` - Feature Pyramid Network
- `PSPNet` - Pyramid Scene Parsing Network
- `MAnet` - Multi-scale Attention Network
- `Segformer` - Transformer-based (efficient for high resolution)
- `UperNet` - Unified Perceptual Parsing

**Encoder Backbones** (via `backbone` parameter):
- ResNet: `resnet18`, `resnet34`, `resnet50`, `resnet101`
- EfficientNet: `efficientnet-b0` through `efficientnet-b7`
- MobileNet: `mobilenet_v2`
- SegFormer: `mit_b0` through `mit_b5`
- Swin Transformer: `swin_base_patch4_window7_224`
- See [segmentation-models-pytorch docs](https://github.com/qubvel/segmentation_models.pytorch) for full list

**Loss Functions** (via `loss` parameter):
- `DiceLoss` - Good for imbalanced datasets
- `LovaszLoss` - Optimizes IoU directly
- `FocalLoss` - Focuses on hard examples
- `JaccardLoss` - IoU loss
- `TverskyLoss` - Generalization of Dice
- `FocalDiceComboLoss` - Combination of Focal and Dice

### Binary vs Multi-class Models

**Binary segmentation** (background vs target):
```yaml
model:
  class_path: "src.models.smp.SMPBinarySegmentationModel"
  init_args:
    num_classes: 1
    loss: "LovaszLoss"
    loss_opts:
      mode: "binary"
```

**Multi-class segmentation** (background + multiple classes):
```yaml
model:
  class_path: "src.models.smp.SMPMulticlassSegmentationModel"
  init_args:
    num_classes: 3  # e.g., background, macrocystis, nereocystis
    class_names: ["bg", "macro", "nereo"]
    loss: "LovaszLoss"
    loss_opts:
      mode: "multiclass"
```

## Model Export for Production

After training, export PyTorch Lightning checkpoints to production-ready formats (ONNX or TorchScript) for deployment in Kelp-o-Matic.

### Export to ONNX

The recommended format for production deployment:

```bash
python -m src.deploy.kom_onnx <config_path> <ckpt_path> <output_path> [--opset 11]
```

**Example:**
```bash
python -m src.deploy.kom_onnx \
  configs/kelp-rgb/segformer_b3.yaml \
  checkpoints/best_model.ckpt \
  models/kelp_segformer_b3.onnx \
  --opset 14
```

**Parameters:**
- `config_path`: Path to the YAML config used for training
- `ckpt_path`: Path to the PyTorch Lightning checkpoint (.ckpt file)
- `output_path`: Where to save the ONNX model
- `--opset`: ONNX opset version (default: 11, recommend 14 for newer models)

The exported ONNX model:
- Strips the Lightning wrapper and extracts just the segmentation model
- Supports dynamic batch size and spatial dimensions
- Outputs raw logits (no activation function applied)

### Export Legacy Models

For backwards compatibility with older Kelp-o-Matic versions:

**Legacy RGB Kelp Models:**
```bash
python -m src.deploy.kom_onnx_legacy_kelp_rgb \
  configs/kelp-rgb/kom_baseline.yaml \
  checkpoints/legacy_model.ckpt \
  models/kelp_legacy_rgb.onnx
```

**Legacy RGBI Kelp Models:**
```bash
python -m src.deploy.kom_onnx_legacy_kelp_rgbi \
  configs/kelp-rgbi/kom_baseline.yaml \
  checkpoints/legacy_model.ckpt \
  models/kelp_legacy_rgbi.onnx
```

### Export to TorchScript

Alternative deployment format (less portable but may be faster in pure PyTorch environments):

```bash
python -m src.deploy.kom_torchscript \
  configs/kelp-rgb/segformer_b3.yaml \
  checkpoints/best_model.ckpt \
  models/kelp_segformer_b3.pt
```

### Deployment

Exported models are typically:
- Uploaded to AWS S3 in the kelp-o-matic bucket for production use
- Integrated into the [Kelp-o-Matic inference pipeline](https://github.com/HakaiInstitute/kelp-o-matic)
- Used via ONNX Runtime for efficient cross-platform inference

Training checkpoints and experiment logs remain in Weights & Biases under the hakai entity.

## Development

### Code Quality

Format and lint with Ruff:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

Run pre-commit hooks on all files:

```bash
pre-commit run --all-files
```

### Key Source Files

**Core Components:**
- `trainer.py` - Lightning CLI entry point
- `src/models/smp.py` - Lightning module wrappers (SMPBinarySegmentationModel, SMPMulticlassSegmentationModel)
- `src/data.py` - DataModule and dataset classes
- `src/losses.py` - Loss function registry
- `src/transforms.py` - Augmentation pipeline helpers

**Data Preparation:**
- `src/prepare/make_chip_dataset.py` - Tile GeoTIFF mosaics into chips
- `src/prepare/remove_bg_only_tiles.py` - Filter background-only chips
- `src/prepare/remove_tiles_with_nodata_areas.py` - Filter chips with nodata
- `src/prepare/channel_stats.py` - Compute normalization statistics

**Model Export:**
- `src/deploy/kom_onnx.py` - Export to ONNX format
- `src/deploy/kom_torchscript.py` - Export to TorchScript
- `src/deploy/kom_onnx_legacy_*.py` - Export legacy model formats

**Configuration:**
- `configs/` - Training configuration files organized by dataset

### Configuration Guidelines

When creating new configuration files:

1. **Use YAML anchors** for parameter reuse:
   ```yaml
   ignore_index: &ignore_index -100
   model:
     init_args:
       ignore_index: *ignore_index
   ```

2. **Include descriptive metadata** in W&B logging:
   - Use clear project names (e.g., `kom-kelp-rgb`)
   - Include dataset version or date in the `group` field
   - Add relevant tags for filtering experiments

3. **Adjust batch size and gradient accumulation** based on GPU memory:
   - Effective batch size = `batch_size` × `accumulate_grad_batches`
   - For large models, use smaller batch_size with larger accumulate_grad_batches

4. **Match image normalization** to your preprocessing:
   - RGB: Use ImageNet stats `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
   - Multispectral: Compute custom stats with `channel_stats.py`

## Weights & Biases

Training metrics, hyperparameters, and model checkpoints are logged to Weights & Biases.

**Access**: Contact Taylor Denouden for access to the hakai entity.

**Organization:**
- Project names follow pattern: `kom-{dataset}-{modality}` (e.g., `kom-kelp-rgb`)
- Checkpoints are uploaded as W&B artifacts when `log_model: true`
- Metrics tracked: IoU, accuracy, precision, recall, F1, loss

**Viewing Results:**
- Navigate to [wandb.ai/hakai](https://wandb.ai/hakai)
- Select your project to view runs and compare experiments
- Download checkpoints from the Artifacts tab

## Technical Stack

- **PyTorch Lightning** - Training orchestration and multi-GPU support
- **segmentation-models-pytorch** - Model architectures and pretrained backbones
- **Albumentations** - Data augmentation
- **TorchGeo** - Geospatial data loading for preprocessing
- **Weights & Biases** - Experiment tracking
- **ONNX** - Model export for production
- **uv** - Fast Python package management

## Related Projects

- [Habitat-Mapper](https://github.com/HakaiInstitute/habitat-mapper) - Production inference pipeline CLI (formerly kelp-o-matic)
- [Hakai Institute](https://hakai.org) - Marine ecology research organization
