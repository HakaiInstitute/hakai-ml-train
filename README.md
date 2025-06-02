# Hakai ML Train

Deep learning models and training configuration for semantic segmentation in marine ecology applications, specifically kelp and mussel detection from aerial/drone imagery. This is the training component of the [Kelp-o-Matic](https://github.com/HakaiInstitute/kelp-o-matic) ecosystem.

## Project Overview

This repository provides PyTorch Lightning-based training infrastructure for computer vision models used in marine ecology research. The models perform semantic segmentation on aerial imagery to detect kelp forests and mussel beds.

**Key Features:**
- Configurable training pipelines via YAML configs
- Support for multiple architectures (UNet++, SegFormer, DeepLabV3+, etc.)
- Integration with Weights & Biases for experiment tracking
- Model export to ONNX and TorchScript for production deployment
- Built-in data augmentation and preprocessing

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

2. **Create and activate UV environment:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # Basic installation
   uv pip install -e .

   # Development installation (includes Jupyter, pre-commit, etc.)
   uv pip install -e .[dev]
   ```

4. **Set up pre-commit hooks (optional but recommended):**
   ```bash
   pre-commit install
   ```

## Data Structure

Training data should be organized in the following directory structure:

```
data/
├── train/
│   ├── x/          # Training images (.tif)
│   └── y/          # Training labels (.tif)
├── val/
│   ├── x/          # Validation images (.tif)
│   └── y/          # Validation labels (.tif)
└── test/
    ├── x/          # Test images (.tif)
    └── y/          # Test labels (.tif)
```

**File naming convention:**
- Images: `image_name.tif`
- Labels: `image_name_label.tif`

**Image specifications:**
- Format: GeoTIFF (.tif)
- Standard size: 1024x1024 pixels
- Bands: 3 (RGB) for standard configs
- Label values: 0 (background), 1 (target class), 2 (ignore/no-data)

## Training

### Running Training

Train a model using a configuration file:

```bash
# Basic training command
python -m train path/to/config.yml

# Example commands
python -m train train/configs/kelp-rgb/pa-unetpp-efficientnetv2-m.yml
python -m train train/configs/mussels-rgb/unetpp-efficientnetv2-m.yml
```

### Configuration System

Training is controlled via YAML configuration files located in `train/configs/`. The configs are organized by task:

- **`kelp-rgb/`**: Kelp detection models using RGB imagery
- **`mussels-rgb/`**: Mussel detection models using RGB imagery

#### Configuration Structure

```yaml
segmentation_model_cls: SMPSegmentationModel  # Model class
enable_logging: true                          # Enable W&B logging

# Anchors for parameter reuse
num_bands: &num_bands 3
num_classes: &num_classes 1
tile_size: &tile_size 1024
batch_size: &batch_size 6
max_epoch: &max_epoch 20

segmentation_config:     # Model-specific parameters
  architecture: "UnetPlusPlus"
  backbone: "tu-tf_efficientnetv2_m"
  num_classes: *num_classes
  lr: 0.0003
  loss:
    name: "LovaszLoss"
    opts:
      mode: "binary"

data_module:            # Data loading configuration
  data_dir: "/path/to/data/"
  tile_size: *tile_size
  batch_size: *batch_size
  num_workers: 4

trainer:                # PyTorch Lightning trainer settings
  accumulate_grad_batches: 32
  max_epochs: *max_epoch
  precision: "16-mixed"

logging:                # W&B logging configuration
  project: "kom-kelp-pa-rgb"
  name: "UNetPP-efficientnetv2-m"
  log_model: true
```

#### Key Configuration Parameters

- **`segmentation_model_cls`**: Model implementation (`SMPSegmentationModel`, `DINOv2Segmentation`, `AerialFormer`)
- **Architecture options**: UnetPlusPlus, DeepLabV3Plus, FPN, MAnet, SegFormer
- **Backbone options**: EfficientNet variants, MobileNet, Vision Transformers
- **Loss functions**: Dice, Focal, Lovász, Tversky, combinations (see `train/losses.py`)

## Model Conversion for Production

After training, convert PyTorch Lightning checkpoints to production formats:

### Convert Checkpoint

```bash
python train/convert_checkpoint.py <wandb_artifact_url> <config_file> --model_name <name> --task <task>
```

**Example:**
```bash
python train/convert_checkpoint.py \
  "hakai/kom-mussels-rgb/model-nzty80z7:v0" \
  train/configs/mussels-rgb/unetpp-efficientnetv2-m.yml \
  --model_name "UNetPlusPlus_EfficientNetV2M" \
  --task "mussel_presence_rgb"
```

This generates:
- **JIT model**: `inference/weights/{model_name}_{task}_jit_dice={score}.pt`
- **ONNX model**: `inference/weights/{model_name}_{task}_dice={score}.onnx`

### Deployment

- **ONNX and TorchScript files** are uploaded to AWS S3 in the kelp-o-matic bucket for production use
- **Raw checkpoint files** and training metrics are stored in Weights & Biases under the hakai-org account

## Test Inference

Run inference on a single image using the included inference script:

```bash
cd inference
python inference.py <input_image.tif> <output_segmentation.tif>
```

**Example:**
```bash
cd inference
python inference.py /path/to/aerial_image.tif /path/to/output_kelp_mask.tif
```

The inference script:
- Uses Bartlett-Hanning windowing for seamless tile processing
- Handles images of arbitrary size with overlapping tile strategy
- Outputs single-band GeoTIFF with pixel values: 0 (background), 1 (target class)
- Includes preprocessing (normalization, padding) and memory-efficient processing

## Development

### Code Quality

```bash
# Run linting
ruff check

# Run linting with auto-fix
ruff check --fix
```

### Testing/Evaluation

Evaluation scripts are located in `eval/` directory:

```bash
python eval/test_kom.py      # Kelp-o-Matic integration tests
python eval/test_new_model.py   # New model validation
```

### Key Development Files

**Core Training Components:**
- **`train/model.py`**: Model definitions (`SMPSegmentationModel`, `DINOv2Segmentation`, `AerialFormer`)
- **`train/datamodule.py`**: PyTorch Lightning DataModule for data loading
- **`train/losses.py`**: Loss function implementations
- **`train/transforms.py`**: Data augmentation and preprocessing
- **`train/configs/config.py`**: Pydantic configuration validation

**Model Architectures:**
- **`train/models/aerial_former/`**: Custom transformer architecture for aerial imagery

**Configuration:**
- **`train/configs/kelp-rgb/`**: Kelp detection model configurations
- **`train/configs/mussels-rgb/`**: Mussel detection model configurations

**Inference:**
- **`inference/inference.py`**: Production inference pipeline
- **`inference/kernels.py`**: Windowing functions for tile-based processing

### Configuration Guidelines

When creating new configurations:

1. **Use YAML anchors** for parameter consistency:
   ```yaml
   num_bands: &num_bands 3
   segmentation_config:
     num_bands: *num_bands
   data_module:
     num_bands: *num_bands
   ```

2. **Follow naming conventions**:
   - `pa-`: Presence/absence detection configurations
   - `sp-`: Species detection configurations (for kelp)
   - Include architecture and backbone in filename

3. **Set appropriate batch sizes** based on model complexity and available GPU memory

4. **Configure logging** with descriptive project and run names for experiment tracking

## Weights & Biases Integration

**Access**: Contact Taylor Denouden for access to the hakai-org W&B account.

**Project Organization:**
- Training metrics, model artifacts, and hyperparameters are automatically logged
- Raw PyTorch Lightning checkpoints are stored as W&B artifacts
- Model performance metrics (IoU, Dice, F1) are tracked across experiments

**Artifact URLs** follow the format: `hakai/<project-name>/model-<id>:v<version>`

## Architecture Overview

The training framework is built on:
- **PyTorch Lightning**: Training orchestration and distributed training support
- **Segmentation Models PyTorch**: Pre-trained encoder backbones and decoder architectures
- **Albumentations**: Data augmentation pipeline
- **Weights & Biases**: Experiment tracking and model versioning
- **Pydantic**: Configuration validation and type safety
