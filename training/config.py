import os
from pathlib import Path
from typing import Optional, Callable

import albumentations as A
import numpy as np
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    data_dir: Path
    num_classes: int
    ignore_index: Optional[int] = None
    project_name: str
    class_labels: dict[int, str]
    extra_transforms: Optional[list[Callable]] = None
    name: str = "UNet++"
    backbone: str = "efficientnet-b4"

    # Dataset config
    num_workers: int = os.cpu_count() // 2
    persistent_workers: bool = True
    pin_memory: bool = True
    tile_size: int = 1024
    batch_size: int = 4
    num_bands: int = 3
    fill_value: int = 0

    # Checkpoint options
    checkpoint_dir: str = "./checkpoints"

    # Training options
    lr: float = 0.0003
    alpha: float = 0.8
    gamma: float = 0.5
    weight_decay: float = 0.0001
    max_epochs: int = 10
    precision: str = "16-mixed"
    sync_batchnorm: bool = True
    warmup_period: float = 1.0 / max_epochs
    enable_logging: bool = True
    gradient_clip_val: float = 0.5
    accumulate_grad_batches: int = 8
    deterministic: bool = True
    benchmark: bool = False

def _remap_species_labels(y: np.ndarray, **kwargs):
    new_y = y.copy()
    new_y[new_y == 0] = 3
    return new_y - 1


species_label_transform = A.Lambda(name="remap_labels", mask=_remap_species_labels)

class KelpPresenceEfficientNetB4Config(TrainingConfig):
    num_classes: int = 3
    ignore_index: int = 2
    class_labels: dict[int, str] = {0: "background", 1: "kelp"}
    backbone: str = "efficientnet-b4"
    name: str = "UNet++_efficientnet-b4"

class KelpSpeciesEfficientNetB4Config(TrainingConfig):
    num_classes: int = 3
    ignore_index: int = 2
    class_labels: dict[int, str] = {0: "macro", 1: "nereo"}
    backbone: str = "efficientnet-b4"
    name: str = "UNet++_efficientnet-b4"
    extra_transforms: Optional[list[Callable]] = [species_label_transform]

kelp_pa_efficientnet_b4_config_rgbi = KelpPresenceEfficientNetB4Config(
    data_dir="/home/taylor/data/KP-ACO-RGBI-Nov2023/",
    project_name="kom-kelp-pa-aco-rgbi",
    num_bands=4,
)

kelp_sp_efficientnet_b4_config_rgbi = KelpSpeciesEfficientNetB4Config(
    data_dir="/home/taylor/data/KS-ACO-RGBI-Nov2023/",
    project_name="kom-kelp-sp-aco-rgbi",
    num_bands=4,
)

kelp_pa_efficientnet_b4_config_rgb = KelpPresenceEfficientNetB4Config(
    data_dir="/home/taylor/data/KP-RGB-Jan2024/",
    project_name="kom-kelp-pa-rgb",
    num_bands=3,
    max_epochs=100,
    warmup_period=0.02,
)

kelp_sp_efficientnet_b4_config_rgb = KelpSpeciesEfficientNetB4Config(
    data_dir="/home/taylor/data/KS-RGB-Jan2024/",
    project_name="kom-kelp-sp-rgb",
    num_bands=3,
    max_epochs=100,
    warmup_period=0.02,
)
