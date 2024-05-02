import os
from pathlib import Path
from typing import Optional, Callable, Any, Iterable

import albumentations as A
import numpy as np
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    data_dir: Path
    project_name: str

    # Model config
    name: str
    architecture: str
    backbone: str
    options_model: dict[str, Any] = None
    num_classes: int
    ignore_index: Optional[int] = None

    # Loss config
    loss_name: str
    loss_opts: dict[str, Any] = None

    # Dataset config
    tile_size: int = 1024
    num_bands: int = 3
    fill_value: int = 0
    batch_size: int = 4
    accumulate_grad_batches: int = 8
    extra_transforms: Iterable[Callable] = ()
    num_workers: int = os.cpu_count() // 2
    persistent_workers: bool = True
    pin_memory: bool = True

    # Checkpoint options
    checkpoint_dir: str = "./checkpoints"

    # Training options
    lr: float = 0.0003
    weight_decay: float = 0.0001
    max_epochs: int = 10
    warmup_period: float = 0.3

    # Tricks
    precision: str = "16-mixed"
    gradient_clip_val: float = 0.5
    sync_batchnorm: bool = True

    deterministic: bool = True
    benchmark: bool = False
    enable_logging: bool = True
    tags: list[str] = []


def _remap_species_labels(y: np.ndarray, **kwargs):
    new_y = y.copy()
    new_y[new_y == 0] = 3
    return new_y - 1


species_label_transform = A.Lambda(name="remap_labels", mask=_remap_species_labels)


class KelpPresenceEfficientNetB4Config(TrainingConfig):
    num_classes: int = 3
    ignore_index: int = 2
    architecture: str = "UnetPlusPlus"
    backbone: str = "efficientnet-b4"
    name: str = "UNetPlusPlus-effb4"
    options_model: dict[str, Any] = dict(decoder_attention_type="scse")
    loss_name: str = "FocalTverskyLoss"
    loss_opts: dict[str, Any] = dict(delta=0.5, gamma=2.0)


class KelpSpeciesEfficientNetB4Config(TrainingConfig):
    num_classes: int = 3
    ignore_index: int = 2
    architecture: str = "UnetPlusPlus"
    backbone: str = "efficientnet-b4"
    name: str = "UNetPlusPlus-effb4"
    options_model: dict[str, Any] = dict(decoder_attention_type="scse")
    extra_transforms: Optional[list[Callable]] = [species_label_transform]
    loss_name: str = "FocalTverskyLoss"
    loss_opts: dict[str, Any] = dict(delta=0.5, gamma=2.0)


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

seagrass_pa_efficientnet_b5_config_rgb = TrainingConfig(
    data_dir="/home/taylor/data/Seagrass_Feb2024_big/",
    project_name="seagrass-pa-rgb",
    tile_size=512,
    num_bands=3,
    batch_size=16,
    accumulate_grad_batches=16,
    max_epochs=20,
    warmup_period=0.05,
    num_classes=1,
    architecture="UnetPlusPlus",
    options_model=dict(decoder_attention_type="scse"),
    backbone="efficientnet-b5",
    name="UNet++_effb5",
    loss_name="DiceLoss",
    loss_opts=dict(mode="binary", from_logits=True, smooth=1e-6),
)

mussels_pa_efficientnet_b4_config_rgb = TrainingConfig(
    data_dir="/home/taylor/data/MP-RGB-May2024-1024-1024",
    project_name="kom-mussels-rgb",
    tile_size=1024,
    num_bands=3,
    batch_size=6,
    accumulate_grad_batches=32,
    max_epochs=20,
    warmup_period=0.05,
    lr=0.003,
    num_classes=1,
    architecture="UnetPlusPlus",
    backbone="efficientnet-b4",
    name="UNetPP-effb4",
    loss_name="DiceLoss",
    loss_opts=dict(mode="binary", from_logits=True, smooth=1.0),
    tags=["mussels", "May2024"],
)
