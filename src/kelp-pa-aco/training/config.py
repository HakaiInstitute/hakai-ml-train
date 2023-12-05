import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class _TrainingConfig(BaseModel):
    data_dir: Path
    num_classes: int
    ignore_index: Optional[int] = None
    project_name: str

    # Dataset config
    num_workers: int = os.cpu_count() // 2
    persistent_workers: bool = True
    pin_memory: bool = True
    tile_size: int = 1024
    batch_size: int = 2
    num_bands: int = 4
    fill_value: int = 0

    # Checkpoint options
    checkpoint_dir: str = "./checkpoints"
    name: str = "UNetPlusPlus-ResNet34"

    # Training options
    lr: float = 0.0003
    alpha: float = 0.8
    gamma: float = 0.5
    weight_decay: float = 0.0001
    max_epochs: int = 10
    precision: str = "16-mixed"
    sync_batchnorm: bool = True
    warmup_period: float = 1. / max_epochs
    enable_logging: bool = True

class PATrainingConfig(_TrainingConfig):
    data_dir: Path = "/home/taylor/data/KP-ACO-RGBI-Nov2023/"
    num_classes: int = 3
    ignore_index: Optional[int] = 2
    project_name: str = "kom-kelp-pa-aco-rgbi"

class SPTrainingConfig(_TrainingConfig):
    data_dir: Path = "/home/taylor/data/KS-ACO-RGBI-Nov2023/"
    num_classes: int = 4
    ignore_index: Optional[int] = 3
    project_name: str = "kom-kelp-sp-aco-rgbi"
