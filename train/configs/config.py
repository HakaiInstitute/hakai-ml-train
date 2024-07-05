import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Extra


class _BaseConfig(BaseModel):
    class Config:
        extra = Extra.forbid


class LossConfig(_BaseConfig):
    name: str
    opts: dict[str, Any] | None = None


class ModelConfig(_BaseConfig):
    num_classes: int
    lr: float
    weight_decay: float = 0
    batch_size: int = 2
    num_bands: int = 3
    tile_size: int = 1024
    max_epochs: int = 100
    warmup_period: float = 0
    loss: LossConfig
    architecture: str = "UnetPlusPlus"
    backbone: str = "dino_v2s"
    freeze_encoder: bool = False
    ignore_index: int | None = None
    opts: dict[str, Any] = None


class DataModuleConfig(_BaseConfig):
    data_dir: str
    num_classes: int
    batch_size: int
    num_workers: int = os.cpu_count()
    pin_memory: bool = True
    persistent_workers: bool = False
    fill_value: int = 0
    tile_size: int = 1024
    seed: int = 42


class TrainerConfig(_BaseConfig):
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 0.5
    deterministic: bool = True
    benchmark: bool = False
    max_epochs: int = 100
    precision: str = "32"


class LoggingConfig(_BaseConfig):
    project: str
    name: str
    save_dir: Path
    log_model: bool = False


class CheckpointConfig(_BaseConfig):
    dirpath: str | Path | None = "./checkpoints"
    filename: str | None = "{val/dice_epoch:.4f}_{epoch}"
    monitor: str = "val/dice_epoch"
    mode: str = "max"
    save_top_k: int = 1
    save_last: bool = True
    save_on_train_epoch_end: bool = False
    every_n_epochs: int = 1
    verbose: bool = False


class Config(_BaseConfig):
    segmentation_model_cls: str
    enable_logging: bool
    tags: list[str] |  None = None
    extra_transforms: list[str] | None = None

    num_bands: int = 3
    num_classes: int = 1
    tile_size: int = 896
    batch_size: int = 4
    max_epoch: int = 20

    segmentation_config: ModelConfig
    data_module: DataModuleConfig
    trainer: TrainerConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig = CheckpointConfig()


def _load_yml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_yml_config(path: Path):
    return Config(**_load_yml(path))


if __name__ == "__main__":
    y = _load_yml("./mussels-rgb/pa-dinov2-s.yml")
    print(Config.validate(y))
