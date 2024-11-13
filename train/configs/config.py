import os
from pathlib import Path
from typing import Any, Literal

import albumentations as A
import yaml
from albumentations.core.serialization import Serializable
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic import (
    GetJsonSchemaHandler,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import Annotated


class _BaseConfig(BaseModel):
    class Config:
        extra = "forbid"


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
    task: Literal["binary", "multiclass"] = "binary"


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


class AlbumentationsAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """
        We return a pydantic_core.CoreSchema that behaves in the following ways:

        * dicts will be parsed as `A.Compose` instances with the dict as the definition
        * `ComposeType` instances will be parsed as `ComposeType` instances without any changes
        * Nothing else will pass validation
        * Serialization will always return a dict
        """

        def validate(value: Any) -> Serializable:
            if isinstance(value, Serializable):
                return value
            if isinstance(value, dict):
                return A.from_dict(value)
            raise ValueError(
                "Value must be a Albumentations Serializable Transform instance or a dict that can be read with `A.from_dict`)"
            )

        return core_schema.json_or_python_schema(
            json_schema=core_schema.dict_schema(),
            python_schema=core_schema.no_info_plain_validator_function(validate),
            serialization=core_schema.plain_serializer_function_ser_schema(A.to_dict),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Use the same schema that would be used for `int`
        return handler(core_schema.dict_schema())


class Config(_BaseConfig):
    segmentation_model_cls: str
    enable_logging: bool
    tags: list[str] | None = None

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
    train_transforms: Annotated[A.Compose, AlbumentationsAnnotation]
    test_transforms: Annotated[A.Compose, AlbumentationsAnnotation]


def _load_yml(path: Path | str):
    with open(path) as f:
        return yaml.safe_load(f)


def load_yml_config(path: Path | str):
    return Config(**_load_yml(path))


if __name__ == "__main__":
    y = _load_yml("./kelp-rgbi/pa-unetpp-efficientnetv2-s.yml")
    config = Config.model_validate(y)

    print(config.model_dump(warnings="none"))
