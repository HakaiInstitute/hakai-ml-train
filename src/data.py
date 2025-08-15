from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import albumentations as A
import lightning.pytorch as pl
import numpy as np
import torch
import torchvision.transforms.functional as f
from albumentations import ToTensorV2, to_dict
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from src.transforms import get_test_transforms, get_train_transforms


class NpzSegmentationDataset(VisionDataset):
    """Load preprocessed image chips. Used during model train and validation phases."""

    def __init__(
        self,
        root: str,
        *args,
        **kwargs,
    ):
        super().__init__(root, *args, **kwargs)
        self.chips = sorted(Path(root).glob(f"*.npz"))

    def __len__(self):
        return len(self.chips)

    # noinspection DuplicatedCode
    def __getitem__(self, idx):
        chip_name = self.chips[idx]
        data = np.load(chip_name)
        if self.transforms is not None:
            with torch.no_grad():
                augmented = self.transforms(image=data["image"], mask=data["label"])
                return augmented["image"], augmented["mask"]

        return data["image"], data["label"]


# noinspection PyAbstractClass
class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_chip_dir: str,
        val_chip_dir: str,
        test_chip_dir: str,
        batch_size: int,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = True,
        persistent_workers: bool = False,
        train_transforms: Any | None = None,
        test_transforms: Any | None = None,
    ):
        super().__init__()
        self.train_data_dir = train_chip_dir
        self.val_data_dir = val_chip_dir
        self.test_data_dir = test_chip_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.train_trans = A.from_dict(train_transforms)
        self.test_trans = A.from_dict(test_transforms)

        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: str | None = None):
        self.ds_train = NpzSegmentationDataset(
            self.train_data_dir,
            transforms=self.train_trans,
        )
        self.ds_val = NpzSegmentationDataset(
            self.val_data_dir,
            transforms=self.test_trans,
        )
        self.ds_test = NpzSegmentationDataset(
            self.test_data_dir,
            transforms=self.test_trans,
        )

    def teardown(self, stage: str | None = None) -> None:
        del self.ds_train
        del self.ds_val
        del self.ds_test

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.ds_train,
            shuffle=True,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader | list[DataLoader]:
        return DataLoader(
            self.ds_val,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader | list[DataLoader]:
        return DataLoader(
            self.ds_test,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        # This runs once at the start
        if not hasattr(self, "_logged_transforms"):
            if (
                self.trainer.logger
                and self.train_trans
                and hasattr(self.trainer.logger.experiment.config, "update")
            ):
                self.trainer.logger.experiment.config.update(
                    {
                        "train_transforms": to_dict(self.train_trans),
                        "test_transforms": to_dict(self.test_trans),
                    },
                    allow_val_change=True,
                )
            self._logged_transforms = True
        return batch


class MAEDataModule(DataModule):
    @property
    def train_trans(self):
        return A.Compose(
            [
                A.D4(),
                # A.RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0), p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.ToTensorV2(),
            ]
        )

    @property
    def test_trans(self):
        return self.train_trans


class KOMBaselineRGBIDataModule(DataModule):
    @staticmethod
    def _rgbi_kelp_transform(img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # to float
        x = img.to(torch.float)
        # min-max scale
        x_unique = x.flatten().unique()
        min_ = x_unique[0]
        if len(x_unique) > 1:
            min_, _ = torch.kthvalue(x_unique, 2)
        max_ = x.flatten().max()
        return torch.clamp((x - min_) / (max_ - min_ + 1e-8), 0, 1)

    @property
    def test_trans(self):
        return A.Compose(
            [
                ToTensorV2(),
                A.Lambda(name="normalize", image=self._rgbi_kelp_transform),
            ],
            p=1,
        )
