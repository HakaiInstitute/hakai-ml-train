import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from training.transforms import get_test_transforms, get_train_transforms


class SegmentationDataset(VisionDataset):
    """Load preprocessed image chips. Used during model train and validation phases."""

    def __init__(self, root: str, *args, ext: str = "tif", **kwargs):
        super().__init__(root, *args, **kwargs)

        self._images = sorted(Path(root).joinpath("x").glob(f"*.{ext}"))
        self._labels = sorted(Path(root).joinpath("y").glob(f"*.{ext}"))

        assert len(self._images) == len(
            self._labels
        ), "There are an unequal number of images and labels!"

    def __len__(self):
        return len(self._images)

    # noinspection DuplicatedCode
    def __getitem__(self, idx):
        img = np.array(Image.open(self._images[idx]))
        target = np.array(Image.open(self._labels[idx]))

        if self.transforms is not None:
            with torch.no_grad():
                transformed = self.transforms(image=img, mask=target)
                img, target = transformed["image"], transformed["mask"]

        return img, target


# noinspection PyAbstractClass
class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            num_classes: int,
            batch_size: int,
            num_workers: int = os.cpu_count(),
            pin_memory: bool = True,
            persistent_workers: bool = False,
            fill_value: int = 0,
            tile_size: int = 1024,
            extra_transforms=None,
            **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.fill_value = fill_value
        self.tile_size = tile_size

        self.train_data_dir = str(Path(data_dir).joinpath("train"))
        self.val_data_dir = str(Path(data_dir).joinpath("val"))
        self.test_data_dir = str(Path(data_dir).joinpath("test"))

        self.train_trans = get_train_transforms(self.tile_size, extra_transforms)
        self.test_trans = get_test_transforms(self.tile_size, extra_transforms)

        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        self.ds_train = SegmentationDataset(
            self.train_data_dir,
            ext="tif",
            transforms=self.train_trans,
        )
        self.ds_val = SegmentationDataset(
            self.val_data_dir,
            ext="tif",
            transforms=self.test_trans,
        )
        self.ds_test = SegmentationDataset(
            self.test_data_dir,
            ext="tif",
            transforms=self.test_trans,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
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

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.ds_val,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.ds_test,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
