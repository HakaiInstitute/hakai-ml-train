import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from src.transforms import get_test_transforms, get_train_transforms


class SegmentationDataset(VisionDataset):
    """Load preprocessed image chips. Used during m odel train and validation phases."""

    def __init__(self, root: str, *args, ext: str = "tif", **kwargs):
        super().__init__(root, *args, **kwargs)
        self._images = sorted(Path(root).joinpath("x").glob(f"*.{ext}"))
        self._labels = sorted(Path(root).joinpath("y").glob(f"*.{ext}"))

        assert len(self._images) == len(self._labels), (
            "There are an unequal number of images and labels!"
        )

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
        train_chip_dir: str,
        val_chip_dir: str,
        test_chip_dir: str,
        batch_size: int,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = True,
        persistent_workers: bool = False,
        fill_value: int = 0,
        fill_mask: int = 0,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.train_data_dir = train_chip_dir
        self.val_data_dir = val_chip_dir
        self.test_data_dir = test_chip_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.fill_value = fill_value
        self.train_trans = get_train_transforms(
            mean=mean, std=std, fill=fill_value, fill_mask=fill_mask
        )
        self.test_trans = get_test_transforms(mean=mean, std=std)

        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: str | None = None):
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
