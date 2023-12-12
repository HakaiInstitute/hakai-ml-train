import os
import warnings
from pathlib import Path
from typing import List, Optional, Union

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset


class SegmentationDataset(VisionDataset):
    """Load preprocessed image chips. Used during model train and validation phases."""

    def __init__(self, root: str, *args, ext: str = "tif", **kwargs):
        super().__init__(root, *args, **kwargs)

        warnings.warn(
            "Using every 2nd image under the assumption that this dataset was "
            "generated with 50% overlap.")
        self._images = sorted(Path(root).joinpath("x").glob(f"*.{ext}"))[::2]
        self._labels = sorted(Path(root).joinpath("y").glob(f"*.{ext}"))[::2]

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
                img, target = transformed['image'], transformed['mask']

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
        if extra_transforms is None:
            extra_transforms = []
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

        self.train_trans = A.Compose([
            *extra_transforms,
            A.ToFloat(p=1),
            A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=45, border_mode=0, value=0,
                               p=0.7),
            A.PadIfNeeded(self.tile_size, self.tile_size, border_mode=0, value=0, p=1.),
            A.RandomCrop(self.tile_size, self.tile_size, p=1.),
            A.Flip(p=0.75),
            A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),
            A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1),

            # Colour transforms
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3,
                                           p=1),
                A.RandomGamma(gamma_limit=(70, 130), p=1),
                # A.ChannelShuffle(p=0.2),
                # A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1),
                # A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
            ], p=0.8),

            # distortion
            A.OneOf([
                A.ElasticTransform(p=1),
                A.OpticalDistortion(p=1),
                A.GridDistortion(p=1),
                A.Perspective(p=1),
            ], p=0.2),

            # noise transforms
            A.OneOf([
                A.GaussNoise(p=1),
                A.MultiplicativeNoise(p=1),
                A.Sharpen(p=1),
                A.GaussianBlur(p=1),
            ], p=0.2),
            ToTensorV2(),
        ], p=1)

        self.test_trans = A.Compose([
            *extra_transforms,
            A.ToFloat(p=1),
            A.PadIfNeeded(self.tile_size, self.tile_size, border_mode=0, value=0, p=1.),
            ToTensorV2(),
        ], p=1)

        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        self.ds_train = SegmentationDataset(
            self.train_data_dir,
            ext='tif',
            transforms=self.train_trans,
        )
        self.ds_val = SegmentationDataset(
            self.val_data_dir,
            ext='tif',
            transforms=self.test_trans,
        )
        self.ds_test = SegmentationDataset(
            self.test_data_dir,
            ext='tif',
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
            shuffle=True,
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
