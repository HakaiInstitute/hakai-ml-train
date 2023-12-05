import os
from pathlib import Path
from typing import List, Union, Any

import pytorch_lightning as pl
import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2
from torchvision.tv_tensors import Image, Mask


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
        img = Image(PILImage.open(self._images[idx]))
        target = Mask(PILImage.open(self._labels[idx]))

        if self.transforms is not None:
            with torch.no_grad():
                img, target = self.transforms(img, target)

        return img, target


class PadOut(object):
    def __init__(self, height: int = 128, width: int = 128, fill_value: int = 0):
        self.height = height
        self.width = width
        self.fill_value = fill_value

    def __call__(self, x: Any) -> Any:
        """
        Pad out a pillow image, so it is the correct size as specified by `self.height` and `self.width`
        """
        h, w = x.shape[-2:]

        if h == self.height and w == self.width:
            return x

        w_pad = self.width - w
        h_pad = self.height - h

        return tv_tensors.wrap(v2.functional.pad(x, [0, 0, w_pad, h_pad], fill=self.fill_value), like=x)


def normalize_min_max(x: tv_tensors.TVTensor) -> tv_tensors.TVTensor:
    mask = torch.all(x == 0, dim=0).unsqueeze(dim=0).repeat(x.shape[0], 1, 1)
    masked_values = x.flatten()[~mask.flatten()]

    min_ = masked_values.min()
    max_ = masked_values.max()
    return tv_tensors.wrap(torch.clamp((x - min_) / (max_ - min_), 0, 1), like=x)


def normalize_min_max2(x: tv_tensors.TVTensor) -> tv_tensors.TVTensor:
    """Get second-smallest value as min to accommodate black backgrounds/nodata areas"""
    min_, _ = torch.kthvalue(x.flatten().unique(), 2)
    max_ = x.flatten().max()
    return tv_tensors.wrap(torch.clamp((x - min_) / (max_ - min_), 0, 1), like=x)


def normalize_percentile(x: tv_tensors.TVTensor, upper=0.99, lower=0.01) -> tv_tensors.TVTensor:
    mask = torch.all(x == 0, dim=0).unsqueeze(dim=0).repeat(x.shape[0], 1, 1)
    masked_values = x.flatten()[~mask.flatten()]

    max_ = torch.quantile(masked_values, upper)
    min_ = torch.quantile(masked_values, lower)
    return tv_tensors.wrap(torch.clamp((x - min_) / (max_ - min_), 0, 1), like=x)


def append_ndvi(x: tv_tensors.TVTensor) -> tv_tensors.TVTensor:
    r, g, b, nir = x
    ndvi = torch.nan_to_num(torch.div((nir - r), (nir + r)))

    # Scale NDVI from [-1, 1] to [0, 255]
    ndvi = (((ndvi + 1) / 2.) * 255).to(torch.uint8)

    new_x = torch.concat((x, ndvi.unsqueeze(dim=0)), dim=0)
    return tv_tensors.wrap(new_x, like=x)


# normalize = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
            tile_size: int = 1024
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

        self.pad_f = PadOut(self.tile_size, self.tile_size, fill_value=self.fill_value)

        self.train_trans = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.Lambda(v2.ToDtype(torch.float32, scale=True), Image),
                v2.Lambda(normalize_min_max2, Image),
                v2.Lambda(v2.ToDtype(torch.long), Mask),
                v2.Lambda(torch.squeeze, Mask),
            ]
        )

        self.test_trans = v2.Compose(
            [
                v2.Lambda(v2.ToDtype(torch.float32, scale=True), Image),
                v2.Lambda(normalize_min_max2, Image),
                v2.Lambda(v2.ToDtype(torch.long), Mask),
                v2.Lambda(torch.squeeze, Mask),
            ]
        )

        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: str):
        if stage == "fit":
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
        if stage == "test":
            self.ds_test = SegmentationDataset(
                self.test_data_dir,
                ext='tif',
                transforms=self.test_trans,
            )

    def teardown(self, stage: str) -> None:
        if stage == "fit":
            del self.ds_train
            del self.ds_val
        if stage == "test":
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
