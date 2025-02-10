import os
from pathlib import Path
from typing import List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset


class SegmentationDataset(VisionDataset):
    """Load preprocessed image chips. Used during m odel train and validation phases."""

    def __init__(self, root: str, *args, ext: str = "tif", **kwargs):
        super().__init__(root, *args, **kwargs)
        self._images = list(sorted(Path(root).joinpath("x").glob(f"*.{ext}")))[::2]
        self._labels = list(sorted(Path(root).joinpath("y").glob(f"*.{ext}")))[::2]

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
        seed: int = 42,
        train_transforms=None,
        test_transforms=None,
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
        self.seed = seed
        self.train_trans = train_transforms
        self.test_trans = test_transforms

        self.train_data_dir = str(Path(data_dir).joinpath("train"))
        self.val_data_dir = str(Path(data_dir).joinpath("val"))
        self.test_data_dir = str(Path(data_dir).joinpath("test"))

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


if __name__ == "__main__":
    from albumentations.pytorch import ToTensorV2
    import albumentations as A
    from tqdm import tqdm
    import math
    import torch
    import numpy as np
    from collections import defaultdict

    # Calculate channel stats
    train_data_dir = "/home/taylor/data/KP-ACO-RGBI-Nov2023/train"
    ds_train = SegmentationDataset(
        train_data_dir,
        ext="tif",
        transforms=A.Compose(
            [
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ]
        ),
    )
    batch_size = 16
    dataloader = DataLoader(
        ds_train,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        num_workers=os.cpu_count() - 2,
        persistent_workers=True,
    )

    num_channels = 4
    n_samples = 0
    channel_sum = torch.zeros(num_channels)
    channel_sum_sq = torch.zeros(num_channels)

    max_pixels = 1024 * 1024
    num_bins = 20
    bin_edges = np.exp(np.linspace(np.log(1), np.log(max_pixels), num_bins + 1)).astype(
        int
    )
    histogram = np.zeros((num_bins,))

    label_count = torch.zeros((max_pixels,))

    for images, labels in tqdm(dataloader, total=math.ceil(len(ds_train) / batch_size)):
        b, c, h, w = images.shape
        n_samples += b * h * w

        # Update channel sums and squared sums
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sum_sq += (images**2).sum(dim=(0, 2, 3))

        labels = np.ma.array(labels.numpy(), mask=(labels == 2).numpy())
        label_sum = labels.sum(axis=(1, 2))
        label_count[label_sum] += 1
        bin_idx = np.digitize(label_sum, bin_edges) - 1
        histogram[bin_idx] += 1

    # Calculate mean and std
    mean = channel_sum / n_samples
    # Var[X] = E[X^2] - E[X]^2
    var = (channel_sum_sq / n_samples) - (mean**2)
    std = torch.sqrt(var)

    print(f"{mean=}")
    print(f"{std=}")

    # Convert to format for visualization
    histogram_data = [
        {
            "range": f"{bin_edges[i]}-{bin_edges[i+1]}",
            "count": histogram[i],
            "binStart": int(bin_edges[i]),
            "binEnd": int(bin_edges[i + 1]),
        }
        for i in range(len(bin_edges) - 1)
    ]

    # Print some statistics
    print("\nHistogram data:")
    for bin_data in histogram_data:
        print(f"Range {bin_data['range']}: {bin_data['count']} images")
