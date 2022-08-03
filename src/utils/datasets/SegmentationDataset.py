import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Load preprocessed image chips. Used during model train and validation phases."""

    def __init__(self, ds_path, ext="png", transform=None, target_transform=None):
        super().__init__()
        self._images = sorted(Path(ds_path).joinpath("x").glob(f"*.{ext}"))
        self._labels = sorted(Path(ds_path).joinpath("y").glob(f"*.{ext}"))

        assert len(self._images) == len(
            self._labels
        ), "There are an unequal number of images and labels!"

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self._images)

    # noinspection DuplicatedCode
    def __getitem__(self, idx):
        img = Image.open(self._images[idx]).convert("RGB")
        target = Image.open(self._labels[idx])

        seed = np.random.randint(2147483647)

        torch.manual_seed(seed)
        random.seed(seed)  # apply this seed to img transforms
        if self.transform is not None:
            img = self.transform(img)

        torch.manual_seed(seed)
        random.seed(seed)  # apply this seed to target transforms
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
