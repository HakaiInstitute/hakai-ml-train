import random

import numpy as np
import torch
from torch.utils.data import Dataset


class TransformDataset(Dataset):
    """Util class to allow binding different data transforms to Subset datasets."""

    def __init__(self, dataset, transform=None, target_transform=None):
        super().__init__()
        self.ds = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        img, label = self.ds[idx]

        seed = np.random.randint(2147483647)

        torch.manual_seed(seed)
        random.seed(seed)  # apply this seed to img transforms
        if self.transform is not None:
            img = self.transform(img)

        torch.manual_seed(seed)
        random.seed(seed)  # apply this seed to target transforms
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.ds)
