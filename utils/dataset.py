from torch.utils.data import Dataset
import torch
from PIL import Image
from pathlib import Path
import numpy as np
import random
from torchvision import transforms as T
from types import SimpleNamespace


def _target_to_tensor_func(mask):
    return (T.functional.to_tensor(mask) * 255).long().squeeze(dim=0)


_target_to_tensor = T.Lambda(_target_to_tensor_func)
_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_inv_normalize = T.Compose([T.Normalize(mean=[ 0., 0., 0.],
                                        std=[1/0.229, 1/0.224, 1/0.225]),
                            T.Normalize(mean=[-0.485, -0.456, -0.406],
                                        std=[1., 1., 1.]),
                            ])
transforms = SimpleNamespace(
    normalize=_normalize,
    inv_normalize=_inv_normalize,
    train_transforms=T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=90),
        T.ToTensor(),
        _normalize,
    ]),
    train_target_transforms=T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=90, fill=(0,)),
        _target_to_tensor,
    ]),
    test_transforms=T.Compose([
        T.ToTensor(),
        _normalize,
    ]),
    test_target_transforms=T.Compose([
        _target_to_tensor,
    ])
)


class SegmentationDataset(Dataset):
    def __init__(self, ds_path, ext=".png", transform=None, target_transform=None):
        super().__init__()
        self._images = sorted(Path(ds_path).joinpath("x").glob("*" + ext))
        self._labels = sorted(Path(ds_path).joinpath("y").glob("*" + ext))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(list(self._images))

    def __getitem__(self, idx):
        img = Image.open(self._images[idx]).convert('RGB')
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