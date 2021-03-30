import os
from pathlib import Path
from typing import List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset.SegmentationDataset import SegmentationDataset
from utils.dataset import transforms as t


class KelpPresenceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = os.cpu_count()):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_data_dir = Path(data_dir).join("train")
        self.val_data_dir = Path(data_dir).join("val")
        self.test_data_dir = Path(data_dir).join("test")

        self.train_transforms = t.Compose([
            t.ImageClip(),
            t.PadOut(512, 512),
            t.RandomHorizontalFlip(),
            t.RandomVerticalFlip(),
            t.RandomRotation(degrees=45),
            t.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            t.ToTensor(),
            t.normalize,
        ])
        self.train_target_transforms = t.Compose([
            t.PadOut(512, 512),
            t.RandomHorizontalFlip(),
            t.RandomVerticalFlip(),
            t.RandomRotation(degrees=45, fill=(0,)),
            t.target_to_tensor,
        ])
        self.test_transforms = t.Compose([
            t.ImageClip(),
            t.PadOut(512, 512),
            t.ToTensor(),
            t.normalize,
        ])
        self.test_target_transforms = t.Compose([
            t.PadOut(512, 512),
            t.target_to_tensor,
        ])

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.ds_train = SegmentationDataset(self.train_data_dir, transform=self.train_transforms,
                                                target_transform=self.train_target_transforms)
            self.ds_val = SegmentationDataset(self.val_data_dir, transform=self.test_transforms,
                                              target_transform=self.test_target_transforms)

            self.dims = tuple(self.ds_train[0][0].shape)

        if stage == 'test' or stage is None:
            self.ds_test = SegmentationDataset(self.val_data_dir, transform=self.test_transforms,
                                               target_transform=self.test_target_transforms)

            self.dims = tuple(self.ds_test[0][0].shape)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.ds_train, shuffle=True, batch_size=self.batch_size, pin_memory=True,
                          drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.ds_val, shuffle=False, batch_size=self.batch_size, pin_memory=True,
                          num_workers=self.num_workers)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.ds_test, shuffle=False, batch_size=self.batch_size, pin_memory=True,
                          num_workers=self.num_workers)
