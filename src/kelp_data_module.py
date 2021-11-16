import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as t

from utils.SegmentationDataset import SegmentationDataset
from utils.transforms import Clamp, ImageClip, PadOut, normalize, target_to_tensor


class KelpDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, num_classes: int, batch_size: int,
                 num_workers: int = os.cpu_count(), pin_memory=True):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_data_dir = Path(data_dir).joinpath("train")
        self.val_data_dir = Path(data_dir).joinpath("eval")

        self.train_trans = t.Compose([
            PadOut(512, 512),
            t.RandomHorizontalFlip(),
            t.RandomVerticalFlip(),
            t.RandomRotation(degrees=45),
            ImageClip(min_=0, max_=255),
            t.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            t.ToTensor(),
            normalize,
        ])
        self.train_target_trans = t.Compose([
            PadOut(512, 512),
            t.RandomHorizontalFlip(),
            t.RandomVerticalFlip(),
            t.RandomRotation(degrees=45, fill=(0,)),
            target_to_tensor,
            Clamp(0, self.num_classes - 1),
        ])
        self.test_trans = t.Compose([
            PadOut(512, 512),
            ImageClip(min_=0, max_=255),
            t.ToTensor(),
            normalize,
        ])
        self.test_target_trans = t.Compose([
            PadOut(512, 512),
            target_to_tensor,
            Clamp(0, self.num_classes - 1),
        ])

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        self.ds_train = SegmentationDataset(self.train_data_dir, transform=self.train_trans,
                                            target_transform=self.train_target_trans)
        eval_full = SegmentationDataset(self.val_data_dir, transform=self.test_trans,
                                        target_transform=self.test_target_trans)

        val_size = int(len(eval_full) * 0.5)
        test_size = len(eval_full) - val_size

        self.ds_val, self.ds_test = random_split(eval_full, [val_size, test_size],
                                                 generator=torch.Generator().manual_seed(42))

        # self.dims = tuple(self.ds_train[0][0].shape)

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.ds_train, shuffle=True, batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          drop_last=True, num_workers=self.num_workers)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.ds_val, shuffle=False, batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.ds_test, shuffle=False, batch_size=self.batch_size,
                          pin_memory=self.pin_memory,
                          num_workers=self.num_workers)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = parent_parser.add_argument_group('KelpDataModule')

        parser.add_argument('--batch_size', type=int, default=32,
                            help="The number of images to process at one time.")
        parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                            help="The number of CPU workers to load images from disk.")
        parser.add_argument('--pin_memory', type=bool, default=True,
                            help="Flag to pin GPU memory for batch loading.")

        return parent_parser
