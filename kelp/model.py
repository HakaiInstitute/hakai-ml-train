import os

import torch
from torch.utils.data import DataLoader

from models.deeplabv3 import DeepLabv3
from utils.dataset.SegmentationDataset import SegmentationDataset
from utils.dataset.transforms import transforms as t


class KelpModel(DeepLabv3):
    def end_epoch_stats(self, losses, ious, phase):
        bg_iou = ious[:, 0][~torch.isnan(ious[:, 0])].mean(dim=0)
        fg_iou = ious[:, 1][~torch.isnan(ious[:, 1])].mean(dim=0)

        miou = torch.stack([bg_iou, fg_iou]).mean()

        key_prefix = 'val_' if phase == 'val' else ''
        return {
            f'{key_prefix}loss': losses.mean(),
            f'{key_prefix}miou': miou,
            f'{key_prefix}bg_iou': bg_iou,
            f'{key_prefix}fg_iou': fg_iou
        }

    def train_dataloader(self):
        ds_train = SegmentationDataset(self.hparams.train_data_dir, transform=t.train_transforms,
                                       target_transform=t.train_target_transforms, ext="tif")
        return DataLoader(ds_train, shuffle=True, batch_size=self.hparams.batch_size, pin_memory=True,
                          drop_last=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        ds_val = SegmentationDataset(self.hparams.val_data_dir, transform=t.test_transforms,
                                     target_transform=t.test_target_transforms, ext="tif")
        return DataLoader(ds_val, shuffle=False, batch_size=self.hparams.batch_size, pin_memory=True,
                          num_workers=os.cpu_count())
