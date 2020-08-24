import torch

from models.deeplabv3 import DeepLabv3


class KelpModel(DeepLabv3):
    def end_epoch_stats(self, losses, ious, phase):
        for i in range(ious.shape[1]):
            ious = ious[:, i][~torch.isnan(ious[:, i])].mean(dim=0)

        key_prefix = 'val_' if phase == 'val' else ''
        return {
            f'{key_prefix}loss': losses.mean(),
            f'{key_prefix}miou': torch.stack(ious).mean(),
            f'{key_prefix}bg_iou': ious[0],
            f'{key_prefix}fg_iou': ious[1]
        }
