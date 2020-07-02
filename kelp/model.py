import torch

from models.deeplabv3 import DeepLabv3


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
