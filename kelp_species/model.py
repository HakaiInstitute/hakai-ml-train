import torch

from models.deeplabv3 import DeepLabv3


class KelpSpeciesModel(DeepLabv3):
    def end_epoch_stats(self, losses, ious, phase):
        bg_iou = ious[:, 0][~torch.isnan(ious[:, 0])].mean(dim=0)
        macro_iou = ious[:, 1][~torch.isnan(ious[:, 1])].mean(dim=0)
        nereo_iou = ious[:, 2][~torch.isnan(ious[:, 2])].mean(dim=0)

        miou = torch.stack([bg_iou, macro_iou, nereo_iou])
        miou = miou[~torch.isnan(miou)].mean()

        key_prefix = 'val_' if phase == 'val' else ''
        return {
            f'{key_prefix}loss': losses.mean(),
            f'{key_prefix}miou': miou,
            f'{key_prefix}bg_iou': bg_iou,
            f'{key_prefix}macro_iou': macro_iou,
            f'{key_prefix}nereo_iou': nereo_iou
        }
