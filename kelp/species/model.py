import torch

from models.deeplabv3 import DeepLabv3


class KelpSpeciesModel(DeepLabv3):
    def end_epoch_stats(self, losses, ious, phase):
        bg_iou = ious[:, 0][~torch.isnan(ious[:, 0])].mean(dim=0)
        macro_iou = ious[:, 1][~torch.isnan(ious[:, 1])].mean(dim=0)
        nereo_iou = ious[:, 2][~torch.isnan(ious[:, 2])].mean(dim=0)

        miou = torch.stack([bg_iou, macro_iou, nereo_iou]).mean()

        key_prefix = 'val_' if phase == 'val' else ''
        return {
            f'{key_prefix}loss': losses.mean(),
            f'{key_prefix}miou': miou,
            f'{key_prefix}bg_iou': bg_iou,
            f'{key_prefix}macro_iou': macro_iou,
            f'{key_prefix}nereo_iou': nereo_iou
        }

    @classmethod
    def load_from_presence_absence_checkpoint(cls, checkpoint_file, hparams):
        actual_num_classes = hparams.num_classes
        weights = torch.load(checkpoint_file)['state_dict']

        # Load presence/absence weights
        hparams.num_classes = 2
        self = cls(hparams)
        self.load_state_dict(weights)
        self.hparams.num_classes = actual_num_classes

        # switch classifier output layer
        in_channels = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = torch.nn.Conv2d(in_channels, actual_num_classes, kernel_size=1, stride=1)
        self.model.classifier.requires_grad_(True)

        # switch aux_classifier output layer
        in_channels = self.model.aux_classifier[-1].in_channels
        self.model.aux_classifier[-1] = torch.nn.Conv2d(in_channels, actual_num_classes, kernel_size=1, stride=1)
        self.model.aux_classifier.requires_grad_(True)

        return self
