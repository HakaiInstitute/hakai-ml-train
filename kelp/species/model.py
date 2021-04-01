import torch

from models.deeplabv3 import DeepLabv3


class KelpSpeciesModel(DeepLabv3):
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
