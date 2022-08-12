# Created by: Taylor Denouden
# Organization: Hakai Institute

import torch
from torch.optim import Optimizer
from torchvision.models import ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_resnet101

from .base_model import BaseModel, Finetuning, WeightsT


# noinspection PyAbstractClass
class DeepLabV3ResNet101(BaseModel):
    def init_model(self):
        self.model = deeplabv3_resnet101(progress=True, weights_backbone=ResNet101_Weights.IMAGENET1K_V1,
                                         num_classes=self.num_classes, aux_loss=False)
        self.model.requires_grad_(True)
        self.model.backbone.requires_grad_(False)
        self.model.backbone.layer3.requires_grad_(True)
        self.model.backbone.layer4.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)["out"]

    def freeze_before_training(self, ft_module: Finetuning) -> None:
        ft_module.freeze(self.model.backbone.layer3, train_bn=False)
        ft_module.freeze(self.model.backbone.layer4, train_bn=False)

    def finetune_function(self, ft_module: Finetuning, epoch: int, optimizer: Optimizer, opt_idx: int) -> None:
        if epoch == ft_module.unfreeze_at_epoch:
            ft_module.unfreeze_and_add_param_group(
                self.model.backbone.layer3,
                optimizer,
                train_bn=ft_module.train_bn)
            ft_module.unfreeze_and_add_param_group(
                self.model.backbone.layer4,
                optimizer,
                train_bn=ft_module.train_bn)

    @staticmethod
    def drop_output_layer_weights(weights: WeightsT) -> WeightsT:
        del weights["model.classifier.low_classifier.weight"]
        del weights["model.classifier.low_classifier.bias"]
        del weights["model.classifier.high_classifier.weight"]
        del weights["model.classifier.high_classifier.bias"]
        return weights
