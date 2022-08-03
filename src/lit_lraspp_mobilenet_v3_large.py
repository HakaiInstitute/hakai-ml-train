# Created by: Taylor Denouden
# Organization: Hakai Institute
from typing import TypeVar

import torch
from torch.optim import Optimizer
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from base_model import BaseFinetuning, BaseModel

T = TypeVar('T')


class LRASPPMobileNetV3Large(BaseModel):
    def init_model(self):
        self.model = lraspp_mobilenet_v3_large(progress=True, num_classes=self.num_classes)
        self.model.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)['out']

    def freeze_before_training(self, ft_module: BaseFinetuning) -> None:
        ft_module.freeze(self.model.backbone, train_bn=False)

    def finetune_function(self, ft_module: BaseFinetuning, epoch: int, optimizer: Optimizer, opt_idx: int) -> None:
        if epoch == ft_module.unfreeze_at_epoch:
            ft_module.unfreeze_and_add_param_group(
                self.model.backbone,
                optimizer,
                train_bn=ft_module.train_bn)

    @staticmethod
    def drop_output_layer_weights(weights: T) -> T:
        del weights["model.classifier.weight"]
        del weights["model.classifier.bias"]
        return weights
