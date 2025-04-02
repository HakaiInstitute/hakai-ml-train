# from unified_focal_loss import FocalTverskyLoss
# from monai.losses import DiceFocalLoss, GeneralizedDiceLoss
# from torchseg.losses import DiceLoss, FocalLoss, JaccardLoss, LovaszLoss, TverskyLoss
from typing import Any

import torch
from segmentation_models_pytorch.losses import (
    DiceLoss,
    FocalLoss,
    JaccardLoss,
    LovaszLoss,
    TverskyLoss,
)


class FocalDiceComboLoss(torch.nn.Module):
    def __init__(self, dice_kwargs: dict[str, Any], focal_kwargs: dict[str, Any]):
        super().__init__()
        self.dice_loss = DiceLoss(**dice_kwargs)
        self.focal_loss = FocalLoss(**focal_kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dl = self.dice_loss(input, target.squeeze(dim=1))
        fl = self.focal_loss(input, target.squeeze(dim=1))
        return dl + fl


__all__ = [
    "DiceLoss",
    "LovaszLoss",
    "FocalLoss",
    "TverskyLoss",
    "JaccardLoss",
    "FocalDiceComboLoss",
]
