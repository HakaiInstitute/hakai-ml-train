# from unified_focal_loss import FocalTverskyLoss
# from monai.losses import DiceFocalLoss, GeneralizedDiceLoss
from torchseg.losses import DiceLoss, LovaszLoss, FocalLoss, TverskyLoss, JaccardLoss

# class FocalDiceLoss(DiceLoss):
#     def __init__(self, *args, gamma: float = 2, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.gamma = gamma
#
#     def forward(self, input: torch.Tensor, target: torch.Tensor):
#         dice_loss = super().forward(input, target)
#         focal_dice_loss = dice_loss ** self.gamma
#         return focal_dice_loss


__all__ = [
    "DiceLoss",
    "LovaszLoss",
    "FocalLoss",
    "TverskyLoss",
    "JaccardLoss",
    # "DiceFocalLoss",
    # "GeneralizedDiceLoss",
]
