from typing import Any

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import (
    DiceLoss,
    FocalLoss,
    JaccardLoss,
    LovaszLoss,
    TverskyLoss,
)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing for segmentation."""

    def __init__(self, mode="multiclass", smoothing=0.1, ignore_index=-100):
        super().__init__()
        assert mode in ("binary", "multiclass")
        self.mode = mode
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) for multiclass or (B, 1, H, W) for binary
            targets: (B, H, W) or (B, 1, H, W) integer class labels
        """
        if self.mode == "binary":
            logits = logits.squeeze(1)
            targets = targets.squeeze(1).float()

        valid = targets != self.ignore_index

        if self.mode == "binary":
            targets_smooth = targets.float().clone()
            targets_smooth[~valid] = 0  # zero out ignore pixels before smoothing
            targets_smooth = (
                targets_smooth * (1 - self.smoothing) + 0.5 * self.smoothing
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, targets_smooth, reduction="none"
            )
            loss = loss[valid]
        else:
            log_probs = F.log_softmax(logits, dim=1)
            safe_targets = targets.clone()
            safe_targets[~valid] = 0
            nll = -log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
            smooth = -log_probs.mean(dim=1)
            loss = (1 - self.smoothing) * nll + self.smoothing * smooth
            loss = loss[valid]

        return loss.mean()


class LabelSmoothingLovasz(nn.Module):
    """Combine label-smoothed CE with Lovász loss."""

    def __init__(
        self, mode="multiclass", smoothing=0.1, lovasz_weight=0.5, ignore_index=-100
    ):
        super().__init__()
        assert mode in ("binary", "multiclass")
        self.ce = LabelSmoothingCrossEntropy(mode, smoothing, ignore_index)
        self.lovasz = smp.losses.LovaszLoss(
            mode=mode, from_logits=True, ignore_index=ignore_index
        )
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        lovasz_loss = self.lovasz(logits, targets)
        return ce_loss + self.lovasz_weight * lovasz_loss


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
    "LabelSmoothingLovasz",
    "LabelSmoothingCrossEntropy",
]
