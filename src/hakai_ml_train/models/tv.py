import torch
import torchvision
from torch import nn

from hakai_ml_train.models.smp import (
    SMPBinarySegmentationModel,
    SMPMulticlassSegmentationModel,
)


class TorchvisionSegmentationNet(nn.Module):
    """Adapts a torchvision segmentation model to the SMP model interface.

    torchvision models return ``OrderedDict(out=...)`` and hold their feature
    extractor at ``.backbone``; SMP models return a tensor and use ``.encoder``.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    @property
    def encoder(self) -> nn.Module:
        return self.net.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)["out"]


class TorchvisionModelMixin:
    """Builds the network from torchvision's segmentation model zoo instead of SMP.

    The torchvision builder is looked up as ``f"{architecture}_{encoder_name}"``,
    e.g. ``architecture: lraspp`` + ``encoder_name: mobilenet_v3_large`` gives
    ``lraspp_mobilenet_v3_large``. ``model_opts`` is passed to that builder
    (e.g. ``weights_backbone: IMAGENET1K_V1``), not to SMP. Layer-wise LR decay
    is not wired up for these backbones.
    """

    def _create_model(self) -> nn.Module:
        net = torchvision.models.get_model(
            f"{self.hparams.architecture}_{self.hparams.encoder_name}",
            num_classes=self.hparams.num_classes,
            **self.hparams.model_opts,
        )
        return TorchvisionSegmentationNet(net)


class TorchvisionBinarySegmentationModel(
    TorchvisionModelMixin, SMPBinarySegmentationModel
):
    """Binary (``num_classes: 1``) torchvision segmentation model."""


class TorchvisionMulticlassSegmentationModel(
    TorchvisionModelMixin, SMPMulticlassSegmentationModel
):
    """Multiclass torchvision segmentation model."""
