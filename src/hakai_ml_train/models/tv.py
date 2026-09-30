import torch
import torchmetrics as tm
import torchmetrics.classification as fm
import torchvision
from segmentation_models_pytorch.losses import LovaszLoss
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


class TorchvisionKelpSpeciesSegmentationModel(TorchvisionMulticlassSegmentationModel):
    """Kelp species model that also trains on kelp of unknown species.

    Targets are ``0`` bg, ``1..num_classes-1`` species, ``unknown_index`` kelp of
    unknown species, and ``ignore_index``. The loss adds two terms:

    1. ``loss`` (e.g. multiclass Lovasz) over pixels whose class is known.
    2. A binary Lovasz hinge on kelp vs. background over every labelled pixel,
       unknown species included, with kelp logit ``logsumexp(species) - bg``.

    An unknown-species pixel therefore only costs something when the model calls
    it background, and the model isn't pushed toward either species.

    Species metrics ignore unknown-species pixels. ``presence_iou`` and
    ``presence_f1`` score kelp vs. background over every kelp class.
    """

    def __init__(
        self,
        *args,
        unknown_index: int = 3,
        presence_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.presence_loss_fn = LovaszLoss(
            mode="binary", ignore_index=self.hparams.ignore_index, from_logits=True
        )
        presence = tm.MetricCollection(
            {
                "presence_iou": fm.BinaryJaccardIndex(
                    ignore_index=self.hparams.ignore_index
                ),
                "presence_f1": fm.BinaryF1Score(ignore_index=self.hparams.ignore_index),
            }
        )
        self.train_presence_metrics = presence.clone(prefix="train/")
        self.val_presence_metrics = presence.clone(prefix="val/")
        self.test_presence_metrics = presence.clone(prefix="test/")

    def _targets(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split targets into species (unknown ignored) and kelp presence."""
        ignore = self.hparams.ignore_index
        species = y.masked_fill(y == self.hparams.unknown_index, ignore)
        presence = torch.where(y == ignore, ignore, (y > 0).long())
        return species, presence

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        logits = self.forward(x)
        species, presence = self._targets(y.long())

        # Explicitly compute loss in f32 (not bf16, etc.)
        logits_f32 = logits.float()
        kelp_logit = torch.logsumexp(logits_f32[:, 1:], dim=1) - logits_f32[:, 0]
        loss_species = self.loss_fn(logits_f32, species)
        loss_presence = self.presence_loss_fn(kelp_logit.unsqueeze(1), presence)
        loss = loss_species + self.hparams.presence_loss_weight * loss_presence
        self.log_dict(
            {
                f"{phase}/loss": loss,
                f"{phase}/loss_species": loss_species,
                f"{phase}/loss_presence": loss_presence,
            },
            prog_bar=(phase == "train"),
            sync_dist=True,
        )

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            getattr(self, f"{phase}_metrics").update(preds, species)
            getattr(self, f"{phase}_presence_metrics").update(
                (preds > 0).long(), presence
            )
        return loss

    def _log_epoch_metrics(self, phase: str) -> None:
        metrics = getattr(self, f"{phase}_metrics")
        presence_metrics = getattr(self, f"{phase}_presence_metrics")
        computed = metrics.compute()
        self.log(
            f"{phase}/accuracy_epoch", computed[f"{phase}/accuracy"], sync_dist=True
        )
        for metric_name in ("iou", "recall", "precision", "f1"):
            per_class = computed[f"{phase}/{metric_name}"]
            for i, class_name in enumerate(self.class_names):
                self.log(
                    f"{phase}/{metric_name}_epoch/{class_name}",
                    per_class[i],
                    sync_dist=True,
                )
            # Mean over the kelp classes, excluding background.
            self.log(
                f"{phase}/{metric_name}_epoch", per_class[1:].mean(), sync_dist=True
            )
        self.log_dict(
            {f"{k}_epoch": v for k, v in presence_metrics.compute().items()},
            sync_dist=True,
        )
        metrics.reset()
        presence_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")
