from typing import Any

import habitat_mapper as hab
import torch
import torchmetrics as tm
from habitat_mapper.model import LegacyKelpRGBIModel, LegacyKelpRGBModel
from lightning import pytorch as pl
from torchmetrics import classification as fm

from hakai_ml_train import losses
from hakai_ml_train.models import configure_optimizers as _configure_optimizers

#: Metrics reported per class rather than aggregated over classes.
_PER_CLASS_METRICS = ("iou", "recall", "precision", "f1")


class HabitatMapperSegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model: str,
        revision: str,
        loss: str,
        loss_opts: dict[str, Any],
        num_classes: int = 2,
        ignore_index: int | None = None,
        optimizer_class: str = "torch.optim.AdamW",
        optimizer_opts: dict[str, Any] | None = None,
        lr_scheduler_class: str = "torch.optim.lr_scheduler.OneCycleLR",
        lr_scheduler_opts: dict[str, Any] | None = None,
        lr_scheduler_interval: str = "step",
        lr_scheduler_monitor: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        task = "binary" if num_classes == 1 else "multiclass"

        self.model = hab.model_registry[model, revision]

        self.loss_fn = losses.__dict__[loss](**loss_opts)

        if task == "binary":
            self.activation_fn = lambda x: torch.sigmoid(x).squeeze(1)
        elif task == "multiclass":
            self.activation_fn = lambda x: torch.softmax(x, dim=1).squeeze(1)
        else:
            raise ValueError("task not supported. Must be 'binary' or 'multiclass'")

        # metrics
        metric_kwargs = dict(
            task=task,
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
        )
        metrics = tm.MetricCollection(
            {
                "accuracy": fm.Accuracy(**metric_kwargs),
                "iou": fm.JaccardIndex(**metric_kwargs),
                "recall": fm.Recall(**metric_kwargs),
                "precision": fm.Precision(**metric_kwargs),
                "f1": fm.F1Score(**metric_kwargs),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    @property
    def backbone(self):
        return self.model.encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.model._predict(x.cpu().numpy()), device=self.device)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="test")

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        logits = self.forward(x)

        # Explicitly compute loss in f32 (not bf16, etc.)
        loss = self.loss_fn(logits.float(), y.long().unsqueeze(1))
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"), sync_dist=True)

        probs = self.activation_fn(logits)
        metrics = getattr(self, f"{phase}_metrics")
        self.log_dict(metrics(probs, y), sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        computed = self.train_metrics.compute()
        self.log_dict(
            {f"{k}_epoch": v for k, v in computed.items()},
            sync_dist=True,
        )
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        computed = self.val_metrics.compute()
        self.log_dict(
            {f"{k}_epoch": v for k, v in computed.items()},
            sync_dist=True,
        )
        self.val_metrics.reset()

    def configure_optimizers(self):
        return _configure_optimizers(self)


class HabitatMapperLegacyKelpModel(HabitatMapperSegmentationModel):
    """Baseline wrapper for habitat_mapper's legacy two-network kelp models.

    `kelp-rgb` and `kelp-rgbi` don't emit one logit per class: the leading
    channel(s) score kelp presence and the trailing two score species, from two
    separately trained networks. Softmaxing all of them together -- as the
    generic wrapper does -- mixes the two heads and gives meaningless
    probabilities, so recombine them the way habitat_mapper does at inference
    time (see `habitat_mapper.model.Legacy*Model._postprocess`):

        P(bg)      = 1 - P(kelp)
        P(species) = P(kelp) * P(species | kelp)

    `forward` therefore returns probabilities rather than logits; configure the
    loss with `from_logits: false`.
    """

    def __init__(
        self, *args, class_names: tuple[str, ...] = ("bg", "macro", "nereo"), **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_names = class_names

        if len(class_names) != self.hparams.num_classes:
            raise ValueError(
                f"class_names has {len(class_names)} entries "
                f"({', '.join(class_names)}) but num_classes is "
                f"{self.hparams.num_classes}."
            )

        if not isinstance(self.model, (LegacyKelpRGBModel, LegacyKelpRGBIModel)):
            raise TypeError(
                f"{type(self).__name__} only handles habitat_mapper's legacy "
                f"two-network kelp models, but "
                f"{self.hparams.model}@{self.hparams.revision} loads as "
                f"{type(self.model).__name__}. Use "
                f"{HabitatMapperSegmentationModel.__name__} instead."
            )

        # The datamodule's Albumentations pipeline already scales and normalizes
        # the chips, so habitat_mapper must not do it a second time. Its default
        # max_pixel_value of "auto" also fails outright on float input, since it
        # infers the scale with np.iinfo(dtype).
        self.model.cfg.normalization = None
        self.model.cfg.max_pixel_value = 1.0

        # forward() already returns probabilities.
        self.activation_fn = lambda x: x

        # metrics -- override the parent's aggregated collection with per-class
        # (average="none") variants, named to match
        # SMPMulticlassSegmentationModel so baseline runs plot against trained ones.
        metric_kwargs = dict(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
        )
        metrics = tm.MetricCollection(
            {
                "accuracy": fm.Accuracy(**metric_kwargs),
                "iou": fm.JaccardIndex(**metric_kwargs, average="none"),
                "recall": fm.Recall(**metric_kwargs, average="none"),
                "precision": fm.Precision(**metric_kwargs, average="none"),
                "f1": fm.F1Score(**metric_kwargs, average="none"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)

        if isinstance(self.model, LegacyKelpRGBModel):
            # Channel 0 is a lone kelp-presence logit, scored with a sigmoid.
            p_kelp = torch.sigmoid(out[:, :1])
        else:
            # Channels 0-1 are (bg, kelp) presence logits, scored with a softmax.
            p_kelp = torch.softmax(out[:, :2], dim=1)[:, 1:]

        p_species = torch.softmax(out[:, -2:], dim=1)
        return torch.cat([1 - p_kelp, p_kelp * p_species], dim=1)

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        probs = self.forward(x)

        # Explicitly compute loss in f32 (not bf16, etc.)
        loss = self.loss_fn(probs.float(), y.long())
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"), sync_dist=True)

        step_values = getattr(self, f"{phase}_metrics")(probs, y)
        self.log(f"{phase}/accuracy", step_values[f"{phase}/accuracy"], sync_dist=True)
        self.log_dict(
            {
                f"{phase}/{metric}_{class_name}": step_values[f"{phase}/{metric}"][i]
                for metric in _PER_CLASS_METRICS
                for i, class_name in enumerate(self.class_names)
            },
            sync_dist=True,
        )
        # Summary IoU excludes the background class, as for the trained models.
        self.log(f"{phase}/iou", step_values[f"{phase}/iou"][1:].mean(), sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")

    def _log_epoch_metrics(self, phase: str) -> None:
        """Log metrics accumulated over the whole epoch, per class, then reset."""
        metrics = getattr(self, f"{phase}_metrics")
        computed = metrics.compute()

        self.log(
            f"{phase}/accuracy_epoch", computed[f"{phase}/accuracy"], sync_dist=True
        )
        self.log_dict(
            {
                f"{phase}/{metric}_epoch/{class_name}": computed[f"{phase}/{metric}"][i]
                for metric in _PER_CLASS_METRICS
                for i, class_name in enumerate(self.class_names)
            },
            sync_dist=True,
        )
        self.log(
            f"{phase}/iou_epoch", computed[f"{phase}/iou"][1:].mean(), sync_dist=True
        )

        metrics.reset()
