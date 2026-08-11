from typing import Any

import habitat_mapper as hab
import torch
import torchmetrics as tm
from lightning import pytorch as pl
from torchmetrics import classification as fm

from hakai_ml_train import losses
from hakai_ml_train.models import configure_optimizers as _configure_optimizers


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
