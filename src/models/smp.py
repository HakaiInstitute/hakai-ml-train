from typing import Any

import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics as tm
import torchmetrics.classification as fm
from huggingface_hub import PyTorchModelHubMixin

from .. import losses
from . import configure_optimizers as _configure_optimizers


class SMPBinarySegmentationModel(
    pl.LightningModule,
    PyTorchModelHubMixin,
    library_name="kelp-o-matic",
    tags=["pytorch", "kelp", "segmentation", "drones", "remote-sensing"],
    repo_url="https://github.com/HakaiInstitute/kelp-o-matic",
    docs_url="https://kelp-o-matic.readthedocs.io/",
):
    def __init__(
        self,
        architecture: str,
        encoder_name: str,
        model_opts: dict[str, Any],
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
        ckpt_path: str | None = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        task = "binary" if num_classes == 1 else "multiclass"

        self.model = smp.create_model(
            arch=architecture,
            encoder_name=encoder_name,
            classes=self.hparams.num_classes,
            **model_opts,
        )
        if ckpt_path is not None:
            ckpt = torch.load(self.hparams.ckpt_path, weights_only=False)
            self.load_state_dict(ckpt["state_dict"])

        for p in self.model.parameters():
            p.requires_grad = True
        if freeze_backbone:
            print("Freezing backbone parameters")
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        self.model = torch.compile(self.model)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="test")

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        logits = self.forward(x)

        loss = self.loss_fn(logits, y.long().unsqueeze(1))
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


class SMPMulticlassSegmentationModel(SMPBinarySegmentationModel):
    def __init__(
        self, *args, class_names: tuple[str, ...] = ("bg", "macro", "nereo"), **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_names = class_names

        # metrics — override parent with per-class (average="none") variants
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

        # Override parent's activation function to remove .squeeze(1) for multiclass
        self.activation_fn = lambda x: torch.softmax(x, dim=1)

    def on_validation_epoch_end(self) -> None:
        computed = self.val_metrics.compute()
        self.log("val/accuracy_epoch", computed["val/accuracy"], sync_dist=True)

        iou_per_class = computed["val/iou"]
        precision_per_class = computed["val/precision"]
        recall_per_class = computed["val/recall"]
        f1_per_class = computed["val/f1"]

        for i, class_name in enumerate(self.class_names):
            self.log(f"val/iou_epoch/{class_name}", iou_per_class[i], sync_dist=True)
            self.log(
                f"val/recall_epoch/{class_name}", recall_per_class[i], sync_dist=True
            )
            self.log(
                f"val/precision_epoch/{class_name}",
                precision_per_class[i],
                sync_dist=True,
            )
            self.log(f"val/f1_epoch/{class_name}", f1_per_class[i], sync_dist=True)
        self.log("val/iou_epoch", iou_per_class[1:].mean(), sync_dist=True)
        self.val_metrics.reset()

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        logits = self.forward(x)

        loss = self.loss_fn(logits, y.long())
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"), sync_dist=True)

        probs = self.activation_fn(logits)
        metrics = getattr(self, f"{phase}_metrics")
        step_values = metrics(probs, y)

        self.log(f"{phase}/accuracy", step_values[f"{phase}/accuracy"], sync_dist=True)

        iou_per_class = step_values[f"{phase}/iou"]
        precision_per_class = step_values[f"{phase}/precision"]
        recall_per_class = step_values[f"{phase}/recall"]
        f1_per_class = step_values[f"{phase}/f1"]
        for i, class_name in enumerate(self.class_names):
            self.log_dict(
                {
                    f"{phase}/iou_{class_name}": iou_per_class[i],
                    f"{phase}/recall_{class_name}": recall_per_class[i],
                    f"{phase}/precision_{class_name}": precision_per_class[i],
                    f"{phase}/f1_{class_name}": f1_per_class[i],
                },
                sync_dist=True,
            )
        self.log(f"{phase}/iou", iou_per_class[1:].mean(), sync_dist=True)

        return loss
