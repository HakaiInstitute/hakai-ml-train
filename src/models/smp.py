from typing import Any

import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics.classification as fm
from huggingface_hub import PyTorchModelHubMixin

from .. import losses


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
        backbone: str,
        model_opts: dict[str, Any],
        loss: str,
        loss_opts: dict[str, Any],
        num_classes: int = 2,
        ignore_index: int | None = None,
        lr: float = 0.0003,
        wd: float = 0,
        b1: float = 0.9,
        b2: float = 0.999,
        amsgrad: bool = False,
        warmup_period: float = 0.1,
        ckpt_path: str | None = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        task = "binary" if num_classes == 1 else "multiclass"

        self.model = smp.create_model(
            arch=architecture,
            encoder_name=backbone,
            classes=self.hparams.num_classes,
            **model_opts,
        )

        if ckpt_path is not None:
            ckpt = torch.load(self.hparams.ckpt_path)
            self.load_state_dict(ckpt["state_dict"])

        for p in self.model.parameters():
            p.requires_grad = True
        if freeze_backbone:
            print("Freezing backbone parameters")
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        self.loss_fn = losses.__dict__[loss](**loss_opts)

        if task == "binary":
            self.activation_fn = lambda x: torch.sigmoid(x).squeeze(1)
        elif task == "multiclass":
            self.activation_fn = lambda x: torch.softmax(x, dim=1).squeeze(1)
        else:
            raise ValueError("task not supported. Must be 'binary' or 'multiclass'")

        # metrics
        self.accuracy = fm.Accuracy(
            task=task,
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
        )
        self.jaccard_index = fm.JaccardIndex(
            task=task,
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
        )
        self.recall = fm.Recall(
            task=task,
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
        )
        self.precision = fm.Precision(
            task=task,
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
        )
        self.f1_score = fm.F1Score(
            task=task,
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
        )

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
        self.log_dict(
            {
                f"{phase}/accuracy": self.accuracy(probs, y),
                f"{phase}/iou": self.jaccard_index(probs, y),
                f"{phase}/recall": self.recall(probs, y),
                f"{phase}/precision": self.precision(probs, y),
                f"{phase}/f1": self.f1_score(probs, y),
            },
            sync_dist=True,
        )

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            {
                "val/accuracy_epoch": self.accuracy,
                "val/iou_epoch": self.jaccard_index,
                "val/recall_epoch": self.recall,
                "val/precision_epoch": self.precision,
                "val/f1_epoch": self.f1_score,
            },
            sync_dist=True,
        )

    def configure_optimizers(self):
        """Init optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
            amsgrad=self.hparams.amsgrad,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.warmup_period,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class SMPMulticlassSegmentationModel(SMPBinarySegmentationModel):
    def __init__(
        self, *args, class_names: tuple[str, ...] = ("bg", "macro", "nereo"), **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_names = class_names

        # metrics
        self.accuracy = fm.Accuracy(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
        )
        self.jaccard_index = fm.JaccardIndex(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
            average="none",
        )
        self.recall = fm.Recall(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
            average="none",
        )
        self.precision = fm.Precision(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
            average="none",
        )
        self.f1_score = fm.F1Score(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
            average="none",
        )

        # Override parent's activation function to remove .squeeze(1) for multiclass
        self.activation_fn = lambda x: torch.softmax(x, dim=1)

    def on_validation_epoch_end(self) -> None:
        self.log("val/accuracy_epoch", self.accuracy, sync_dist=True)

        iou_per_class = self.jaccard_index.compute()
        precision_per_class = self.precision.compute()
        recall_per_class = self.recall.compute()
        f1_per_class = self.f1_score.compute()

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
        self.log(f"val/iou_epoch", iou_per_class[1:].mean(), sync_dist=True)

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        logits = self.forward(x)

        loss = self.loss_fn(logits, y.long())
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"), sync_dist=True)

        probs = self.activation_fn(logits)
        self.log(f"{phase}/accuracy", self.accuracy(probs, y), sync_dist=True)

        iou_per_class = self.jaccard_index(probs, y)
        precision_per_class = self.precision(probs, y)
        recall_per_class = self.recall(probs, y)
        f1_per_class = self.f1_score(probs, y)
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

    def configure_optimizers(self):
        """Init optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
            amsgrad=self.hparams.amsgrad,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="max",
        #     factor=0.1,
        #     patience=2,
        #     threshold=0.0001,
        #     cooldown=3,
        # )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=100,
        #     T_mult=1,
        #     eta_min=self.hparams.lr*100,
        #     last_epoch=-1,
        # )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            pct_start=0.1,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # "monitor": "val/iou_epoch",
                # "interval": "epoch",
                # "frequency": 1,
                "interval": "step",
            },
        }
