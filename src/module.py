from typing import Any

import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics.classification as fm
from huggingface_hub import PyTorchModelHubMixin

from . import losses


class SMPSegmentationModel(
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
        b2: float = 0.99,
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
        for p in self.model.parameters():
            p.requires_grad = True

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
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"))

        probs = self.activation_fn(logits)
        self.log_dict(
            {
                f"{phase}/accuracy": self.accuracy(probs, y),
                f"{phase}/iou": self.jaccard_index(probs, y),
                f"{phase}/recall": self.recall(probs, y),
                f"{phase}/precision": self.precision(probs, y),
                f"{phase}/f1": self.f1_score(probs, y),
            }
        )

        return loss

    # def on_fit_start(self) -> None:
    #     # Save transforms
    #     self.trainer.logger.log_hyperparams(
    #         {
    #             "train_trans": self.train_dataloader().dataset.transforms.to_dict(),
    #             "val_trans": self.val_dataloader().dataset.transforms.to_dict(),
    #         }
    #     )

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            {
                "val/accuracy_epoch": self.accuracy,
                "val/iou_epoch": self.jaccard_index,
                "val/recall_epoch": self.recall,
                "val/precision_epoch": self.precision,
                "val/f1_epoch": self.f1_score,
            }
        )

    def configure_optimizers(self):
        """Init optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            [
                param
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
