from typing import Optional, Any

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics.classification as fm
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

from . import losses


class SegmentationModel(
    pl.LightningModule,
    PyTorchModelHubMixin,
    library_name="kelp-o-matic",
    tags=["pytorch", "kelp", "segmentation", "drones", "remote-sensing"],
    repo_url="https://github.com/HakaiInstitute/kelp-o-matic",
    docs_url="https://kelp-o-matic.readthedocs.io/",
):
    def __init__(
        self,
        num_classes: int = 2,
        ignore_index: Optional[int] = None,
        lr: float = 0.35,
        weight_decay: float = 0,
        max_epochs: int = 100,
        warmup_period: float = 0.3,
        batch_size: int = 2,
        num_bands: int = 3,
        tile_size: int = 1024,
        architecture: str = "UnetPlusPlus",
        backbone: str = "resnet34",
        options_model: dict[str, Any] = None,
        loss_name: str = "DiceLoss",
        loss_opts: dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_period = warmup_period
        self.batch_size = batch_size
        self.num_bands = num_bands
        self.tile_size = tile_size
        self.n = num_classes - int(
            self.ignore_index is not None
        )  # Sub 1 if ignore_index is set
        if options_model is None:
            options_model = {}

        self.model = smp.__dict__[architecture](
            backbone,
            in_channels=self.num_bands,
            classes=self.n,
            **options_model,
        )
        for p in self.model.parameters():
            p.requires_grad = True

        self.loss_fn = losses.__dict__[loss_name](**(loss_opts or {}))

        # metrics
        self.accuracy = fm.BinaryAccuracy()
        self.jaccard_index = fm.BinaryJaccardIndex()
        self.recall = fm.BinaryRecall()
        self.precision = fm.BinaryPrecision()
        self.f1_score = fm.BinaryF1Score()
        self.dice = fm.Dice()

    @property
    def example_input_array(self) -> Any:
        return torch.ones(
            (self.batch_size, self.num_bands, self.tile_size, self.tile_size),
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def remove_ignore_pixels(self, logits: torch.Tensor, y: torch.Tensor):
        mask = y != self.ignore_index
        return logits[mask], y[mask]

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="test")

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        logits = self.forward(x)
        # if phase == "val" and batch_idx == 0:
        #     self.log_image_samples(batch, preds)

        # Flatten and eliminate ignore class instances
        y = rearrange(y, "b h w -> (b h w)").long()
        logits = rearrange(logits, "b c h w -> (b h w) c")
        if self.ignore_index is not None:
            logits, y = self.remove_ignore_pixels(logits, y)

        if len(y) == 0:
            print("0 length y!")
            return 0

        loss = self.loss_fn(logits, y.long().unsqueeze(1))
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"))

        probs = torch.sigmoid(logits).squeeze(1)
        self.log_dict(
            {
                f"{phase}/accuracy": self.accuracy(probs, y),
                f"{phase}/iou": self.jaccard_index(probs, y),
                f"{phase}/recall": self.recall(probs, y),
                f"{phase}/precision": self.precision(probs, y),
                f"{phase}/f1": self.f1_score(probs, y),
                f"{phase}/dice": self.dice(probs, y),
            }
        )

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            {
                f"val/accuracy_epoch": self.accuracy,
                f"val/iou_epoch": self.jaccard_index,
                f"val/recall_epoch": self.recall,
                f"val/precision_epoch": self.precision,
                f"val/f1_epoch": self.f1_score,
                f"val/dice_epoch": self.dice,
            }
        )

    def configure_optimizers(self):
        """Init optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=True,
        )

        steps = self.trainer.estimated_stepping_batches

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=steps, pct_start=self.warmup_period
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
