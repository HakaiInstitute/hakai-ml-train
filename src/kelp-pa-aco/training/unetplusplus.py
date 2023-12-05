from typing import Optional, Any

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics.functional as fm
from einops import rearrange
from pytorch_lightning.loggers import WandbLogger
from unified_focal_loss import FocalTverskyLoss

from config import TrainingConfig

config = TrainingConfig()


class UNetPlusPlus(pl.LightningModule):
    def __init__(self, num_classes: int = 2, ignore_index: Optional[int] = None, lr: float = 0.35,
                 weight_decay: float = 0, loss_delta: float = 0.7, loss_gamma: float = 4.0 / 3.0, max_epochs: int = 100,
                 warmup_period: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_period = warmup_period

        self.loss_delta = loss_delta
        if self.ignore_index is not None:
            self.n = num_classes - 1
        else:
            self.n = num_classes

        self.model = smp.UnetPlusPlus('resnet34', in_channels=config.num_bands,
                                      classes=self.n, decoder_attention_type="scse")
        for p in self.model.parameters():
            p.requires_grad = True
        self.model = torch.compile(self.model, fullgraph=False, mode="max-autotune")
        # self.loss_fn = AsymmetricUnifiedFocalLoss(delta=loss_delta, gamma=loss_gamma)
        self.loss_fn = FocalTverskyLoss(delta=loss_delta, gamma=loss_gamma)

    @property
    def example_input_array(self) -> Any:
        return torch.ones((config.batch_size, config.num_bands, config.img_shape, config.img_shape), device=self.device, dtype=self.dtype)

    @property
    def class_labels(self) -> dict[int, str]:
        return {
            0: "background",
            1: "kelp"
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def remove_ignore_pixels(self, logits: torch.Tensor, y: torch.Tensor):
        mask = (y != self.ignore_index)
        return logits[mask], y[mask]

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="test")

    def log_image_samples(self, batch: torch.Tensor, preds: torch.Tensor):
        has_wandb_logger = isinstance(self.trainer.logger, pl.loggers.WandbLogger)
        if not has_wandb_logger:
            "No wandb logger found, skipping image logging."
            return

        x, y = batch

        images = [im for im in x[:, :3, :, :]]
        masks = [
            {
                "predictions": {
                    "mask_data": pr.detach().cpu().numpy(),
                    "class_labels": self.class_labels
                },
                "ground_truth": {
                    "mask_data": gt.detach().cpu().numpy(),
                    "class_labels": self.class_labels
                },
            } for gt, pr in zip(y, preds)
        ]
        self.logger.log_image("Predictions", images=images, masks=masks)

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        logits = self.forward(x)
        preds = logits.argmax(dim=1)

        if phase == "val" and batch_idx == 0:
            self.log_image_samples(batch, preds)

        # Flatten and eliminate ignore class instances
        y = rearrange(y, 'b h w -> (b h w)').long()
        logits = rearrange(logits, 'b c h w -> (b h w) c')
        if self.ignore_index is not None:
            logits, y = self.remove_ignore_pixels(logits, y)

        if len(y) == 0:
            print("0 length y!")
            return 0

        probs = torch.softmax(logits, dim=1)

        loss = self.loss_fn(probs, y.long())

        accuracy = fm.accuracy(probs, y, task="multiclass", num_classes=self.n, average='macro')
        miou = fm.jaccard_index(probs, y, task="multiclass", num_classes=self.n, average='macro')
        ious = fm.jaccard_index(probs, y, task="multiclass", num_classes=self.n, average='none')
        recalls = fm.recall(probs, y, task="multiclass", num_classes=self.n, average='none')
        precisions = fm.precision(probs, y, task="multiclass", num_classes=self.n, average='none')
        f1s = fm.f1_score(probs, y, task="multiclass", num_classes=self.n, average='none')

        is_training = phase == "train"
        self.log(f"{phase}/loss", loss, prog_bar=is_training)
        self.log(f"{phase}/miou", miou, prog_bar=is_training)
        self.log(f"{phase}/accuracy", accuracy)

        for c in range(self.n):
            clsx = f"cls{c + 1}" if self.ignore_index and c >= self.ignore_index else f"cls{c}"
            self.log(f"{phase}/{clsx}_iou", ious[c], prog_bar=is_training)
            self.log(f"{phase}/{clsx}_recall", recalls[c])
            self.log(f"{phase}/{clsx}_precision", precisions[c])
            self.log(f"{phase}/{clsx}_f1", f1s[c])

        return loss

    def configure_optimizers(self):
        """Init optimizer and scheduler"""
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                      lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

        steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(steps * self.warmup_period)

        linear_warmup_sch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=warmup_steps)
        cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(steps - warmup_steps))
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_warmup_sch, cosine_sch],
                                                             milestones=[warmup_steps])

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
