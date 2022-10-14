# Created by: Taylor Denouden
# Organization: Hakai Institute
from abc import abstractmethod
from typing import Any, Optional, TypeVar

import pytorch_lightning as pl
import torch
import torchmetrics.functional as fm
from einops import rearrange
from pytorch_lightning.callbacks import BaseFinetuning
from torch.optim import Optimizer

from utils.loss import focal_tversky_loss

WeightsT = TypeVar('WeightsT')


class BaseModel(pl.LightningModule):
    @abstractmethod
    def init_model(self):
        raise NotImplementedError

    @abstractmethod
    def freeze_before_training(self, ft_module: 'Finetuning') -> None:
        raise NotImplementedError

    @abstractmethod
    def finetune_function(self, ft_module: 'Finetuning', epoch: int, optimizer: Optimizer, opt_idx: int) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def drop_output_layer_weights(weights: WeightsT) -> WeightsT:
        raise NotImplementedError

    def __init__(self, num_classes: int = 2, ignore_index: Optional[int] = None, lr: float = 0.35,
                 weight_decay: float = 0, loss_alpha: float = 0.7, loss_gamma: float = 4.0 / 3.0, max_epochs: int = 100):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.loss_alpha = loss_alpha
        self.loss_beta = 1 - loss_alpha
        self.loss_gamma = loss_gamma

        if self.ignore_index is not None:
            self.n = num_classes - 1
        else:
            self.n = num_classes

        # Create model from pre-trained UNet
        self.model = None
        self.init_model()

    @property
    def example_input_array(self) -> Any:
        return torch.ones((2, 3, 512, 512))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    def remove_ignore_pixels(self, probs, y):
        mask = (y != self.ignore_index)
        return probs[mask], y[mask]

    def training_step(self, batch, batch_idx):
        return self._phase_step(batch, batch_idx, phase="train")

    def validation_step(self, batch, batch_idx):
        return self._phase_step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx):
        return self._phase_step(batch, batch_idx, phase="test")

    # noinspection PyUnusedLocal,DuplicatedCode
    def _phase_step(self, batch, batch_idx, phase):
        x, y = batch
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)

        # Flatten and eliminate ignore class instances
        y = rearrange(y, 'b h w -> (b h w)').long()
        probs = rearrange(probs, 'b c h w -> (b h w) c')

        if self.ignore_index is not None:
            probs, y = self.remove_ignore_pixels(probs, y)

        # Update metrics
        loss = focal_tversky_loss(probs, y, alpha=self.loss_alpha, beta=(1 - self.loss_alpha), gamma=self.loss_gamma)
        ious = fm.jaccard_index(probs, y, num_classes=self.n, average='none')
        miou = fm.jaccard_index(probs, y, num_classes=self.n, average='macro')
        acc = fm.accuracy(probs, y, num_classes=self.n)
        avg_precision = fm.precision(probs, y, num_classes=self.n, average='micro')
        avg_recall = fm.recall(probs, y, num_classes=self.n, average='micro')

        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True),
        self.log(f"{phase}/miou", miou, on_step=False, on_epoch=True),
        self.log(f"{phase}/accuracy", acc, on_step=False, on_epoch=True)
        self.log(f"{phase}/average_precision", avg_precision, on_step=False, on_epoch=True)
        self.log(f"{phase}/average_recall", avg_recall, on_step=False, on_epoch=True)

        for c in range(self.n):
            if self.ignore_index and c >= self.ignore_index:
                self.log(f"{phase}/cls{c + 1}_iou", ious[c], on_step=False, on_epoch=True)
            else:
                self.log(f"{phase}/cls{c}_iou", ious[c], on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Init optimizer and scheduler"""
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                      lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                                           total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


class Finetuning(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10, train_bn=True):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: BaseModel) -> None:
        pl_module.freeze_before_training(ft_module=self)

    def finetune_function(self, pl_module: BaseModel, epoch: int, optimizer: Optimizer, opt_idx: int) -> None:
        pl_module.finetune_function(ft_module=self, epoch=epoch, optimizer=optimizer, opt_idx=opt_idx)
