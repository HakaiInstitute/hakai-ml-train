# Created by: Taylor Denouden
# Organization: Hakai Institute
from abc import abstractmethod
from typing import Optional, TypeVar

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BaseFinetuning
from torch.optim import Optimizer
from torchmetrics import Accuracy, JaccardIndex, Precision, Recall

from utils.loss import FocalTverskyLoss

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

        # Create model from pre-trained UNet
        self.model = None
        self.init_model()

        # Loss function
        self.focal_tversky_loss = FocalTverskyLoss(self.num_classes, ignore_index=self.ignore_index,
                                                   alpha=loss_alpha, beta=(1 - loss_alpha), gamma=loss_gamma)
        self.accuracy_metric = Accuracy(num_classes=self.num_classes, ignore_index=self.ignore_index, mdmc_average='global')
        self.iou_metric = JaccardIndex(num_classes=self.num_classes, ignore_index=self.ignore_index, average="none")
        self.precision_metric = Precision(num_classes=self.num_classes, ignore_index=self.ignore_index,
                                          average="none", mdmc_average='global')
        self.recall_metric = Recall(num_classes=self.num_classes, ignore_index=self.ignore_index,
                                    average="none", mdmc_average='global')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    # noinspection DuplicatedCode
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
        loss = self.focal_tversky_loss(probs, y)

        ious = self.iou_metric(probs, y)
        acc = self.accuracy_metric(probs, y)
        precisions = self.precision_metric(probs, y)
        recalls = self.recall_metric(probs, y)
        stats = self.stat_scores(probs, y)

        # Filter nan values before averaging
        miou = ious[~ious.isnan()].mean()
        avg_precision = precisions[~precisions.isnan()].mean()
        avg_recall = recalls[~recalls.isnan()].mean()

        if phase == 'val':
            self.log(f"hp_metric", miou, sync_dist=True)

        self.log(f"{phase}_loss", loss, sync_dist=True)
        self.log(f"{phase}_miou", miou, sync_dist=True),
        self.log(f"{phase}_accuracy", acc, sync_dist=True)
        self.log(f"{phase}_average_precision", avg_precision, sync_dist=True)
        self.log(f"{phase}_average_recall", avg_recall, sync_dist=True)

        for c in range(len(ious)):
            i = c + 1 if c >= self.ignore_index else c
            self.log(f"{phase}_cls{i}_iou", ious[c], sync_dist=True)

        for c in range(len(precisions)):
            if c == self.ignore_index:
                continue
            if not precisions[c].isnan():
                self.log(f"{phase}_cls{c}_precision", precisions[c], sync_dist=True)
            if not recalls[c].isnan():
                self.log(f"{phase}_cls{c}_recall", recalls[c], sync_dist=True)

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
