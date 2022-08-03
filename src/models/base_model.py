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
        self.accuracy_metric = Accuracy(num_classes=self.num_classes, ignore_index=self.ignore_index,
                                        mdmc_average='global')
        self.iou_metric = JaccardIndex(num_classes=self.num_classes, ignore_index=self.ignore_index,
                                       average="none")
        self.precision_metric = Precision(num_classes=self.num_classes, ignore_index=self.ignore_index,
                                          average='weighted', mdmc_average='global')
        self.recall_metric = Recall(num_classes=self.num_classes, ignore_index=self.ignore_index,
                                    average='weighted', mdmc_average='global')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    # noinspection DuplicatedCode
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(probs, y)

        preds = logits.argmax(dim=1)
        ious = self.iou_metric(preds, y)
        acc = self.accuracy_metric(preds, y)

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_miou", ious.mean(), sync_dist=True)
        self.log("train_accuracy", acc, sync_dist=True)
        for c in range(len(ious)):
            name = f"train_cls{(c + 1) if (self.ignore_index and c >= self.ignore_index) else c}_iou"
            self.log(name, ious[c], sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._val_test_step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx):
        return self._val_test_step(batch, batch_idx, phase="test")

    # noinspection PyUnusedLocal,DuplicatedCode
    def _val_test_step(self, batch, batch_idx, phase="val"):
        x, y = batch
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(probs, y)

        preds = logits.argmax(dim=1)
        ious = self.iou_metric(preds, y)
        miou = ious.mean()
        acc = self.accuracy_metric(preds, y)
        precision = self.precision_metric(preds, y)
        recall = self.recall_metric(preds, y)

        if phase == 'val':
            self.log(f"hp_metric", miou)

        self.log(f"{phase}_loss", loss, sync_dist=True)
        self.log(f"{phase}_miou", miou, sync_dist=True)
        self.log(f"{phase}_accuracy", acc, sync_dist=True)
        self.log(f"{phase}_precision", precision, sync_dist=True)
        self.log(f"{phase}_recall", recall, sync_dist=True)

        for c in range(len(ious)):
            name = f"{phase}_cls{(c + 1) if (self.ignore_index and c >= self.ignore_index) else c}_iou"
            self.log(name, ious[c], sync_dist=True)

        return loss

    @property
    def estimated_stepping_batches(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.max_epochs

    def configure_optimizers(self):
        """Init optimizer and scheduler"""
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                    lr=self.lr, weight_decay=self.weight_decay, nesterov=True, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                                           total_steps=self.estimated_stepping_batches)
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
