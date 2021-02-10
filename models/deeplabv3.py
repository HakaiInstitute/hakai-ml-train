# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-23
# Description:
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.metrics.functional import iou
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

from models.mixins import GeoTiffPredictionMixin
from utils.loss import FocalTverskyMetric


class DeepLabv3(GeoTiffPredictionMixin, pl.LightningModule):
    def __init__(self, hparams):
        """hparams must be a dict of
                    aux_loss_factor
                    weight_decay
                    lr
                    unfreeze_backbone_epoch
                    aux_loss_factor
                    num_classes
                """
        super().__init__()
        self.save_hyperparameters(hparams)

        # Create model from pre-trained DeepLabv3
        self.model = deeplabv3_resnet101(pretrained=True, progress=True)
        self.model.requires_grad_(False)
        self.model.classifier = DeepLabHead(2048, self.hparams.num_classes)
        self.model.classifier.requires_grad_(True)

        self.model.aux_classifier = FCNHead(1024, self.hparams.num_classes)
        self.model.aux_classifier.requires_grad_(True)

        if self.hparams.unfreeze_backbone_epoch == 0:
            self.model.backbone.layer3.requires_grad_(True)
            self.model.backbone.layer4.requires_grad_(True)

        # Loss function
        self.focal_tversky_loss = FocalTverskyMetric(self.hparams.num_classes, alpha=0.7, beta=0.3, gamma=4. / 3.)

    @auto_move_data
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model.forward(x)['out'], dim=1)

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, amsgrad=True,
                                      weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr,
                                                           epochs=self.trainer.max_epochs,
                                                           steps_per_epoch=self.num_training_steps)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        logits = y_hat['out']
        preds = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(preds, y)

        aux_logits = y_hat['aux']
        aux_preds = torch.softmax(aux_logits, dim=1)
        aux_loss = self.focal_tversky_loss(aux_preds, y)

        loss = loss + self.hparams.aux_loss_factor * aux_loss
        ious = iou(logits.argmax(dim=1), y, num_classes=self.hparams.num_classes, reduction='none')

        self.log('train_loss', loss, on_epoch=True)
        self.log('train_miou', ious.mean(), on_epoch=True)
        for c in range(len(ious)):
            self.log(f'train_c{c}_iou', ious[c], on_epoch=True)

        return loss

    def training_epoch_end(self, outputs):
        # Allow fine-tuning of backbone layers after some epochs
        if self.current_epoch >= self.hparams.unfreeze_backbone_epoch - 1:
            self.model.backbone.layer3.requires_grad_(True)
            self.model.backbone.layer4.requires_grad_(True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        logits = y_hat['out']
        preds = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(preds, y)
        ious = iou(logits.argmax(dim=1), y, num_classes=self.hparams.num_classes, reduction='none')

        self.log('val_loss', loss)
        self.log('val_miou', ious.mean())
        for c in range(len(ious)):
            self.log(f'val_cls{c}_iou', ious[c])

        return loss

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--num_workers', type=int, default=os.cpu_count())
        parser.add_argument('--unfreeze_backbone_epoch', type=int, default=0)
        parser.add_argument('--aux_loss_factor', type=float, default=0.3)

        return parser
