# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-23
# Description:

import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.metrics.functional import iou
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

from models.mixins import GeoTiffPredictionMixin
from utils.loss import focal_tversky_loss


class DeepLabv3(GeoTiffPredictionMixin, pl.LightningModule):
    def __init__(self, steps_per_epoch, num_classes=2, max_epochs=100, lr=0.001, weight_decay=0.001,
                 unfreeze_backbone_epoch=0, aux_loss_factor=0.3):
        super().__init__()
        self.aux_loss_factor = aux_loss_factor
        self.weight_decay = weight_decay
        self.lr = lr
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.steps_per_epoch = steps_per_epoch
        self.unfreeze_backbone_epoch = unfreeze_backbone_epoch
        self.save_hyperparameters()

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

    @auto_move_data
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model.forward(x)['out'], dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, amsgrad=True,
                                      weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr,
                                                           epochs=self.hparams.max_epochs,
                                                           steps_per_epoch=self.hparams.steps_per_epoch)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    @staticmethod
    def calc_loss(p, g):
        return focal_tversky_loss(torch.softmax(p, dim=1), g, alpha=0.7, beta=0.3, gamma=4. / 3.)

    @staticmethod
    def calc_iou(p, g):
        num_classes = p.shape[1]
        p_hat = p.argmax(dim=1)
        return iou(p_hat, g, num_classes=num_classes, reduction='none')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        logits = y_hat['out']
        loss = self.calc_loss(logits, y)

        aux_logits = y_hat['aux']
        aux_loss = self.calc_loss(aux_logits, y)

        loss = loss + self.hparams.aux_loss_factor * aux_loss
        ious = self.calc_iou(logits, y)

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
        loss = self.calc_loss(logits, y)
        ious = self.calc_iou(logits, y)

        self.log('val_loss', loss)
        self.log('val_miou', ious.mean())
        for c in range(len(ious)):
            self.log(f'val_cls{c}_iou', ious[c])

        return loss
