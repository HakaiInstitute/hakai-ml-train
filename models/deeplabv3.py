# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-23
# Description:

import math
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.metrics.functional import iou
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

from models.mixins import GeoTiffPredictionMixin
from utils.dataset.SegmentationDataset import SegmentationDataset
from utils.dataset.transforms import transforms as t
from utils.loss import focal_tversky_loss


class DeepLabv3(GeoTiffPredictionMixin, pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

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
        steps_per_epoch = math.floor(len(self.train_dataloader()) / max(torch.cuda.device_count(), 1))
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr,
                                                           epochs=self.hparams.epochs, steps_per_epoch=steps_per_epoch)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        ds_train = SegmentationDataset(self.hparams.train_data_dir, transform=t.train_transforms,
                                       target_transform=t.train_target_transforms)
        return DataLoader(ds_train, shuffle=True, batch_size=self.hparams.batch_size, pin_memory=True,
                          drop_last=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        ds_val = SegmentationDataset(self.hparams.val_data_dir, transform=t.test_transforms,
                                     target_transform=t.test_target_transforms)
        return DataLoader(ds_val, shuffle=False, batch_size=self.hparams.batch_size, pin_memory=True,
                          num_workers=os.cpu_count())

    @staticmethod
    def calc_loss(p, g):
        return focal_tversky_loss(torch.softmax(p, dim=1), g, alpha=0.7, beta=0.3, gamma=4. / 3.)

    @staticmethod
    def calc_iou(p, g):
        num_classes = p.shape[1]
        return iou(p.argmax(dim=1), g, num_classes=num_classes, reduction='none')

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        logits = y_hat['out']
        loss = self.calc_loss(logits, y)

        aux_logits = y_hat['aux']
        aux_loss = self.calc_loss(aux_logits, y)

        loss = loss + self.hparams.aux_loss_factor * aux_loss
        ious = self.calc_iou(logits, y)
        return {'loss': loss, 'ious': ious}

    def training_epoch_end(self, outputs):
        # Allow fine-tuning of backbone layers after some epochs
        if self.current_epoch >= self.hparams.unfreeze_backbone_epoch - 1:
            self.model.backbone.layer3.requires_grad_(True)
            self.model.backbone.layer4.requires_grad_(True)

        losses = torch.stack([x['loss'] for x in outputs])
        ious = torch.stack([x['ious'] for x in outputs])
        stats = self.end_epoch_stats(losses, ious, phase='train')
        return {'loss': stats['loss'], 'log': stats}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        logits = y_hat['out']
        loss = self.calc_loss(logits, y)

        ious = self.calc_iou(logits, y)
        return {'loss': loss, 'ious': ious}

    def validation_epoch_end(self, outputs):
        losses = torch.stack([x['loss'] for x in outputs])
        ious = torch.stack([x['ious'] for x in outputs])
        stats = self.end_epoch_stats(losses, ious, phase="val")
        return {'loss': stats['val_loss'], 'log': stats}

    def end_epoch_stats(self, losses, ious, phase):
        for i in range(ious.shape[1]):
            ious = ious[:, i][~torch.isnan(ious[:, i])].mean(dim=0)

        key_prefix = 'val_' if phase == 'val' else ''
        log_dict = {
            f'{key_prefix}loss': losses.mean(),
            f'{key_prefix}miou': torch.stack(ious).mean(),
        }
        for i in range(len(ious)):
            log_dict[f'{key_prefix}cls{i}_iou'] = ious[i]

        return log_dict
