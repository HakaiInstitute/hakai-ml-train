import os

import pytorch_lightning as pl
import torch
from pl_bolts.models.vision import UNet as UNet_Base
from torchmetrics.functional import accuracy, iou

from utils.loss import FocalTverskyMetric


class UNet(pl.LightningModule):
    def __init__(self, hparams):
        self.save_hyperparameters(hparams)

        super().__init__()
        self.model = UNet_Base(hparams.num_classes, hparams.input_channels, hparams.num_layers,
                               hparams.features_start, hparams.bilinear)

        # Loss function
        self.focal_tversky_loss = FocalTverskyMetric(self.hparams.num_classes, alpha=0.7, beta=0.3, gamma=4. / 3.)

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

    @property
    def steps_per_epoch(self) -> int:
        return self.num_training_steps // self.trainer.max_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, amsgrad=True,
                                      weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr,
                                                           epochs=self.trainer.max_epochs,
                                                           steps_per_epoch=self.steps_per_epoch)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        logits = y_hat
        preds = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(preds, y)

        preds = logits.argmax(dim=1)
        ious = iou(preds, y, num_classes=self.hparams.num_classes, reduction='none')
        acc = accuracy(preds, y)

        self.log('train_loss', loss, on_epoch=True)
        self.log('train_miou', ious.mean(), on_epoch=True)
        self.log('train_accuracy', acc, on_epoch=True)
        for c in range(len(ious)):
            self.log(f'train_c{c}_iou', ious[c], on_epoch=True)

        return loss

    def val_test_step(self, batch, batch_idx, phase='val'):
        x, y = batch
        y_hat = self.model(x)

        logits = y_hat
        probs = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(probs, y)

        preds = logits.argmax(dim=1)
        ious = iou(preds, y, num_classes=self.hparams.num_classes, reduction='none')
        acc = accuracy(preds, y)

        self.log(f'{phase}_loss', loss)
        self.log(f'{phase}_miou', ious.mean())
        self.log(f'{phase}_accuracy', acc)
        for c in range(len(ious)):
            self.log(f'{phase}_cls{c}_iou', ious[c])

        return loss

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, phase='val')

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, phase='test')

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--num_workers', type=int, default=os.cpu_count())

        parser.add_argument('--input_channels', type=int, default=3)
        parser.add_argument('--num_layers', type=int, default=5)
        parser.add_argument('--features_start', type=int, default=64)
        parser.add_argument('--bilinear', type=bool, default=False)

        return parser
