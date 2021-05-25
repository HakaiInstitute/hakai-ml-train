# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2021-05-17
# Description:


import pytorch_lightning as pl
import torch
from pytorch_lightning.core.decorators import auto_move_data
from torchmetrics import Accuracy, IoU
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from models.mixins import GeoTiffPredictionMixin
from utils.loss import FocalTverskyMetric


class LRASPPMobileNetV3Large(GeoTiffPredictionMixin, pl.LightningModule):
    def __init__(self, hparams):
        """hparams must be a dict of
                    weight_decay
                    lr
                    num_classes
                """
        super().__init__()
        self.save_hyperparameters(hparams)

        # Create model from pre-trained DeepLabv3
        self.model = lraspp_mobilenet_v3_large(progress=True, num_classes=self.hparams.num_classes)
        self.model.requires_grad_(True)

        # Loss function
        self.focal_tversky_loss = FocalTverskyMetric(self.hparams.num_classes, alpha=0.7, beta=0.3,
                                                     gamma=4. / 3.)
        self.accuracy_metric = Accuracy()
        self.iou_metric = IoU(num_classes=self.hparams.num_classes, reduction='none')

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
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(
            limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    @property
    def steps_per_epoch(self) -> int:
        return self.num_training_steps // self.trainer.max_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, amsgrad=True,
                                      weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr,
                                                           total_steps=self.num_training_steps)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        logits = y_hat['out']
        probs = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(probs, y)

        preds = logits.argmax(dim=1)
        ious = self.iou_metric(preds, y)
        acc = self.accuracy_metric(preds, y)

        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        self.log('train_miou', ious.mean(), on_epoch=True, sync_dist=True)
        self.log('train_accuracy', acc, on_epoch=True, sync_dist=True)
        for c in range(len(ious)):
            self.log(f'train_c{c}_iou', ious[c], on_epoch=True, sync_dist=True)

        return loss

    def val_test_step(self, batch, batch_idx, phase='val'):
        x, y = batch
        y_hat = self.model(x)

        logits = y_hat['out']
        probs = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(probs, y)

        preds = logits.argmax(dim=1)
        ious = self.iou_metric(preds, y)
        acc = self.accuracy_metric(preds, y)

        self.log(f'{phase}_loss', loss, sync_dist=True)
        self.log(f'{phase}_miou', ious.mean(), sync_dist=True)
        self.log(f'{phase}_accuracy', acc, sync_dist=True)
        for c in range(len(ious)):
            self.log(f'{phase}_cls{c}_iou', ious[c], sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, phase='val')

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, phase='test')

    @classmethod
    def from_presence_absence_weights(cls, pt_weights_file, hparams):
        self = cls(hparams)
        weights = torch.load(pt_weights_file)

        # Remove trained weights for previous classifier output layers
        del weights['model.classifier.low_classifier.weight']
        del weights['model.classifier.low_classifier.bias']
        del weights['model.classifier.high_classifier.weight']
        del weights['model.classifier.high_classifier.bias']

        self.load_state_dict(weights, strict=False)
        return self

    @staticmethod
    def add_argparse_args(parser):
        group = parser.add_argument_group('L-RASPP-MobileNet-V3-Large')

        group.add_argument('--num_classes', type=int, default=2,
                           help="The number of image classes, including background.")
        group.add_argument('--lr', type=float, default=0.001, help="the learning rate")
        group.add_argument('--weight_decay', type=float, default=1e-3,
                           help="The weight decay factor for L2 regularization.")

        return parser
