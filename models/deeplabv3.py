# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-23
# Description:

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.core.decorators import auto_move_data
from torch.optim import Optimizer
from torchmetrics import Accuracy, IoU
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
        self.model.aux_classifier = FCNHead(1024, self.hparams.num_classes)
        self.model.classifier = DeepLabHead(2048, self.hparams.num_classes)

        # Loss function
        self.focal_tversky_loss = FocalTverskyMetric(self.hparams.num_classes, alpha=0.7, beta=0.3, gamma=4. / 3.,
                                                     dist_sync_on_step=True)
        self.accuracy_metric = Accuracy(dist_sync_on_step=True)
        self.iou_metric = IoU(num_classes=self.hparams.num_classes, reduction='none', dist_sync_on_step=True)

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

    @property
    def steps_per_epoch(self) -> int:
        return self.num_training_steps // self.trainer.max_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                      amsgrad=True, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr,
                                                           epochs=self.trainer.max_epochs,
                                                           steps_per_epoch=self.steps_per_epoch)

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
        preds = logits.argmax(dim=1)
        ious = self.iou_metric(preds, y)
        acc = self.accuracy_metric(preds, y)

        self.log('train_loss', loss, on_epoch=True)
        self.log('train_miou', ious.mean(), on_epoch=True)
        self.log('train_accuracy', acc, on_epoch=True)
        for c in range(len(ious)):
            self.log(f'train_c{c}_iou', ious[c], on_epoch=True)

        return loss

    # def training_epoch_end(self, outputs):
    #     # Allow fine-tuning of backbone layers after some epochs
    #     if self.current_epoch >= self.hparams.unfreeze_backbone_epoch - 1:
    #         self.model.backbone.layer3.requires_grad_(True)
    #         self.model.backbone.layer4.requires_grad_(True)

    def val_test_step(self, batch, batch_idx, phase='val'):
        x, y = batch
        y_hat = self.model(x)

        logits = y_hat['out']
        probs = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(probs, y)

        preds = logits.argmax(dim=1)
        ious = self.iou_metric(preds, y)
        acc = self.accuracy_metric(preds, y)

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
        group = parser.add_argument_group('DeeplabV3')

        group.add_argument('--num_classes', type=int, default=2,
                           help="The number of image classes, including background.")
        group.add_argument('--lr', type=float, default=0.001, help="the learning rate")
        group.add_argument('--weight_decay', type=float, default=1e-3,
                           help="The weight decay factor for L2 regularization.")
        group.add_argument('--aux_loss_factor', type=float, default=0.3,
                           help="The proportion of loss backpropagated to classifier built only on early layers.")

        return parser


class DeepLabv3FineTuningCallback(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=100):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module: LightningModule):
        for layer in pl_module.model.backbone:
            self.freeze(pl_module.model.backbone[layer], train_bn=False)

    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        if epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(pl_module.model.backbone.layer3, optimizer=optimizer,
                                              train_bn=True)
            self.unfreeze_and_add_param_group(pl_module.model.backbone.layer4, optimizer=optimizer,
                                              train_bn=True)

    @staticmethod
    def add_argparse_args(parser):
        group = parser.add_argument_group('DeeplabV3FineTuningCallback')

        group.add_argument('--unfreeze_backbone_epoch', type=int, default=0,
                           help="The training epoch to unfreeze earlier layers of Deeplabv3 for fine tuning.")
        # group.add_argument('--train_backbone_bn', dest='train_backbone_bn', action='store_true',
        #                    help="Flag to indicate if backbone batch norm layers should be trained.")
        # group.add_argument('--no_train_backbone_bn', dest='train_backbone_bn', action='store_false',
        #                    help="Flag to indicate if backbone batch norm layers should not be trained.")
        # group.set_defaults(train_backbone_bn=False)

        return parser
