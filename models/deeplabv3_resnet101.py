# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-23
# Description:

import itertools

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.core.decorators import auto_move_data
from torchmetrics import Accuracy, IoU
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

from models.mixins import GeoTiffPredictionMixin
from utils.loss import FocalTverskyMetric


class DeepLabv3ResNet101(GeoTiffPredictionMixin, pl.LightningModule):
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
        self.model.requires_grad_(True)
        self.model.aux_classifier = FCNHead(1024, self.hparams.num_classes)
        self.model.aux_classifier.requires_grad_(True)
        self.model.classifier = DeepLabHead(2048, self.hparams.num_classes)
        self.model.classifier.requires_grad_(True)

        BaseFinetuning.freeze((self.model.backbone[a] for a in self.model.backbone),
                              train_bn=self.hparams.train_backbone_bn)

        if self.hparams.unfreeze_backbone_epoch == 0:
            BaseFinetuning.make_trainable([self.model.backbone.layer4, self.model.backbone.layer3])

        # Loss function
        self.focal_tversky_loss = FocalTverskyMetric(self.hparams.num_classes, alpha=0.7, beta=0.3, gamma=4. / 3.)
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
        head_params = itertools.chain(self.model.classifier.parameters(), self.model.aux_classifier.parameters())
        backbone_params = itertools.chain(self.model.backbone.layer4.parameters(),
                                          self.model.backbone.layer3.parameters())

        optimizer = torch.optim.AdamW([
            {"params": head_params},
            {"params": backbone_params, "lr": self.hparams.backbone_lr},
        ], lr=self.hparams.lr, amsgrad=True, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                           max_lr=[self.hparams.lr, self.hparams.backbone_lr],
                                                           total_steps=self.num_training_steps)

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

        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        self.log('train_miou', ious.mean(), on_epoch=True, sync_dist=True)
        self.log('train_accuracy', acc, on_epoch=True, sync_dist=True)
        for c in range(len(ious)):
            self.log(f'train_c{c}_iou', ious[c], on_epoch=True, sync_dist=True)

        return loss

    def training_epoch_end(self, outputs):
        # Allow fine-tuning of backbone layers after some epochs
        if self.current_epoch == self.hparams.unfreeze_backbone_epoch:
            BaseFinetuning.make_trainable([self.model.backbone.layer4, self.model.backbone.layer3])

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

    @staticmethod
    def ckpt2pt(ckpt_file, pt_path):
        checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
        torch.save(checkpoint['state_dict'], pt_path)

    @classmethod
    def from_presence_absence_weights(cls, pt_weights_file, hparams):
        actual_num_classes = hparams.num_classes
        weights = torch.load(pt_weights_file)

        # Load kelp_presence_scripts/absence weights
        hparams.num_classes = 2
        self = cls(hparams)
        self.load_state_dict(weights)
        self.hparams.num_classes = actual_num_classes

        # switch classifier output layer
        in_channels = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = torch.nn.Conv2d(in_channels, actual_num_classes, kernel_size=1, stride=1)
        self.model.classifier.requires_grad_(True)

        # switch aux_classifier output layer
        in_channels = self.model.aux_classifier[-1].in_channels
        self.model.aux_classifier[-1] = torch.nn.Conv2d(in_channels, actual_num_classes, kernel_size=1, stride=1)
        self.model.aux_classifier.requires_grad_(True)

        return self

    @staticmethod
    def add_argparse_args(parser):
        group = parser.add_argument_group('DeeplabV3')

        group.add_argument('--num_classes', type=int, default=2,
                           help="The number of image classes, including background.")
        group.add_argument('--lr', type=float, default=0.001, help="the learning rate")
        group.add_argument('--backbone_lr', type=float, default=0.0001, help="the learning rate for backbone layers")
        group.add_argument('--weight_decay', type=float, default=1e-3,
                           help="The weight decay factor for L2 regularization.")
        group.add_argument('--aux_loss_factor', type=float, default=0.3,
                           help="The proportion of loss backpropagated to classifier built only on early layers.")
        group.add_argument('--unfreeze_backbone_epoch', type=int, default=0,
                           help="The training epoch to unfreeze earlier layers of Deeplabv3 for fine tuning.")
        group.add_argument('--train_backbone_bn', dest='train_backbone_bn', action='store_true',
                           help="Flag to indicate if backbone batch norm layers should be trained.")
        group.add_argument('--no_train_backbone_bn', dest='train_backbone_bn', action='store_false',
                           help="Flag to indicate if backbone batch norm layers should not be trained.")
        group.set_defaults(train_backbone_bn=True)

        return parser

# class DeepLabv3FineTuningCallback(BaseFinetuning):
#     def __init__(self, unfreeze_at_epoch=100, train_bn=True):
#         super().__init__()
#         self._unfreeze_at_epoch = unfreeze_at_epoch
#         self._train_bn = train_bn
#
#     def freeze_before_training(self, pl_module: LightningModule):
#         for layer in pl_module.model.backbone:
#             self.freeze(pl_module.model.backbone[layer], train_bn=self._train_bn)
#
#     def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
#         if epoch == self._unfreeze_at_epoch:
#             self.make_trainable([pl_module.model.backbone.layer4, pl_module.model.backbone.layer3])
#
#     @staticmethod
#     def add_argparse_args(parser):
#         group = parser.add_argument_group('DeeplabV3FineTuningCallback')
#
#         group.add_argument('--unfreeze_backbone_epoch', type=int, default=0,
#                            help="The training epoch to unfreeze earlier layers of Deeplabv3 for fine tuning.")
#         group.add_argument('--train_backbone_bn', dest='train_backbone_bn', action='store_true',
#                            help="Flag to indicate if backbone batch norm layers should be trained.")
#         group.add_argument('--no_train_backbone_bn', dest='train_backbone_bn', action='store_false',
#                            help="Flag to indicate if backbone batch norm layers should not be trained.")
#         group.set_defaults(train_backbone_bn=True)
#
#         return parser
