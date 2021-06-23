# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-23
# Description:
import itertools
import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.loggers import TestTubeLogger
from torchmetrics import Accuracy, IoU
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

from kelp_data_module import KelpDataModule
from loss import FocalTverskyMetric
from utils.mixins import GeoTiffPredictionMixin


class DeepLabv3ResNet101(GeoTiffPredictionMixin, pl.LightningModule):
    def __init__(self, hparams):
        """hparams must be a dict of
                    weight_decay
                    lr
                    unfreeze_backbone_epoch
                    aux_loss_factor
                    num_classes
                    train_backbone_bn
                """
        super().__init__()
        self.save_hyperparameters(hparams)

        # Create model from pre-trained DeepLabv3
        self.model = deeplabv3_resnet101(pretrained=True, progress=True)
        self.model.aux_classifier = FCNHead(1024, self.hparams.num_classes)
        self.model.classifier = DeepLabHead(2048, self.hparams.num_classes)

        # Setup trainable layers
        self.model.requires_grad_(True)

        BaseFinetuning.freeze((self.model.backbone[a] for a in self.model.backbone),
                              train_bn=self.hparams.train_backbone_bn)

        if self.hparams.unfreeze_backbone_epoch == 0:
            BaseFinetuning.make_trainable([self.model.backbone.layer4, self.model.backbone.layer3])

        # Loss function and metrics
        self.focal_tversky_loss = FocalTverskyMetric(self.hparams.num_classes,
                                                     alpha=0.7, beta=0.3, gamma=4. / 3.,
                                                     ignore_index=self.hparams.get("ignore_index"))
        self.accuracy_metric = Accuracy(ignore_index=self.hparams.get("ignore_index"))
        self.iou_metric = IoU(num_classes=self.hparams.num_classes, reduction='none',
                              ignore_index=self.hparams.get("ignore_index"))

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
        # Get trainable params
        head_params = itertools.chain(self.model.classifier.parameters(),
                                      self.model.aux_classifier.parameters())
        backbone_params = itertools.chain(self.model.backbone.layer4.parameters(),
                                          self.model.backbone.layer3.parameters())

        # Init optimizer and scheduler
        optimizer = torch.optim.AdamW([
            {"params": head_params},
            {"params": backbone_params, "lr": self.hparams.backbone_lr},
        ], lr=self.hparams.lr, amsgrad=True, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                           max_lr=[self.hparams.lr,
                                                                   self.hparams.backbone_lr],
                                                           total_steps=self.num_training_steps)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        logits = y_hat['out']
        probs = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(probs, y)

        aux_logits = y_hat['aux']
        aux_probs = torch.softmax(aux_logits, dim=1)
        aux_loss = self.focal_tversky_loss(aux_probs, y)

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
        self = cls(hparams)
        weights = torch.load(pt_weights_file)

        # Remove trained weights for previous classifier output layers
        del weights['model.classifier.4.weight']
        del weights['model.classifier.4.bias']
        del weights['model.aux_classifier.4.weight']
        del weights['model.aux_classifier.4.bias']

        self.load_state_dict(weights, strict=False)
        return self

    @staticmethod
    def add_argparse_args(parser):
        group = parser.add_argument_group('DeeplabV3')

        group.add_argument('--num_classes', type=int, default=2,
                           help="The number of image classes, including background.")
        group.add_argument('--lr', type=float, default=0.001, help="the learning rate")
        group.add_argument('--backbone_lr', type=float, default=0.0001,
                           help="the learning rate for backbone layers")
        group.add_argument('--weight_decay', type=float, default=1e-3,
                           help="The weight decay factor for L2 regularization.")
        group.add_argument('--ignore_index', type=int,
                           help="Label of any class to ignore.")
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


def cli_main(argv=None):
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser(name='train', help="Train the model.")
    parser_train.add_argument('data_dir', type=str,
                              help="The path to a data directory with subdirectories 'train' and "
                                   "'eval', each with 'x' and 'y' subdirectories containing image "
                                   "crops and labels, respectively.")
    parser_train.add_argument('checkpoint_dir', type=str, help="The path to save training outputs")
    parser_train.add_argument('--initial_weights_ckpt', type=str,
                              help="Path to checkpoint file to load as initial model weights")
    parser_train.add_argument('--initial_weights', type=str,
                              help="Path to pytorch weights to load as initial model weights")
    parser_train.add_argument('--pa_weights', type=str,
                              help="Presence/Absence model weights to use as initial model weights")
    parser_train.add_argument('--name', type=str, default="",
                              help="Identifier used when creating files and directories for this "
                                   "training run.")

    parser_train = KelpDataModule.add_argparse_args(parser_train)
    parser_train = DeepLabv3ResNet101.add_argparse_args(parser_train)
    # parser_train = DeepLabv3FineTuningCallback.add_argparse_args(parser_train)
    parser_train = pl.Trainer.add_argparse_args(parser_train)
    parser_train.set_defaults(func=train)

    parser_pred = subparsers.add_parser(name='pred', help='Predict kelp presence in an image.')
    parser_pred.add_argument('seg_in', type=str,
                             help="Path to a *.tif image to do segmentation on in pred mode.")
    parser_pred.add_argument('seg_out', type=str,
                             help="Path to desired output *.tif created by the model in pred mode.")
    parser_pred.add_argument('weights', type=str,
                             help="Path to a model weights file (*.pt). "
                                  "Required for eval and pred mode.")
    parser_pred.add_argument('--batch_size', type=int, default=2,
                             help="The batch size per GPU (default 2).")
    parser_pred.add_argument('--crop_pad', type=int, default=128,
                             help="The amount of padding added for classification context to each "
                                  "image crop. The output classification on this crop area is not "
                                  "output by the model but will influence the classification of "
                                  "the area in the (crop_size x crop_size) window "
                                  "(defaults to 128).")
    parser_pred.add_argument('--crop_size', type=int, default=256,
                             help="The crop size in pixels for processing the image. Defines the "
                                  "length and width of the individual sections the input .tif "
                                  "image is cropped to for processing (defaults 256).")
    parser_pred = DeepLabv3ResNet101.add_argparse_args(parser_pred)
    parser_pred.set_defaults(func=pred)

    args = parser.parse_args(argv)
    args.func(args)


def pred(args):
    seg_in, seg_out = Path(args.seg_in), Path(args.seg_out)
    seg_out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        logger.warning("Could not find GPU device")

    # ------------
    # model
    # ------------
    print("Loading model:", args.weights)
    if Path(args.weights).suffix == "ckpt":
        model = DeepLabv3ResNet101.load_from_checkpoint(args.weights, batch_size=args.batch_size,
                                                        crop_size=args.crop_size,
                                                        padding=args.crop_pad)
    else:  # Assumes .pt
        model = DeepLabv3ResNet101(args)
        model.load_state_dict(torch.load(args.weights), strict=False)

    model.freeze()
    model = model.to(device)

    # ------------
    # inference
    # ------------
    model.predict_geotiff(str(seg_in), str(seg_out))


def train(args):
    pl.seed_everything(0)

    # ------------
    # data
    # ------------
    kelp_data = KelpDataModule(args.data_dir, num_classes=args.num_classes,
                               batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_HOME'] = str(Path(args.checkpoint_dir).parent)

    if args.initial_weights_ckpt:
        print("Loading initial weights ckpt:", args.initial_weights_ckpt)
        model = DeepLabv3ResNet101.load_from_checkpoint(args.initial_weights_ckpt)
    elif args.pa_weights:
        print("Loading presence/absence weights:", args.pa_weights)
        model = DeepLabv3ResNet101.from_presence_absence_weights(args.pa_weights, args)
    else:
        model = DeepLabv3ResNet101(args)

        if args.initial_weights:
            print("Loading initial weights:", args.initial_weights)
            model.load_state_dict(torch.load(args.initial_weights))

    # ------------
    # callbacks
    # ------------
    logger_cb = TestTubeLogger(args.checkpoint_dir, name=args.name)
    lr_monitor_cb = pl.callbacks.LearningRateMonitor()
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        verbose=True,
        monitor='val_miou',
        mode='max',
        filename='best-{val_miou:.4f}-{epoch}-{step}',
        save_top_k=1,
        save_last=True,
    )

    callbacks = [
        lr_monitor_cb,
        checkpoint_cb,
        # DeepLabv3FineTuningCallback(args.unfreeze_backbone_epoch, args.train_backbone_bn)
    ]
    if isinstance(args.gpus, int):
        callbacks.append(pl.callbacks.GPUStatsMonitor())

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, logger=logger_cb, callbacks=callbacks)
    # resume_from_checkpoint=checkpoint)
    # Tune params
    # trainer.tune(model, datamodule=kelp_data)

    # Training
    trainer.fit(model, datamodule=kelp_data)

    # Validation and Test stats
    trainer.validate(datamodule=kelp_data, ckpt_path=checkpoint_cb.best_model_path)
    trainer.test(datamodule=kelp_data, ckpt_path=checkpoint_cb.best_model_path)

    # Save final weights only
    model.load_from_checkpoint(checkpoint_cb.best_model_path)
    torch.save(model.state_dict(), Path(checkpoint_cb.best_model_path).with_suffix('.pt'))


if __name__ == '__main__':
    if os.getenv('DEBUG', False):
        # cli_main([
        #     'pred',
        #     'scripts/presence/train_input/Triquet_kelp_U0653.tif',
        #     'scripts/presence/train_output/Triquet_kelp_U0653_kelp.tif',
        #     'species/train_input/data/best-val_miou=0.9393-epoch=97-step=34789.pt',
        #     # '--batch_size=2',
        #     # '--crop_size=256',
        #     # '--crop_pad=128'
        # ])
        # cli_main([
        #     'train',
        #     'scripts/presence/train_input/data',
        #     'scripts/presence/train_output/checkpoints',
        #     '--name=TEST', '--num_classes=2', '--lr=0.001', '--backbone_lr=0.00001',
        #     '--weight_decay=0.001', '--gradient_clip_val=0.5', '--auto_select_gpus', '--gpus=-1',
        #     '--benchmark', '--max_epochs=100', '--batch_size=2', "--unfreeze_backbone_epoch=100",
        #     '--log_every_n_steps=5', '--overfit_batches=1', '--no_train_backbone_bn'
        # ])
        # cli_main([
        #     'train',
        #     'scripts/species/train_input/data',
        #     'scripts/species/train_output/checkpoints',
        #     '--name=TEST', '--num_classes=3', '--lr=0.001', '--backbone_lr=0.00001',
        #     '--weight_decay=0.001', '--gradient_clip_val=0.5',
        #     '--max_epochs=2', '--batch_size=2', "--unfreeze_backbone_epoch=100",
        #     '--log_every_n_steps=5',
        #     # '--overfit_batches=20',
        #     '--no_train_backbone_bn',
        #     '--benchmark', '--auto_select_gpus', '--gpus=-1',
        #     '--pa_weights=scripts/species/train_input/data/best-val_miou=0.9393-epoch=97-step=34789.pt'
        # ])
        cli_main([
            'train',
            'scripts/mussels/train_input/data',
            'scripts/mussels/train_output/checkpoints',
            '--name=TEST', '--num_classes=2', '--lr=0.001', '--backbone_lr=0.00001',
            '--weight_decay=0.001', '--gradient_clip_val=0.5', '--auto_select_gpus', '--gpus=-1',
            '--benchmark', '--max_epochs=10', '--batch_size=2', "--unfreeze_backbone_epoch=100",
            '--log_every_n_steps=5', '--overfit_batches=1', '--no_train_backbone_bn'
        ])
    else:
        cli_main()
