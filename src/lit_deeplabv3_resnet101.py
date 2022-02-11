# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-23
# Description:
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, JaccardIndex
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

from kelp_data_module import KelpDataModule
from utils import callbacks as cb
from utils.loss import FocalTverskyMetric
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
        self.model.backbone.requires_grad_(False)

        # Loss function and metrics
        self.focal_tversky_loss = FocalTverskyMetric(
            self.hparams.num_classes,
            alpha=0.7,
            beta=0.3,
            gamma=4.0 / 3.0,
            ignore_index=self.hparams.get("ignore_index"),
        )
        self.accuracy_metric = Accuracy(ignore_index=self.hparams.get("ignore_index"))
        self.iou_metric = JaccardIndex(
            num_classes=self.hparams.num_classes,
            reduction="none",
            ignore_index=self.hparams.get("ignore_index"),
        )

    @property
    def example_input_array(self) -> Any:
        return torch.rand(2, 3, 512, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)["out"]

    @property
    def steps_per_epoch(self) -> int:
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
        return batches // effective_accum

    def configure_optimizers(self):
        """Init optimizer and scheduler"""
        optimizer = torch.optim.SGD([
            {"params": self.model.aux_classifier.parameters(), "lr": self.hparams.lr},
            {"params": self.model.classifier.parameters(), "lr": self.hparams.lr},
            {"params": self.model.backbone.layer3.parameters(), "lr": self.hparams.lr / 10.0},
            {"params": self.model.backbone.layer4.parameters(), "lr": self.hparams.lr / 10.0},
        ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # return optimizer
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[
                self.hparams.lr,
                self.hparams.lr,
                self.hparams.lr / 10.0,
                self.hparams.lr / 10.0
            ],
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.hparams.max_epochs,
        )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        logits = y_hat["out"]
        probs = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(probs, y)

        aux_logits = y_hat["aux"]
        aux_probs = torch.softmax(aux_logits, dim=1)
        aux_loss = self.focal_tversky_loss(aux_probs, y)

        loss = loss + self.hparams.aux_loss_factor * aux_loss
        preds = logits.argmax(dim=1)
        ious = self.iou_metric(preds, y)
        acc = self.accuracy_metric(preds, y)

        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.log("train_miou", ious.mean(), on_epoch=True, sync_dist=True)
        self.log("train_accuracy", acc, on_epoch=True, sync_dist=True)
        for c in range(len(ious)):
            self.log(f"train_c{c}_iou", ious[c], on_epoch=True, sync_dist=True)

        return loss

    def val_test_step(self, batch, batch_idx, phase="val"):
        x, y = batch
        y_hat = self.model(x)

        logits = y_hat["out"]
        probs = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(probs, y)

        preds = logits.argmax(dim=1)
        ious = self.iou_metric(preds, y)
        acc = self.accuracy_metric(preds, y)

        self.log(f"{phase}_loss", loss, sync_dist=True)
        self.log(f"{phase}_miou", ious.mean(), sync_dist=True)
        self.log(f"{phase}_accuracy", acc, sync_dist=True)
        for c in range(len(ious)):
            self.log(f"{phase}_cls{c}_iou", ious[c], sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, phase="test")

    @staticmethod
    def ckpt2pt(ckpt_file, pt_path):
        checkpoint = torch.load(ckpt_file, map_location=torch.device("cpu"))
        torch.save(checkpoint["state_dict"], pt_path)

    @classmethod
    def from_presence_absence_weights(cls, pt_weights_file, hparams):
        self = cls(hparams)
        weights = torch.load(pt_weights_file)

        # Remove trained weights for previous classifier output layers
        del weights["model.classifier.4.weight"]
        del weights["model.classifier.4.bias"]
        del weights["model.aux_classifier.4.weight"]
        del weights["model.aux_classifier.4.bias"]

        self.load_state_dict(weights, strict=False)
        return self

    @staticmethod
    def add_argparse_args(parser):
        group = parser.add_argument_group("DeeplabV3")

        group.add_argument(
            "--num_classes",
            type=int,
            default=2,
            help="The number of image classes, including background.",
        )
        group.add_argument("--lr", type=float, default=0.001, help="the learning rate")
        group.add_argument(
            "--weight_decay",
            type=float,
            default=1e-3,
            help="The weight decay factor for L2 regularization.",
        )
        group.add_argument(
            "--ignore_index", type=int, help="Label of any class to ignore."
        )
        group.add_argument(
            "--aux_loss_factor",
            type=float,
            default=0.3,
            help="The proportion of loss backpropagated to classifier built only on early layers.",
        )

        return parser


def cli_main(argv=None):
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser(name="train", help="Train the model.")
    parser_train.add_argument(
        "data_dir",
        type=str,
        help="The path to a data directory with subdirectories 'train' and "
             "'eval', each with 'x' and 'y' subdirectories containing image "
             "crops and labels, respectively.",
    )
    parser_train.add_argument(
        "checkpoint_dir", type=str, help="The path to save training outputs"
    )
    parser_train.add_argument(
        "--initial_weights_ckpt",
        type=str,
        help="Path to checkpoint file to load as initial model weights",
    )
    parser_train.add_argument(
        "--initial_weights",
        type=str,
        help="Path to pytorch weights to load as initial model weights",
    )
    parser_train.add_argument(
        "--pa_weights",
        type=str,
        help="Presence/Absence model weights to use as initial model weights",
    )
    parser_train.add_argument(
        "--name",
        type=str,
        default="",
        help="Identifier used when creating files and directories for this "
             "training run.",
    )

    parser_train = KelpDataModule.add_argparse_args(parser_train)
    parser_train = DeepLabv3ResNet101.add_argparse_args(parser_train)
    parser_train = cb.Deeplabv3Resnet101Finetuning.add_argparse_args(parser_train)
    parser_train = pl.Trainer.add_argparse_args(parser_train)
    parser_train.set_defaults(func=train)

    parser_pred = subparsers.add_parser(
        name="pred", help="Predict kelp presence in an image."
    )
    parser_pred.add_argument(
        "seg_in",
        type=str,
        help="Path to a *.tif image to do segmentation on in pred mode.",
    )
    parser_pred.add_argument(
        "seg_out",
        type=str,
        help="Path to desired output *.tif created by the model in pred mode.",
    )
    parser_pred.add_argument(
        "weights",
        type=str,
        help="Path to a model weights file (*.pt). " "Required for eval and pred mode.",
    )
    parser_pred.add_argument(
        "--batch_size", type=int, default=2, help="The batch size per GPU (default 2)."
    )
    parser_pred.add_argument(
        "--crop_pad",
        type=int,
        default=128,
        help="The amount of padding added for classification context to each "
             "image crop. The output classification on this crop area is not "
             "output by the model but will influence the classification of "
             "the area in the (crop_size x crop_size) window "
             "(defaults to 128).",
    )
    parser_pred.add_argument(
        "--crop_size",
        type=int,
        default=256,
        help="The crop size in pixels for processing the image. Defines the "
             "length and width of the individual sections the input .tif "
             "image is cropped to for processing (defaults 256).",
    )
    parser_pred = DeepLabv3ResNet101.add_argparse_args(parser_pred)
    parser_pred.set_defaults(func=pred)

    args = parser.parse_args(argv)
    args.func(args)


def pred(args):
    seg_in, seg_out = Path(args.seg_in), Path(args.seg_out)
    seg_out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ------------
    # model
    # ------------
    print("Loading model:", args.weights)
    if Path(args.weights).suffix == "ckpt":
        model = DeepLabv3ResNet101.load_from_checkpoint(
            args.weights,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            padding=args.crop_pad,
        )
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
    kelp_data = KelpDataModule(
        args.data_dir, num_classes=args.num_classes, batch_size=args.batch_size
    )

    # ------------
    # model
    # ------------
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
    logger_cb = TensorBoardLogger(args.checkpoint_dir, name=args.name)
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        verbose=True,
        monitor="val_miou",
        mode="max",
        filename="best-{val_miou:.4f}-{epoch}-{step}",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [
        cb.Deeplabv3Resnet101Finetuning(unfreeze_at_epoch=args.unfreeze_backbone_epoch,
                                        train_bn=args.train_backbone_bn),
        pl.callbacks.LearningRateMonitor(),
        checkpoint_cb,
        cb.SaveBestStateDict(),
        cb.SaveBestTorchscript(method='trace'),
        cb.SaveBestOnnx(opset_version=11),
    ]

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, logger=logger_cb, callbacks=callbacks)

    # Tune params
    # trainer.tune(model, datamodule=kelp_data)

    # Training
    trainer.fit(model, datamodule=kelp_data)

    # Validation and Test stats
    trainer.validate(model, datamodule=kelp_data, ckpt_path="best")
    trainer.test(model, datamodule=kelp_data, ckpt_path="best")


if __name__ == "__main__":
    if os.getenv("DEBUG", False):
        cli_main(
            [
                "train",
                "scripts/presence/train_input/data",
                "scripts/presence/train_output/checkpoints",
                "--name=DEEPLAB_TEST",
                "--num_classes=2",
                "--lr=0.35",
                "--weight_decay=3e-6",
                "--gradient_clip_val=0.5",
                "--max_epochs=100",
                "--batch_size=2",
                "--unfreeze_backbone_epoch=100",
                "--log_every_n_steps=5",
                '--overfit_batches=1',
                "--no_train_backbone_bn",
                "--gpus=-1",
                # "--pa_weights=scripts/species/train_input/data/best-val_miou=0.9393-epoch=97-step=34789.pt",
            ]
        )
    else:
        cli_main()
