# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-23
# Description:
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, JaccardIndex
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from typing import Any, Optional

from kelp_data_module import KelpDataModule
from utils import callbacks as cb
from utils.loss import FocalTverskyMetric


class DeepLabv3ResNet101(pl.LightningModule):
    def __init__(self, num_classes: int = 2, ignore_index: Optional[int] = None, lr: float = 0.001,
                 weight_decay: float = 0.001, aux_loss_factor: float = 0.3):

        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        self.weight_decay = weight_decay
        self.aux_loss_factor = aux_loss_factor

        # Create model from pre-trained DeepLabv3
        self.model = deeplabv3_resnet101(pretrained=True, progress=True)
        self.model.aux_classifier = FCNHead(1024, self.num_classes)
        self.model.classifier = DeepLabHead(2048, self.num_classes)

        # Setup trainable layers
        self.model.requires_grad_(True)
        self.model.backbone.requires_grad_(False)

        # Loss function and metrics
        self.focal_tversky_loss = FocalTverskyMetric(
            self.num_classes,
            alpha=0.7,
            beta=0.3,
            gamma=4.0 / 3.0,
            ignore_index=self.ignore_index,
        )
        self.accuracy_metric = Accuracy(ignore_index=self.ignore_index)
        self.iou_metric = JaccardIndex(
            num_classes=self.num_classes,
            reduction="none",
            ignore_index=self.ignore_index,
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
            {"params": self.model.aux_classifier.parameters(), "lr": self.lr},
            {"params": self.model.classifier.parameters(), "lr": self.lr},
            {"params": self.model.backbone.layer3.parameters(), "lr": self.lr / 10.0},
            {"params": self.model.backbone.layer4.parameters(), "lr": self.lr / 10.0},
        ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # return optimizer
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[
                self.lr,
                self.lr,
                self.lr / 10.0,
                self.lr / 10.0
            ],
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.trainer.max_epochs,
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

        loss = loss + self.aux_loss_factor * aux_loss
        preds = logits.argmax(dim=1)
        ious = self.iou_metric(preds, y)
        acc = self.accuracy_metric(preds, y)

        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.log("train_miou", ious.mean(), on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train_accuracy", acc, on_epoch=True, sync_dist=True)
        for c in range(len(ious)):
            self.log(f"train_c{c}_iou", ious[c], on_epoch=True, sync_dist=True, prog_bar=True)

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
    def from_presence_absence_weights(cls, pt_weights_file, args):
        self = cls(num_classes=args.num_classes, ignore_index=args.ignore_index,
                   lr=args.lr, weight_decay=args.weight_decay)
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

        group.add_argument("--num_classes", type=int, default=2,
                           help="The number of image classes, including background.")
        group.add_argument("--lr", type=float, default=0.001, help="the learning rate")
        group.add_argument("--weight_decay", type=float, default=1e-3,
                           help="The weight decay factor for L2 regularization.")
        group.add_argument("--ignore_index", type=int, help="Label of any class to ignore.")
        group.add_argument("--aux_loss_factor", type=float, default=0.3,
                           help="The proportion of loss backpropagated to classifier built only on "
                                "early layers.")
        return parser


def train():
    """Dummy method to enable loading old checkpoint files."""
    pass


def cli_main(argv=None):
    pl.seed_everything(0)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    parser.add_argument("data_dir", type=str,
                        help="The path to a data directory with subdirectories 'train', 'val' and "
                             "'test', each with 'x' and 'y' subdirectories containing image crops "
                             "and labels, respectively.")
    parser.add_argument("checkpoint_dir", type=str, help="The path to save training outputs")
    parser.add_argument("--weights", type=str,
                        help="Path to pytorch weights to load as initial model weights")
    parser.add_argument("--pa_weights", type=str,
                        help="Presence/Absence model weights to use as initial model weights")
    parser.add_argument("--name", type=str, default="",
                        help="Identifier used when creating files and directories for this "
                             "training run.")
    parser.add_argument("--test-only", action="store_true", help="Only run the test dataset")

    parser = KelpDataModule.add_argparse_args(parser)
    parser = DeepLabv3ResNet101.add_argparse_args(parser)
    parser = cb.Deeplabv3Resnet101Finetuning.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args(argv)

    # ------------
    # data
    # ------------
    kelp_data = KelpDataModule(
        args.data_dir, num_classes=args.num_classes, batch_size=args.batch_size
    )

    # ------------
    # model
    # ------------
    if args.pa_weights:
        print("Loading presence/absence weights:", args.pa_weights)
        model = DeepLabv3ResNet101.from_presence_absence_weights(args.pa_weights, args)
    elif args.weights and Path(args.weights).suffix == ".ckpt":
        print("Loading checkpoint:", args.weights)
        model = DeepLabv3ResNet101.load_from_checkpoint(args.weights)
    else:
        model = DeepLabv3ResNet101(num_classes=args.num_classes, ignore_index=args.ignore_index,
                                   lr=args.lr, weight_decay=args.weight_decay)
    if args.weights and Path(args.weights).suffix == ".pt":
        print("Loading state_dict:", args.weights)
        model.load_state_dict(torch.load(args.weights), strict=False)

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
    ]

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, logger=logger_cb, callbacks=callbacks)
    if args.test_only:
        with torch.no_grad():
            trainer.validate(model, datamodule=kelp_data)
            trainer.test(model, datamodule=kelp_data)
    else:
        trainer.fit(model, datamodule=kelp_data)
        trainer.test(model, datamodule=kelp_data, ckpt_path="best")


if __name__ == "__main__":
    if os.getenv("DEBUG", False):
        cli_main(
            [
                "data/kelp_pa/Feb2022",
                "checkpoints/kelp_pa",
                "--test-only",
                "--weights=/home/taylor/PycharmProjects/hakai-ml-train/checkpoints/kelp_pa/DEEPLAB/deeplabv3_kelp_200704.ckpt",
                "--name=DEEPLAB",
                "--num_classes=2",
                "--lr=0.35",
                "--weight_decay=3e-6",
                "--gradient_clip_val=0.5",
                "--no_train_backbone_bn",
                "--unfreeze_backbone_epoch=100",
                "--accelerator=gpu",
                "--gpus=-1",
                "--benchmark",
                "--max_epochs=10",
                "--batch_size=1",
                # "--pa_weights=scripts/species/train_input/data/best-val_miou=0.9393-epoch=97-step=34789.pt",
            ]
        )
    else:
        cli_main()
