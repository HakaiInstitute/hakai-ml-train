# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2021-05-17
# Description:
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, JaccardIndex, Precision, Recall
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from typing import Any, Optional

from kelp_data_module import KelpDataModule
from utils.loss import FocalTverskyMetric


class LRASPPMobileNetV3Large(pl.LightningModule):
    def __init__(self, num_classes: int = 2, ignore_index: Optional[int] = None, lr: float = 0.35,
                 weight_decay: float = 3e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        self.weight_decay = weight_decay

        # Create model from pre-trained DeepLabv3
        self.model = lraspp_mobilenet_v3_large(progress=True, num_classes=self.num_classes)
        self.model.requires_grad_(True)

        # Loss function
        self.focal_tversky_loss = FocalTverskyMetric(self.num_classes, alpha=0.7, beta=0.3,
                                                     gamma=4.0 / 3.0,
                                                     ignore_index=self.ignore_index)
        self.accuracy_metric = Accuracy(ignore_index=self.ignore_index)
        self.iou_metric = JaccardIndex(num_classes=self.num_classes, reduction="none",
                                       ignore_index=self.ignore_index)
        self.precision_metric = Precision(num_classes=self.num_classes, ignore_index=self.ignore_index,
                                          average='weighted', mdmc_average='global')
        self.recall_metric = Recall(num_classes=self.num_classes, ignore_index=self.ignore_index,
                                    average='weighted', mdmc_average='global')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)["out"]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        logits = y_hat["out"]
        probs = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(probs, y)

        preds = logits.argmax(dim=1)
        ious = self.iou_metric(preds, y)
        acc = self.accuracy_metric(preds, y)

        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.log("train_miou", ious.mean(), on_epoch=True, sync_dist=True)
        self.log("train_accuracy", acc, on_epoch=True, sync_dist=True)
        for c in range(len(ious)):
            self.log(f"train_c{c}_iou", ious[c], on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._val_test_step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx):
        return self._val_test_step(batch, batch_idx, phase="test")

    def _val_test_step(self, batch, batch_idx, phase="val"):
        x, y = batch
        y_hat = self.model(x)

        logits = y_hat["out"]
        probs = torch.softmax(logits, dim=1)
        loss = self.focal_tversky_loss(probs, y)

        preds = logits.argmax(dim=1)
        ious = self.iou_metric(preds, y)
        miou = ious.mean()
        acc = self.accuracy_metric(preds, y)
        precision = self.precision_metric(preds, y)
        recall = self.recall_metric(preds, y)

        if phase == 'val':
            self.log(f"hp_metric", miou)
        self.log(f"{phase}_loss", loss, sync_dist=True)
        self.log(f"{phase}_miou", miou, sync_dist=True)
        self.log(f"{phase}_accuracy", acc, sync_dist=True)
        self.log(f"{phase}_precision", precision, sync_dist=True)
        self.log(f"{phase}_recall", recall, sync_dist=True)
        for c in range(len(ious)):
            self.log(f"{phase}_cls{c}_iou", ious[c], sync_dist=True)

        return loss

    # @torch.jit.export
    def predict_step(self, batch: Any) -> Any:
        mc_iteration = 5

        # enable Monte Carlo Dropout
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                print(m)
                m.train()

        # take average of `self.mc_iteration` iterations
        pred = torch.vstack([self.forward(batch).unsqueeze(0) for _ in range(mc_iteration)]).mean(
            dim=0)
        return pred

    def configure_optimizers(self):
        """Init optimizer and scheduler"""
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                    lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]

    @classmethod
    def from_presence_absence_weights(cls, pt_weights_file, args):
        self = cls(num_classes=args.num_classes, ignore_index=args.ignore_index,
                   lr=args.lr, weight_decay=args.weight_decay)
        weights = torch.load(pt_weights_file)

        # Remove trained weights for previous classifier output layers
        del weights["model.classifier.low_classifier.weight"]
        del weights["model.classifier.low_classifier.bias"]
        del weights["model.classifier.high_classifier.weight"]
        del weights["model.classifier.high_classifier.bias"]

        self.load_state_dict(weights, strict=False)
        return self

    @staticmethod
    def add_argparse_args(parser):
        group = parser.add_argument_group("LR-ASPP-MobileNet-V3-Large")

        group.add_argument("--num_classes", type=int, default=2,
                           help="The number of image classes, including background.")
        group.add_argument("--lr", type=float, default=0.001, help="the learning rate")
        group.add_argument("--weight_decay", type=float, default=1e-3,
                           help="The weight decay factor for L2 regularization.")
        group.add_argument("--ignore_index", type=int, default=None,
                           help="Label of any class to ignore.")
        return parser


def cli_main(argv=None):
    pl.seed_everything(0)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    parser.add_argument("data_dir", type=str,
                        help="The path to a data directory with subdirectories 'train', 'val', and "
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
    # parser.add_argument("--swa_epoch_start", type=int, default=75,
    #                     help="The epoch at which to start the stochastic weight averaging "
    #                          "procedure.")
    parser.add_argument("--test-only", action="store_true", help="Only run the test dataset")

    parser = KelpDataModule.add_argparse_args(parser)
    parser = LRASPPMobileNetV3Large.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(argv)

    # ------------
    # data
    # ------------
    kelp_data = KelpDataModule(args.data_dir, num_classes=args.num_classes,
                               batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    if args.pa_weights:
        print("Loading presence/absence weights:", args.pa_weights)
        model = LRASPPMobileNetV3Large.from_presence_absence_weights(args.pa_weights, args)
    elif args.weights and Path(args.weights).suffix == ".ckpt":
        print("Loading checkpoint:", args.weights)
        model = LRASPPMobileNetV3Large.load_from_checkpoint(args.weights)
    else:
        model = LRASPPMobileNetV3Large(num_classes=args.num_classes, ignore_index=args.ignore_index,
                                       lr=args.lr, weight_decay=args.weight_decay)
    if args.weights and Path(args.weights).suffix == ".pt":
        print("Loading state_dict:", args.weights)
        model.load_state_dict(torch.load(args.weights), strict=False)

    # ------------
    # callbacks
    # ------------
    logger_cb = TensorBoardLogger(args.checkpoint_dir, name=args.name, default_hp_metric=False)
    checkpoint_cb = pl.callbacks.ModelCheckpoint(verbose=True, monitor="val_miou", mode="max",
                                                 filename="best-{val_miou:.4f}-{epoch}-{step}", save_top_k=1, save_last=True, )
    callbacks = [  # pl.callbacks.StochasticWeightAveraging(swa_epoch_start=args.swa_epoch_start),
        pl.callbacks.LearningRateMonitor(), checkpoint_cb, ]

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


def train():
    pass


if __name__ == "__main__":
    debug = os.getenv("DEBUG", False)
    if debug:
        cli_main(["data/kelp_pa/Feb2022",
                  "checkpoints/kelp_pa",
                  # "--test-only",
                  # "--weights=/home/taylor/PycharmProjects/hakai-ml-train/checkpoints/kelp_pa/last.ckpt",
                  "--name=LR_ASPP", "--num_classes=2", "--lr=0.35", "--weight_decay=3e-6",
                  "--gradient_clip_val=0.5", "--accelerator=gpu", "--gpus=-1", "--benchmark",
                  "--max_epochs=10", "--batch_size=2",
                  "--overfit_batches=10",
                  ])
    else:
        cli_main()
