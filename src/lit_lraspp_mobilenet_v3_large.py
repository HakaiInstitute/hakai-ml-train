# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2021-05-17
# Description:

import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, IoU
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from kelp_data_module import KelpDataModule
from utils.callbacks.SaveBestStateDictCallback import SaveBestStateDictCallback
from utils.loss import FocalTverskyMetric
from utils.mixins import GeoTiffPredictionMixin


class LRASPPMobileNetV3Large(GeoTiffPredictionMixin, pl.LightningModule):
    def __init__(self, hparams):
        """hparams must be a dict of {weight_decay, lr, num_classes}"""
        super().__init__()
        self.save_hyperparameters(hparams)

        # Create model from pre-trained DeepLabv3
        self.model = lraspp_mobilenet_v3_large(
            progress=True, num_classes=self.hparams.num_classes
        )
        self.model.requires_grad_(True)

        # Loss function
        self.focal_tversky_loss = FocalTverskyMetric(
            self.hparams.num_classes,
            alpha=0.7,
            beta=0.3,
            gamma=4.0 / 3.0,
            ignore_index=self.hparams.get("ignore_index"),
        )
        self.accuracy_metric = Accuracy(ignore_index=self.hparams.get("ignore_index"))
        self.iou_metric = IoU(
            num_classes=self.hparams.num_classes,
            reduction="none",
            ignore_index=self.hparams.get("ignore_index"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model.forward(x)["out"], dim=1)

    @property
    def num_training_steps(self) -> int:
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
        return (batches // effective_accum) * self.trainer.max_epochs

    @property
    def steps_per_epoch(self) -> int:
        return self.num_training_steps // self.trainer.max_epochs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            amsgrad=True,
            weight_decay=self.hparams.weight_decay,
        )
        # Stochastic Weight Averaging setup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

        # One Cycle LR setup
        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer, max_lr=self.hparams.lr, total_steps=self.num_training_steps
        # )
        #
        # return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

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

    @classmethod
    def from_presence_absence_weights(cls, pt_weights_file, hparams):
        self = cls(hparams)
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
        group = parser.add_argument_group("L-RASPP-MobileNet-V3-Large")

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
            "--ignore_index",
            type=int,
            default=None,
            help="Label of any class to ignore.",
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
        help="Identifier used when creating files and directories for this training run.",
    )
    parser_train.add_argument(
        "--swa_epoch_start",
        type=Union[int, float],
        default=0.75,
        help="If provided as int, the procedure will start from the `swa_epoch_start`-th epoch. If "
        "provided as float between 0 and 1, the procedure will start from "
        "`int(swa_epoch_start * max_epochs)` epoch",
    )

    parser_train = KelpDataModule.add_argparse_args(parser_train)
    parser_train = LRASPPMobileNetV3Large.add_argparse_args(parser_train)
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
        help="Path to a model weights file (*.pt). Required for eval and pred " "mode.",
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
        default=512,
        help="The crop size in pixels for processing the image. Defines the "
        "length and width of the individual sections the input .tif "
        "image is cropped to for processing (default 512).",
    )
    parser_pred = LRASPPMobileNetV3Large.add_argparse_args(parser_pred)
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
        model = LRASPPMobileNetV3Large.load_from_checkpoint(
            args.weights,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            padding=args.crop_pad,
        )
    else:  # Assumes .pt
        model = LRASPPMobileNetV3Large(args)
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
        model = LRASPPMobileNetV3Large.load_from_checkpoint(args.initial_weights_ckpt)
    elif args.pa_weights:
        print("Loading presence/absence weights:", args.pa_weights)
        model = LRASPPMobileNetV3Large.from_presence_absence_weights(
            args.pa_weights, args
        )
    else:
        model = LRASPPMobileNetV3Large(args)

    if args.initial_weights:
        print("Loading initial weights:", args.initial_weights)
        model.load_state_dict(torch.load(args.initial_weights))

    # ------------
    # callbacks
    # ------------
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        verbose=True,
        monitor="val_miou",
        mode="max",
        filename="best-{val_miou:.4f}-{epoch}-{step}",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [
        pl.callbacks.StochasticWeightAveraging(swa_epoch_start=args.swa_epoch_start),
        pl.callbacks.LearningRateMonitor(),
        checkpoint_cb,
        SaveBestStateDictCallback(),
    ]
    # if isinstance(args.gpus, int):
    #     callbacks.append(pl.callbacks.DeviceStatsMonitor())

    logger_cb = TensorBoardLogger(args.checkpoint_dir, name=args.name)
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
    # trainer.validate(model, datamodule=kelp_data, ckpt_path="best")
    # trainer.test(model, datamodule=kelp_data, ckpt_path="best")

    # Save final weights only
    # model.load_from_checkpoint(checkpoint_cb.best_model_path)
    # torch.save(
    #     model.state_dict(), Path(checkpoint_cb.best_model_path).with_suffix(".pt")
    # )


if __name__ == "__main__":
    debug = os.getenv("DEBUG", False)
    if debug:
        # cli_main([
        #     'pred',
        #     '/home/taylor/Desktop/Sept27P1125_Orthomosaic_export_FriApr03192436.053821.tif',
        #     '/home/taylor/Desktop/Sept27P1125_Orthomosaic_export_FriApr03192436.053821_species2.tif',
        #     'scripts/species/train_output/checkpoints/L_RASPP_MobileNetV3/version_0/checkpoints/best-val_miou=0.8945-epoch=198-step=58107.pt',
        #     '--num_classes=4',
        #     '--ignore_index=1',
        #     '--batch_size=1',
        #     '--crop_size=512',
        #     '--stride=256'
        #     # '--crop_pad=0'
        # ])
        # cli_main([
        #     'train',
        #     'scripts/species/train_input/data',
        #     'scripts/species/train_output/checkpoints',
        #     '--name=L_RASPP_TEST', '--num_classes=4', '--ignore_index=1', '--lr=0.001',
        #     '--weight_decay=0.001', '--gradient_clip_val=0.5',
        #     '--max_epochs=100', '--log_every_n_steps=5',
        #     '--batch_size=8',
        #     # '--auto_scale_batch_size',
        #     # '--pa_weights=scripts/presence/train_output/checkpoints/LRASPP_MobileNetV3/version_0/'
        #     # 'checkpoints/best-val_miou=0.9218-epoch=196-step=69934.pt',
        #     '--benchmark', '--auto_select_gpus', '--gpus=-1',
        #     # '--overfit_batches=1',
        #     # '--num_workers=0'
        # ])
        cli_main(
            [
                "train",
                "scripts/mussels/train_input/data",
                "scripts/mussels/train_output/checkpoints",
                "--name=L_RASPP_TEST",
                "--num_classes=2",
                "--lr=0.001",
                "--weight_decay=0.001",
                "--gradient_clip_val=0.5",
                "--accelerator=gpu",
                "--gpus=-1",
                "--benchmark",
                "--max_epochs=10",
                "--batch_size=2",
                "--overfit_batches=2",
                "--log_every_n_steps=4",
            ]
        )
    else:
        cli_main()