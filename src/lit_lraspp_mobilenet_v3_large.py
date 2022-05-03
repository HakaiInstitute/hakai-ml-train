# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2021-05-17
# Description:
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from torchmetrics import Accuracy, JaccardIndex, Precision, Recall
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from kelp_data_module import KelpDataModule
from utils.loss import FocalTverskyMetric


class LRASPPMobileNetV3Large(pl.LightningModule):
    def __init__(self, num_classes: int = 2, ignore_index: Optional[int] = None, lr: float = 0.35,
                 weight_decay: float = 0):
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
                                    lr=self.lr, weight_decay=self.weight_decay, nesterov=True, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                                           total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    @classmethod
    def from_presence_absence_weights(cls, pt_weights_file, args):
        self = cls(num_classes=args.num_classes, ignore_index=args.ignore_index,
                   lr=args.lr, weight_decay=args.weight_decay)
        weights = torch.load(pt_weights_file)

        # Remove trained weights for previous classifier output layers
        if args.keep_pretrained_output_layers:
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
        group.add_argument("--weight_decay", type=float, default=0,
                           help="The weight decay factor for L2 regularization.")
        group.add_argument("--ignore_index", type=int, default=None,
                           help="Label of any class to ignore.")
        group.add_argument("--keep_pretrained_output_layers", action="store_true", default=False,
                           help="Label of any class to ignore.")
        group.add_argument("--backbone_finetuning_epoch", type=int, default=None,
                           help="Set a value to unlock the epoch that the backbone network should be unfrozen."
                                "Leave as None to train all layers from the start.")
        return parser


class Finetuning(pl.callbacks.BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=30):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.model.backbone, train_bn=True)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.backbone,
                optimizer=optimizer,
                train_bn=False,
            )


def train(config, args):
    # ------------
    # data
    # ------------
    kelp_data = KelpDataModule(args.data_dir,
                               num_classes=args.num_classes,
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
                                       lr=config['lr'], weight_decay=config['weight_decay'])
    if args.weights and Path(args.weights).suffix == ".pt":
        print("Loading state_dict:", args.weights)
        model.load_state_dict(torch.load(args.weights), strict=False)

    # ------------
    # callbacks
    # ------------
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            verbose=True,
            monitor="val_miou", mode="max",
            filename="best-{val_miou:.4f}-{epoch}-{step}",
            save_top_k=1, save_last=True,
            save_on_train_epoch_end=False,
            every_n_epochs=1,
        ),
        TuneReportCallback(
            {
                "loss": "val_loss",
                "miou": "val_miou",
                "accuracy": "val_accuracy",
                "precision": "val_precision",
                "recall": "val_recall",
                "cls0_iou": "val_cls0_iou",
                "cls1_iou": "val_cls1_iou"
            },
            on="validation_end"
        )
    ]

    if args.backbone_finetuning_epoch is not None:
        callbacks.append(Finetuning(unfreeze_at_epoch=args.backbone_finetuning_epoch))
    if args.swa_epoch_start:
        callbacks.append(
            pl.callbacks.StochasticWeightAveraging(swa_lrs=args.swa_lrs, swa_epoch_start=args.swa_epoch_start))

    tensorboard_logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".", default_hp_metric=False)

    trainer = pl.Trainer.from_argparse_args(args, logger=tensorboard_logger, callbacks=callbacks, enable_progress_bar=False)
    trainer.fit(model, datamodule=kelp_data)


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
    parser.add_argument("--swa_epoch_start", type=float,
                        help="The epoch at which to start the stochastic weight averaging procedure.")
    parser.add_argument("--swa_lrs", type=float, default=0.05,
                        help="The lr to start the annealing procedure for stochastic weight averaging.")
    parser.add_argument("--tune_lr", action='store_true', default=False,
                        help="Flag to run the lr finder procedure before training.")
    parser.add_argument("--test-only", action="store_true", help="Only run the test dataset")

    parser = KelpDataModule.add_argparse_args(parser)
    parser = LRASPPMobileNetV3Large.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(argv)

    # ------------
    # training
    # ------------
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
    }
    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=3,
        reduction_factor=2
    )
    reporter = CLIReporter(
        parameter_columns=["lr", "weight_decay"],
        metric_columns=["loss", "miou", "training_iteration"],
    )

    train_fn_with_parameters = tune.with_parameters(train, args=args)

    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial={"cpu": 8, "gpu": 1},
        metric="miou",
        mode="max",
        config=config,
        num_samples=20,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=args.name,
        local_dir=args.checkpoint_dir
    )

    print("Best hyperparameters found were: ", analysis.best_config)

    # if args.test_only:
    #     with torch.no_grad():
    #         trainer.validate(model, datamodule=kelp_data)
    #         trainer.test(model, datamodule=kelp_data)
    # else:
    #     if args.tune_lr:
    #         lr_finder = trainer.tuner.lr_find(model, datamodule=kelp_data, early_stop_threshold=None)
    #         lr = lr_finder.suggestion()
    #         print(f"Found lr: {lr}")
    #         model.lr = lr
    #
    #     trainer.fit(model, datamodule=kelp_data)
    #     trainer.test(model, datamodule=kelp_data, ckpt_path="best")


# def train():
#     pass


if __name__ == "__main__":
    debug = os.getenv("DEBUG", False)
    if debug:
        cli_main([
            "/home/taylor/PycharmProjects/hakai-ml-train/data/kelp_pa/Feb2022",
            "/home/taylor/PycharmProjects/hakai-ml-train/checkpoints/kelp_pa/",
            # "--test-only",
            "--weights=/home/taylor/PycharmProjects/hakai-ml-train/checkpoints/kelp_pa/LRASPP/"
            "best-val_miou=0.8023-epoch=18-step=17593.pt",
            "--name=LR_ASPP_ACO_DEV", "--num_classes=2", "--lr=0.0035",
            "--weight_decay=3e-6",
            "--gradient_clip_val=0.5",
            "--benchmark",
            "--accelerator=gpu",
            # "--accelerator=cpu",
            "--devices=auto",
            # "--strategy=ddp_find_unused_parameters_false",
            "--keep_pretrained_output_layers",
            "--max_epochs=10",
            # "--batch_size=2",
            # "--overfit_batches=10",
            '--limit_train_batches=10',
            "--limit_val_batches=10",
            "--limit_test_batches=10",
            "--log_every_n_steps=5",
            # "--swa_epoch_start=0.8"
        ])
    else:
        cli_main()
