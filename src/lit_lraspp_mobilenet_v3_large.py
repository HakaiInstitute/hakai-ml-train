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
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.skopt import SkOptSearch
from torchmetrics import Accuracy, JaccardIndex, Precision, Recall
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from kelp_data_module import KelpDataModule
from utils.loss import FocalTverskyLoss


class LRASPPMobileNetV3Large(pl.LightningModule):
    def __init__(self, num_classes: int = 2, ignore_index: Optional[int] = None, lr: float = 0.35,
                 weight_decay: float = 0, loss_alpha: float = 0.7, loss_gamma: float = 4.0 / 3.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        self.weight_decay = weight_decay

        # Create model from pre-trained DeepLabv3
        self.model = lraspp_mobilenet_v3_large(progress=True, num_classes=self.num_classes)
        self.model.requires_grad_(True)

        # Loss function
        self.focal_tversky_loss = FocalTverskyLoss(self.num_classes, ignore_index=self.ignore_index,
                                                   alpha=loss_alpha, beta=(1 - loss_alpha), gamma=loss_gamma)
        self.accuracy_metric = Accuracy(num_classes=self.num_classes, ignore_index=self.ignore_index,
                                        mdmc_average='global')
        self.iou_metric = JaccardIndex(num_classes=self.num_classes, ignore_index=self.ignore_index,
                                       average="none")
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

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_miou", ious.mean(), sync_dist=True)
        self.log("train_accuracy", acc, sync_dist=True)
        for c in range(len(ious)):
            name = f"train_cls{(c + 1) if (self.ignore_index and c >= self.ignore_index) else c}_iou"
            self.log(name, ious[c], sync_dist=True)

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
            name = f"{phase}_cls{(c + 1) if (self.ignore_index and c >= self.ignore_index) else c}_iou"
            self.log(name, ious[c], sync_dist=True)

        return loss

    # @torch.jit.export
    def predict_step(self, batch: Any) -> Any:
        """Work in Progress. Use regular forward method for now."""
        mc_iteration = 5

        # enable Monte Carlo Dropout
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
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

    @staticmethod
    def add_argparse_args(parser):
        group = parser.add_argument_group("LR-ASPP-MobileNet-V3-Large")

        group.add_argument("--num_classes", type=int, default=2,
                           help="The number of image classes, including background.")
        group.add_argument("--ignore_index", type=int, default=None,
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
    pl.seed_everything(0)

    # ------------
    # data
    # ------------
    kelp_data = KelpDataModule(
        args.data_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size
    )

    # ------------
    # model
    # ------------
    model = LRASPPMobileNetV3Large(
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        lr=config['lr'],
        loss_alpha=config['alpha'],
        weight_decay=config['weight_decay'],
        # loss_gamma=config['gamma']
    )

    if args.weights and Path(args.weights).suffix == ".pt":
        print("Loading state_dict:", args.weights)
        weights = torch.load(args.weights)

        # Remove trained weights for previous classifier output layers
        if args.drop_output_layer_weights:
            del weights["model.classifier.low_classifier.weight"]
            del weights["model.classifier.low_classifier.bias"]
            del weights["model.classifier.high_classifier.weight"]
            del weights["model.classifier.high_classifier.bias"]

        model.load_state_dict(weights, strict=False)

    elif args.weights and Path(args.weights).suffix == ".ckpt":
        print("Loading checkpoint:", args.weights)
        model = LRASPPMobileNetV3Large.load_from_checkpoint(args.weights)

    # ------------
    # callbacks
    # ------------
    callbacks = [
        TuneReportCheckpointCallback(
            metrics=dict(
                [
                    ("loss", "val_loss"),
                    ("miou", "val_miou"),
                    ("accuracy", "val_accuracy"),
                    ("precision", "val_precision"),
                    ("recall", "val_recall"),
                ] + [
                    (f"cls{i}_iou", f"val_cls{i}_iou") for i in
                    filter(lambda i: i != args.ignore_index,
                           range(args.num_classes))
                ]),
            filename="checkpoint",
            on="validation_end"
        ),
        # pl.callbacks.LearningRateMonitor()
    ]

    if args.backbone_finetuning_epoch is not None:
        callbacks.append(Finetuning(unfreeze_at_epoch=args.backbone_finetuning_epoch))
    if args.swa_epoch_start:
        callbacks.append(
            pl.callbacks.StochasticWeightAveraging(swa_lrs=args.swa_lrs, swa_epoch_start=args.swa_epoch_start))

    tensorboard_logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".", default_hp_metric=False)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tensorboard_logger,
        callbacks=callbacks,
        enable_progress_bar=False
    )

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
    parser.add_argument("--name", type=str, default="",
                        help="Identifier used when creating files and directories for this training run.")
    parser.add_argument("--weights", type=str,
                        help="Path to pytorch weights to load as initial model weights")
    parser.add_argument("--drop_output_layer_weights", action="store_true", default=False,
                        help="Drop the output layer weights before restoring them. "
                             "Use for finetuning to different class outputs.")

    parser.add_argument("--swa_epoch_start", type=float,
                        help="The epoch at which to start the stochastic weight averaging procedure.")
    parser.add_argument("--swa_lrs", type=float, default=0.05,
                        help="The lr to start the annealing procedure for stochastic weight averaging.")

    parser.add_argument("--tune_trials", type=int, default=30,
                        help="Number of Ray Tune trials to run.")
    parser.add_argument("--init_lr", type=float, default=0.028147503791496848,
                        help="The initial LR to test with Ray Tune.")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="The lower limit of the range of LRs to optimize with Ray Tune.")
    parser.add_argument("--max_lr", type=float, default=0.1,
                        help="The upper limit of the range of LRs to optimize with Ray Tune.")
    parser.add_argument("--init_alpha", type=float, default=0.777442699607719,
                        help="The initial alpha (a FTLoss hyperparameter) to test with Ray Tune.")
    parser.add_argument("--min_alpha", type=float, default=0.1,
                        help="The lower limit of the range of alpha hyperparameters to optimize with Ray Tune.")
    parser.add_argument("--max_alpha", type=float, default=0.9,
                        help="The upper limit of the range of alpha hyperparameters to optimize with Ray Tune.")
    parser.add_argument("--init_weight_decay", type=float, default=2.7891888551808663e-06,
                        help="The initial weight decay to test with Ray Tune.")
    parser.add_argument("--min_weight_decay", type=float, default=1e-8,
                        help="The lower limit of the range of weight decay values to optimize with Ray Tune.")
    parser.add_argument("--max_weight_decay", type=float, default=1e-3,
                        help="The upper limit of the range of weight decay values to optimize with Ray Tune.")
    parser.add_argument("--test-only", action="store_true", help="Only run the test dataset")

    parser = KelpDataModule.add_argparse_args(parser)
    parser = LRASPPMobileNetV3Large.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(argv)

    # ------------
    # training
    # ------------
    config = {
        "alpha": tune.uniform(args.min_alpha, args.max_alpha),
        # "gamma": tune.choice([4.0 / 3.0]),
        "lr": tune.loguniform(args.min_lr, args.max_lr),
        "weight_decay": tune.loguniform(args.min_weight_decay, args.max_weight_decay),
    }
    scheduler = ASHAScheduler(
        max_t=args.max_epochs
    )
    reporter = CLIReporter(
        parameter_columns=["lr", "alpha", "weight_decay"],
        metric_columns=(["miou", "loss", "training_iteration"] + [f"cls{i}_iou" for i in
                                                                  filter(lambda i: i != args.ignore_index,
                                                                         range(args.num_classes))]),
    )

    train_fn_with_parameters = tune.with_parameters(train, args=args)

    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial={"cpu": 8, "gpu": 1},
        metric="miou",
        mode="max",
        config=config,
        search_alg=SkOptSearch(
            points_to_evaluate=[{'lr': args.init_lr, 'alpha': args.init_alpha, 'weight_decay': args.init_weight_decay}]),
        num_samples=args.tune_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=args.name,
        local_dir=args.checkpoint_dir,
    )

    print("Best hyperparameters found were: ", analysis.best_config)

    checkpoint_path = os.path.join(analysis.best_checkpoint.local_path, "checkpoint")
    checkpoint = torch.load(checkpoint_path)

    save_path_name = Path(args.checkpoint_dir).joinpath(f"best-miou={analysis.best_result['miou']:.4f}")
    torch.save(checkpoint, save_path_name.with_suffix(".ckpt"))
    torch.save(checkpoint['state_dict'], save_path_name.with_suffix(".pt"))
    analysis.best_result_df.to_csv(save_path_name.with_suffix(".csv"))

    # if args.test_only:
    #     with torch.no_grad():
    #         trainer.validate(model, datamodule=kelp_data)
    #         trainer.test(model, datamodule=kelp_data)
    # else:
    #     trainer.fit(model, datamodule=kelp_data)
    #     trainer.test(model, datamodule=kelp_data, ckpt_path="best")


if __name__ == "__main__":
    debug = os.getenv("DEBUG", False)
    if debug:
        cli_main([
            "/home/taylor/PycharmProjects/hakai-ml-train/data/kelp_species",
            "/home/taylor/PycharmProjects/hakai-ml-train/checkpoints/kelp_species",
            "--name=lr_aspp_kelp_species_DEV",
            "--num_classes=4",
            "--ignore_index=1",
            "--weights=/home/taylor/PycharmProjects/hakai-ml-train/checkpoints/kelp_pa/LRASPP/"
            "best-val_miou=0.8023-epoch=18-step=17593.pt",
            "--drop_output_layer_weights",
            "--batch_size=2",
            "--gradient_clip_val=0.5",
            # "--accelerator=gpu",
            "--accelerator=cpu",
            # "--backbone_finetuning_epoch=1",
            "--devices=auto",
            # "--strategy=ddp_find_unused_parameters_false",
            # "--sync_batchnorm",
            "--max_epochs=10",
            # '--limit_train_batches=10',
            # "--limit_val_batches=10",
            # "--limit_test_batches=10",
            "--log_every_n_steps=5",
        ])
    else:
        cli_main()
