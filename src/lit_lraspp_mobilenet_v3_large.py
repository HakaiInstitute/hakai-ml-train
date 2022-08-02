# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2021-05-17
# Description:
import argparse
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Optional

import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy, JaccardIndex, Precision, Recall
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from kelp_data_module import KelpDataModule
from utils.loss import FocalTverskyLoss


class LRASPPMobileNetV3Large(pl.LightningModule):
    def __init__(self, num_classes: int = 2, ignore_index: Optional[int] = None, lr: float = 0.35,
                 weight_decay: float = 0, loss_alpha: float = 0.7, loss_gamma: float = 4.0 / 3.0, max_epochs: int = 100):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

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
        pred = torch.vstack([self.forward(batch).unsqueeze(0) for _ in range(mc_iteration)]).mean(dim=0)
        return pred

    @property
    def estimated_stepping_batches(self) -> int:
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
        return (batches // effective_accum) * self.max_epochs

    def configure_optimizers(self):
        """Init optimizer and scheduler"""
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                    lr=self.lr, weight_decay=self.weight_decay, nesterov=True, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                                           total_steps=self.estimated_stepping_batches)
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


# Objective function to be maximized by Optuna
class Objective(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def __call__(self, trial: optuna.trial.Trial):
        args = self.args

        # ------------
        # data
        # ------------
        kelp_data = KelpDataModule(
            args.data_dir,
            # num_workers=0,
            # pin_memory=False,
            num_classes=args.num_classes,
            batch_size=args.batch_size
        )

        # ------------
        # hyperparameter search space
        # ------------
        lr = trial.suggest_float('lr', args.min_lr, args.max_lr, log=True)
        alpha = trial.suggest_float('alpha', args.min_alpha, args.max_alpha)
        weight_decay = trial.suggest_float('weight_decay', args.min_weight_decay, args.max_weight_decay)

        # ------------
        # model
        # ------------
        model = LRASPPMobileNetV3Large(
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            lr=lr,
            loss_alpha=alpha,
            weight_decay=weight_decay,
            max_epochs=args.max_epochs,
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
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            # verbose=True,
            monitor="val_miou", mode="max",
            filename="best-{val_miou:.4f}-{epoch}",
            save_top_k=1, save_last=True,
            save_on_train_epoch_end=False,
            every_n_epochs=1,
        )
        callbacks = [
            checkpoint_callback,
            pl.callbacks.LearningRateMonitor(),
            PyTorchLightningPruningCallback(trial, monitor='val_miou'),
        ]

        if args.backbone_finetuning_epoch is not None:
            callbacks.append(Finetuning(unfreeze_at_epoch=args.backbone_finetuning_epoch))
        if args.swa_epoch_start:
            callbacks.append(
                pl.callbacks.StochasticWeightAveraging(swa_lrs=args.swa_lrs, swa_epoch_start=args.swa_epoch_start))

        logger = TensorBoardLogger(save_dir=args.checkpoint_dir, name=f'{args.name}/trial_{trial.number}', default_hp_metric=False)
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=callbacks,
        )
        trainer.logger.log_hyperparams({
            'lr': lr,
            'alpha': alpha,
            'weight_decay': weight_decay,
            'batch_size': args.batch_size,
        })
        trainer.fit(model, datamodule=kelp_data)

        return checkpoint_callback.best_model_score.detach().cpu()


def cli_main(argv=None):
    pl.seed_everything(0)
    # torch.multiprocessing.set_sharing_strategy('file_system')

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
    parser.add_argument("--init_lr", type=float, default=0.03,
                        help="The initial LR to test with Ray Tune.")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="The lower limit of the range of LRs to optimize with Ray Tune.")
    parser.add_argument("--max_lr", type=float, default=0.1,
                        help="The upper limit of the range of LRs to optimize with Ray Tune.")
    parser.add_argument("--init_alpha", type=float, default=0.4,
                        help="The initial alpha (a FTLoss hyperparameter) to test with Ray Tune.")
    parser.add_argument("--min_alpha", type=float, default=0.1,
                        help="The lower limit of the range of alpha hyperparameters to optimize with Ray Tune.")
    parser.add_argument("--max_alpha", type=float, default=0.9,
                        help="The upper limit of the range of alpha hyperparameters to optimize with Ray Tune.")
    parser.add_argument("--init_weight_decay", type=float, default=0,
                        help="The initial weight decay to test with Ray Tune.")
    parser.add_argument("--min_weight_decay", type=float, default=0,
                        help="The lower limit of the range of weight decay values to optimize with Ray Tune.")
    parser.add_argument("--max_weight_decay", type=float, default=1e-3,
                        help="The upper limit of the range of weight decay values to optimize with Ray Tune.")
    parser.add_argument("--test-only", action="store_true", help="Only run the test dataset")

    parser = KelpDataModule.add_argparse_args(parser)
    parser = LRASPPMobileNetV3Large.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(argv)

    # Make checkpoint directory
    Path(args.checkpoint_dir, args.name).mkdir(exist_ok=True, parents=True)

    objective = Objective(args)
    pruner: optuna.pruners.BasePruner = optuna.pruners.SuccessiveHalvingPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner, storage=f'sqlite:///{args.checkpoint_dir}/{args.name}/hyper_opt.db')
    study.enqueue_trial({'lr': args.init_lr, 'alpha': args.init_alpha, 'weight_decay': args.init_weight_decay})
    study.optimize(objective, n_trials=args.tune_trials, gc_after_trial=True)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    # print("Best hyperparameters found were: ", analysis.best_config)

    # checkpoint_path = os.path.join(analysis.best_checkpoint.local_path, "checkpoint")
    # checkpoint = torch.load(checkpoint_path)

    # save_path_name = Path(args.checkpoint_dir).joinpath(f"best-miou={analysis.best_result['miou']:.4f}")
    # torch.save(checkpoint, save_path_name.with_suffix(".ckpt"))
    # torch.save(checkpoint['state_dict'], save_path_name.with_suffix(".pt"))
    # analysis.best_result_df.to_csv(save_path_name.with_suffix(".csv"))

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
            "/home/taylor/PycharmProjects/hakai-ml-train/data/kelp_pa_aco",
            "/home/taylor/PycharmProjects/hakai-ml-train/checkpoints/kelp_pa",
            "--name=LRASPP_ACO_DEV",
            "--num_classes=2",
            "--batch_size=2",
            "--gradient_clip_val=0.5",
            "--accelerator=gpu",
            # "--accelerator=cpu",
            "--backbone_finetuning_epoch=100",
            "--devices=auto",
            # "--strategy=ddp_find_unused_parameters_false",
            # "--sync_batchnorm",
            "--max_epochs=3",
            '--limit_train_batches=10',
            "--limit_val_batches=10",
            "--limit_test_batches=10",
            "--log_every_n_steps=5",
        ])
    else:
        cli_main()
