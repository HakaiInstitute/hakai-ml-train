# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-23
# Description:
import argparse
import os
from argparse import ArgumentParser
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Optimizer
from torchvision.models import ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_resnet101

from base_model import BaseFinetuning, BaseModel
from kelp_data_module import KelpDataModule


class DeepLabv3ResNet101(BaseModel):
    def init_model(self):
        self.model = deeplabv3_resnet101(progress=True, weights_backbone=ResNet101_Weights.IMAGENET1K_V1,
                                         num_classes=self.num_classes, aux_loss=False)
        self.model.requires_grad_(True)
        self.model.backbone.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)["out"]

    def freeze_before_training(self, ft_module: BaseFinetuning) -> None:
        ft_module.freeze(self.model.backbone.layer3, train_bn=False)
        ft_module.freeze(self.model.backbone.layer4, train_bn=False)

    def finetune_function(self, ft_module: BaseFinetuning, epoch: int, optimizer: Optimizer, opt_idx: int) -> None:
        if epoch == ft_module.unfreeze_at_epoch:
            ft_module.unfreeze_and_add_param_group(
                self.model.backbone.layer3,
                optimizer,
                train_bn=ft_module.train_bn)
            ft_module.unfreeze_and_add_param_group(
                self.model.backbone.layer4,
                optimizer,
                train_bn=ft_module.train_bn)


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
        model = DeepLabv3ResNet101(
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
            model = DeepLabv3ResNet101.load_from_checkpoint(args.weights)

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
            pl.callbacks.EarlyStopping(monitor="val_miou", mode="max", patience=10),
            PyTorchLightningPruningCallback(trial, monitor='val_miou'),
        ]

        if args.backbone_finetuning_epoch is not None:
            callbacks.append(BaseFinetuning(unfreeze_at_epoch=args.backbone_finetuning_epoch))
        if args.swa_epoch_start:
            callbacks.append(
                pl.callbacks.StochasticWeightAveraging(swa_lrs=args.swa_lrs, swa_epoch_start=args.swa_epoch_start))

        logger = TensorBoardLogger(save_dir=args.checkpoint_dir, name=f'{args.name}/trial_{trial.number}',
                                   default_hp_metric=False)
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
    parser = DeepLabv3ResNet101.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(argv)

    # Make checkpoint directory
    Path(args.checkpoint_dir, args.name).mkdir(exist_ok=True, parents=True)

    objective = Objective(args)
    pruner: optuna.pruners.BasePruner = optuna.pruners.SuccessiveHalvingPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner,
                                storage=f'sqlite:///{args.checkpoint_dir}/{args.name}/hyper_opt.db')
    study.enqueue_trial({'lr': args.init_lr, 'alpha': args.init_alpha, 'weight_decay': args.init_weight_decay})
    study.optimize(objective, n_trials=args.tune_trials, gc_after_trial=True)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    debug = os.getenv("DEBUG", False)
    if debug:
        cli_main(
            [
                "/home/taylor/PycharmProjects/hakai-ml-train/data/kelp_pa_aco",
                "/home/taylor/PycharmProjects/hakai-ml-train/checkpoints/kelp_pa",
                "--name=DEEPLAB_DEV",
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
            ]
        )
    else:
        cli_main()
