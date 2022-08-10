# Created by: Taylor Denouden
# Organization: Hakai Institute
import argparse
import os
from argparse import ArgumentParser
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import TensorBoardLogger

from kelp_data_module import KelpDataModule
from models.base_model import Finetuning
from models.lit_deeplabv3_resnet101 import DeepLabV3ResNet101
from models.lit_lraspp_mobilenet_v3_large import LRASPPMobileNetV3Large
from models.lit_unet import UnetEfficientnet
from utils.git_hash import get_git_revision_hash


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
            pin_memory=True,
            persistent_workers=True,
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
        if args.model == "unet":
            model = UnetEfficientnet(
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                lr=lr,
                loss_alpha=alpha,
                weight_decay=weight_decay,
                max_epochs=args.max_epochs,
            )
        elif args.model == "deeplab":
            model = DeepLabV3ResNet101(
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                lr=lr,
                loss_alpha=alpha,
                weight_decay=weight_decay,
                max_epochs=args.max_epochs,
            )
        elif args.model == "lraspp":
            model = LRASPPMobileNetV3Large(
                num_classes=args.num_classes,
                ignore_index=args.ignore_index,
                lr=lr,
                loss_alpha=alpha,
                weight_decay=weight_decay,
                max_epochs=args.max_epochs,
            )
        else:
            raise ValueError(f"No model for {args.model}")

        if args.weights and Path(args.weights).suffix == ".pt":
            print("Loading state_dict:", args.weights)
            weights = torch.load(args.weights)

            # Remove trained weights for previous classifier output layers
            if args.drop_output_layer_weights:
                weights = model.drop_output_layer_weights(weights)
            model.load_state_dict(weights, strict=False)

        elif args.weights and Path(args.weights).suffix == ".ckpt":
            print("Loading checkpoint:", args.weights)
            model = model.load_from_checkpoint(args.weights)

        # ------------
        # callbacks
        # ------------
        checkpoint_options = {
            # "verbose": True,
            "monitor": "val_miou",
            "mode": "max",
            "filename": "{val_miou:.4f}_{epoch}",
            "save_top_k": 1,
            "save_last": True,
            "save_on_train_epoch_end": False,
            "every_n_epochs": 1,
        }
        checkpoint_callback = pl.callbacks.ModelCheckpoint(**checkpoint_options, verbose=False, )
        checkpoint_weights_callback = pl.callbacks.ModelCheckpoint(**checkpoint_options, save_weights_only=True)
        checkpoint_weights_callback.FILE_EXTENSION = ".pt"

        callbacks = [
            checkpoint_callback,
            checkpoint_weights_callback,
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.EarlyStopping(monitor="val_miou", mode="max", patience=10),
            PyTorchLightningPruningCallback(trial, monitor='val_miou'),
        ]

        if args.backbone_finetuning_epoch is not None:
            callbacks.append(Finetuning(unfreeze_at_epoch=args.backbone_finetuning_epoch))
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
            'sha': get_git_revision_hash(),
        })
        trainer.fit(model, datamodule=kelp_data)

        return checkpoint_callback.best_model_score.detach().cpu()


def cli_main(argv=None):
    pl.seed_everything(0)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("model", type=str, choices=["unet", "lraspp", "deeplab"],
                        help="The name of the model to train")
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

    parser.add_argument("--num_classes", type=int, default=2,
                        help="The number of image classes, including background.")
    parser.add_argument("--ignore_index", type=int, default=None,
                        help="Label of any class to ignore.")
    parser.add_argument("--backbone_finetuning_epoch", type=int, default=None,
                        help="Set a value to unlock the epoch that the backbone network should be unfrozen."
                             "Leave as None to train all layers from the start.")

    parser.add_argument("--swa_epoch_start", type=float,
                        help="The epoch at which to start the stochastic weight averaging procedure.")
    parser.add_argument("--swa_lrs", type=float, default=0.05,
                        help="The lr to start the annealing procedure for stochastic weight averaging.")

    parser.add_argument("--tune_trials", type=int, default=30,
                        help="Number of Tune trials to run.")
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
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(argv)

    # Make checkpoint directory
    Path(args.checkpoint_dir, args.name).mkdir(exist_ok=True, parents=True)

    objective = Objective(args)
    pruner: optuna.pruners.BasePruner = optuna.pruners.SuccessiveHalvingPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.enqueue_trial({'lr': args.init_lr, 'alpha': args.init_alpha, 'weight_decay': args.init_weight_decay})
    study.optimize(objective, n_trials=args.tune_trials)

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
        cli_main([
            "lraspp",
            "/home/taylor/PycharmProjects/hakai-ml-train/data/kelp_pa/July2022",
            "/home/taylor/PycharmProjects/hakai-ml-train/checkpoints/kelp_pa",
            "--name=LRASPP_DEV",
            "--num_classes=2",
            "--batch_size=2",
            "--gradient_clip_val=0.5",
            "--accelerator=gpu",
            "--devices=auto",
            "--max_epochs=10",
            '--limit_train_batches=10',
            "--limit_val_batches=10",
            "--limit_test_batches=10",
            "--log_every_n_steps=5",
            # "--backbone_finetuning_epoch=10",
        ])
    else:
        cli_main()
