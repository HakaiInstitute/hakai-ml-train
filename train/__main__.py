#!/usr/bin/env python
# coding: utf-8

import shutil
from pathlib import Path

import albumentations as A
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from wandb import AlertLevel

from .config import (
    TrainingConfig,
    kelp_pa_efficientnet_b4_config_rgbi,
    kelp_sp_efficientnet_b4_config_rgbi,
    kelp_sp_efficientnet_b4_config_rgb,
    kelp_pa_efficientnet_b4_config_rgb,
    seagrass_pa_efficientnet_b5_config_rgb,
    mussels_pa_efficientnet_b4_config_rgb,
)
from .datamodule import DataModule
from .model import SegmentationModel
from .transforms import get_test_transforms, get_train_transforms


def train(config: TrainingConfig):
    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("medium")

    # Make checkpoint directory
    Path(config.checkpoint_dir, config.name).mkdir(exist_ok=True, parents=True)

    # Setup Callbacks and Trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/dice_epoch",
        mode="max",
        filename="{val/dice_epoch:.4f}_{epoch}",
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=False,
        every_n_epochs=1,
        verbose=False,
    )

    if config.enable_logging:
        logger = WandbLogger(
            name=config.name,
            project=config.project_name,
            save_dir=config.checkpoint_dir,
            log_model=True,
        )
        logger.experiment.config["batch_size"] = config.batch_size
    else:
        logger = pl.loggers.CSVLogger(save_dir="/tmp/")

    def compute_amount(epoch):
        # the sum of all returned values need to be smaller than 1
        if epoch == 1:
            return 0.5
        elif epoch == 5:
            return 0.25
        elif 7 < epoch < 9:
            return 0.01

    trainer = pl.Trainer(
        # overfit_batches=10,
        # log_every_n_steps=3,
        # limit_train_batches=3,
        # limit_val_batches=3,
        # accelerator='cpu',
        # fast_dev_run=True,
        deterministic=config.deterministic,
        benchmark=config.benchmark,
        max_epochs=config.max_epochs,
        precision=config.precision,
        logger=logger,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=[
            checkpoint_callback,
            pl.callbacks.LearningRateMonitor(),
            # pl.callbacks.ModelPruning("l1_unstructured", amount=compute_amount),
        ],
    )

    # Load data augmentation
    train_trans = get_train_transforms(config.tile_size, config.extra_transforms)
    test_trans = get_test_transforms(config.tile_size, config.extra_transforms)

    # Save to WandB
    if config.enable_logging:
        A.save(train_trans, "./train_transforms.json")
        A.save(test_trans, "./test_transforms.json")

        artifact = wandb.Artifact(name=f"{wandb.run.id}-transforms", type="config")
        artifact.add_file("./train_transforms.json")
        artifact.add_file("./test_transforms.json")

        wandb.run.log_artifact(artifact)

    # Load dataset
    data_module = DataModule(
        train_transforms=train_trans, tests_transforms=test_trans, **dict(config)
    )

    # Load model
    model = SegmentationModel(**dict(config))

    # Train
    if config.enable_logging:
        wandb.run.config.update(dict(config))
        # wandb.run.config.update(
        #     {
        #         "train_transforms": train_trans.to_dict(),
        #         "test_transforms": test_trans.to_dict(),
        #     }
        # )
        wandb.run.tags += tuple(config.tags)

    try:
        trainer.fit(model, datamodule=data_module)

        if not trainer.fast_dev_run and config.enable_logging:
            best_dice = checkpoint_callback.best_model_score.detach().cpu()
            print("Best dice:", best_dice)
            wandb.alert(
                title="Training complete",
                text=f"Best dice: {best_dice}",
                level=AlertLevel.INFO,
            )

    finally:
        if config.enable_logging:
            wandb.finish()
        else:
            shutil.rmtree(logger.log_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=str, choices=["kelp", "seagrass", "mussels"])
    parser.add_argument("objective", type=str, choices=["pa", "sp"])
    parser.add_argument("--bands", type=int, default=3)
    args = parser.parse_args()

    train_config = None
    if args.model_type == "kelp":
        if args.objective == "pa":
            if args.bands == 3:
                train_config = kelp_pa_efficientnet_b4_config_rgb
            elif args.bands == 4:
                train_config = kelp_pa_efficientnet_b4_config_rgbi

        elif args.objective == "sp":
            if args.bands == 3:
                train_config = kelp_sp_efficientnet_b4_config_rgb
            elif args.bands == 4:
                train_config = kelp_sp_efficientnet_b4_config_rgbi

    elif args.model_type == "seagrass":
        if args.bands == 3 and args.objective == "pa":
            train_config = seagrass_pa_efficientnet_b5_config_rgb

    elif args.model_type == "mussels":
        if args.bands == 3 and args.objective == "pa":
            train_config = mussels_pa_efficientnet_b4_config_rgb

    if train_config is None:
        raise ValueError("Invalid model_type or band_type")

    train(train_config)
