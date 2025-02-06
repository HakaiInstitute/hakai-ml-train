#!/usr/bin/env python

from pathlib import Path

import albumentations as A
import lightning.pytorch as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from wandb import AlertLevel

from . import model as models
from .configs.config import Config, load_yml_config
from .datamodule import DataModule
from .transforms import extra_transforms, get_test_transforms, get_train_transforms


def train(config: Config):
    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("medium")

    # Make checkpoint directory
    Path(config.checkpoint.dirpath, config.logging.project).mkdir(
        exist_ok=True, parents=True
    )

    # Setup Callbacks and Trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**config.checkpoint.dict())

    if config.enable_logging:
        logger = WandbLogger(**config.logging.dict())
        logger.experiment.config["batch_size"] = config.data_module.batch_size
    else:
        logger = pl.loggers.CSVLogger(save_dir="/tmp/")

    # def compute_amount(epoch):
    #     # the sum of all returned values need to be smaller than 1
    #     if epoch == 2:
    #         return 0.5
    #     elif epoch == 10:
    #         return 0.25
    #     elif 14 < epoch < 18:
    #         return 0.01

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            checkpoint_callback,
            pl.callbacks.LearningRateMonitor(),
        ],
        **config.trainer.dict(),
        # overfit_batches=10,
        # log_every_n_steps=3,
        # limit_train_batches=3,
        # limit_val_batches=3,
        # accelerator='cpu',
        # fast_dev_run=True,
    )

    # Load data augmentation
    extra_trans = []
    if config.extra_transforms is not None:
        for k in config.extra_transforms:
            extra_trans.append(extra_transforms[k])

    train_trans = get_train_transforms(config.data_module.tile_size, extra_trans)
    test_trans = get_test_transforms(config.data_module.tile_size, extra_trans)

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
        train_transforms=train_trans,
        test_transforms=test_trans,
        **config.data_module.dict(),
    )

    # Load model
    model_cls = models.__dict__[config.segmentation_model_cls]
    model = model_cls(**config.segmentation_config.dict())

    if config.segmentation_config.freeze_encoder:
        model.model.encoder.model.requires_grad_(False)

    # Train
    if config.enable_logging:
        wandb.run.config.update(config.dict())
        wandb.run.config.update(
            {
                "train_transforms": train_trans.to_dict(),
                "test_transforms": test_trans.to_dict(),
            }
        )
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
        # else:
        #     shutil.rmtree(logger.log_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = load_yml_config(args.config)
    train(config)
