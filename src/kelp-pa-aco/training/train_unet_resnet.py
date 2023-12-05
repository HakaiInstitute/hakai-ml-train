#!/usr/bin/env python
# coding: utf-8

import os
import shutil
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from wandb import AlertLevel

from config import PATrainingConfig, SPTrainingConfig
from datamodule import DataModule
from unetplusplus import UNetPlusPlus


def train(config: Union[PATrainingConfig, SPTrainingConfig]):
    pl.seed_everything(0, workers=True)

    # Get path to this notebook
    notebook_path = __file__ if "__file__" in globals() else os.path.abspath(os.getcwd())
    os.environ['WANDB_NOTEBOOK_NAME'] = notebook_path

    # Make checkpoint directory
    Path(config.checkpoint_dir, config.name).mkdir(exist_ok=True, parents=True)

    # Setup Callbacks and Trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/miou",
        mode="max",
        filename="{val/miou:.4f}_{epoch}",
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

    trainer = pl.Trainer(
        # overfit_batches=10,
        # log_every_n_steps=3,
        # limit_train_batches=3,
        # limit_val_batches=3,
        # accelerator='cpu',
        # fast_dev_run=True,
        deterministic=True,
        benchmark=True,
        max_epochs=config.max_epochs,
        precision=config.precision,
        logger=logger,
        gradient_clip_val=0.5,
        accumulate_grad_batches=8,
        callbacks=[
            checkpoint_callback,
            pl.callbacks.LearningRateMonitor(),
        ],
    )

    # # Load dataset
    data_module = DataModule(**dict(config))

    # Load model
    model = UNetPlusPlus(**dict(config))

    # Train
    torch.set_float32_matmul_precision("medium")

    try:
        trainer.fit(model, datamodule=data_module)

        if not trainer.fast_dev_run and config.enable_logging:
            best_miou = checkpoint_callback.best_model_score.detach().cpu()
            print("Best mIoU:", best_miou)
            wandb.alert(
                title="Training complete",
                text=f"Best mIoU: {best_miou}",
                level=AlertLevel.INFO,
            )

    finally:
        if config.enable_logging:
            wandb.finish()
        else:
            shutil.rmtree(logger.log_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=str, choices=["pa", "sp"])
    args = parser.parse_args()

    if args.model_type == "pa":
        train_config = PATrainingConfig()
    elif args.model_type == "sp":
        train_config = SPTrainingConfig
    else:
        raise ValueError("Invalid model type")

    train(train_config)
