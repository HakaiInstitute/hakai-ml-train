#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import lightning.pytorch as pl
import torch
import wandb

from .config import (
    TrainingConfig,
    kelp_pa_efficientnet_b4_config_rgbi,
    kelp_sp_efficientnet_b4_config_rgbi,
    kelp_sp_efficientnet_b4_config_rgb,
    kelp_pa_efficientnet_b4_config_rgb,
)
from .datamodule import DataModule
from .model import SMPSegmentationModel


def test(config: TrainingConfig, checkpoint_url: str):
    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("medium")

    # Download .ckpt file from W&B
    api = wandb.Api()
    artifact = api.artifact(checkpoint_url, type="model")
    artifact_dir = artifact.download()
    ckpt_file = Path(artifact_dir) / "model.ckpt"

    trainer = pl.Trainer(
        deterministic=config.deterministic,
        benchmark=config.benchmark,
        max_epochs=config.max_epochs,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
    )

    # Load dataset
    data_module = DataModule(**dict(config))

    # Load model
    model = SMPSegmentationModel(**dict(config))

    results = trainer.test(model, datamodule=data_module, ckpt_path=str(ckpt_file))

    print(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=str, choices=["pa", "sp"])
    parser.add_argument("band_type", type=str, choices=["rgb", "rgbi"])
    parser.add_argument("checkpoint_url", type=str)
    args = parser.parse_args()

    train_config = None
    if args.band_type == "rgb":
        if args.model_type == "pa":
            train_config = kelp_pa_efficientnet_b4_config_rgb
        elif args.model_type == "sp":
            train_config = kelp_sp_efficientnet_b4_config_rgb
    elif args.band_type == "rgbi":
        if args.model_type == "pa":
            train_config = kelp_pa_efficientnet_b4_config_rgbi
        elif args.model_type == "sp":
            train_config = kelp_sp_efficientnet_b4_config_rgbi

    if train_config is None:
        raise ValueError("Invalid model_type or band_type")

    test(train_config, checkpoint_url=args.checkpoint_url)
