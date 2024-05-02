import os.path
from pathlib import Path

import torch
import wandb

from .config import (
    seagrass_pa_efficientnet_b5_config_rgb,
    kelp_pa_efficientnet_b4_config_rgbi,
    kelp_sp_efficientnet_b4_config_rgbi,
    TrainingConfig,
    kelp_pa_efficientnet_b4_config_rgb,
    kelp_sp_efficientnet_b4_config_rgb,
    mussels_pa_efficientnet_b4_config_rgb,
)
from .model import SegmentationModel

DEVICE = torch.device("cpu")


def convert_checkpoint(checkpoint_url: str, config: TrainingConfig):
    # Download .ckpt file from W&B
    api = wandb.Api()
    artifact = api.artifact(checkpoint_url, type="model")
    dice = round(artifact.metadata["score"], 4)
    artifact_dir = artifact.download()
    ckpt_file = Path(artifact_dir) / "model.ckpt"

    # Set output paths
    output_path_jit = (
        f"../inference/weights/"
        f"UNetPlusPlus_EfficientNetB5_seagrass_presence_rgb_jit_dice={dice:.4f}.pt"
    )
    output_path_onnx = (
        f"../inference/weights/"
        f"UNetPlusPlus_EfficientNetB5_seagrass_presence_rgb_dice={dice:.4f}.onnx"
    )

    # Load stripped back model
    model = SegmentationModel.load_from_checkpoint(ckpt_file, **dict(config))
    # Have to deactivate SWISH for EfficientNet to export to TorchScript
    model.model.encoder.set_swish(False)

    # Export as JIT
    x = torch.rand(
        1,
        config.num_bands,
        config.tile_size,
        config.tile_size,
        device=DEVICE,
        requires_grad=False,
    )

    # save for use in production environment
    traced_model = model.to_torchscript(method="trace", example_inputs=x)
    torch.jit.save(traced_model, output_path_jit)
    print(f"Saved JIT model to {os.path.abspath(output_path_jit)}")

    # Export as ONNX
    model.to_onnx(output_path_onnx, x, export_params=True)
    print(f"Saved ONNX model to {os.path.abspath(output_path_onnx)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(("checkpoint_url"), type=str)
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

    convert_checkpoint(args.checkpoint_url, train_config)
