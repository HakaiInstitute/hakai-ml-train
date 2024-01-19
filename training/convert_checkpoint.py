import os.path
from pathlib import Path

import torch
import wandb

from config import kelp_pa_efficientnet_b4_config_rgbi, kelp_sp_efficientnet_b4_config_rgbi, TrainingConfig
from unetplusplus import UNetPlusPlus

DEVICE = torch.device("cpu")


def convert_checkpoint(checkpoint_url: str, config: TrainingConfig):
    # Download .ckpt file from W&B
    api = wandb.Api()
    artifact = api.artifact(checkpoint_url, type="model")
    miou = round(artifact.metadata["score"], 4)
    artifact_dir = artifact.download()
    ckpt_file = Path(artifact_dir) / "model.ckpt"

    # Set output paths
    output_path_jit = f"../inference/weights/UNetPlusPlus_EfficientNetB4_kelp_species_rgbi_jit_miou={miou:.4f}.pt"
    output_path_onnx = f"../inference/weights/UNetPlusPlus_EfficientNetB4_kelp_species_rgbi_miou={miou:.4f}.onnx"

    # Load stripped back model
    model = UNetPlusPlus.load_from_checkpoint(ckpt_file, **dict(config))
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
    parser.add_argument("model_type", type=str, choices=["pa", "sp"])
    args = parser.parse_args()

    train_config = {"pa": kelp_pa_efficientnet_b4_config_rgbi, "sp": kelp_sp_efficientnet_b4_config_rgbi}[
        args.model_type
    ]
    convert_checkpoint(args.checkpoint_url, train_config)
