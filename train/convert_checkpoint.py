import os.path
from collections import OrderedDict
from pathlib import Path

import torch
import wandb

from .configs.config import load_yml_config, Config
from .model import SMPSegmentationModel

DEVICE = torch.device("cpu")


def convert_checkpoint(
    checkpoint_url: str,
    config: Config,
    model_name: str = "UNetPlusPlus_EfficientNetB4",
    model_task: str = "kelp_presence_rgb",
):
    # Download .ckpt file from W&B
    api = wandb.Api()
    artifact = api.artifact(checkpoint_url, type="model")
    dice = round(artifact.metadata["score"], 4)
    artifact_dir = artifact.download()
    ckpt_file = Path(artifact_dir) / "model.ckpt"

    # Set output paths
    output_path_jit = (
        f"../inference/weights/{model_name}_{model_task}_jit_dice={dice:.4f}.pt"
    )
    output_path_onnx = (
        f"../inference/weights/{model_name}_{model_task}_dice={dice:.4f}.onnx"
    )

    # Remove ._orig_mod from state dict
    ckpt = torch.load(ckpt_file)
    state_dict = ckpt["state_dict"]
    state_dict = OrderedDict([(k.replace("._orig_mod", ""), v) for k, v in state_dict.items()])
    ckpt["state_dict"] = state_dict

    ckpt_file_clean = Path(ckpt_file).with_stem(Path(ckpt_file).stem + "_clean")
    torch.save(ckpt, ckpt_file_clean)

    # Load stripped back model
    model = SMPSegmentationModel.load_from_checkpoint(
        ckpt_file_clean, **config.segmentation_config.dict()
    )



    # Have to deactivate SWISH for EfficientNet to export to TorchScript
    # model.model.encoder.set_swish(False)

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
    parser.add_argument("checkpoint_url", type=str)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model_name", type=str, default="UNetPlusPlus_EfficientNetB4")
    parser.add_argument("--task", type=str, default="kelp_presence_rgb")
    args = parser.parse_args()

    config = load_yml_config(args.config)

    convert_checkpoint(args.checkpoint_url, config, args.model_name, args.task)
