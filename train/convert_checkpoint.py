import json
import os.path
from collections import OrderedDict
from pathlib import Path

import onnx
import torch
import wandb

from .configs.config import Config, load_yml_config
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
    ckpt = torch.load(ckpt_file, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["state_dict"]
    state_dict = OrderedDict(
        [(k.replace("._orig_mod", ""), v) for k, v in state_dict.items()]
    )
    ckpt["state_dict"] = state_dict

    ckpt_file_clean = Path(ckpt_file).with_stem(Path(ckpt_file).stem + "_clean")
    torch.save(ckpt, ckpt_file_clean)

    # Load stripped back model
    model = SMPSegmentationModel.load_from_checkpoint(
        ckpt_file_clean, **config.segmentation_config.model_dump()
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

    # Define dynamic axes for input and output
    dynamic_axes = {
        "input": {
            0: "batch_size",
            # 2: "height",
            # 3: "width",
        },  # Dynamic batch size, height and width
        "output": {
            0: "batch_size",
            # 2: "height",
            # 3: "width",
        },  # Dynamic batch size, height and width
    }
    input_names = ["input"]
    output_names = ["output"]

    class DeepnessModel(torch.nn.Module):
        def __init__(self, model):
            super(DeepnessModel, self).__init__()
            self.model = model

        def forward(self, x):
            logits = self.model(x)
            probs = torch.sigmoid(logits)  # Shape [batch_size, 1, height, width]

            # Convert class 1 probabilities to 2 class probs shape: # [batch_size, 2, height, width]
            probs = torch.cat(
                [
                    1 - probs,  # Background class
                    probs,  # Kelp class
                ],
                dim=1,
            )

            return probs

    model = DeepnessModel(model.model)

    torch.onnx.export(
        model,  # Model to export
        x,  # Example input
        output_path_onnx,  # Output file path
        export_params=True,  # Store model weights in the model file
        opset_version=11,  # ONNX opset version
        do_constant_folding=True,  # Optimize constants
        input_names=input_names,  # Input tensor names
        output_names=output_names,  # Output tensor names
        dynamic_axes=dynamic_axes,  # Dynamic axes specification
        verbose=False,
    )
    onnx_model = onnx.load(output_path_onnx)

    class_names = {
        0: "_background",
        1: "mussels",
    }

    m1 = onnx_model.metadata_props.add()
    m1.key = "model_type"
    m1.value = json.dumps("Segmentor")

    m2 = onnx_model.metadata_props.add()
    m2.key = "class_names"
    m2.value = json.dumps(class_names)

    m3 = onnx_model.metadata_props.add()
    m3.key = "resolution"
    m3.value = json.dumps(0.5)  # cm/px

    m4 = onnx_model.metadata_props.add()
    m4.key = "tiles_overlap"
    m4.value = json.dumps(40)  # 40% overlap

    m5 = onnx_model.metadata_props.add()
    m5.key = "standardization_mean"
    m5.value = json.dumps([0.485, 0.456, 0.406])

    m6 = onnx_model.metadata_props.add()
    m6.key = "standardization_std"
    m6.value = json.dumps([0.229, 0.224, 0.225])

    onnx.save(onnx_model, output_path_onnx)
    onnx.checker.check_model(onnx_model)

    print(
        f"Saved ONNX model with dynamic dimensions to {os.path.abspath(output_path_onnx)}"
    )


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
