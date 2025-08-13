import argparse
import importlib
from pathlib import Path

import torch
import yaml


class ONNXModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

        # if logits.shape[1] == 1:
        #     probs = torch.sigmoid(logits)
        #     # If the model outputs a single channel, convert to two channels
        #     probs = torch.cat([1 - probs, probs], dim=1)
        #     print(probs.shape)
        #     return probs
        # else:
        #     return torch.softmax(logits, dim=1)


def main(
    config_path: Path, ckpt_path: Path, output_path: Path, opset: int = 11
) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    class_path = model_config["class_path"]
    init_args = model_config["init_args"]

    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    model = model_class(**init_args)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    model = ONNXModel(model.model)

    x = torch.rand(
        1,
        3,
        224,
        224,
        device=torch.device("cpu"),
        requires_grad=False,
    )

    # Define dynamic axes for input and output
    dynamic_axes = {
        "input": {
            0: "batch_size",
            2: "tile_size",
            3: "tile_size",
        },  # Dynamic batch size, height and width
        "output": {
            0: "batch_size",
            2: "tile_size",
            3: "tile_size",
        },  # Dynamic batch size, height and width
    }
    input_names = ["input"]
    output_names = ["output"]

    # Export the segmentation model to ONNX format
    torch.onnx.export(
        model,  # Model to export
        x,  # Example input
        output_path,  # Output file path
        export_params=True,  # Store model weights in the model file
        opset_version=opset,  # ONNX opset version
        do_constant_folding=True,  # Optimize constants
        input_names=input_names,  # Input tensor names
        output_names=output_names,  # Output tensor names
        dynamic_axes=dynamic_axes,  # Dynamic axes specification
        verbose=False,
    )

    print(f"ONNX model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path, help="Path to config YAML file")
    parser.add_argument(
        "ckpt_path", type=Path, help="Path to PyTorch Lightning checkpoint"
    )
    parser.add_argument("output_path", type=Path, help="Path to save the ONNX model")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")

    args = parser.parse_args()

    main(args.config_path, args.ckpt_path, args.output_path, args.opset)
