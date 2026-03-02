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


def main(
    config_path: Path,
    ckpt_path: Path,
    output_path: Path,
    opset: int = 17,
    dynamic_spatial: bool = True,
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

    # Unwrap any torch.compile'd submodules — torch.export (used internally by
    # torch.onnx.export) conflicts with compiled graphs.
    for name, submodule in list(model.named_children()):
        if hasattr(submodule, "_orig_mod"):
            setattr(model, name, submodule._orig_mod)

    num_channels = init_args.get("model_opts", {}).get("in_channels", 3)
    image_size = init_args.get("image_size", 640)
    onnx_model = ONNXModel(model)

    x = torch.rand(
        1,
        num_channels,
        image_size,
        image_size,
        device=torch.device("cpu"),
        requires_grad=False,
    )

    if dynamic_spatial:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }
    else:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        onnx_model,
        x,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=False,
        dynamo=False,
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
    parser.add_argument(
        "--no-dynamic-spatial",
        action="store_true",
        help="Fix spatial dimensions (height, width) in the exported model",
    )

    args = parser.parse_args()

    main(
        args.config_path,
        args.ckpt_path,
        args.output_path,
        args.opset,
        dynamic_spatial=not args.no_dynamic_spatial,
    )
