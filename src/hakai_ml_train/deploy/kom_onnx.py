import argparse
import importlib
from pathlib import Path

import torch
import yaml
from torch.export import Dim


class ONNXModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def encoder_min_image_size(model: torch.nn.Module, default: int = 32) -> int:
    """Smallest tile the encoder can consume, i.e. its total downsampling factor.

    Below this, the deepest stage's feature map collapses to zero size and
    export fails. SMP encoders expose this as `output_stride`.
    """
    encoder = getattr(getattr(model, "model", model), "encoder", None)
    return int(getattr(encoder, "output_stride", default) or default)


def load_model_from_config(config_path: Path, ckpt_path: Path):
    """Instantiate the Lightning module described by a training config and load weights.

    Returns the eval-mode module plus the `model.init_args` dict, which callers use
    to recover input channels / image size.
    """
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

    # Unwrap any torch.compile'd submodules — torch.export (used internally by
    # torch.onnx.export) conflicts with compiled graphs.
    for name, submodule in list(model.named_children()):
        if hasattr(submodule, "_orig_mod"):
            setattr(model, name, submodule._orig_mod)

    model.eval()
    return model, init_args


def main(
    config_path: Path,
    ckpt_path: Path,
    output_path: Path,
    dynamic_spatial: bool = True,
    dynamo: bool = True,
    min_image_size: int | None = None,
) -> None:
    model, init_args = load_model_from_config(config_path, ckpt_path)

    if min_image_size is None:
        min_image_size = encoder_min_image_size(model)

    num_channels = init_args.get("model_opts", {}).get("in_channels", 3)
    image_size = init_args.get("image_size", 640)
    onnx_model = ONNXModel(model)
    onnx_model.eval()

    x = torch.rand(
        2,
        num_channels,
        image_size,
        image_size,
        device=torch.device("cpu"),
        requires_grad=False,
    )

    if dynamo:
        if dynamic_spatial:
            img_shape = (
                Dim("batch", min=1),
                Dim.STATIC,
                Dim("height", min=min_image_size),
                Dim("width", min=min_image_size),
            )
        else:
            img_shape = (Dim("batch", min=1), Dim.STATIC, Dim.STATIC, Dim.STATIC)
        extra_kwargs = dict(dynamic_shapes={"x": img_shape}, dynamo=True)
    else:
        img_dynamic_axes = {0: "batch_size"}
        if dynamic_spatial:
            img_dynamic_axes |= {2: "height", 3: "width"}
        extra_kwargs = dict(
            do_constant_folding=True,
            dynamic_axes={"input": img_dynamic_axes, "output": img_dynamic_axes},
            dynamo=False,
        )

    with torch.no_grad():
        torch.onnx.export(
            onnx_model,
            (x,),
            output_path,
            export_params=True,
            input_names=["input"],
            output_names=["output"],
            verbose=False,
            **extra_kwargs,
        )

    print(f"ONNX model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path, help="Path to config YAML file")
    parser.add_argument(
        "ckpt_path", type=Path, help="Path to PyTorch Lightning checkpoint"
    )
    parser.add_argument("output_path", type=Path, help="Path to save the ONNX model")
    parser.add_argument(
        "--no-dynamic-spatial",
        action="store_true",
        help="Fix spatial dimensions (height, width) in the exported model",
    )
    parser.add_argument(
        "--no-dynamo",
        action="store_true",
        help="Disable Dynamo and use TorchScript tracing export (default: False)",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=None,
        help=(
            "Minimum image size for spatial dimensions "
            "(default: the encoder's output_stride)"
        ),
    )

    args = parser.parse_args()

    main(
        args.config_path,
        args.ckpt_path,
        args.output_path,
        dynamic_spatial=not args.no_dynamic_spatial,
        dynamo=not args.no_dynamo,
        min_image_size=args.min_image_size,
    )
