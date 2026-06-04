import argparse
import importlib
from pathlib import Path

import torch
import yaml


def main(config_path: Path, ckpt_path: Path, output_path: Path) -> None:
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
    torchscript_model = torch.jit.script(model.model)

    torch.jit.save(torchscript_model, output_path)
    print(f"TorchScript model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path, help="Path to config YAML file")
    parser.add_argument(
        "ckpt_path", type=Path, help="Path to PyTorch Lightning checkpoint"
    )
    parser.add_argument("output_path", type=Path, help="Path to save the ONNX model")

    args = parser.parse_args()

    main(args.config_path, args.ckpt_path, args.output_path)
