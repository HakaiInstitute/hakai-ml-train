import argparse
import re
from pathlib import Path

import segmentation_models_pytorch as smp
import torch


class ONNXModel(torch.nn.Module):
    pa_params_file = "artifacts/model-8a1r0k6h:v0/model.ckpt"
    sp_params_file = "artifacts/model-x7m8097u:v0/model.ckpt"

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

        pa_params = torch.load(
            self.pa_params_file, map_location=self.device, weights_only=False
        )["state_dict"]
        # Remove 'model.' prefix from start of keys
        pa_params = {re.sub(r"^model\.", "", k): v for k, v in pa_params.items()}

        self.pa_model = smp.UnetPlusPlus(
            encoder_name="tu-tf_efficientnetv2_m",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        self.pa_model.load_state_dict(pa_params, strict=True)

        sp_params = torch.load(
            self.sp_params_file, map_location=self.device, weights_only=False
        )["state_dict"]
        # Remove 'model.' prefix from start of keys
        sp_params = {re.sub(r"^model\.", "", k): v for k, v in sp_params.items()}
        self.sp_model = smp.UnetPlusPlus(
            encoder_name="tu-tf_efficientnetv2_m",
            encoder_weights=None,
            in_channels=3,
            classes=2,
            activation=None,
        )
        self.sp_model.load_state_dict(sp_params, strict=True)

    def forward(self, x):
        presence_logits = self.pa_model(x)
        species_logits = self.sp_model(x)

        return torch.cat([presence_logits, species_logits], dim=1)

        # if logits.shape[1] == 1:
        #     probs = torch.sigmoid(logits)
        #     # If the model outputs a single channel, convert to two channels
        #     probs = torch.cat([1 - probs, probs], dim=1)
        #     print(probs.shape)
        #     return probs
        # else:
        #     return torch.softmax(logits, dim=1)


def main(
    # config_path: Path,
    # ckpt_path: Path,
    output_path: Path,
    opset: int = 11,
) -> None:
    # with open(config_path, "r") as f:
    #     config = yaml.safe_load(f)
    #
    # model_config = config["model"]
    # class_path = model_config["class_path"]
    # init_args = model_config["init_args"]
    #
    # module_name, class_name = class_path.rsplit(".", 1)
    # module = importlib.import_module(module_name)
    # model_class = getattr(module, class_name)
    #
    # model = model_class(**init_args)
    #
    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    # model.load_state_dict(checkpoint["state_dict"])
    #
    # model.eval()
    model = ONNXModel()

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
    # parser.add_argument("config_path", type=Path, help="Path to config YAML file")
    # parser.add_argument(
    #     "ckpt_path", type=Path, help="Path to PyTorch Lightning checkpoint"
    # )
    parser.add_argument("output_path", type=Path, help="Path to save the ONNX model")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")

    args = parser.parse_args()

    main(
        # args.config_path,
        # args.ckpt_path,
        args.output_path,
        args.opset,
    )
