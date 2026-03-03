import cv2
import torch
from lightning.pytorch.cli import LightningCLI

cv2.setNumThreads(0)
torch.set_float32_matmul_precision("medium")


def cli_main():
    """
    Command-line interface to run SMPSegmentationModel with DataModule.
    """
    cli = LightningCLI(save_config_kwargs={"overwrite": True})
    return cli


if __name__ == "__main__":
    cli_main()

    print("Done!")
