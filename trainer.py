import torch
from lightning.pytorch.cli import LightningCLI

from src.datamodule import DataModule
from src.module import KelpSpeciesModel, SMPSegmentationModel

torch.set_float32_matmul_precision("medium")


def cli_main():
    """
    Command-line interface to run SMPSegmentationModel with DataModule.
    """
    cli = LightningCLI(
        datamodule_class=DataModule,
        save_config_kwargs={"overwrite": True},
    )
    return cli


if __name__ == "__main__":
    cli_main()

    print("Done!")
