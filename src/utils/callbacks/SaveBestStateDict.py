import warnings

import pytorch_lightning as pl
import torch
from pathlib import Path
from typing import Union


class SaveBestStateDict(pl.callbacks.Callback):
    def __init__(self, file_path: Union[str, Path, None] = None):
        super().__init__()
        self.file_path = file_path

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.checkpoint_callback:
            warnings.warn("SaveBestOnnx callback requires a checkpoint callback to be present")
            return

        pl_module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        if not self.file_path:
            filename = Path(trainer.checkpoint_callback.best_model_path).stem + "-state-dict.pt"
            self.file_path = Path(trainer.checkpoint_callback.best_model_path).with_name(filename)

        torch.save(pl_module.state_dict(), str(self.file_path))
