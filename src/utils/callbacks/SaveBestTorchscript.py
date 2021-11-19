import warnings

import pytorch_lightning as pl
from pathlib import Path
from typing import Any, Optional, Union


class SaveBestTorchscript(pl.callbacks.Callback):
    def __init__(self, file_path: Union[str, Path, None] = None, method: Optional[str] = 'script',
                 example_inputs: Optional[Any] = None, **kwargs):
        super().__init__()
        self.file_path = file_path
        self.method = method
        self.example_inputs = example_inputs
        self.kwargs = kwargs



    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.checkpoint_callback:
            warnings.warn("SaveBestOnnx callback requires a checkpoint calback to be present")
            return

        pl_module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        if not self.file_path:
            fname = Path(trainer.checkpoint_callback.best_model_path).stem + "-torchscript.pt"
            self.file_path = Path(trainer.checkpoint_callback.best_model_path).with_name(fname)

        pl_module.eval()
        pl_module.to_torchscript(self.file_path, self.method, self.example_inputs, **self.kwargs)
