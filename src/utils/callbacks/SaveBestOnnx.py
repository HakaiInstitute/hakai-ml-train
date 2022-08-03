import warnings
from pathlib import Path
from typing import Any, Optional, Union

import pytorch_lightning as pl


class SaveBestOnnx(pl.callbacks.Callback):
    def __init__(self, file_path: Union[str, Path, None] = None, input_sample: Optional[Any] = None,
                 **kwargs):
        super().__init__()
        self.file_path = file_path
        self.input_sample = input_sample
        self.kwargs = kwargs

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.checkpoint_callback:
            warnings.warn("SaveBestOnnx callback requires a checkpoint callback to be present")
            return

        pl_module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        if not self.file_path:
            self.file_path = Path(trainer.checkpoint_callback.best_model_path).with_suffix(".onnx")

        pl_module.to_onnx(self.file_path, self.input_sample, **self.kwargs)
