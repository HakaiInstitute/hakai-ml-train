import pytorch_lightning as pl
import torch
from pathlib import Path


class SaveBestStateDictCallback(pl.callbacks.Callback):
    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        pl_module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        torch.save(
            pl_module.state_dict(),
            Path(trainer.checkpoint_callback.best_model_path).with_suffix(".pt"),
        )
