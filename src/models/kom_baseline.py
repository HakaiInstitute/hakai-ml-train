import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import torch
import torchmetrics as tm
from lightning import pytorch as pl
from torchmetrics import classification as fm

from src import losses
from src.models import configure_optimizers as _configure_optimizers

S3_BUCKET = "https://kelp-o-matic.s3.amazonaws.com/pt_jit"
CACHE_DIR = Path("~/.cache/kelp_o_matic").expanduser()


def download_file(url: str, filename: Path):
    # Make a request to the URL
    response = urllib.request.urlopen(url)

    # Download the file
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            # Read data in chunks (e.g., 1024 bytes)
            while True:
                chunk = response.read(1024)
                if not chunk:
                    break
                f.write(chunk)

            # Move the file to the cache directory once downloaded
            f.flush()
            os.fsync(f.fileno())

        shutil.move(f.name, filename)
    except Exception as e:
        if f and os.path.exists(f.name):
            os.remove(f.name)
        raise e


def lazy_load_params(object_name: str):
    object_name = object_name
    remote_url = f"{S3_BUCKET}/{object_name}"
    local_file = CACHE_DIR / object_name

    # Create cache directory if it doesn't exist
    if not CACHE_DIR.is_dir():
        CACHE_DIR.mkdir(parents=True)

    # Download file if it doesn't exist
    if not local_file.is_file():
        download_file(remote_url, local_file)

    return local_file


class KomRGBSpeciesBaselineModel(pl.LightningModule):
    pa_params_file = lazy_load_params(
        "UNetPlusPlus_EfficientNetV2_m_kelp_presence_rgb_jit_dice=0.8703.pt"
    )
    sp_params_file = lazy_load_params(
        "UNetPlusPlus_EfficientNetV2_m_kelp_species_rgb_jit_dice=0.9881.pt"
    )

    def __init__(
        self,
        loss: str,
        loss_opts: dict[str, Any],
        num_classes: int = 2,
        ignore_index: int | None = None,
        optimizer_class: str = "torch.optim.AdamW",
        optimizer_opts: dict[str, Any] | None = None,
        lr_scheduler_class: str = "torch.optim.lr_scheduler.OneCycleLR",
        lr_scheduler_opts: dict[str, Any] | None = None,
        lr_scheduler_interval: str = "step",
        lr_scheduler_monitor: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pa_model = torch.jit.load(self.pa_params_file, map_location=self.device)
        self.sp_model = torch.jit.load(self.sp_params_file, map_location=self.device)

        for p in self.pa_model.parameters():
            p.requires_grad = False

        for p in self.sp_model.parameters():
            p.requires_grad = False

        self.loss_fn = losses.__dict__[loss](**loss_opts)

        self.class_names = ["bg", "macro", "nereo"]

        # metrics
        metric_kwargs = dict(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
        )
        metrics = tm.MetricCollection(
            {
                "accuracy": fm.Accuracy(**metric_kwargs),
                "iou": fm.JaccardIndex(**metric_kwargs, average="none"),
                "recall": fm.Recall(**metric_kwargs, average="none"),
                "precision": fm.Precision(**metric_kwargs, average="none"),
                "f1": fm.F1Score(**metric_kwargs, average="none"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # Override parent's activation function to remove .squeeze(1) for multiclass
        self.activation_fn = lambda x: torch.softmax(x, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        presence_logits = self.pa_model(x)
        species_logits = self.sp_model(x)

        kelp_probs = torch.sigmoid(presence_logits).squeeze(1)  # P(kelp)
        bg_probs = 1 - kelp_probs  # P(bg)
        marginal_probs = torch.softmax(species_logits, dim=1)  # P(macro,nereo|kelp)

        probs = torch.stack(
            [
                bg_probs,
                marginal_probs[:, 0] * kelp_probs,
                marginal_probs[:, 1] * kelp_probs,
            ],
            dim=1,
        )
        return probs

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="test")

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        computed = self.val_metrics.compute()
        self.log("val/accuracy_epoch", computed["val/accuracy"])

        iou_per_class = computed["val/iou"]
        precision_per_class = computed["val/precision"]
        recall_per_class = computed["val/recall"]
        f1_per_class = computed["val/f1"]

        for i, class_name in enumerate(self.class_names):
            self.log(f"val/iou_epoch/{class_name}", iou_per_class[i])
            self.log(f"val/recall_epoch/{class_name}", recall_per_class[i])
            self.log(f"val/precision_epoch/{class_name}", precision_per_class[i])
            self.log(f"val/f1_epoch/{class_name}", f1_per_class[i])
        self.log("val/iou_epoch", iou_per_class[1:].mean())
        self.val_metrics.reset()

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        probs = self.forward(x)

        loss = self.loss_fn(probs, y.long())
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"))

        metrics = getattr(self, f"{phase}_metrics")
        step_values = metrics(probs, y)

        self.log(f"{phase}/accuracy", step_values[f"{phase}/accuracy"])

        iou_per_class = step_values[f"{phase}/iou"]
        precision_per_class = step_values[f"{phase}/precision"]
        recall_per_class = step_values[f"{phase}/recall"]
        f1_per_class = step_values[f"{phase}/f1"]
        for i, class_name in enumerate(self.class_names):
            self.log_dict(
                {
                    f"{phase}/iou_{class_name}": iou_per_class[i],
                    f"{phase}/recall_{class_name}": recall_per_class[i],
                    f"{phase}/precision_{class_name}": precision_per_class[i],
                    f"{phase}/f1_{class_name}": f1_per_class[i],
                }
            )
        self.log(f"{phase}/iou", iou_per_class[1:].mean())

        return loss

    def configure_optimizers(self):
        return _configure_optimizers(self)


class KomMusselsBaselineModel(pl.LightningModule):
    def __init__(
        self,
        loss: str,
        loss_opts: dict[str, Any],
        num_classes: int = 1,
        ignore_index: int | None = None,
        optimizer_class: str = "torch.optim.AdamW",
        optimizer_opts: dict[str, Any] | None = None,
        lr_scheduler_class: str = "torch.optim.lr_scheduler.OneCycleLR",
        lr_scheduler_opts: dict[str, Any] | None = None,
        lr_scheduler_interval: str = "step",
        lr_scheduler_monitor: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        pa_params_file = lazy_load_params(
            "UNetPlusPlus_EfficientNetB4_mussel_presence_rgb_jit_dice=0.9269.pt"
        )
        self.pa_model = torch.jit.load(pa_params_file, map_location=self.device)

        for p in self.pa_model.parameters():
            p.requires_grad = False

        self.loss_fn = losses.__dict__[loss](**loss_opts)

        self.class_names = ["bg", "mussels"]

        # metrics
        metric_kwargs = dict(
            task="binary",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
        )
        metrics = tm.MetricCollection(
            {
                "accuracy": fm.Accuracy(**metric_kwargs),
                "iou": fm.JaccardIndex(**metric_kwargs),
                "recall": fm.Recall(**metric_kwargs),
                "precision": fm.Precision(**metric_kwargs),
                "f1": fm.F1Score(**metric_kwargs),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # Override parent's activation function to remove .squeeze(1) for binary
        self.activation_fn = lambda x: torch.sigmoid(x).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pa_model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="test")

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        computed = self.val_metrics.compute()
        self.log_dict(
            {f"{k}_epoch": v for k, v in computed.items()},
        )
        self.val_metrics.reset()

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        logits = self.forward(x)
        probs = self.activation_fn(logits)

        loss = self.loss_fn(logits, y.long())
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"))

        metrics = getattr(self, f"{phase}_metrics")
        self.log_dict(metrics(probs, y))

        return loss

    def configure_optimizers(self):
        return _configure_optimizers(self)


class KomRGBISpeciesBaselineModel(KomRGBSpeciesBaselineModel):
    pa_params_file = lazy_load_params(
        "UNetPlusPlus_EfficientNetB4_kelp_presence_rgbi_jit_miou=0.8785.pt"
    )
    sp_params_file = lazy_load_params(
        "UNetPlusPlus_EfficientNetB4_kelp_species_rgbi_jit_miou=0.8432.pt"
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        presence_logits = self.pa_model(x)
        species_logits = self.sp_model(x)

        pa_probs = torch.softmax(presence_logits, dim=1)  # P(bg,kelp)
        bg_probs = pa_probs[:, 0]  # P(bg)
        kelp_probs = pa_probs[:, 1]  # P(kelp)
        marginal_probs = torch.softmax(species_logits, dim=1)  # P(macro,nereo|kelp)

        probs = torch.stack(
            [
                bg_probs,  # P(bg)
                marginal_probs[:, 0] * kelp_probs,  # P(macro)
                marginal_probs[:, 1] * kelp_probs,  # P(nereo)
            ],
            dim=1,
        )
        return probs
