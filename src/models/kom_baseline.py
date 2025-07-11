import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import torch
from lightning import pytorch as pl
from torchmetrics import classification as fm

from src import losses

S3_BUCKET = "https://kelp-o-matic.s3.amazonaws.com/pt_jit"
CACHE_DIR = Path("~/.cache/kelp_o_matic").expanduser()


class KomSpeciesBaselineModel(pl.LightningModule):
    def __init__(
        self,
        loss: str,
        loss_opts: dict[str, Any],
        num_classes: int = 2,
        ignore_index: int | None = None,
        lr: float = 0.0003,
        wd: float = 0,
        b1: float = 0.9,
        b2: float = 0.99,
    ):
        super().__init__()
        self.save_hyperparameters()

        pa_params_file = self.lazy_load_params(
            "UNetPlusPlus_EfficientNetV2_m_kelp_presence_rgb_jit_dice=0.8703.pt"
        )
        sp_params_file = self.lazy_load_params(
            "UNetPlusPlus_EfficientNetV2_m_kelp_species_rgb_jit_dice=0.9881.pt"
        )
        self.pa_model = torch.jit.load(pa_params_file, map_location=self.device)
        self.sp_model = torch.jit.load(sp_params_file, map_location=self.device)

        for p in self.pa_model.parameters():
            p.requires_grad = False

        for p in self.sp_model.parameters():
            p.requires_grad = False

        self.loss_fn = losses.__dict__[loss](**loss_opts)

        self.class_names = ["bg", "macro", "nereo"]

        # metrics
        self.accuracy = fm.Accuracy(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
        )
        self.jaccard_index = fm.JaccardIndex(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
            average="none",
        )
        self.recall = fm.Recall(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
            average="none",
        )
        self.precision = fm.Precision(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
            average="none",
        )
        self.f1_score = fm.F1Score(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            ignore_index=self.hparams.ignore_index,
            average="none",
        )

        # Override parent's activation function to remove .squeeze(1) for multiclass
        self.activation_fn = lambda x: torch.softmax(x, dim=1)

    @staticmethod
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

    @staticmethod
    def lazy_load_params(object_name: str):
        object_name = object_name
        remote_url = f"{S3_BUCKET}/{object_name}"
        local_file = CACHE_DIR / object_name

        # Create cache directory if it doesn't exist
        if not CACHE_DIR.is_dir():
            CACHE_DIR.mkdir(parents=True)

        # Download file if it doesn't exist
        if not local_file.is_file():
            KomSpeciesBaselineModel.download_file(remote_url, local_file)

        return local_file

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        presence_logits = self.pa_model(x)
        species_logits = self.sp_model(x)

        presence = (torch.sigmoid(presence_logits)[:, 0] > 0.5).to(
            torch.long
        )  # 0: bg, 1: kelp
        species = torch.argmax(species_logits, dim=1) + 1  # 1: macro, 2: nereo
        label = torch.mul(presence, species)  # 0: bg, 1: macro, 2: nereo

        return label

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="test")

    def on_validation_epoch_end(self) -> None:
        self.log("val/accuracy_epoch", self.accuracy)

        iou_per_class = self.jaccard_index.compute()
        precision_per_class = self.precision.compute()
        recall_per_class = self.recall.compute()
        f1_per_class = self.f1_score.compute()

        for i, class_name in enumerate(self.class_names):
            self.log(f"val/iou_epoch/{class_name}", iou_per_class[i])
            self.log(f"val/recall_epoch/{class_name}", recall_per_class[i])
            self.log(f"val/precision_epoch/{class_name}", precision_per_class[i])
            self.log(f"val/f1_epoch/{class_name}", f1_per_class[i])
        self.log(f"val/iou_epoch", iou_per_class[1:].mean())

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        classification = self.forward(x)
        probs = (
            torch.nn.functional.one_hot(
                classification, num_classes=self.hparams.num_classes
            )
            .permute(0, 3, 1, 2)
            .to(torch.float)
        )

        loss = self.loss_fn(probs, y.long())
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"))

        self.log(f"{phase}/accuracy", self.accuracy(probs, y))

        iou_per_class = self.jaccard_index(probs, y)
        precision_per_class = self.precision(probs, y)
        recall_per_class = self.recall(probs, y)
        f1_per_class = self.f1_score(probs, y)
        for i, class_name in enumerate(self.class_names):
            self.log_dict(
                {
                    f"{phase}/iou-{class_name}": iou_per_class[i],
                    f"{phase}/recall-{class_name}": recall_per_class[i],
                    f"{phase}/precision-{class_name}": precision_per_class[i],
                    f"{phase}/f1-{class_name}": f1_per_class[i],
                }
            )
        self.log(f"{phase}/iou", iou_per_class[1:].mean())

        return loss

    def configure_optimizers(self):
        """Init optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            [
                param
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
