from abc import ABCMeta, abstractmethod
from typing import Optional, Any

import lightning.pytorch as pl
import torch
import torchmetrics.classification as fm
import torchseg
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from . import losses


class _SegmentationModelBase(
    pl.LightningModule,
    PyTorchModelHubMixin,
    library_name="kelp-o-matic",
    tags=["pytorch", "kelp", "segmentation", "drones", "remote-sensing"],
    repo_url="https://github.com/HakaiInstitute/kelp-o-matic",
    docs_url="https://kelp-o-matic.readthedocs.io/",
    metaclass=ABCMeta,
):
    def __init__(
        self,
        num_classes: int = 2,
        ignore_index: Optional[int] = None,
        lr: float = 0.35,
        weight_decay: float = 0,
        max_epochs: int = 100,
        warmup_period: float = 0.3,
        batch_size: int = 2,
        num_bands: int = 3,
        tile_size: int = 1024,
        loss: dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()
        # self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_period = warmup_period
        self.batch_size = batch_size
        self.num_bands = num_bands
        self.tile_size = tile_size
        self.n = num_classes #- int(self.ignore_index is not None)

        self.loss_fn = losses.__dict__[loss["name"]](**(loss["opts"] or {}))

        # metrics
        self.accuracy = fm.BinaryAccuracy()
        self.jaccard_index = fm.BinaryJaccardIndex()
        self.recall = fm.BinaryRecall()
        self.precision = fm.BinaryPrecision()
        self.f1_score = fm.BinaryF1Score()
        self.dice = fm.Dice()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def example_input_array(self) -> Any:
        return torch.ones(
            (self.batch_size, self.num_bands, self.tile_size, self.tile_size),
            dtype=self.dtype,
            device=self.device,
        )

    def remove_ignore_pixels(self, logits: torch.Tensor, y: torch.Tensor):
        mask = y != self.ignore_index
        return logits[mask], y[mask]

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="test")

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        logits = self.forward(x)
        # if phase == "val" and batch_idx == 0:
        #     self.log_image_samples(batch, preds)

        # Flatten and eliminate ignore class instances
        y = rearrange(y, "b h w -> (b h w)").long()
        logits = rearrange(logits, "b c h w -> (b h w) c")
        if self.ignore_index is not None:
            logits, y = self.remove_ignore_pixels(logits, y)

        if len(y) == 0:
            print("0 length y!")
            return 0

        loss = self.loss_fn(logits, y.long().unsqueeze(1))
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"))

        probs = torch.sigmoid(logits).squeeze(1)
        self.log_dict(
            {
                f"{phase}/accuracy": self.accuracy(probs, y),
                f"{phase}/iou": self.jaccard_index(probs, y),
                f"{phase}/recall": self.recall(probs, y),
                f"{phase}/precision": self.precision(probs, y),
                f"{phase}/f1": self.f1_score(probs, y),
                f"{phase}/dice": self.dice(probs, y),
            }
        )

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            {
                "val/accuracy_epoch": self.accuracy,
                "val/iou_epoch": self.jaccard_index,
                "val/recall_epoch": self.recall,
                "val/precision_epoch": self.precision,
                "val/f1_epoch": self.f1_score,
                "val/dice_epoch": self.dice,
            }
        )

    def configure_optimizers(self):
        """Init optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=True,
        )

        steps = self.trainer.estimated_stepping_batches

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=steps, pct_start=self.warmup_period
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        # return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]


class SMPSegmentationModel(_SegmentationModelBase):
    def __init__(
        self,
        *args,
        architecture: str = "UnetPlusPlus",
        backbone: str = "resnet34",
        opts: dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if opts is None:
            opts = {}

        self.model = torchseg.__dict__[architecture](
            backbone,
            in_channels=self.num_bands,
            classes=self.n,
            **opts,
        )

        for p in self.model.parameters():
            p.requires_grad = True

        # self.model = torch.compile(self.model, fullgraph=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


dino_backbones = {
    "dinov2_s": {"name": "dinov2_vits14", "embedding_size": 384, "patch_size": 14},
    "dinov2_b": {"name": "dinov2_vitb14", "embedding_size": 768, "patch_size": 14},
    "dinov2_l": {"name": "dinov2_vitl14", "embedding_size": 1024, "patch_size": 14},
    "dinov2_g": {"name": "dinov2_vitg14", "embedding_size": 1536, "patch_size": 14},
    "dinov2_s_reg": {
        "name": "dinov2_vits14_reg",
        "embedding_size": 384,
        "patch_size": 14,
    },
    "dinov2_b_reg": {
        "name": "dinov2_vitb14_reg",
        "embedding_size": 768,
        "patch_size": 14,
    },
    "dinov2_l_reg": {
        "name": "dinov2_vitl14_reg",
        "embedding_size": 1024,
        "patch_size": 14,
    },
    "dinov2_g_reg": {
        "name": "dinov2_vitg14_reg",
        "embedding_size": 1536,
        "patch_size": 14,
    },
}


class DINOv2Segmentation(_SegmentationModelBase):
    def __init__(
        self,
        *args,
        backbone: str = "dino_v2s",
        opts: dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if opts is None:
            opts = {}

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            dino_backbones[backbone]["name"],
            **opts,
        )
        self.backbone.eval()
        self.embedding_size = dino_backbones[backbone]["embedding_size"]
        self.patch_size = dino_backbones[backbone]["patch_size"]
        self.head = nn.Sequential(
            nn.ConvTranspose2d(
                self.embedding_size, 256, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.n, (3, 3), padding=(1, 1)),
            nn.Upsample(scale_factor=1.75, mode="bilinear", align_corners=False),
        )
        for p in self.head.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size)
        with torch.no_grad():
            x = self.backbone.forward_features(x.cuda())
            x = x["x_norm_patchtokens"]
            x = x.permute(0, 2, 1)
            x = x.reshape(
                batch_size, self.embedding_size, int(mask_dim[0]), int(mask_dim[1])
            )
        return self.head(x)

    @property
    def example_input_array(self) -> Any:
        return torch.ones(
            (self.batch_size, self.num_bands, 896, 896),
            dtype=self.dtype,
            device=self.device,
        )
