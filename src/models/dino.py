from typing import Any, Literal

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm
import torchmetrics.classification as fm
from huggingface_hub import PyTorchModelHubMixin
from transformers import (
    DINOv3ViTConfig,
    DINOv3ViTModel,
    EomtDinov3Config,
    EomtDinov3ForUniversalSegmentation,
)

from . import configure_optimizers as _configure_optimizers


def _load_dinov3_backbone_weights(
    model: EomtDinov3ForUniversalSegmentation,
    pretrained_model_name_or_path: str,
) -> None:
    """Load DINOv3ViT backbone weights into an EomtDinov3 model.

    The DINOv3ViT and EomtDinov3 have nearly identical backbone param names
    differing only by pluralization: layer.N.* -> layers.N.*, norm.* -> layernorm.*.
    """
    vit = DINOv3ViTModel.from_pretrained(pretrained_model_name_or_path)
    vit_state = vit.state_dict()

    renamed = {}
    for key, value in vit_state.items():
        if key == "embeddings.mask_token":
            continue
        new_key = key.replace("layer.", "layers.").replace("norm.", "layernorm.")
        renamed[new_key] = value

    # Filter to only keys present in the model — the pretrained ViT may use a
    # different MLP variant (e.g. SwiGLU gate_proj) not present in EomtDinov3.
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in renamed.items() if k in model_keys}

    model.load_state_dict(filtered, strict=False)


def _convert_semantic_labels(
    y: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert (B, H, W) class-index labels to EoMT format.

    Returns:
        mask_labels: list of B tensors, each (K, H, W) binary masks
        class_labels: list of B tensors, each (K,) class indices
    """
    B, H, W = y.shape
    mask_labels = []
    class_labels = []

    for i in range(B):
        yi = y[i]  # (H, W)
        masks = []
        classes = []

        for c in range(num_classes):
            mask = (yi == c).float()  # (H, W)
            if ignore_index is not None:
                mask[yi == ignore_index] = 0.0
            masks.append(mask)
            classes.append(c)

        mask_labels.append(torch.stack(masks))  # (K, H, W)
        class_labels.append(torch.tensor(classes, dtype=torch.long, device=y.device))

    return mask_labels, class_labels


def _get_semantic_probs(
    outputs,
    num_classes: int,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Post-process EoMT outputs into (B, C, H, W) semantic probability maps.

    Reproduces the EoMT post-processing: class_probs @ mask_probs -> semantic map.
    """
    # class_queries_logits: (B, Q, num_classes+1) — last class is no-object
    # masks_queries_logits: (B, Q, h, w) — low-res mask logits
    class_logits = outputs.class_queries_logits  # (B, Q, C+1)
    mask_logits = outputs.masks_queries_logits  # (B, Q, h, w)

    # Get class probabilities, excluding the no-object class
    class_probs = class_logits.softmax(dim=-1)[..., :-1]  # (B, Q, C)

    # Get mask probabilities
    mask_probs = mask_logits.sigmoid()  # (B, Q, h, w)

    # Combine: (B, Q, C) x (B, Q, h, w) -> (B, C, h, w)
    semantic = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)

    # Interpolate to target size
    semantic = F.interpolate(
        semantic, size=target_size, mode="bilinear", align_corners=False
    )

    return semantic


class DinoBinarySegmentationModel(
    pl.LightningModule,
    PyTorchModelHubMixin,
    library_name="habitat-mapper",
    tags=["pytorch", "kelp", "segmentation", "drones", "remote-sensing"],
    repo_url="https://github.com/HakaiInstitute/habitat-mapper",
    docs_url="https://habitat-mapper.readthedocs.io/",
):
    def __init__(
        self,
        model_opts: dict[str, Any],
        image_size: int = 640,
        num_classes: int = 2,
        num_queries: int = 100,
        ignore_index: int | None = None,
        optimizer_class: str = "torch.optim.AdamW",
        optimizer_opts: dict[str, Any] | None = None,
        lr_scheduler_class: str = "torch.optim.lr_scheduler.OneCycleLR",
        lr_scheduler_opts: dict[str, Any] | None = None,
        lr_scheduler_interval: str = "step",
        lr_scheduler_monitor: str | None = None,
        ckpt_path: str | None = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        task: Literal["binary", "multiclass", "multilabel"] = (
            "binary" if num_classes == 1 else "multiclass"
        )

        pretrained_path = model_opts["pretrained_model_name_or_path"]

        # Build EomtDinov3Config from the DINOv3ViT pretrained config
        vit_config = DINOv3ViTConfig.from_pretrained(pretrained_path)
        eomt_config = EomtDinov3Config(
            hidden_size=vit_config.hidden_size,
            intermediate_size=vit_config.intermediate_size,
            num_hidden_layers=vit_config.num_hidden_layers,
            num_attention_heads=vit_config.num_attention_heads,
            patch_size=vit_config.patch_size,
            num_channels=vit_config.num_channels,
            num_register_tokens=vit_config.num_register_tokens,
            hidden_act=vit_config.hidden_act,
            layer_norm_eps=vit_config.layer_norm_eps,
            layerscale_value=vit_config.layerscale_value,
            drop_path_rate=vit_config.drop_path_rate,
            image_size=image_size,
            num_labels=num_classes,
            num_queries=num_queries,
        )

        # Create model with random init, then load backbone weights
        self.model = EomtDinov3ForUniversalSegmentation(eomt_config)
        _load_dinov3_backbone_weights(self.model, pretrained_path)

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            self.load_state_dict(ckpt["state_dict"])

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if name.startswith(("embeddings.", "layers.", "layernorm.")):
                    param.requires_grad = False

        self.model = torch.compile(self.model)

        # Metrics
        metric_kwargs = dict(
            task=task,
            num_classes=num_classes,
            ignore_index=ignore_index,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        outputs = self.model(pixel_values=x)
        return _get_semantic_probs(outputs, self.hparams.num_classes, (H, W))

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self._phase_step(batch, batch_idx, phase="test")

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        _, _, H, W = x.shape

        mask_labels, class_labels = _convert_semantic_labels(
            y, self.hparams.num_classes, self.hparams.ignore_index
        )

        outputs = self.model(
            pixel_values=x,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        loss = outputs.loss
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"), sync_dist=True)

        probs = _get_semantic_probs(outputs, self.hparams.num_classes, (H, W))

        metrics = getattr(self, f"{phase}_metrics")
        self.log_dict(metrics(probs, y), sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        computed = self.val_metrics.compute()
        self.log_dict(
            {f"{k}_epoch": v for k, v in computed.items()},
            sync_dist=True,
        )
        self.val_metrics.reset()

    def configure_optimizers(self):
        return _configure_optimizers(self)


class DinoMulticlassSegmentationModel(DinoBinarySegmentationModel):
    def __init__(
        self, *args, class_names: tuple[str, ...] = ("bg", "macro", "nereo"), **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_names = class_names

        # Override metrics with per-class (average="none") variants
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

    def on_validation_epoch_end(self) -> None:
        computed = self.val_metrics.compute()
        self.log("val/accuracy_epoch", computed["val/accuracy"], sync_dist=True)

        iou_per_class = computed["val/iou"]
        precision_per_class = computed["val/precision"]
        recall_per_class = computed["val/recall"]
        f1_per_class = computed["val/f1"]

        for i, class_name in enumerate(self.class_names):
            self.log(f"val/iou_epoch/{class_name}", iou_per_class[i], sync_dist=True)
            self.log(
                f"val/recall_epoch/{class_name}", recall_per_class[i], sync_dist=True
            )
            self.log(
                f"val/precision_epoch/{class_name}",
                precision_per_class[i],
                sync_dist=True,
            )
            self.log(f"val/f1_epoch/{class_name}", f1_per_class[i], sync_dist=True)
        self.log("val/iou_epoch", iou_per_class[1:].mean(), sync_dist=True)
        self.val_metrics.reset()

    def _phase_step(self, batch: torch.Tensor, batch_idx: int, phase: str):
        x, y = batch
        _, _, H, W = x.shape

        mask_labels, class_labels = _convert_semantic_labels(
            y, self.hparams.num_classes, self.hparams.ignore_index
        )

        outputs = self.model(
            pixel_values=x,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        loss = outputs.loss
        self.log(f"{phase}/loss", loss, prog_bar=(phase == "train"), sync_dist=True)

        probs = _get_semantic_probs(outputs, self.hparams.num_classes, (H, W))

        metrics = getattr(self, f"{phase}_metrics")
        step_values = metrics(probs, y)

        self.log(f"{phase}/accuracy", step_values[f"{phase}/accuracy"], sync_dist=True)

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
                },
                sync_dist=True,
            )
        self.log(f"{phase}/iou", iou_per_class[1:].mean(), sync_dist=True)

        return loss
