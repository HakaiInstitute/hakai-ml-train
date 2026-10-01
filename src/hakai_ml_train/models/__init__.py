import importlib
import inspect
import re

import torch
import torch.nn as nn
from timm.models import group_parameters


def _import_class(class_path: str):
    """Import a class from a dotted path string."""
    module_path, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


# Layer grouping for ViT-shaped backbones that don't ship a timm `group_matcher`
# (e.g. torchgeo's DOFA). Mirrors timm's own ViT matcher: stem -> 0,
# blocks.{i} -> i + 1. DOFA's wavelength encoder lives under
# `patch_embed.weight_generator`, so the stem pattern already covers it.
_VIT_GROUP_MATCHER = {
    "stem": r"^cls_token|pos_embed|reg_token|patch_embed",
    # Only `blocks.(\d+)` may capture -- timm floats every regex group it finds.
    # (99999,) is timm's MATCH_PREV_GROUP: fold into the last block rather than
    # opening a new level. The classification head is vestigial when the ViT is
    # used as a feature extractor, so it must not add depth of its own.
    "blocks": [
        (r"^blocks\.(\d+)", None),
        (r"^(?:norm|fc_norm|head)", (99999,)),
    ],
}


# Learned token embeddings that timm's ViT/EVA matchers omit from their `stem`
# pattern. Left alone they fall through to the top layer, which both gives them
# full LR and adds a phantom depth level that shrinks every real layer's scale.
# `reg_token` affects the whole DINOv3 family (timm 1.x vision_transformer/eva).
_STEM_TOKEN_NAMES = ("reg_token",)


def _with_stem_tokens(matcher):
    """Extend a timm `group_matcher`'s stem pattern to cover stray token embeddings."""
    if not isinstance(matcher, dict) or not isinstance(matcher.get("stem"), str):
        return matcher
    extra = "".join(f"|{name}" for name in _STEM_TOKEN_NAMES)
    return {**matcher, "stem": matcher["stem"] + extra}


def _has_vit_stem(backbone: nn.Module) -> bool:
    """True if `_VIT_GROUP_MATCHER`'s stem pattern recognizes this backbone.

    Guards against applying the fallback to something merely `.blocks`-shaped.
    A CNN like EfficientNet has `.blocks` but a `conv_stem`, which the pattern
    misses -- the stem would silently land in the *top* group and train at full
    LR, inverting the whole point of layer decay.
    """
    stem = re.compile(_VIT_GROUP_MATCHER["stem"])
    return any(stem.match(name) for name, _ in backbone.named_parameters())


# timm's feature-extraction wrappers rebuild a backbone as a flat container, so
# `tu-convnext_*` encoders arrive as a `FeatureListNet` with no `group_matcher`
# and with `stages.0` rewritten to `stages_0`. This mirrors timm's own ConvNeXt
# matcher against the flattened names -- the only change is the separator.
_CONVNEXT_FLAT_STAGE_RE = re.compile(r"^stages_\d+\.blocks\.\d+")


def _convnext_flat_group_matcher(coarse: bool = False):
    if coarse:
        return {"stem": r"^stem", "blocks": r"^stages_(\d+)"}
    return {
        "stem": r"^stem",
        # (0,) puts a stage's downsample ahead of its blocks; (99999,) is timm's
        # MATCH_PREV_GROUP, folding the final norm into the last block rather
        # than opening a level of its own.
        "blocks": [
            (r"^stages_(\d+)\.downsample", (0,)),
            (r"^stages_(\d+)\.blocks\.(\d+)", None),
            (r"^norm_pre", (99999,)),
        ],
    }


def _has_flat_convnext_stages(backbone: nn.Module) -> bool:
    """True if this is a timm ConvNeXt flattened by a feature-extraction wrapper."""
    names = [name for name, _ in backbone.named_parameters()]
    return any(_CONVNEXT_FLAT_STAGE_RE.match(n) for n in names) and any(
        n.startswith("stem") for n in names
    )


class LayerDecayMixin:
    """Exposes timm's layer-wise LR decay hooks on a composite segmentation model.

    ``timm.optim.create_optimizer_v2(..., layer_decay=x)`` derives per-layer LR
    scales from ``model.group_matcher()``. timm's built-in fallback keys off
    ``pretrained_cfg['classifier']``, which a LightningModule doesn't have, so
    without this mixin every parameter lands in one group and ``layer_decay``
    silently does nothing.

    Subclasses set ``backbone_module_path`` to the dotted path of the backbone
    holding the transformer blocks. Grouping is delegated to that backbone's own
    timm ``group_matcher`` when it has one, so per-architecture knowledge stays
    in timm. Everything outside the backbone (necks, decoder, head) trains at
    full LR.
    """

    #: Dotted path from `self` to the module holding the transformer blocks.
    backbone_module_path: str = ""

    def _resolve_backbone(self) -> tuple[nn.Module | None, str]:
        """Return (backbone module, its parameter-name prefix), or (None, "")."""
        if not self.backbone_module_path:
            return None, ""
        obj = self
        for attr in self.backbone_module_path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                return None, ""
        return obj, f"{self.backbone_module_path}."

    def no_weight_decay(self) -> set[str]:
        """Backbone params to exempt from weight decay, as full parameter names.

        Read by timm's optimizer factory. Covers things weight_decay_exclude_1d
        misses, notably `pos_embed` and `cls_token` (both >1-D).
        """
        backbone, prefix = self._resolve_backbone()
        if backbone is None or not hasattr(backbone, "no_weight_decay"):
            return set()
        return {f"{prefix}{name}" for name in backbone.no_weight_decay()}

    def group_matcher(self, coarse: bool = False):
        """Return a callable mapping parameter name -> layer index (0 = earliest)."""
        backbone, prefix = self._resolve_backbone()
        if backbone is None:
            raise ValueError(
                f"Layer-wise LR decay needs a backbone module, but "
                f"{type(self).__name__}.backbone_module_path="
                f"{self.backbone_module_path!r} does not resolve to one. "
                f"SMP's native encoders (mit_b*, resnet*, mobilenet_v2, ...) hold "
                f"no submodule there; layer decay is currently wired up for ViT "
                f"encoders (`tu-vit_*`, `tu-deit_*`) and DOFA -- drop "
                f"`layer_decay` from optimizer_opts for this backbone."
            )

        if hasattr(backbone, "group_matcher"):
            matcher = _with_stem_tokens(backbone.group_matcher(coarse=coarse))
        elif hasattr(backbone, "blocks") and _has_vit_stem(backbone):
            matcher = _VIT_GROUP_MATCHER
        elif _has_flat_convnext_stages(backbone):
            matcher = _convnext_flat_group_matcher(coarse)
        else:
            raise ValueError(
                f"Layer-wise LR decay cannot order the layers of "
                f"{self.backbone_module_path} ({type(backbone).__name__}): it "
                f"provides no timm `group_matcher`, and neither fallback "
                f"applies -- the ViT one needs `.blocks` plus a recognizable "
                f"ViT stem (cls_token/pos_embed/patch_embed), the ConvNeXt one "
                f"needs flattened `stem*`/`stages_N.blocks.M` names. Layer decay "
                f"is currently wired up for ViT encoders (`tu-vit_*`, "
                f"`tu-deit_*`), `tu-convnext_*` and DOFA -- drop `layer_decay` "
                f"from optimizer_opts for this backbone."
            )

        # Layer ids are resolved against the backbone's own parameter names, so
        # timm's regexes (anchored with ^) apply unchanged.
        layer_map = group_parameters(backbone, matcher, reverse=True)
        top = max(layer_map.values(), default=0) + 1

        def match(name: str) -> int:
            if not name.startswith(prefix):
                return top
            return layer_map.get(name.removeprefix(prefix), top)

        return match


_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def _param_groups(module):
    """Return two parameter groups: norm/bias params with weight_decay=0, rest unchanged."""
    no_wd_ids = set()
    for mod in module.modules():
        if isinstance(mod, _NORM_TYPES):
            no_wd_ids.update(id(p) for p in mod.parameters())
    for name, param in module.named_parameters():
        if name.endswith(".bias"):
            no_wd_ids.add(id(param))

    decay = [
        p for p in module.parameters() if p.requires_grad and id(p) not in no_wd_ids
    ]
    no_decay = [
        p for p in module.parameters() if p.requires_grad and id(p) in no_wd_ids
    ]
    return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]


def _is_param_factory(fn) -> bool:
    """True for timm-style factories that build their own param groups from a model.

    ``timm.optim.create_optimizer_v2`` takes ``model_or_params`` first and only
    applies ``filter_bias_and_bn`` / ``layer_decay`` when handed an ``nn.Module``
    -- passing it a pre-built group list silently discards both.
    """
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return False
    return bool(params) and params[0] == "model_or_params"


def configure_optimizers(module):
    """Configure optimizer and LR scheduler from module hparams.

    Expects hparams to contain:
    - optimizer_class: dotted path (e.g., "torch.optim.AdamW")
    - optimizer_opts: dict of kwargs (e.g., {"lr": 3e-4, "weight_decay": 0.01})
    - lr_scheduler_class: dotted path (e.g., "torch.optim.lr_scheduler.OneCycleLR")
    - lr_scheduler_opts: dict of kwargs (e.g., {"max_lr": 3e-4, "pct_start": 0.1})
    - lr_scheduler_interval: "step" or "epoch"
    - lr_scheduler_monitor: metric name or None (required for ReduceLROnPlateau)

    Layer-wise LR decay is opt-in via a timm optimizer factory, e.g.
    ``optimizer_class: timm.optim.create_optimizer_v2`` with
    ``optimizer_opts: {opt: adamw, lr: 3e-4, layer_decay: 0.7}``. The model must
    provide a `group_matcher` (see `LayerDecayMixin`).
    """
    optimizer_cls = _import_class(module.hparams.optimizer_class)
    optimizer_opts = dict(module.hparams.optimizer_opts or {})

    if _is_param_factory(optimizer_cls):
        # Hand the module over so timm builds the weight-decay / layer-decay groups.
        if optimizer_opts.get("layer_decay") is not None and not hasattr(
            module, "group_matcher"
        ):
            raise ValueError(
                f"optimizer_opts['layer_decay'] is set but "
                f"{type(module).__name__} has no `group_matcher`, so timm would "
                f"put every parameter in one group and apply no decay. Add "
                f"`LayerDecayMixin` to the model class and set "
                f"`backbone_module_path`."
            )
        optimizer = optimizer_cls(module, **optimizer_opts)
    else:
        optimizer = optimizer_cls(_param_groups(module), **optimizer_opts)

    # timm's layer-decay groups carry `lr_scale` and expect the *scheduler* to
    # multiply it in (see timm.scheduler.Scheduler.update_groups). torch and
    # Lightning schedulers don't, so fold it into `lr` now -- before any
    # scheduler captures base_lrs from the groups.
    lr_scales = [g.get("lr_scale", 1.0) for g in optimizer.param_groups]
    for group, scale in zip(optimizer.param_groups, lr_scales, strict=True):
        group["lr"] = group["lr"] * scale

    scheduler_cls = _import_class(module.hparams.lr_scheduler_class)
    scheduler_kwargs = dict(module.hparams.lr_scheduler_opts or {})

    # OneCycleLR ignores initial_lr, so scaled LRs must go in via a max_lr list.
    if scheduler_cls is torch.optim.lr_scheduler.OneCycleLR and any(
        scale != 1.0 for scale in lr_scales
    ):
        max_lr = scheduler_kwargs.get("max_lr")
        if isinstance(max_lr, (int, float)):
            scheduler_kwargs["max_lr"] = [max_lr * scale for scale in lr_scales]

    # Auto-inject total_steps/T_max for schedulers that need the total
    # training step count (e.g., OneCycleLR, CosineAnnealingLR).
    # Only injected when not explicitly provided in lr_scheduler_opts.
    # For epoch-interval schedulers, use max_epochs since the scheduler
    # is stepped once per epoch rather than once per optimization step.
    interval = module.hparams.lr_scheduler_interval
    if interval == "epoch":
        total = module.trainer.max_epochs
    else:
        total = module.trainer.estimated_stepping_batches
    sig = inspect.signature(scheduler_cls)
    if "total_steps" in sig.parameters and "total_steps" not in scheduler_kwargs:
        scheduler_kwargs["total_steps"] = total
    if "T_max" in sig.parameters and "T_max" not in scheduler_kwargs:
        scheduler_kwargs["T_max"] = total
    if (
        "steps_per_epoch" in sig.parameters
        and "steps_per_epoch" not in scheduler_kwargs
    ):
        scheduler_kwargs["steps_per_epoch"] = (
            module.trainer.estimated_stepping_batches // module.trainer.max_epochs
        )

    scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

    config = {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": module.hparams.lr_scheduler_interval,
        },
    }
    if module.hparams.lr_scheduler_monitor:
        config["lr_scheduler"]["monitor"] = module.hparams.lr_scheduler_monitor

    return config
