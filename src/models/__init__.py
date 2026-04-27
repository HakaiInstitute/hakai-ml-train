import importlib
import inspect

import torch.nn as nn


def _import_class(class_path: str):
    """Import a class from a dotted path string."""
    module_path, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


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


def configure_optimizers(module):
    """Configure optimizer and LR scheduler from module hparams.

    Expects hparams to contain:
    - optimizer_class: dotted path (e.g., "torch.optim.AdamW")
    - optimizer_opts: dict of kwargs (e.g., {"lr": 3e-4, "weight_decay": 0.01})
    - lr_scheduler_class: dotted path (e.g., "torch.optim.lr_scheduler.OneCycleLR")
    - lr_scheduler_opts: dict of kwargs (e.g., {"max_lr": 3e-4, "pct_start": 0.1})
    - lr_scheduler_interval: "step" or "epoch"
    - lr_scheduler_monitor: metric name or None (required for ReduceLROnPlateau)
    """
    optimizer_cls = _import_class(module.hparams.optimizer_class)
    optimizer_opts = module.hparams.optimizer_opts or {}
    if getattr(optimizer_cls, "_TAKES_MODULE", False):
        optimizer = optimizer_cls(module, **optimizer_opts)
    else:
        optimizer = optimizer_cls(_param_groups(module), **optimizer_opts)

    scheduler_cls = _import_class(module.hparams.lr_scheduler_class)
    scheduler_kwargs = dict(module.hparams.lr_scheduler_opts or {})

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
