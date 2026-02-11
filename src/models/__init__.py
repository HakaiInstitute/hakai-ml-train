import importlib
import inspect
from functools import wraps


def _import_class(class_path: str):
    """Import a class from a dotted path string."""
    module_path, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _patch_scheduler_for_finetuning(scheduler):
    """Patch a scheduler so it doesn't override lr for externally managed param groups.

    When BackboneFinetuning (or similar callbacks) add param groups after the
    scheduler is created, those groups' lr should be managed by the callback,
    not the scheduler. This patches step() to save and restore lr values for
    any param groups added after scheduler creation.
    """
    n_original = len(scheduler.optimizer.param_groups)
    original_step = scheduler.step

    @wraps(original_step)
    def _step(*args, **kwargs):
        extra_groups = scheduler.optimizer.param_groups[n_original:]
        saved_lrs = [g["lr"] for g in extra_groups]
        original_step(*args, **kwargs)
        for g, lr in zip(extra_groups, saved_lrs, strict=False):
            g["lr"] = lr

    scheduler.step = _step
    return scheduler


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
    optimizer = optimizer_cls(
        filter(lambda p: p.requires_grad, module.parameters()),
        **(module.hparams.optimizer_opts or {}),
    )

    scheduler_cls = _import_class(module.hparams.lr_scheduler_class)
    scheduler_kwargs = dict(module.hparams.lr_scheduler_opts or {})

    # Auto-inject total_steps/T_max for schedulers that need the total
    # training step count (e.g., OneCycleLR, CosineAnnealingLR).
    # Only injected when not explicitly provided in lr_scheduler_opts.
    sig = inspect.signature(scheduler_cls)
    if "total_steps" in sig.parameters and "total_steps" not in scheduler_kwargs:
        scheduler_kwargs["total_steps"] = module.trainer.estimated_stepping_batches
    if "T_max" in sig.parameters and "T_max" not in scheduler_kwargs:
        scheduler_kwargs["T_max"] = module.trainer.estimated_stepping_batches

    scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
    _patch_scheduler_for_finetuning(scheduler)

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
