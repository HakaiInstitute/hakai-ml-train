import importlib
import inspect


def _import_class(class_path: str):
    """Import a class from a dotted path string."""
    module_path, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


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
