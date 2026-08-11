import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupCosineDecayLR(LRScheduler):
    """Linear warmup followed by cosine decay learning rate scheduler.

    Exactly one of ``warmup_steps`` or ``warmup_epochs`` must be provided.
    When using ``warmup_epochs``, ``steps_per_epoch`` is also required so the
    warmup duration can be converted to optimizer steps.

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Total number of training steps (auto-injected by
            configure_optimizers when not provided).
        warmup_steps: Number of steps for linear warmup from 0 to base lr.
            Mutually exclusive with ``warmup_epochs``.
        warmup_epochs: Number of epochs for linear warmup from 0 to base lr.
            Mutually exclusive with ``warmup_steps``. Requires
            ``steps_per_epoch``.
        steps_per_epoch: Number of optimizer steps per epoch. Required when
            ``warmup_epochs`` is provided; ignored otherwise.
        min_lr: Minimum learning rate at the end of cosine decay.
        last_epoch: The index of last epoch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int | None = None,
        warmup_epochs: int | None = None,
        steps_per_epoch: int | None = None,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if (warmup_steps is None) == (warmup_epochs is None):
            raise ValueError(
                "Exactly one of warmup_steps or warmup_epochs must be provided."
            )
        if warmup_epochs is not None:
            if steps_per_epoch is None:
                raise ValueError(
                    "steps_per_epoch must be provided when using warmup_epochs."
                )
            self.warmup_steps = warmup_epochs * steps_per_epoch
        else:
            assert warmup_steps is not None
            self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        # Handle dynamically added param groups (e.g., from BackboneFinetuning)
        if len(self.optimizer.param_groups) > len(self.base_lrs):
            for group in self.optimizer.param_groups[len(self.base_lrs) :]:
                self.base_lrs.append(group.get("initial_lr", group["lr"]))

        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]

        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_scale
            for base_lr in self.base_lrs
        ]


__all__ = ["LinearWarmupCosineDecayLR"]
