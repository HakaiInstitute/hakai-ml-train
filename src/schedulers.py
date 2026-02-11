import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupCosineDecayLR(LRScheduler):
    """Linear warmup followed by cosine decay learning rate scheduler.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of steps for linear warmup from 0 to base lr.
        total_steps: Total number of training steps (auto-injected by
            configure_optimizers when not provided).
        min_lr: Minimum learning rate at the end of cosine decay.
        last_epoch: The index of last epoch.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup: scale from 0 to base_lr
            scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay from base_lr to min_lr
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            scale = self.min_lr + (1.0 - self.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
            # scale is already in absolute terms when min_lr > 0, adjust
            return [
                self.min_lr
                + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]
        return [base_lr * scale for base_lr in self.base_lrs]
