"""Hybrid optimizer: Muon for ndim>=2 weights, AdamW for the rest."""

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .muon import Muon


class MuonAdamWAux(Optimizer):
    """Apply Muon to matrix-shaped parameters and AdamW to the rest.

    Parameters with ``ndim >= 2`` (Linear/Conv weights) go to Muon. Remaining
    1D parameters (biases, normalization scales, 1D embeddings) go to AdamW.

    Inherits from ``torch.optim.Optimizer`` so PyTorch LR schedulers and
    Lightning recognize it as an optimizer. Internally holds two child
    optimizers; ``param_groups`` is the concatenation of their groups so
    schedulers can adjust each independently, and ``step``/``state_dict`` /
    ``load_state_dict`` delegate to both children.
    """

    _TAKES_MODULE = True

    def __init__(
        self,
        module: nn.Module,
        muon_lr: float = 0.02,
        muon_weight_decay: float = 0.0,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_adjust_lr_fn: str | None = None,
        adamw_lr: float = 3e-4,
        adamw_weight_decay: float = 0.0,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
    ) -> None:
        muon_params: list[torch.Tensor] = []
        adamw_params: list[torch.Tensor] = []
        for p in module.parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        # Initialize the Optimizer base with all trainable params so isinstance
        # checks (LRScheduler, Lightning) and base-class step/hook machinery work.
        # We replace param_groups below with the inner optimizers' groups so
        # LR schedulers can adjust Muon and AdamW LRs independently.
        super().__init__(muon_params + adamw_params, defaults={})

        self.muon = Muon(
            muon_params,
            lr=muon_lr,
            weight_decay=muon_weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            adjust_lr_fn=muon_adjust_lr_fn,
        )
        self.adamw = torch.optim.AdamW(
            adamw_params,
            lr=adamw_lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=adamw_weight_decay,
        )
        self.param_groups = list(self.muon.param_groups) + list(self.adamw.param_groups)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])

    def add_param_group(self, param_group: dict) -> None:
        if not hasattr(self, "muon"):
            # Called from Optimizer.__init__ before inner optimizers exist.
            super().add_param_group(param_group)
            return
        params = param_group["params"]
        if not isinstance(params, list):
            params = list(params)
        if any(p.ndim < 2 for p in params):
            self.adamw.add_param_group(param_group)
        else:
            self.muon.add_param_group(param_group)
        self.param_groups = list(self.muon.param_groups) + list(self.adamw.param_groups)


__all__ = ["MuonAdamWAux"]
