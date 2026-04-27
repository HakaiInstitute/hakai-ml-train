"""Hybrid optimizer: Muon for ndim>=2 weights, AdamW for the rest."""

from collections import ChainMap

import torch
import torch.nn as nn

from .muon import Muon


class MuonAdamWAux:
    """Apply Muon to matrix-shaped parameters and AdamW to the rest.

    Parameters with ``ndim >= 2`` (Linear/Conv weights) go to Muon. Remaining
    1D parameters (biases, normalization scales, 1D embeddings) go to AdamW.

    The wrapper exposes the standard ``torch.optim.Optimizer`` duck-type
    interface (``param_groups``, ``state``, ``step``, ``zero_grad``,
    ``state_dict``, ``load_state_dict``, ``add_param_group``) so Lightning's
    automatic optimization, gradient clipping, gradient accumulation and
    standard LR schedulers operate on it transparently.
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
        self._optimizers: tuple[torch.optim.Optimizer, ...] = (self.muon, self.adamw)
        self.param_groups = list(self.muon.param_groups) + list(self.adamw.param_groups)
        self.defaults = dict(self.muon.defaults)

    @property
    def state(self):
        return ChainMap(self.muon.state, self.adamw.state)

    def step(self, closure=None) -> float | None:
        loss: float | None = None
        if closure is not None:
            loss = closure()
        for opt in self._optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self._optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])

    def add_param_group(self, group: dict) -> None:
        params = group["params"]
        if not isinstance(params, list):
            params = list(params)
        if any(p.ndim < 2 for p in params):
            self.adamw.add_param_group(group)
        else:
            self.muon.add_param_group(group)
        self.param_groups = list(self.muon.param_groups) + list(self.adamw.param_groups)


__all__ = ["MuonAdamWAux"]
