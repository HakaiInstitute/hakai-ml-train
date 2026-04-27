"""Muon optimizer supporting parameters with ndim >= 2.

`torch.optim.Muon` rejects non-2D parameters. This variant flattens any
higher-dimensional parameter (e.g. a 4D Conv2d weight of shape
``(out_ch, in_ch, kH, kW)``) to ``(out_ch, in_ch * kH * kW)`` before the
Newton-Schulz orthogonalization, then reshapes the orthogonalized update
back to the original shape. The Newton-Schulz iteration and shape-aware
learning-rate adjustment are reused from ``torch.optim._muon``.
"""

import torch
from torch.optim import Optimizer
from torch.optim._muon import (
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_C,
    DEFAULT_NS_STEPS,
    EPS,
    _adjust_lr,
    _zeropower_via_newtonschulz,
)


class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if momentum < 0.0:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in (
            "original",
            "match_rms_adamw",
        ):
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(
                        f"Muon requires ndim >= 2 parameters, got shape {tuple(p.size())}"
                    )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            adjust_lr_fn = group["adjust_lr_fn"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if torch.is_complex(p):
                    raise RuntimeError("Muon does not support complex parameters")
                if p.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )
                buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf.clone()

                orig_shape = update.shape
                update_2d = update.reshape(orig_shape[0], -1)
                ortho_2d = _zeropower_via_newtonschulz(
                    update_2d, ns_coefficients, ns_steps, eps
                ).to(update.dtype)
                ortho = ortho_2d.reshape(orig_shape)

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                lr_adj = _adjust_lr(lr, adjust_lr_fn, update_2d.shape)
                p.add_(ortho, alpha=-lr_adj)

        return loss


__all__ = ["Muon"]
