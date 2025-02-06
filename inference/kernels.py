from abc import ABCMeta, abstractmethod

import numpy as np
import torch

# Implementation of paper:
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0229839#pone.0229839.ref007


class Kernel(torch.nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        size: int = 512,
        device: torch.device.type = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cpu")

        self.size = size
        self.wi = self._init_wi(size, device)
        self.wj = self.wi.clone()

    @staticmethod
    @abstractmethod
    def _init_wi(size: int, device: torch.device.type) -> torch.Tensor:
        raise NotImplementedError

    def get_kernel(
        self,
        top: bool = False,
        bottom: bool = False,
        left: bool = False,
        right: bool = False,
    ) -> torch.Tensor:
        assert not (top and bottom), "top and bottom are mutually exclusive"
        assert not (left and right), "left and right are mutually exclusive"
        wi, wj = self.wi.clone(), self.wj.clone()

        if top:
            wi[: self.size // 2] = 1
        elif bottom:
            wi[self.size // 2 :] = 1

        if left:
            wj[: self.size // 2] = 1
        elif right:
            wj[self.size // 2 :] = 1

        return wi.unsqueeze(1) @ wj.unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        top: bool = False,
        bottom: bool = False,
        left: bool = False,
        right: bool = False,
    ) -> torch.Tensor:
        kernel = self.get_kernel(top=top, bottom=bottom, left=left, right=right)
        return torch.mul(x, kernel)


class HanningKernel(Kernel):
    @staticmethod
    def _init_wi(size: int, device: torch.device.type) -> torch.Tensor:
        i = torch.arange(0, size, device=device)
        return (1 - ((2 * np.pi * i) / (size - 1)).cos()) / 2


class BartlettHanningKernel(Kernel):
    @staticmethod
    def _init_wi(size: int, device: torch.device.type) -> torch.Tensor:
        # Follows original paper:
        # Ha YH, Pearce JA. A new window and comparison to standard windows.
        # IEEE Transactions on Acoustics, Speech, and Signal Processing.
        # 1989;37(2):298–301.
        i = torch.arange(0, size, device=device)
        return (
            0.62
            - 0.48 * (i / size - 1 / 2).abs()
            + 0.38 * (2 * np.pi * (i / size - 1 / 2).abs()).cos()
        )


class TriangularKernel(Kernel):
    @staticmethod
    def _init_wi(size: int, device: torch.device.type) -> torch.Tensor:
        i = torch.arange(0, size, device=device)
        return 1 - (2 * i / size - 1).abs()
