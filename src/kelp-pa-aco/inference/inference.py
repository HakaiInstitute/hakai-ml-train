#!/usr/bin/env python
# coding: utf-8
import argparse
import math
from pathlib import Path

import rasterio
import torch
from rasterio.windows import Window
from tqdm import tqdm

from kernels import Kernel, BartlettHanningKernel

CLIP_SIZE = 1024
NUM_CLASSES = 2
DEVICE = torch.device('cuda')
MODEL_WEIGHTS = "./UNetPlusPlus_Resnet34_kelp_presence_aco_jit_miou=0.8340.pt"


# Classification v1.1
class TorchMemoryRegister(object):
    def __init__(self, image_path: str, num_classes: int, window_size: int, device: torch.device.type):
        super().__init__()
        self.image_path = image_path
        self.n = num_classes
        self.ws = window_size
        self.hws = window_size // 2
        self.device = device

        # Copy metadata from img
        with rasterio.open(image_path, 'r') as src:
            src_width = src.width

        self.height = self.ws
        self.width = (math.ceil(src_width / self.ws) * self.ws) + self.hws
        self.register = torch.zeros((self.n, self.height, self.width), device=self.device)

    @property
    def _zero_chip(self):
        return torch.zeros((self.n, self.hws, self.hws), dtype=torch.float, device=self.device)

    def step(self, new_logits: torch.Tensor, img_window: Window):
        # 1. Read data from the registry to update with the new logits
        # |a|b| |
        # |c|d| |
        with torch.no_grad():
            logits_abcd = self.register[:, :, img_window.col_off:img_window.col_off + self.ws].clone()
            logits_abcd += new_logits

        # Move data around and write to the registry to make new space for the next row of processing windows
        # |c|b| | + pop a
        # |0|d| |
        logits_a = logits_abcd[:, :self.hws, :self.hws]
        logits_c = logits_abcd[:, self.hws:, :self.hws]
        logits_c0 = torch.concat([logits_c, self._zero_chip], dim=1)
        logits_bd = logits_abcd[:, :, self.hws:]

        # write c0
        self.register[:, :, img_window.col_off:img_window.col_off + self.hws] = logits_c0

        # write bd
        col_off_bd = img_window.col_off + self.hws
        self.register[:, :, col_off_bd:col_off_bd + self.hws] = logits_bd

        # Return the information complete predictions (argmax of a, stripped over overflowing padding)
        preds_win = Window(col_off=img_window.col_off, row_off=img_window.row_off,
                           height=min(self.hws, img_window.height), width=min(self.hws, img_window.width))
        preds = logits_a[:, :img_window.height, :img_window.width].softmax(axis=0)

        return preds, preds_win


class HanningWindowSegmentation(object):
    def __init__(self, model: torch.nn.Module, kernel: Kernel,
                 num_classes: int, window_size: int, device: torch.device.type):
        super().__init__()
        self.model = model
        self.kernel = kernel(window_size, device)
        self.n = num_classes
        self.ws = window_size
        self.hws = window_size // 2
        self.device = device

    @property
    def _pure_black_pred(self):
        """Shortcut prediction for any tiles that are all black. Equal to a pred of BG for all pixels"""
        pred = torch.zeros((self.n, self.ws, self.ws), device=self.device)
        pred[0] = 1
        return pred

    def _window_generator(self, img_height: int, img_width: int):
        for row_start in range(0, img_height, self.hws):
            for col_start in range(0, img_width, self.hws):
                row_stop = min(row_start + self.ws, img_height)
                col_stop = min(col_start + self.ws, img_width)
                yield Window.from_slices((row_start, row_stop), (col_start, col_stop))

    def _pad_tile(self, x: torch.Tensor) -> torch.Tensor:
        # zero pad image data to consistent window size
        _, th, tw = x.shape
        return torch.nn.functional.pad(x, (0, self.ws - tw, 0, self.ws - th), mode='constant', value=0)

    @staticmethod
    def _preprocess(x: torch.Tensor) -> torch.Tensor:
        # to float
        x = x.to(torch.float) / 255
        # min-max scale
        min_, _ = torch.kthvalue(x.flatten().unique(), 2)
        max_ = x.flatten().max()
        x = torch.clamp((x - min_) / (max_ - min_ + 1e-8), 0, 1)

        # batch_size=1
        x = x.unsqueeze(dim=0)
        return x

    def __call__(self, img_path: str, output_path: str):
        # Initialize the output file and working file
        self._init_output_file(img_path, output_path)

        # Initialize the working data register
        register = TorchMemoryRegister(img_path, self.n, self.ws, self.device)

        # Classification
        with rasterio.open(img_path) as src, rasterio.open(output_path, 'r+') as dest:
            windows = list(self._window_generator(src.height, src.width))
            last_col_off = windows[-1].col_off
            last_row_off = windows[-1].row_off

            for window in tqdm(windows):
                x = torch.tensor(src.read(window=window), device=self.device, dtype=torch.uint8)
                x = self._pad_tile(x)

                if (x == 0).all():
                    # Mock softmax outputs for BG with 100% confidence
                    pred = self._pure_black_pred
                else:
                    # Prediction
                    x = self._preprocess(x)
                    with torch.no_grad():
                        pred = self.model.forward(x).squeeze(dim=0)

                # Multiply by hann window
                with torch.no_grad():
                    kernel_pred = self.kernel(
                        pred,
                        top=(window.row_off == 0),
                        bottom=(window.row_off == last_row_off),
                        left=(window.col_off == 0),
                        right=(window.col_off == last_col_off)
                    )

                # Update the registry with intermediate data and recieve complete predictions (softmax output)
                preds, pred_win = register.step(kernel_pred, window)
                dest.write(preds.argmax(dim=0).detach().cpu().numpy(), indexes=1, window=pred_win)

    @staticmethod
    def _init_output_file(img_path: str, output_path: str):
        """Write a new single band geotiff with the same size and CRS as the src image and all pixels==0"""
        with rasterio.open(img_path) as src:
            # Initialize final output file
            output_profile = src.profile
            output_profile.update({"count": 1, "dtype": "uint8", "bigtiff": "IF_SAFER"})
            with rasterio.open(output_path, 'w', **output_profile):
                pass

    @classmethod
    def run(cls, img_path: str, output_path: str, model: torch.nn.Module, kernel: Kernel, num_classes: int,
            window_size: int, device: torch.device.type):
        self = cls(model, kernel, num_classes, window_size, device)
        return self(img_path, output_path)


def main(img_path, output_path):
    with open(MODEL_WEIGHTS, "rb") as f:
        model = torch.jit.load(f, map_location=DEVICE).eval()

    segmentor = HanningWindowSegmentation(model, BartlettHanningKernel, NUM_CLASSES, CLIP_SIZE, DEVICE)
    segmentor(img_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=Path, required=True)
    parser.add_argument("output_path", type=Path, required=True)
    args = parser.parse_args()

    main(**vars(args))
