"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2021-04-06
Description: Get stats over all image bands
"""

from functools import cached_property

import fire
import numpy as np
import rasterio
import rasterio.features
import rasterio.warp


class ImageStats(object):
    def __init__(self, img_path):
        super().__init__()
        self.dataset = rasterio.open(img_path)
        self.bands = self.dataset.count
        self.size = self.dataset.width * self.dataset.height
        self.width = self.dataset.width
        self.height = self.dataset.height

    @cached_property
    def count(self):
        return self.dataset.read(masked=True).count()

    @cached_property
    def mean(self):
        return self.dataset.read(masked=True).mean()

    @cached_property
    def std(self):
        return self.dataset.read(masked=True).std()

    @cached_property
    def max(self):
        return self.dataset.read(masked=True).max()

    @cached_property
    def min(self):
        return self.dataset.read(masked=True).min()

    def percentile(self, min_percent=1, max_percent=98):
        """Returns the low and high pixel values of the image corresponding to the specified percentile range."""
        return np.percentile(self.dataset.read(masked=True), (min_percent, max_percent))

    def __del__(self):
        self.dataset.close()

    @classmethod
    def max_cli(cls, img_path):
        print(cls(img_path).max)

    @classmethod
    def min_cli(cls, img_path):
        print(cls(img_path).min)

    @classmethod
    def mean_cli(cls, img_path):
        print(cls(img_path).mean)

    @classmethod
    def std_cli(cls, img_path):
        print(cls(img_path).std)

    @classmethod
    def percentile_low_cli(cls, img_path, min_percent=1, max_percent=98):
        low, high = cls(img_path).percentile(min_percent, max_percent)
        print(low)

    @classmethod
    def percentile_high_cli(cls, img_path, min_percent=1, max_percent=98):
        low, high = cls(img_path).percentile(min_percent, max_percent)
        print(high)


if __name__ == '__main__':
    fire.Fire({
        "mean": ImageStats.mean_cli,
        "std": ImageStats.std_cli,
        "max": ImageStats.max_cli,
        "min": ImageStats.min_cli,
        "percentile_low": ImageStats.percentile_low_cli,
        "percentile_high": ImageStats.percentile_high_cli,
    })
