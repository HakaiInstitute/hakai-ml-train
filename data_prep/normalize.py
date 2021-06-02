"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2021-04-06
Description: Get stats over all image bands
"""

from functools import cached_property

import fire
import rasterio
import rasterio.features
import rasterio.warp


class Normalize(object):
    def __init__(self, img_path, outfile):
        super().__init__()
        self.dataset = rasterio.open(img_path)
        self.outfile = outfile
        self.bands = self.dataset.count

    def __del__(self):
        self.dataset.close()

    def mean_std_scale(self):
        with rasterio.Env():
            with rasterio.open(self.outfile, 'w', **self.dataset.profile) as dst:
                for b in range(1, self.bands + 1):
                    band = self.dataset.read(b, masked=True).squeeze()
                    dst.write((band - band.mean()) / band.std(), b)

    @classmethod
    def mean_std_scale_cli(cls, img_path, outfile):
        cls(img_path, outfile).mean_std_scale()


if __name__ == '__main__':
    fire.Fire({
        "mean_std_scale": Normalize.mean_std_scale_cli,
    })
