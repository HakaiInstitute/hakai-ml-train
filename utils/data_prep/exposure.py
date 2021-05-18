from functools import cached_property

import fire
import numpy as np
import rasterio
from skimage.exposure import adjust_gamma, equalize_adapthist, rescale_intensity


class Exposure(object):
    def __init__(self, img_path, out_path):
        super().__init__()
        self.dataset = rasterio.open(img_path)
        self.bands = self.dataset.count
        self.out_path = out_path

    def __del__(self):
        self.dataset.close()

    @cached_property
    def img(self):
        return self.dataset.read(masked=True)

    @classmethod
    def equalize_adapthist(cls, img_path, out_path, clip_limit=0.01, kernel_size=(64, 64)):
        self = cls(img_path, out_path)
        img = self.img.transpose(1, 2, 0)  # CWH -> WHC
        out_data = equalize_adapthist(img, clip_limit=clip_limit, kernel_size=kernel_size)
        self.write_data(out_data.transpose(2, 0, 1))  # WHC -> CWH

    @classmethod
    def contrast_stretch(cls, img_path, out_path, min_=0.0, max_=99.8):
        self = cls(img_path, out_path)
        img = self.img.transpose(1, 2, 0)
        v_min, v_max = np.percentile(img, (min_, max_))
        print(v_min, v_max)
        out_data = rescale_intensity(img, in_range=(v_min, v_max))
        self.write_data(out_data.transpose(2, 0, 1))  # WHC -> CWH

    @classmethod
    def match_hist(cls, img_path, out_path, ref_path):
        self = cls(img_path, out_path)
        img = self.img.transpose(1, 2, 0)  # CWH -> WHC
        reference = io.imread(ref_path) / 255
        mask = img.mask

        matched = np.ma.array(np.empty(img.shape, dtype=img.dtype), mask=mask,
                              fill_value=img.fill_value)
        for ch in range(img.shape[-1]):
            matched_ch = match_histograms(img[..., ch].compressed(), reference[..., ch].ravel())

            # Re-insert masked background
            mask_ch = mask[..., ch]
            matched[..., ch][~mask_ch] = matched_ch.ravel()

        out_data = matched.filled()
        self.write_data(out_data.transpose(2, 0, 1))  # WHC -> CWH

    @classmethod
    def adjust_gamma(cls, img_path, out_path, gamma=1 / 3):
        self = cls(img_path, out_path)
        img = self.img.transpose(1, 2, 0)
        out_data = adjust_gamma(img, gamma=gamma)
        self.write_data(out_data.transpose(2, 0, 1))  # WHC -> CWH

    def write_data(self, data):
        with rasterio.Env():
            with rasterio.open(self.out_path, 'w', **self.dataset.profile) as dst:
                for b in range(self.bands):
                    dst.write(data[b, ...], b + 1)


if __name__ == '__main__':
    fire.Fire({
        "adjust_gamma": Exposure.adjust_gamma,
        "contrast_stretch": Exposure.contrast_stretch,
        "equalize_adapthist": Exposure.equalize_adapthist,
        "match_hist": Exposure.match_hist
    })
