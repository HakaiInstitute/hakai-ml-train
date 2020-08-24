import itertools

import numpy as np
from geotiff_crop_dataset import CropDatasetReader
from tqdm.contrib import concurrent


class GeoTiffReader(CropDatasetReader):
    """Load GeoTiff images for inference.

    For efficiency, blank image areas (sections that contain only black or white pixels) are filtered out in advanced.

    Assumes that the image has values in domain [0, 255]. All other values are clipped, so it may be necessary to
    convert 16 or 32-bit images to 8-bit apriori.
    """

    def __init__(self, *args, min_value=0, max_value=255, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_value = min_value
        self._max_value = max_value

        mask = concurrent.thread_map(self._should_keep, range(len(self)),
                                     desc="Filtering blank areas")
        self.y0x0 = list(itertools.compress(self.y0x0, mask))

    def _get_np(self, idx: int) -> np.ndarray:
        y0, x0 = self.y0x0[idx]

        crop = self.raster.read(
            window=((y0, y0 + self.crop_size), (x0, x0 + self.crop_size)),
            masked=True, boundless=True, fill_value=self.fill_value
        )
        return crop.filled()

    def _should_keep(self, idx):
        img = np.clip(self._get_np(idx), self._min_value, self._max_value)
        is_blank = np.all((img == self._min_value) | (img == self._max_value))
        return not is_blank
