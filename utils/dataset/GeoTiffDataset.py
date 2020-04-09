from torch.utils.data import Dataset
import rasterio
import itertools
import numpy as np
from PIL import Image


def _pad_out(crop, crop_size):
    """Pads image crop, which is a numpy array of size (h, w, c), with zeros so that h and w equal crop_size."""
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        if len(crop.shape) == 2:
            padding = ((0, crop_size - crop.shape[0]), (0, crop_size - crop.shape[1]))
        else:
            padding = ((0, crop_size - crop.shape[0]), (0, crop_size - crop.shape[1]), (0, 0))

        return np.pad(crop, padding, mode='constant', constant_values=0)
    else:
        return crop


class GeoTiffDataset(Dataset):
    def __init__(self, img_path, transform=None, crop_size=200, pad=0, mode='RGB', channels=(1, 2, 3)):
        super().__init__()
        self.img_path = img_path
        self.transform = transform
        self.crop_size = crop_size
        self.pad = pad
        self.mode = mode
        self.channels = channels
        self.raster = rasterio.open(self.img_path)

        self._y0s = range(0, self.raster.height, self.crop_size)
        self._x0s = range(0, self.raster.width, self.crop_size)
        self._y0x0s = list(itertools.product(self._y0s, self._x0s))

    def height(self):
        return self.raster.height

    def width(self):
        return self.raster.width

    def get_origin(self, item):
        return self._y0x0s[item]

    def __len__(self):
        return len(self._y0x0s)

    def __getitem__(self, item):
        y0, x0 = self._y0x0s[item]
        window = ((np.max((y0 - self.pad, 0)), np.min((y0 + self.crop_size + self.pad, self.raster.height))),
                  (np.max((x0 - self.pad, 0)), np.min((x0 + self.crop_size + self.pad, self.raster.width))))
        subset = self.raster.read(self.channels, window=window)

        if len(subset.shape) == 3:
            subset = np.moveaxis(subset, 0, 2)  # (c, h, w) => (h, w, c)

        # Pad out sections that are too small
        t_pad = np.max((self.pad - y0, 0))
        b_pad = np.max(((y0 + self.crop_size + self.pad) - self.raster.height, 0))
        l_pad = np.max((self.pad - x0, 0))
        r_pad = np.max(((x0 + self.crop_size + self.pad) - self.raster.width, 0))

        if len(subset.shape) == 2:
            padding = ((t_pad, b_pad), (l_pad, r_pad))
        else:
            padding = ((t_pad, b_pad), (l_pad, r_pad), (0, 0))

        subset = np.pad(subset, padding, mode='reflect')

        subset = np.clip(subset, 0, 255).astype(np.uint8)
        img = Image.fromarray(subset, self.mode)

        if self.transform:
            img = self.transform(img)

        return img

    def __del__(self):
        if hasattr(self, 'raster') and not self.raster.closed:
            self.raster.close()


class GeoTiffWriter(object):
    def __init__(self, geotiff_ds, out_path):
        super().__init__()
        self.geotiff_ds = geotiff_ds
        self.out_path = out_path
        self.crop_size = geotiff_ds.crop_size
        self.pad = geotiff_ds.pad

        self.out_raster = rasterio.open(
            self.out_path, 'w',
            driver='GTiff',
            height=self.geotiff_ds.height(),
            width=self.geotiff_ds.width(),
            count=1,
            dtype='uint8',
            crs=self.geotiff_ds.raster.crs,
            transform=self.geotiff_ds.raster.transform
        )

    def height(self):
        return self.out_raster.height

    def width(self):
        return self.out_raster.width

    def __del__(self):
        if hasattr(self, 'raster') and not self.out_raster.closed:
            self.out_raster.close()

    def write_index(self, idx, segmentation):
        y0, x0 = self.geotiff_ds.get_origin(idx)
        window = ((y0, np.min((y0 + self.crop_size, self.height()))),
                  (x0, np.min((x0 + self.crop_size, self.width()))))
        segmentation = segmentation[:, self.pad:-self.pad, self.pad:-self.pad]
        self.out_raster.write(segmentation, window=window)


if __name__ == '__main__':
    ds = GeoTiffDataset("../../data/RPAS/NW_Calvert_2012/NWCalvert_2012.tif")
    ds[len(ds) // 2].show()
    del ds
