from torch.utils.data import Dataset
import rasterio
import itertools
import numpy as np
from PIL import Image
from utils.image import pad_out


class GeoTiffDataset(Dataset):
    def __init__(self, img_path, transform=None, crop_size=200, mode='RGB', channels=(1, 2, 3)):
        super().__init__()
        self.img_path = img_path
        self.transform = transform
        self.crop_size = crop_size
        self.mode = mode
        self.channels = channels
        self.raster = rasterio.open(self.img_path)

        self._y0s = range(0, self.raster.height, self.crop_size)
        self._x0s = range(0, self.raster.width, self.crop_size)
        self._y0x0s = list(itertools.product(self._y0s, self._x0s))

    def height(self):
        return len(list(self._y0s)) * self.crop_size

    def width(self):
        return len(list(self._x0s)) * self.crop_size

    def get_origin(self, item):
        return self._y0x0s[item]

    def __len__(self):
        return len(self._y0x0s)

    def __getitem__(self, item):
        y0, x0 = self._y0x0s[item]
        window = ((y0, y0 + self.crop_size), (x0, x0 + self.crop_size))
        subset = self.raster.read(self.channels, window=window)
        subset = np.moveaxis(subset, 0, 2)  # (c, h, w) = (h, w, c)
        subset = np.clip(subset, 0, 255).astype(np.uint8)
        subset = pad_out(subset, self.crop_size)
        img = Image.fromarray(subset, self.mode)

        if self.transform:
            img = self.transform(img)

        return img

    def __del__(self):
        self.raster.close()


class GeoTiffWriter(object):
    def __init__(self, geotiff_ds, out_path):
        super().__init__()
        self.geotiff_ds = geotiff_ds
        self.out_path = out_path
        self.crop_size = geotiff_ds.crop_size

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

    def __del__(self):
        self.out_raster.close()

    def write_index(self, idx, segmentation):
        y0, x0 = self.geotiff_ds.get_origin(idx)
        window = ((y0, y0 + self.crop_size), (x0, x0 + self.crop_size))
        self.out_raster.write(segmentation, window=window)

        
if __name__ == '__main__':
    ds = GeoTiffDataset("../data/RPAS/NW_Calvert_2012/NWCalvert_2012.tif")
    ds[len(ds) // 2].show()
    del ds
