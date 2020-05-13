#!/usr/bin/env python
from pathlib import Path

import fire
from osgeo import gdal

import data_prep as dp


def make(img, kelp, out, crop_size=513, stride=None):
    """
    Create tiled png images from drone imagery with kelp labels. Useful for creating a dataset for ML learning.

    Args:
        img: The drone imagery to make the tiled dataset from.
        kelp: A raster delineating the kelp beds. Used to create tiled label rasters for ML algorithms.
        out: The directory to save the output dataset and intermediate files.
        crop_size: The size of the tiled dataset images. Used for both length and width.
        stride: The difference in x0 and y0 positions for adjacent cropped image chips. Defaults to crop_size.

    Returns: None. Creates a tiled dataset at location `out`.

    """
    if stride is None:
        stride = crop_size

    # Create out directory if not already exists
    Path(out).mkdir(parents=True, exist_ok=True)
    print("Creating file:", out)

    # Make the output directories if they don't exist
    dest_x = str(Path(out).joinpath('x'))
    dest_y = str(Path(out).joinpath('y'))
    Path(dest_x).mkdir(parents=True, exist_ok=True)
    Path(dest_y).mkdir(parents=True, exist_ok=True)

    # Crop kelp raster to img extent
    print("Clipping kelp raster to image extent...")
    clipped_kelp = str(Path(out).joinpath("kelp_clipped.tif"))
    extent = dp.get_raster_extent(img)
    h = dp.get_raster_height(img)
    w = dp.get_raster_width(img)
    dp.clip_raster_by_extent(clipped_kelp, kelp, extent=extent, height=h, width=w)

    print("Creating image patches dataset...")
    # Slice the image into fixed width and height sections
    dp.check_same_extent(img, clipped_kelp)
    dp.slice_and_dice_image(img, dest_x, mode='RGB', crop_size=crop_size, stride=stride)

    print("Creating label patches dataset...")
    dp.slice_and_dice_image(clipped_kelp, dest_y, mode='L', crop_size=crop_size, stride=stride)

    print("Deleting extra labels")
    dp.del_extra_labels(out)


class GdalErrorHandler(object):
    def __init__(self):
        self.err_level = gdal.CE_None
        self.err_no = 0
        self.err_msg = ''

    def handler(self, err_level, err_no, err_msg):
        self.err_level = err_level
        self.err_no = err_no
        self.err_msg = err_msg


if __name__ == '__main__':
    err = GdalErrorHandler()
    handler = err.handler  # Note don't pass class method directly or python segfaults
    # due to a reference counting bug
    # http://trac.osgeo.org/gdal/ticket/5186#comment:4

    gdal.PushErrorHandler(handler)
    gdal.UseExceptions()  # Exceptions will get raised on anything >= gdal.CE_Failure

    fire.Fire(make)
