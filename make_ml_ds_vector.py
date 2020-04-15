#!/usr/bin/env python
from pathlib import Path

import geopandas as gpd
from osgeo import gdal

import utils.data_prep as ut


def make(img, shapefile, out, crop_size=513):
    """
    Create tiled png images from drone imagery with seagrass labels. Useful for creating a dataset for ML learning.

    Args:
        img: The drone imagery to make the tiled dataset from.
        shapefile: A shapefile delineating the seagrass beds. Used to create tiled label rasters for ML algorithms.
        out: The directory to save the output dataset and intermediate files.
        crop_size: The size of the tiled dataset images. Used for both length and width.
        mask: Optional shapefile to clip the drone imagery with. Useful for clipping out boundary artifacts.

    Returns: None. Creates a tiled dataset at location `out`.

    """
    # Create out directory if not already exists
    Path(out).mkdir(parents=True, exist_ok=True)
    print("Creating file:", out)

    # Make the output directories if they don't exist
    dest_x = str(Path(out).joinpath('x'))
    dest_y = str(Path(out).joinpath('y'))
    Path(dest_x).mkdir(parents=True, exist_ok=True)
    Path(dest_y).mkdir(parents=True, exist_ok=True)

    print("Adding label field to Kelp data..")
    # Create and populate a new label attribute for shapefile class
    df = gpd.read_file(shapefile)

    # Kelp is 1, not seagrass is 0
    labels = [1 for _, row in df.iterrows()]

    df['label'] = labels
    seagrass_s = str(Path(out).joinpath("seagrass_labelled.shp"))
    df.to_file(seagrass_s)

    # Convert the seagrass shapefile to a raster
    print("Rasterizing seagrass shapefile...")
    seagrass_r = str(Path(out).joinpath('./seagrass.tif'))
    ut.shp2tiff(seagrass_s, seagrass_r, img, label_attr="label")

    # Crop seagrass raster to img extent
    print("Clipping seagrass raster to image extent...")
    clipped_seagrass = str(Path(out).joinpath("seagrass_clipped.tif"))
    extent = ut.get_raster_extent(img)
    ut.clip_raster_by_extent(clipped_seagrass, seagrass_r, extent=extent)

    print("Creating image patches dataset...")
    # Slice the image into fixed width and height sections
    ut.check_same_extent(img, clipped_seagrass)
    ut.slice_and_dice_image(img, dest_x, mode='RGB', crop_size=crop_size)

    print("Creating label patches dataset...")
    ut.slice_and_dice_image(clipped_seagrass, dest_y, mode='L', crop_size=crop_size)

    print("Deleting extra labels")
    ut.del_extra_labels(out)


def main():
    # SEAGRASS DATA
    dsets = [
        "beaumont_2017",
        "bennet_bay_2018",
        "cabbage_2017",
        "choked_pass_2016",
        "choked_pass_2017",
        "goose_se_2015",
        "goose_sw_2015",
        "james_bay_2018",
        "koeye_2015",
        "koeye_2017",
        "koeye_2018",
        "koeye_2019",
        "lyall_harbour_2018",
        "marna_2018",
        "mcmullin_north_2015",
        "mcmullin_south_2015",
        "nalau_2019",
        "narvaez_bay_2018",
        "pruth_bay_2016",
        "pruth_bay_2017",
        "selby_cove_2017",
        "stirling_bay_2019",
        "triquet_bay_2016",
        "triquet_north_2016",
        "tumbo_2018",
        "underhill_2019",
        "west_bay_mcnaughton_2019",
    ]
    for d in dsets:
        make(
            f"data/seagrass/raw/{d}/image_wgs.tif",
            f"data/seagrass/raw/{d}/seagrass_wgs.shp",
            f"data/seagrass/processed/{d}",
            crop_size=513
        )


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

    # fire.Fire(make)
    main()
