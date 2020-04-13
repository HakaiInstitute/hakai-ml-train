#!/usr/bin/env python
from pathlib import Path
import utils.data_prep.utils as ut
from osgeo import gdal


def make(img, kelp, out, crop_size=513):
    """
    Create tiled png images from drone imagery with kelp labels. Useful for creating a dataset for ML learning.

    Args:
        img: The drone imagery to make the tiled dataset from.
        kelp: A raster delineating the kelp beds. Used to create tiled label rasters for ML algorithms.
        out: The directory to save the output dataset and intermediate files.
        crop_size: The size of the tiled dataset images. Used for both length and width.

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

    # Crop kelp raster to img extent
    print("Clipping kelp raster to image extent...")
    clipped_kelp = str(Path(out).joinpath("kelp_clipped.tif"))
    extent = ut.get_raster_extent(img)
    ut.clip_raster_by_extent(clipped_kelp, kelp, extent=extent)

    print("Creating image patches dataset...")
    # Slice the image into fixed width and height sections
    ut.check_same_extent(img, clipped_kelp)
    ut.slice_and_dice_image(img, dest_x, mode='RGB', crop_size=crop_size)

    print("Creating label patches dataset...")
    ut.slice_and_dice_image(clipped_kelp, dest_y, mode='L', crop_size=crop_size)

    print("Deleting extra labels")
    ut.del_extra_labels(out)


def main():
    # # KELP DATA
    # # Calvert 2012
    # make(
    #     "data/kelp/nw_calvert_2012/imagery/Calvert_ortho_2012_Web_NAD83.tif",
    #     "data/kelp/nw_calvert_2012/raster/2012_Kelp_Water_RC_1_byte_clean.tif",
    #     "data/datasets/kelp/nw_calvert_2012",
    #     crop_size=513
    # )
    # # # Calvert 2014
    # # # make(
    # # #     "data/kelp/nw_calvert_2014/imagery/",
    # # #     "data/kelp/nw_calvert_2014/raster/",
    # # #     "data/datasets/nw_calvert_2014",
    # # # )
    # # Calvert 2015
    # make(
    #     "data/kelp/nw_calvert_2015/imagery/calvert_nwcalvert15_CSRS_mos_U0015.tif",
    #     "data/kelp/nw_calvert_2015/raster/2015_U0015_kelp_byte_clean.tif",
    #     "data/datasets/kelp/nw_calvert_2015",
    #     crop_size=513
    # )
    # # Choked Pass 2016
    # make(
    #     "data/kelp/choked_pass_2016/imagery/20160803_Calvert_ChokedNorthBeach_georef_MOS_Cropped_U0069.tif",
    #     "data/kelp/choked_pass_2016/raster/2016_U069_Kelp_RC_1_byte_clean.tif",
    #     "data/datasets/kelp/choked_pass_2016",
    #     crop_size=513
    # )
    # # West Beach 2016
    # make(
    #     "data/kelp/west_beach_2016/imagery/20160804_Calvert_WestBeach_Georef_mos_U0070.tif",
    #     "data/kelp/west_beach_2016/raster/2016_U070_Kelp_RC_1_byte_clean.tif",
    #     "data/datasets/kelp/west_beach_2016",
    #     crop_size=513
    # )
    # # McNaughton 2017
    # # make(
    # #     "data/kelp/mcnaughton_2017/imagery/CentralCoast_McNaughtonGroup_MOS_U0168.tif",
    # #     "data/kelp/mcnaughton_2017/raster/McNaughton_kelp_byte_clean.tif",
    # #     "data/datasets/kelp/mcnaughton_2017",
    # #     crop_size=513
    # # )


    # SEAGRASS DATA
    dsets = [
        "goose_sw_2015",
        "goose_se_2015",
        "mcmullin_north_2015",
        "mcmullin_south_2015",
        "koeye_2015",
        "triquet_bay_2016",
        "triquet_north_2016",
        "choked_pass_2017",
        "pruth_bay_2016",
        "choked_pass_2017",
        "pruth_bay_2017",
        "selby_cove_2017",
        "beaumont_2017",
        "cabbage_2017",
        "koeye_2017",
        "james_bay_2018",
        "bennet_bay_2018",
        "lyall_harbour_2018",
        "koeye_2018",
        "narvaez_bay_2018",
        "tumbo_2018",
        "marna_2018",
        "stirling_bay_2019",
        "nalau_2019",
        "underhill_2019",
        "west_bay_mcnaughton_2019",
        "koeye_2019",
    ]
    for d in dsets:
        make(
            f"data/seagrass/{d}/imagery/image.tif",
            f"data/seagrass/{d}/raster/seagrass.shp",
            f"data/datasets/seagrass/{d}",
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

