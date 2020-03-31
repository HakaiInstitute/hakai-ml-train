from pathlib import Path
from utils import data_prep
from osgeo import gdal


def make(img, kelp, out, crop_size=200):
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
    extent = data_prep.get_raster_extent(img)
    data_prep.clip_raster_by_extent(clipped_kelp, kelp, extent=extent)

    print("Creating image patches dataset...")
    # Slice the image into fixed width and height sections
    data_prep.check_same_extent(img, clipped_kelp)
    data_prep.slice_and_dice_image(img, dest_x, mode='RGB', crop_size=crop_size)

    print("Creating label patches dataset...")
    data_prep.slice_and_dice_image(clipped_kelp, dest_y, mode='L', crop_size=crop_size)

    print("Deleting extra labels")
    data_prep.del_extra_labels(out)


def main():
    # # Calvert 2012
    # make(
    #     "data/kelp/nw_calvert_2012/imagery/Calvert_ortho_2012_Web_NAD83.tif",
    #     "data/kelp/nw_calvert_2012/raster/2012_Kelp_Water_RC_1_byte_clean.tif",
    #     "data/datasets/kelp/nw_calvert_2012",
    # )
    # # Calvert 2014
    # # make(
    # #     "data/kelp/nw_calvert_2014/imagery/",
    # #     "data/kelp/nw_calvert_2014/raster/",
    # #     "data/datasets/nw_calvert_2014",
    # # )
    # Calvert 2015
    # make(
    #     "data/kelp/nw_calvert_2015/imagery/calvert_nwcalvert15_CSRS_mos_U0015.tif",
    #     "data/kelp/nw_calvert_2015/raster/2015_U0015_kelp_byte_clean.tif",
    #     "data/datasets/kelp/nw_calvert_2015",
    # )
    # Choked Pass 2016
    # make(
    #     "data/kelp/choked_pass_2016/imagery/20160803_Calvert_ChokedNorthBeach_georef_MOS_Cropped_U0069.tif",
    #     "data/kelp/choked_pass_2016/raster/2016_U069_Kelp_RC_1_byte_clean.tif",
    #     "data/datasets/kelp/choked_pass_2016",
    # )
    # West Beach 2016
    # make(
    #     "data/kelp/west_beach_2016/imagery/20160804_Calvert_WestBeach_Georef_mos_U0070.tif",
    #     "data/kelp/west_beach_2016/raster/2016_U070_Kelp_RC_1_byte_clean.tif",
    #     "data/datasets/kelp/west_beach_2016",
    # )
    # McNaughton 2017
    make(
        "data/kelp/mcnaughton_2017/imagery/CentralCoast_McNaughtonGroup_MOS_U0168.tif",
        "data/kelp/mcnaughton_2017/raster/McNaughton_kelp_byte_clean.tif",
        "data/datasets/kelp/mcnaughton_2017",
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

