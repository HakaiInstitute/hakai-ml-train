import fire
from pathlib import Path
import geopandas as gpd
import utils as ut
from osgeo import gdal


def make(img, kelp, out, crop_size=200, mask=None):
    """
    Create tiled png images from drone imagery with kelp labels. Useful for creating a dataset for ML learning.

    Args:
        img: The drone imagery to make the tiled dataset from.
        kelp: A shapefile delineating the kelp beds. Used to create tiled label rasters for ML algorithms.
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
    df = gpd.read_file(kelp)
    # Species and density labels
    # labels = [ut.kelp.get_species_density_label(row["Species"], row["Density"]) for _, row in df.iterrows()]

    # Not kelp 0, macro 1, nereo 2, mixed 3
    # labels = [ut.kelp.get_species_label(row["Species"]) for _, row in df.iterrows()]

    # Kelp is 1, not kelp is 0
    labels = [1 for _, row in df.iterrows()]

    df['label'] = labels
    kelp_s = str(Path(out).joinpath("kelp.shp"))
    df.to_file(kelp_s)

    # Convert the kelp shapefile to a raster
    print("Rasterizing kelp shapefile...")
    kelp_r = str(Path(out).joinpath('./kelp.tif'))
    ut.data_prep.shp2tiff(kelp_s, kelp_r, img, label_attr="label")

    # Crop the image using the mask if given
    if mask is not None:
        print("Clipping imagery raster to mask...")
        clipped_img = str(Path(out).joinpath(f"{Path(img).stem}_clipped.tif"))
        ut.data_prep.clip_raster_with_shp_mask(clipped_img, img, mask)
    else:
        clipped_img = img

    # Crop kelp raster to img extent
    print("Clipping kelp raster to image extent...")
    clipped_kelp = str(Path(out).joinpath("kelp_clipped.tif"))
    extent = ut.data_prep.get_raster_extent(clipped_img)
    ut.data_prep.clip_raster_by_extent(clipped_kelp, kelp_r, extent=extent)

    print("Creating image patches dataset...")
    # Slice the image into fixed width and height sections
    ut.data_prep.check_same_extent(clipped_img, clipped_kelp)
    ut.data_prep.slice_and_dice_image(clipped_img, dest_x, mode='RGB', crop_size=crop_size)

    print("Creating label patches dataset...")
    ut.data_prep.slice_and_dice_image(clipped_kelp, dest_y, mode='L', crop_size=crop_size)

    # Delete blank images
    # print("Deleting completely black image crops")
    # ut.data_prep.filter_blank_images(out)

    print("Deleting extra labels")
    ut.data_prep.del_extra_labels(out)


def main():
    make(
        "data/RPAS/NW_Calvert_2012/NWCalvert_2012.tif",
        "data/RPAS/NW_Calvert_2012/2012_Kelp_Extent_FINAL21072016.shp",
        "data/datasets/RPAS/Calvert_2012",
    )

    make(
        "data/RPAS/NW_Calvert_2015/calvert_choked15_CSRS_mos_U0015.tif",
        "data/RPAS/NW_Calvert_2015/2015_Kelp_Extent_FINAL21072016.shp",
        "data/datasets/RPAS/Calvert_2015",
    )

    make(
        "data/RPAS/NW_Calvert_2016/20160804_Calvert_WestBeach_Georef_mos_U0070.tif",
        "data/RPAS/NW_Calvert_2016/2016_Kelp_Extent_KH_May15_2017.shp",
        "data/datasets/RPAS/Calvert_WestBeach_2016",
    )

    make(
        "data/RPAS/NW_Calvert_2016/20160803_Calvert_ChokedNorthBeach_georef_MOS_U0069.tif",
        "data/RPAS/NW_Calvert_2016/2016_Kelp_Extent_KH_May15_2017.shp",
        "data/datasets/RPAS/Calvert_ChokedNorthBeach_2016",
        mask="data/RPAS/NW_Calvert_2016/Calvert_ChokedNorthBeach_2016_Mask.shp",
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

