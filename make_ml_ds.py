import fire
from pathlib import Path

import geopandas as gpd
import utils as ut
from osgeo import gdal
# Make GDAL raise python exceptions for errors (warnings won't raise an exception)
gdal.UseExceptions()
# Stop GDAL printing both warnings and errors to STDERR
gdal.PushErrorHandler('CPLQuietErrorHandler')


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

    # Make the output directories if they don't exist
    dest_x = str(Path(out).joinpath('x'))
    dest_y = str(Path(out).joinpath('y'))
    Path(dest_x).mkdir(parents=True, exist_ok=True)
    Path(dest_y).mkdir(parents=True, exist_ok=True)

    print("Adding label field to Kelp data..")
    # Create and populate a new label attribute for shapefile class
    df = gpd.read_file(kelp)
    labels = [ut.kelp.get_label(row["Species"], row["Density"]) for _, row in df.iterrows()]
    df['label'] = labels
    kelp_s = str(Path(out).joinpath("kelp.shp"))
    df.to_file(kelp_s)

    # Convert the kelp shapefile to a raster
    print("Rasterizing kelp shapefile...")
    kelp_r = str(Path(out).joinpath('./kelp.tif'))
    ut.convert.shp2tiff(kelp_s, kelp_r, img, label_attr="label")

    # Crop the image using the mask if given
    if mask is not None:
        print("Clipping imagery raster to mask...")
        clipped_img = str(Path(out).joinpath(f"{Path(img).stem}_clipped.tif"))
        ut.image.clip_raster_with_shp_mask(clipped_img, img, mask)
    else:
        clipped_img = img

    # Crop kelp raster to img extent
    print("Clipping kelp raster to image extent...")
    clipped_kelp = str(Path(out).joinpath("kelp_clipped.tif"))
    extent = ut.image.get_raster_extent(clipped_img)
    ut.image.clip_raster_by_extent(clipped_kelp, kelp_r, extent=extent)

    print("Creating image patches dataset...")
    # Slice the image into fixed width and height sections
    ut.image.check_same_extent(clipped_img, clipped_kelp)
    ut.image.slice_and_dice_image(clipped_img, dest_x, crop_size=crop_size)

    print("Creating label patches dataset...")
    ut.image.slice_and_dice_image(clipped_kelp, dest_y, crop_size=crop_size)


if __name__ == '__main__':
    # fire.Fire(make)

    make(
        "data/NW_Calvert/2016/20160804_Calvert_WestBeach_Georef_mos_U0070.tif",
        "data/NW_Calvert/2016/2016_Kelp_Extent_KH_May15_2017.shp",
        "data/datasets/Calvert_WestBeach_2016",
    )

    make(
        "data/NW_Calvert/2016/20160803_Calvert_ChokedNorthBeach_georef_MOS_U0069.tif",
        "data/NW_Calvert/2016/2016_Kelp_Extent_KH_May15_2017.shp",
        "data/datasets/Calvert_ChokedNorthBeach_2016",
        mask="data/NW_Calvert/2016/Calvert_ChokedNorthBeach_2016_Mask.shp"
    )

    make(
        "data/NW_Calvert/2015/calvert_choked15_CSRS_mos_U0015.tif",
        "data/NW_Calvert/2015/2015_Kelp_Extent_FINAL21072016.shp",
        "data/datasets/Calvert_2015",
    )

    make(
        "data/NW_Calvert/2012/NWCalvert_2012.tif",
        "data/NW_Calvert/2012/2012_Kelp_Extent_FINAL21072016.shp",
        "data/datasets/Calvert_2012"
    )

    # make(
    #     "data/McNaughtons/CentralCoast_McNaughtonGroup_MOS_U0168_ForDerekJ.tif",
    #     "data/McNaughtons/McNaughtons_Group_Kelp_2017_forDerekJ.shp",
    #     "data/datasets/McNaughtons_2017"
    # )

    # make(
    #     "data/Manley_Womanley/centralcoast_stirling_mos_U0061.tif",
    #     "data/Manley_Womanley/Kelp_20160706_CentralCoast_U0061.shp",
    #     "data/datasets/Manley_Womanley_2016"
    # )
