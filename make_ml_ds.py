import fire
from pathlib import Path

import geopandas as gpd
import utils as ut


def make(img, mask, kelp, out, crop_size=200):
    """
    Create tiled png images from drone imagery with kelp labels. Useful for creating a dataset for ML learning.

    :param img: The drone imagery to make the tiled dataset from.
    :param mask: A shapefile to clip the drone imagery with. Useful for clipping out boundary artifacts.
    :param kelp: A shapefile delineating the kelp beds. Used to create tiled label rasters for ML algorithms.
    :param out: The directory to save the output dataset and intermediate files.
    :param crop_size: The size of the tiled dataset images. Used for both length and width.
    :return: None. Creates a tiled dataset at location `out`.
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

    # Crop the image using the mask
    print("Clipping imagery raster to mask...")
    clipped_img = str(Path(out).joinpath(f"{Path(img).stem}_clipped.tif"))
    ut.image.clip_raster_with_shp_mask(clipped_img, img, mask)

    # Crop kelp raster to img extent
    print("Clipping kelp raster to image extent...")
    clipped_kelp = str(Path(out).joinpath("kelp_clipped.tif"))
    extent = ut.image.get_raster_extent(clipped_img)
    ut.image.clip_raster_by_extent(clipped_kelp, kelp_r, extent=extent)

    # Slice the image into fixed width and height sections
    ut.image.slice_and_dice_image(clipped_img, clipped_kelp, dest_x, dest_y, crop_size=crop_size)


if __name__ == '__main__':
    fire.Fire(make)
