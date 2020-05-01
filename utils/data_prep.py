from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from osgeo import gdal, osr, ogr
from rasterio.windows import Window
from tqdm.auto import tqdm
from tqdm.contrib.itertools import product as tproduct


def check_same_extent(src_a, src_b):
    """
    Check that the projected extents of the geo-located images at src_img and src_labels are the same.
    Args:
        src_a: Path to first geo-located image
        src_b: Path to second geo-located image

    Returns: True if images have the same extent
    """
    with rasterio.open(src_a) as img:
        w = img.width
        h = img.height

    with rasterio.open(src_b) as img:
        if w != img.width or h != img.height:
            raise RuntimeError("Label dataset must have the same extent as the image dataset")

    return True


def _pad_out(crop, crop_size):
    """Pads numpy arrays so that cropped sections are all exactly crop_size*crop_size in shape"""
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        if len(crop.shape) == 2:
            padding = ((0, crop_size - crop.shape[0]), (0, crop_size - crop.shape[1]))
        else:
            padding = ((0, crop_size - crop.shape[0]), (0, crop_size - crop.shape[1]), (0, 0))

        return np.pad(crop, padding, mode='constant', constant_values=0)
    else:
        return crop


def slice_and_dice_image(src_img, dest_d, mode='L', crop_size=200):
    """
    Create a machine learning dataset of image patches. Chops src_img into square sections of length/width crop_size
    and saves the patches and intermediate files to direction dest_d. Uses `cpus` count of cpus to process image in
    parallel.
    Args:
        src_img: Path to the image to crop into square sections
        dest_d: Path to directory to save the processed image
        mode: The Pillow image write mode. E.g. 'RGB', 'L' (for BW)
        crop_size: The length and width of cropped sections to create

    Returns: None
    """
    with rasterio.open(src_img) as dataset:
        x0s = range(0, dataset.width, crop_size)
        y0s = range(0, dataset.height, crop_size)

        dest_d = Path(dest_d)
        for i, (x0, y0) in enumerate(tproduct(x0s, y0s)):
            dest = str(dest_d.joinpath(f"{i}.png"))
            window = Window(x0, y0, crop_size, crop_size)

            if mode == 'L':
                subset = dataset.read(1, window=window, masked=True)
                subset = subset.filled(0)  # Fill nodata areas with 0

                if len(subset.shape) > 2:
                    subset = np.squeeze(subset)
                subset = np.clip(subset, 0, 255).astype(np.uint8)
                subset = _pad_out(subset, crop_size)
                Image.fromarray(subset, mode).save(dest)
            else:  # mode == 'RGB':
                subset = dataset.read([1, 2, 3], window=window)
                subset = np.moveaxis(subset, 0, 2)  # (c, h, w) = (h, w, c)
                subset = np.clip(subset, 0, 255).astype(np.uint8)
                if np.any(subset > 0):
                    subset = _pad_out(subset, crop_size)
                    Image.fromarray(subset, mode).save(dest)


def clip_raster_with_shp_mask(dest, src, mask):
    """
    Clip a raster at path src to shapefile at path mask and save at location dest.
    Args:
        dest: The path to the desired save location
        src: The path to raster to clip
        mask: The path to the shapefile to use for clipping

    Returns: None
    """
    opts = gdal.WarpOptions(format="GTiff", cutlineDSName=mask, cutlineLayer=Path(mask).stem, cropToCutline=True)
    ds = gdal.Warp(dest, src, options=opts)
    ds = None
    opts = None


def get_raster_width(src):
    """
        Get the pixel width of a raster by path.
        Args:
            src: Path to the raster for which pixel width will be returned

        Returns: int pixel width of raster
        """
    with rasterio.open(src) as dataset:
        width = dataset.width
    return width


def get_raster_height(src):
    """
    Get the pixel height of a raster by path.
    Args:
        src: Path to the raster for which pixel height will be returned

    Returns: int pixel height of raster
    """
    with rasterio.open(src) as dataset:
        width = dataset.height
    return width


def clip_raster_by_extent(dest, src, extent, height, width):
    """
    Clips raster at location src to extent and saves at location dest.
    Args:
        dest: Path to save clipped raster
        src: Source raster to clip
        extent: Either an array of extent in format [ulx, uly, lrx, lry]
        width: The width in pixels of the clipped raster
        height: The height in pixels of the clipped raster
    Returns: None
    """
    opts = gdal.TranslateOptions(format="GTiff", projWin=extent, outputBounds=extent, height=height, width=width)
    ds = gdal.Translate(dest, src, options=opts)
    ds = None
    opts = None


def get_raster_corners(src):
    """
    Returns the projected corner coordinates of the images located at src.
    Args:
        src: Path to a geo-located raster image.

    Returns: Coordinates of projected corners in same srs as src. Format is [ulx, uly, llx, lly, lrx, lry, urx, ury].
    """
    gdalSrc = gdal.Open(src, gdal.GA_ReadOnly)
    upx, xres, xskew, upy, yskew, yres = gdalSrc.GetGeoTransform()
    cols = gdalSrc.RasterXSize
    rows = gdalSrc.RasterYSize

    ulx = upx + 0 * xres + 0 * xskew
    uly = upy + 0 * yskew + 0 * yres

    llx = upx + 0 * xres + rows * xskew
    lly = upy + 0 * yskew + rows * yres

    lrx = upx + cols * xres + rows * xskew
    lry = upy + cols * yskew + rows * yres

    urx = upx + cols * xres + 0 * xskew
    ury = upy + cols * yskew + 0 * yres

    return [ulx, uly, llx, lly, lrx, lry, urx, ury]


def get_raster_extent(src):
    """
    Get the upper left and lower right projected coordinates of the image at path src.
    Args:
        src: Path to a geo-located raster image.

    Returns: Extent coordinates in same srs as src. Format is [ulx, uly, lrx, lry].

    DocTests:
    >>> get_raster_extent("../data/NW_Calvert/2016/20160803_Calvert_ChokedNorthBeach_georef_MOS_U0069_clipped.tif")
    [558095.357204558, 5726228.007563976, 561383.0880298427, 5723000.81666507]
    """
    ulx, uly, _, _, lrx, lry, _, _ = get_raster_corners(src)
    return [ulx, uly, lrx, lry]


def get_raster_srs(src):
    """
    Get the spatial reference system code for the geo-located image at path src.
    Args:
        src: Path to a geo-located raster image.

    Returns: EPSG code at a string

    DocTests:
    >>> get_raster_srs("../data/NW_Calvert/2016/20160803_Calvert_ChokedNorthBeach_georef_MOS_U0069_clipped.tif")
    '3156'
    """
    d = gdal.Open(src)
    proj = osr.SpatialReference(wkt=d.GetProjection())
    return proj.GetAttrValue('AUTHORITY', 1)


def shp2tiff(in_shp, out_tiff, ref_tiff, label_attr="label"):
    """
    Convert in_shp shapefile to geotiff using the ref_tiff projection and scale.
    Args:
        in_shp: Path to the shapefile to convert
        out_tiff: Path to save the raster file that is created
        ref_tiff: Path to a geo-referenced tiff to use as a reference for the desired srs and extent
        label_attr: The attr of the shp file to use to populate the raster pixel values

    Returns: None
    """
    gdal_format = 'GTiff'
    datatype = gdal.GDT_Byte

    # Get projection info from reference image
    image = gdal.Open(ref_tiff, gdal.GA_ReadOnly)

    # Open Shapefile
    shapefile = ogr.Open(in_shp)
    shapefile_layer = shapefile.GetLayer()

    # Rasterise
    output = gdal.GetDriverByName(gdal_format).Create(out_tiff, image.RasterXSize, image.RasterYSize, 1, datatype,
                                                      options=['COMPRESS=DEFLATE'])

    output.SetProjection(image.GetProjectionRef())
    output.SetGeoTransform(image.GetGeoTransform())

    # Write data to band 1
    band = output.GetRasterBand(1)
    band.SetNoDataValue(0)

    # rasterize_opts = gdal.RasterizeOptions({'attribute': label_attr, "noData": -1})
    ds = gdal.RasterizeLayer(output, [1], shapefile_layer, options=[f"ATTRIBUTE={label_attr}", f"NODATA=-1"])

    # Close datasets
    ds = None
    band = None
    output = None
    image = None
    shapefile = None


def filter_blank_images(dataset):
    imgs_dir = Path(dataset).joinpath("x")
    labels_dir = Path(dataset).joinpath("y")

    imgs = list(imgs_dir.glob("*.png"))
    print(len(list(imgs)), "Total images in dataset", dataset)

    removed = 0
    for img_path in tqdm(imgs, total=len(imgs)):
        img = Image.open(img_path)
        if img.getbbox() is None:
            removed += 1
            img_path.unlink()

    print(removed, "imgs removed")


def del_extra_labels(dataset):
    imgs_dir = Path(dataset).joinpath("x")
    labels_dir = Path(dataset).joinpath("y")

    imgs = list(imgs_dir.glob("*.png"))
    print(len(list(imgs)), "Total images in dataset", dataset)

    labels = list(labels_dir.glob("*.png"))
    print(len(list(labels)), "Total labels in dataset", dataset)

    img_names = [l.name for l in imgs]

    removed = 0
    for label in labels:
        if label.name not in img_names:
            removed += 1
            label.unlink()

    print(removed, "labels removed")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
