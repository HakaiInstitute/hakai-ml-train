from osgeo import gdal, osr, ogr
import rasterio
from pathlib import Path
import os
import itertools
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm


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


def _crop_to_png(src, dest, x0, y0, crop_size):
    """
    Crop the raster at location src and save at location dest to square section of length and width crop_size from
    origin (x0, y0). Crops from src coordinates, not projected coordinates
    Args:
        src: Path to the image to crop
        dest: Path to save the cropped image to
        x0: Leftmost x coordinate to crop to
        y0: Topmost y coordinate to crop to
        crop_size: The length and width of the cropped section

    Returns: None
    """
    options = gdal.TranslateOptions(outputType=gdal.GDT_Byte, format="PNG", srcWin=[x0, y0, crop_size, crop_size])
    gdal.Translate(dest, src, options=options)


def slice_and_dice_image(src_img, dest_d, crop_size=200, cpus=os.cpu_count()):
    """
    Create a machine learning dataset of image patches. Chops src_img into square sections of length/width crop_size
    and saves the patches and intermediate files to direction dest_d. Uses `cpus` count of cpus to process image in
    parallel.
    Args:
        src_img: Path to the image to crop into square sections
        dest_d: Path to directory to save the processed image
        crop_size: The length and width of cropped sections to create
        cpus: The number of cpus to parallelize the processing task on. Defaults to all available

    Returns: None
    """
    # Create small image dataset
    dest_d = Path(dest_d)

    # Get X and Y dimensions of image
    with rasterio.open(src_img) as img:
        w = img.width
        h = img.height

    x0s = list(range(0, w, crop_size))
    y0s = list(range(0, h, crop_size))
    origins = tuple(itertools.product(x0s, y0s))

    # Crop img sections and save
    # progress = tqdm(total=len(origins))
    # with ProcessPoolExecutor(max_workers=cpus) as pool:
    #     futures = []
    #     for i, (x0, y0) in enumerate(origins):
    #         dest = str(dest_d.joinpath(f"{i}.png"))
    #         future = pool.submit(_crop_to_png, src_img, dest, x0, y0, crop_size)
    #         future.add_done_callback(lambda p: progress.update())
    #         futures.append(future)
    #
    #     for future in futures:
    #         try:
    #             result = future.result()
    #         except Exception as e:
    #             print(e)
    #             import sys
    #             sys.exit(1)

    for i, (x0, y0) in enumerate(tqdm(origins, total=len(origins))):
        dest = str(dest_d.joinpath(f"{i}.png"))
        _crop_to_png(src_img, dest, x0, y0, crop_size)

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
    gdal.Warp(dest, src, options=opts)


def clip_raster_by_extent(dest, src, extent):
    """
    Clips raster at location src to extent and saves at location dest.
    Args:
        dest: Path to save clipped raster
        src: Source raster to clip
        extent: Either an array of extent in format [ulx, uly, lrx, lry]

    Returns: None
    """
    opts = gdal.TranslateOptions(format="GTiff", projWin=extent)
    gdal.Translate(dest, src, options=opts)


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
    gdal.RasterizeLayer(output, [1], shapefile_layer, options=[f"ATTRIBUTE={label_attr}", f"NODATA=-1"])

    # Close datasets
    band = None
    output = None
    image = None
    shapefile = None


if __name__ == "__main__":
    import doctest
    doctest.testmod()
