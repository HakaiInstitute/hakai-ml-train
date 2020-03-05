from osgeo import gdal, osr, ogr
import rasterio
from pathlib import Path
import os
from tqdm.auto import tqdm, trange
import itertools
from multiprocessing import Semaphore, Process


def _crop_to_png(sem, dest, src, x0, y0, crop_size):
    sem.acquire()
    options_list = [
        '-ot Byte',
        '-of PNG',
        '-scale',
        f'-srcwin {x0} {y0} {crop_size} {crop_size}'
    ]
    gdal.Translate(dest, src, options=" ".join(options_list))
    sem.release()


def slice_and_dice_image(src_img, src_labels, dest_x, dest_y, crop_size=200, cpus=os.cpu_count()):
    semaphore = Semaphore(value=cpus)

    # Create small image dataset
    dest_x = Path(dest_x)
    dest_y = Path(dest_y)

    # Get X and Y dimensions of image
    with rasterio.open(src_img) as img:
        w = img.width
        h = img.height

    with rasterio.open(src_labels) as img:
        if w != img.width or h != img.height:
            raise RuntimeError("Label dataset must have the same extent as the image dataset")

    x0s = list(range(0, w, crop_size))
    y0s = list(range(0, h, crop_size))
    origins = itertools.product(x0s, y0s)

    for i, (x0, y0) in tqdm(enumerate(origins), total=len(x0s) * len(y0s)):
        # Crop label sections and save
        dest = str(dest_x.joinpath(f"{i}.png"))
        Process(target=_crop_to_png, args=(semaphore, dest, src_img, x0, y0, crop_size)).start()

        dest = str(dest_y.joinpath(f"{i}.png"))
        Process(target=_crop_to_png, args=(semaphore, dest, src_labels, x0, y0, crop_size)).start()


def clip_raster_with_shp_mask(dest, src, mask):
    opts = gdal.WarpOptions(format="GTiff", cutlineDSName=mask, cutlineLayer=Path(mask).stem, cropToCutline=True)
    gdal.Warp(dest, src, options=opts)


def clip_raster_by_extent(dest, src, extent):
    """
    Clips raster at location src to extent and saves at location dest
    :param dest: Path to save clipped raster
    :param src: Source raster to clip
    :param extent: Either an array of extent in format [ulx, uly, lrx, lry]
    :return: None
    """
    opts = gdal.TranslateOptions(format="GTiff", projWin=extent)
    gdal.Translate(dest, src, options=opts)


def get_raster_corners(src):
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
    >>> get_raster_extent("../data/NW_Calvert/2016/20160803_Calvert_ChokedNorthBeach_georef_MOS_U0069_clipped.tif")
    [558095.357204558, 5726228.007563976, 561383.0880298427, 5723000.81666507]
    """
    ulx, uly, _, _, lrx, lry, _, _ = get_raster_corners(src)
    return [ulx, uly, lrx, lry]


def get_raster_crs(src):
    """
    >>> get_raster_crs("../data/NW_Calvert/2016/20160803_Calvert_ChokedNorthBeach_georef_MOS_U0069_clipped.tif")
    '3156'
    """
    d = gdal.Open(src)
    proj = osr.SpatialReference(wkt=d.GetProjection())
    return proj.GetAttrValue('AUTHORITY', 1)


if __name__ == "__main__":
    import doctest
    doctest.testmod()