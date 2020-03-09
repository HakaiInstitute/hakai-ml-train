# A script to rasterise a shapefile to the same projection & pixel resolution as a reference image.
from osgeo import gdal, ogr


def shp2tiff(in_shp, out_tiff, ref_tiff, label_attr="label"):
    """Convert in_shp shapefile to geotiff using the ref_tiff projection and scale."""
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


if __name__ == '__main__':
    shp_f = '../data/NW_Calvert/2016/2016_Kelp_Extent_KH_May15_2017.shp'
    out_f = '../data/kelp.py.tif'
    ref_f = '../data/NW_Calvert/2016/20160803_Calvert_ChokedNorthBeach_georef_MOS_U0069.tif'

    shp2tiff(shp_f, out_f, ref_f)
