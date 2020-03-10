from osgeo import gdal, ogr


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
