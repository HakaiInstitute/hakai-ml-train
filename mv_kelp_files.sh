# Mount the samba data server
#sudo mkdir -p /mnt/H
#sudo mount -t cifs -o user=taylor.denouden,domain=victoria.hakai.org //10.10.1.50/Geospatial /mnt/H

# Execute all following instructions in the uav-classif/data/kelp/raw directory
mkdir -p nw_calvert_2012 nw_calvert_2015 choked_pass_2016 west_beach_2016 mcnaughton_2017

# Copy all the image tifs to local drive with multiprocessing
echo "nw_calvert_2012	/mnt/H/Internal/RS/Airborne/Air_Photos/Orthophotos_Processed/Calvert_Island_2012/Calvert_ortho_2012_Web_NAD83 \
nw_calvert_2015	/mnt/H/Internal/RS/UAV/Files/Calvert/2015/20150812_NWCalvertFinal_U0015/Products/calvert_nwcalvert15_CSRS_mos_U0015 \
choked_pass_2016	/mnt/H/Internal/RS/UAV/Files/Calvert/2016/20160803_choked_U0069/Products/20160803_Calvert_ChokedNorthBeach_georef_MOS_U0069 \
west_beach_2016	/mnt/H/Internal/RS/UAV/Files/Calvert/2016/20160804_westbeach_U0070/Products/20160804_Calvert_WestBeach_Georef_mos_U0070 \
mcnaughton_2017	/mnt/H/Internal/RS/UAV/Files/CentralCoast/2017/20170527_McNaughtonGroup_U0168/Products/CentralCoast_McNaughtonGroup_MOS_U0168" | xargs -n2 -P8 sh -c \
  'fname="image"; \
cp -u -v "$2.tfw" "./$1/$fname.tfw"; \
cp -u -v "$2.tif" "./$1/$fname.tif"; \
cp -u -v "$2.tif.aux.xml" "./$1/$fname.tif.aux.xml"; \
cp -u -v "$2.tif.xml" "./$1/$fname.tif.xml"' sh

# Copy all the kelp tifs to local drive with multiprocessing
echo "nw_calvert_2012	/mnt/H/Working/For_Taylor/2012_Kelp/2012_Kelp_Water_RC_1 \
nw_calvert_2015	/mnt/H/Working/For_Taylor/2015_Kelp/2015_U0015_kelp \
choked_pass_2016	/mnt/H/Working/For_Taylor/2016_Kelp/2016_U069_Kelp_RC_1 \
west_beach_2016	/mnt/H/Working/For_Taylor/2016_Kelp/2016_U070_Kelp_RC_1 \
mcnaughton_2017	/mnt/H/Working/For_Taylor/McNaughtons_U0168/McNaughton_kelp" | xargs -n2 -P8 sh -c \
  'fname="kelp"; \
cp -u -v "$2.tfw" "./$1/$fname.tfw"; \
cp -u -v "$2.tif" "./$1/$fname.tif"; \
cp -u -v "$2.tif.aux.xml" "./$1/$fname.tif.aux.xml"; \
cp -u -v "$2.tif.xml" "./$1/$fname.tif.xml"' sh

# Convert dataset to the cropped format NOTE: McNaughton 2017 retained for qualitative validation
# conda activate uav
for DIR_NAME in nw_calvert_2012 nw_calvert_2015 choked_pass_2016 west_beach_2016; do
  # Remove any weird noData values
  gdal_edit.py "./$DIR_NAME/kelp.tif" -unsetnodata

  # Let any pixel > 0 be kelp and set to value 1
  gdal_calc.py -A "./$DIR_NAME/kelp.tif" --outfile="./$DIR_NAME/kelp_scaled.tif" --calc="A==1" --type="Byte"
  rm "./$DIR_NAME/kelp.tif"

  # Convert all CRS to EPSG:4326 WGS84
  gdalwarp -t_srs EPSG:4326 -r near -of GTiff "./$DIR_NAME/image.tif" "./$DIR_NAME/image_wgs.tif"
  rm "./$DIR_NAME/image.tif"

  gdalwarp -t_srs EPSG:4326 -r near -of GTiff "./$DIR_NAME/kelp_scaled.tif" "./$DIR_NAME/kelp_wgs.tif"
  rm "./$DIR_NAME/kelp_scaled.tif"

  python ../../../make_ml_ds_raster.py \
    "./$DIR_NAME/image_wgs.tif" \
    "./$DIR_NAME/kelp_wgs.tif" \
    "../processed/$DIR_NAME" \
    --crop_size=513
done

python ../../../sagemaker_kelp_dataset.py \
  ../processed/nw_calvert_2012 \
  ../processed/nw_calvert_2015 \
  ../processed/choked_pass_2016 \
  ../processed/west_beach_2016 \
  - --out_dir=../../../deeplabv3/kelp/train_input/data
