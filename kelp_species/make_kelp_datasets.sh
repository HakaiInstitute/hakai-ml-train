THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
echo $THIS_DIR
PROJECT_DIR=$(realpath "$THIS_DIR/..")
echo $PROJECT_DIR
WORKING_DIR="/mnt/Z/kelp_species_data"

# Mount the samba data server
#sudo mkdir -p /mnt/H
#sudo mount -t cifs -o user=taylor.denouden,domain=victoria.hakai.org //10.10.1.50/Geospatial /mnt/H

# Execute all following instructions in the uav-classif/deeolabv3/kelp directory
mkdir -p "$WORKING_DIR/raw_data"
cd "$WORKING_DIR/raw_data" || exit 1

mkdir -p \
  choked_2014 \
  choked_2016 \
  crabapple_2014 \
  manly_kildidt_2016 \
  nw_calvert_2012 \
  nw_calvert_2015 \
  west_beach_2014 \
  west_beach_2016

# Copy all the image tifs to local drive with multiprocessing
echo "choked_2014 choked_2016 crabapple_2014 manly_kildidt_2016 nw_calvert_2012 nw_calvert_2015 \
  west_beach_2014 west_beach_2016" | xargs -n1 -P8 sh -c \
  'fname="image"; \
cp -u -v "/mnt/H/Working/Taylor/KelpSpecies/$1/$fname.tfw" "./$1/$fname.tfw"; \
cp -u -v "/mnt/H/Working/Taylor/KelpSpecies/$1/$fname.tif" "./$1/$fname.tif"; \
cp -u -v "/mnt/H/Working/Taylor/KelpSpecies/$1/$fname.tif.aux.xml" "./$1/$fname.tif.aux.xml"; \
cp -u -v "/mnt/H/Working/Taylor/KelpSpecies/$1/$fname.tif.xml" "./$1/$fname.tif.xml"' sh

# Copy all the kelp tifs to local drive with multiprocessing
echo "choked_2014 choked_2016 crabapple_2014 manly_kildidt_2016 nw_calvert_2012 nw_calvert_2015 \
  west_beach_2014 west_beach_2016" | xargs -n1 -P8 sh -c \
  'fname="kelp"; \
cp -u -v "/mnt/H/Working/Taylor/KelpSpecies/$1/$fname.tfw" "./$1/$fname.tfw"; \
cp -u -v "/mnt/H/Working/Taylor/KelpSpecies/$1/$fname.tif" "./$1/$fname.tif"; \
cp -u -v "/mnt/H/Working/Taylor/KelpSpecies/$1/$fname.tif.aux.xml" "./$1/$fname.tif.aux.xml"; \
cp -u -v "/mnt/H/Working/Taylor/KelpSpecies/$1/$fname.tif.xml" "./$1/$fname.tif.xml"' sh

# Convert dataset to the cropped format
# shellcheck disable=SC1090
source "$HOME/anaconda3/bin/activate uav"

# conda activate uav
for DIR_NAME in choked_2014 choked_2016 crabapple_2014 manly_kildidt_2016 nw_calvert_2012 nw_calvert_2015 west_beach_2014 west_beach_2016; do
  # Remove any weird noData values
  gdal_edit.py "./$DIR_NAME/kelp.tif" -unsetnodata
  gdal_edit.py "./$DIR_NAME/image.tif" -unsetnodata

  # Let any pixel > 0 be kelp and set to value 1
  gdal_calc.py -A "./$DIR_NAME/kelp.tif" --outfile="./$DIR_NAME/kelp_scaled.tif" --overwrite \
    --calc="nan_to_num(A==1)" --type="Byte"
  rm "./$DIR_NAME/kelp.tif"

  # Convert all CRS to EPSG:4326 WGS84
  gdalwarp -t_srs EPSG:4326 -r near -of GTiff -overwrite "./$DIR_NAME/image.tif" "./$DIR_NAME/image_wgs.tif"
  rm "./$DIR_NAME/image.tif"

  gdalwarp -t_srs EPSG:4326 -r near -of GTiff -overwrite "./$DIR_NAME/kelp_scaled.tif" "./$DIR_NAME/kelp_wgs.tif"
  rm "./$DIR_NAME/kelp_scaled.tif"

  python "$PROJECT_DIR/utils/dice_kelp_img_and_label.py" \
    "./$DIR_NAME/image_wgs.tif" \
    "./$DIR_NAME/kelp_wgs.tif" \
    "./$DIR_NAME" \
    --crop_size=512 \
    --stride=256
done

python "$PROJECT_DIR/utils/combine_filter_upload_kelp_data.py" \
  choked_2014 \
  choked_2016 \
  crabapple_2014 \
  manly_kildidt_2016 \
  nw_calvert_2012 \
  nw_calvert_2015 \
  west_beach_2014 \
  west_beach_2016 \
  --out_dir=../train_input/data --dataset_name="kelp_species"

cd - || exit 1
