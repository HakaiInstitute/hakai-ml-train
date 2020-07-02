THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_DIR=$(realpath "$THIS_DIR/../..")
WORKING_DIR="/mnt/Z/kelp_presence_data"
S3_BUCKET=s3://hakai_deep_learning_dataset/kelp

DATASETS=(
  choked_2014
  golden_monitoring_2018
  manley_kildidt_2016
  mcnaughton_2017
  nw_calvert_2015
  simmonds_area_kelp_2019
  simmonds_monitoring_2018
  stryker_monitoring_2018
  stryker_monitoring_low_2018
  triquet_2019
  west_beach_2016
)

# Mount the samba data server
#sudo mkdir -p /mnt/H
#sudo mount -t cifs -o user=taylor.denouden,domain=victoria.hakai.org //10.10.1.50/Geospatial /mnt/H

# Execute all following instructions in the uav-classif/deeolabv3/kelp directory
mkdir -p "$WORKING_DIR/raw_data"
cd "$WORKING_DIR/raw_data" || exit 1

for DATASET in "${DATASETS[@]}"; do
  mkdir -p "$DATASET"
done

# Copy all the image tifs files to local drive with multiprocessing
echo "${DATASETS[@]}" | tr -d '\n' | xargs -d ' ' -i -P8 sh -c 'cp -u -v /mnt/H/Working/Taylor/KelpPresence/{}/image.* ./{}/'

# Copy all the kelp tifs files to local drive with multiprocessing
echo "${DATASETS[@]}" | tr -d '\n' | xargs -d ' ' -i -P8 sh -c 'cp -u -v /mnt/H/Working/Taylor/KelpPresence/{}/kelp.* ./{}/'

# Convert dataset to the cropped format
for DATASET in "${DATASETS[@]}"; do
  printf "\nProcessing dataset: %s\n" "$DATASET" | cat
  echo "---------------------------------------------------------------"

  # Remove any weird noData values
  echo "Un-setting noData values"
  gdal_edit.py "./$DATASET/kelp.tif" -unsetnodata
  gdal_edit.py "./$DATASET/image.tif" -unsetnodata

  # Convert all CRS to EPSG:4326 WGS84
  echo "Converting image CRS"
  gdalwarp \
    -wo NUM_THREADS=ALL_CPUS \
    -multi \
    -t_srs EPSG:4326 \
    -r near \
    -of GTiff \
    -overwrite \
    "./$DATASET/image.tif" "./$DATASET/image_wgs.tif"
  #  rm "./$DATASET/image.tif"

  # Get image extent
  ul=$(gdalinfo "$DATASET/image_wgs.tif" | grep "Upper Left")
  lr=$(gdalinfo "$DATASET/image_wgs.tif" | grep "Lower Right")
  x_min=$(echo "$ul" | grep -P '(?<=\()([-]?\d{1,3}\.\d+)(?=.*)' -o)
  x_max=$(echo "$lr" | grep -P '(?<=\()([-]?\d{1,3}\.\d+)(?=.*)' -o)
  y_min=$(echo "$lr" | grep -P '(?<=,\s{2})([-]?\d{1,3}\.\d+)(?=.*)' -o)
  y_max=$(echo "$ul" | grep -P '(?<=,\s{2})([-]?\d{1,3}\.\d+)(?=.*)' -o)

  # Get the image res
  res=$(gdalinfo "$DATASET/image_wgs.tif" | grep "Pixel Size")
  x_res=$(echo "$res" | grep -P '(?<=\()([-]?\d\.\d+)' -o)
  y_res=$(echo "$res" | grep -P '(?<=,)([-]?\d\.\d+)' -o)

  echo "Converting label CRS and adjusting extent"
  gdalwarp \
    -wo NUM_THREADS=ALL_CPUS \
    -multi \
    -t_srs EPSG:4326 \
    -te "$x_min" "$y_min" "$x_max" "$y_max" \
    -r near \
    -of GTiff \
    -overwrite \
    -tr "$x_res" "$y_res" \
    -tap \
    "./$DATASET/kelp.tif" "./$DATASET/kelp_wgs.tif"
  #  rm "./$DATASET/kelp.tif"

  # Get kelp extent
  x_min=$(gdalinfo "$DATASET/kelp_wgs.tif" | grep "Upper Left" | grep -P '(?<=\()([-]?\d{1,3}\.\d+)(?=.*)' -o) \
  x_max=$(gdalinfo "$DATASET/kelp_wgs.tif" | grep "Upper Right" | grep -P '(?<=\()([-]?\d{1,3}\.\d+)(?=.*)' -o) \
  y_min=$(gdalinfo "$DATASET/kelp_wgs.tif" | grep "Lower Left" | grep -P '(?<=,\s{2})([-]?\d{1,3}\.\d+)(?=.*)' -o) \
  y_max=$(gdalinfo "$DATASET/kelp_wgs.tif" | grep "Upper Left" | grep -P '(?<=,\s{2})([-]?\d{1,3}\.\d+)(?=.*)' -o)

  echo "Adjusting image extent"
  gdalwarp \
    -wo NUM_THREADS=ALL_CPUS \
    -multi \
    -te_srs EPSG:4326 \
    -te "$x_min" "$y_min" "$x_max" "$y_max" \
    -of GTiff \
    -overwrite \
    "./$DATASET/image_wgs.tif" "./$DATASET/${DATASET}.tif"
  rm "./$DATASET/image_wgs.tif"

  # Set values above 1 to 0 as well as set nodata values (i.e 255) to 0
  echo "Cleaning label values"
  gdal_calc.py \
    -A "./$DATASET/kelp_wgs.tif" \
    --overwrite \
    --calc="nan_to_num(A*(A<2))" \
    --type="Byte" \
    --outfile="./$DATASET/label_${DATASET}.tif"
  rm "./$DATASET/kelp_wgs.tif"

  # Dice up the image
  echo "Creating image tiles"
  rm -rf "$DATASET/x"
  mkdir -p "$DATASET/x"
  gdal_retile.py \
    -ps 512 512 \
    -overlap 256 \
    -ot Byte \
    -targetDir "$DATASET/x" \
    "$DATASET/${DATASET}.tif"

  # Dice up the label
  echo "Creating label tiles"
  rm -rf "$DATASET/y"
  mkdir -p "$DATASET/y"
  gdal_retile.py \
    -ps 512 512 \
    -overlap 256 \
    -ot Byte \
    -targetDir "$DATASET/y" \
    "$DATASET/label_${DATASET}.tif"

  # Filter tiles
  echo "Deleting tile pairs containing only the BG class"
  python "$PROJECT_DIR/utils/data_prep/filter_datasets.py" "bg_only_labels" "$DATASET"

  echo "Deleting tile pairs with blank image data"
  python "$PROJECT_DIR/utils/data_prep/filter_datasets.py" "blank_imgs" "$DATASET"

  # Split to train/test set
  echo "Splitting to 80/20 train/test sets"
  rm -rf "$WORKING_DIR/train"
  rm -rf "$WORKING_DIR/eval"
  python "$PROJECT_DIR/utils/data_prep/train_test_split.py" "$DATASET" "$WORKING_DIR" --train_size=0.8
done

# Upload data to S3
aws s3 rsync "$WORKING_DIR/train" "${S3_BUCKET}/train"
aws s3 rsync "$WORKING_DIR/eval" "${S3_BUCKET}/eval"

cd - || exit 1
