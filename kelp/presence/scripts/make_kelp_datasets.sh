THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_DIR=$(realpath "$THIS_DIR/../../..")
TILED_OUTPUT_DIR="/mnt/Scratch/Taylor/ml/kelp_presence_data"
RAW_IMAGES_DIR="/mnt/geospatial/Working/Taylor/2021/KelpML/raw_datasets/presence"
S3_BUCKET=s3://hakai-deep-learning-datasets/kelp

DATASETS=(
Simmonds_kelp_MS_U0794
Stryker_kelp_U0797
Golden_kelp_U0796
Triquet_kelp_U0776
ManleyWomanley_kelp_U0775
NorthBeach_kelp_U0770
WestBeach_kelp_U0772
Meay_kelp_U0778
McMullinsCenter_kelp_U0728
Edna_kelp_U0722
SurfPass_kelp_U0668
Stirling_kelp_U0661
Starfish_kelp_U0666
Odlum_kelp_U0681
NorthBeach_kelp_U0669
Meay_kelp_U0659
ChokedSouth_kelp_U0667
Breaker_kelp_U0663
Boas_kelp_U0663
AdamsHarbour_kelp_U0682
AdamsFringe_kelp_U0665
McMullinsNereo_kelp_U0729
Stryker_kelp_U0655
Simmonds_kelp_U0650
Golden_kelp_U0656
Triquet_kelp_U0653
ManleyWomanley_kelp_U0652
)

# Execute all following instructions in the uav-classif/deeplabv3/kelp directory
# Uncomment to rebuild all datasets
#rm -rf "$TILED_OUTPUT_DIR/train"
#rm -rf "$TILED_OUTPUT_DIR/eval"

mkdir -p "$TILED_OUTPUT_DIR/raw_data"
cd "$TILED_OUTPUT_DIR/raw_data" || exit 1

for DATASET in "${DATASETS[@]}"; do
  mkdir -p "$DATASET"
done

# Copy all the image tifs files to local drive with multiprocessing
echo "${DATASETS[@]}" | tr -d '\n' | xargs -d ' ' -i -P8 sh -c "cp -u -v $RAW_IMAGES_DIR/{}/image.* ./{}/"

# Copy all the kelp tifs files to local drive with multiprocessing
echo "${DATASETS[@]}" | tr -d '\n' | xargs -d ' ' -i -P8 sh -c "cp -u -v $RAW_IMAGES_DIR/{}/label.* ./{}/"

# Convert dataset to the cropped format
for DATASET in "${DATASETS[@]}"; do
  printf "\nProcessing dataset: %s\n" "$DATASET" | cat
  echo "---------------------------------------------------------------"

  # Remove any weird noData values
  echo "Un-setting noData values"
  gdal_edit.py "./$DATASET/label.tif" -unsetnodata
  gdal_edit.py "./$DATASET/image.tif" -unsetnodata

  # Downscale images to U8 and keep only first 4 bands
  echo "Downscaling bit depth"
  gdal_translate \
    -scale \
    -b 1 -b 2 -b 3 -b 4 \
    -ot 'Byte' \
    "./$DATASET/image.tif" "./$DATASET/image_u8.tif"

  # Convert all CRS to EPSG:4326 WGS84
#  echo "Converting image CRS"
#  gdalwarp \
#    -wo NUM_THREADS=ALL_CPUS \
#    -multi \
#    -t_srs EPSG:4326 \
#    -r near \
#    -of GTiff \
#    -overwrite \
#    "./$DATASET/image_u8.tif" "./$DATASET/image_u8.tif"
#    rm "./$DATASET/image_u8.tif"

  # Get image extent
  ul=$(gdalinfo "$DATASET/image_u8.tif" | grep "Upper Left")
  lr=$(gdalinfo "$DATASET/image_u8.tif" | grep "Lower Right")
  x_min=$(echo "$ul" | grep -P '(?<=\()([-]?\d{1,3}\.\d+)(?=.*)' -o)
  x_max=$(echo "$lr" | grep -P '(?<=\()([-]?\d{1,3}\.\d+)(?=.*)' -o)
  y_min=$(echo "$lr" | grep -P '(?<=,\s{2})([-]?\d{1,3}\.\d+)(?=.*)' -o)
  y_max=$(echo "$ul" | grep -P '(?<=,\s{2})([-]?\d{1,3}\.\d+)(?=.*)' -o)

  # Get the image res
  res=$(gdalinfo "$DATASET/image_u8.tif" | grep "Pixel Size")
  x_res=$(echo "$res" | grep -P '(?<=\()([-]?\d\.\d+)' -o)
  y_res=$(echo "$res" | grep -P '(?<=,)([-]?\d\.\d+)' -o)

  echo "Adjusting label extent"
  gdalwarp \
    -wo NUM_THREADS=ALL_CPUS \
    -multi \
    -te "$x_min" "$y_min" "$x_max" "$y_max" \
    -r near \
    -of GTiff \
    -overwrite \
    -tr "$x_res" "$y_res" \
    -tap \
    "./$DATASET/label.tif" "./$DATASET/label_res.tif"
    rm "./$DATASET/label.tif"

  # Get label extent
  ul=$(gdalinfo "$DATASET/label_res.tif" | grep "Upper Left")
  lr=$(gdalinfo "$DATASET/label_res.tif" | grep "Lower Right")
  x_min=$(echo "$ul" | grep -P '(?<=\()([-]?\d{1,3}\.\d+)(?=.*)' -o)
  x_max=$(echo "$lr" | grep -P '(?<=\()([-]?\d{1,3}\.\d+)(?=.*)' -o)
  y_min=$(echo "$lr" | grep -P '(?<=,\s{2})([-]?\d{1,3}\.\d+)(?=.*)' -o)
  y_max=$(echo "$ul" | grep -P '(?<=,\s{2})([-]?\d{1,3}\.\d+)(?=.*)' -o)

  echo "Adjusting image extent"
  gdalwarp \
    -wo NUM_THREADS=ALL_CPUS \
    -multi \
    -te "$x_min" "$y_min" "$x_max" "$y_max" \
    -of GTiff \
    -overwrite \
    "./$DATASET/image_u8.tif" "./$DATASET/${DATASET}.tif"
  rm "./$DATASET/image_u8.tif"

  # Set values >= 10 to 1, else set to 0
  echo "Cleaning label values"
  gdal_calc.py \
    -A "./$DATASET/label_res.tif" \
    --overwrite \
    --calc="where(logical_and(nan_to_num(A)>0, nan_to_num(A)<=10), 1, 0)" \
    --type="Byte" \
    --outfile="./$DATASET/label_${DATASET}.tif"
  rm "./$DATASET/label_res.tif"

  # Dice up the image
  echo "Creating image tiles"
  rm -rf "$DATASET/x"
  mkdir -p "$DATASET/x"
  gdal_retile.py \
    -ps 512 512 \
    -overlap 256 \
    -ot Byte \
    -of PNG \
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
    -of PNG \
    -targetDir "$DATASET/y" \
    "$DATASET/label_${DATASET}.tif"

  # Filter tiles
  echo "Deleting tile pairs containing only the BG class"
  python "$PROJECT_DIR/utils/data_prep/filter_datasets.py" "bg_only_labels" "$DATASET"

  echo "Deleting tile pairs with blank image data"
  python "$PROJECT_DIR/utils/data_prep/filter_datasets.py" "blank_imgs" "$DATASET"

  echo "Deleting tile pairs less than half the required size"
  python "$PROJECT_DIR/utils/data_prep/filter_datasets.py" "skinny_labels" "$DATASET" --min_height=256 --min_width=256

  # Pad images that aren't 512 x 512 shaped
  echo "Padding incorrectly shaped images."
  python "$PROJECT_DIR/utils/data_prep/preprocess_chips.py" "expand_chips" "$DATASET" --size=512

  # Strip any channels that aren't the first 3 RGB channels.
  echo "Stripping extra channels."
  python "$PROJECT_DIR/utils/data_prep/preprocess_chips.py" "strip_extra_channels" "$DATASET"

  # Split to train/test set
  echo "Splitting to 70/30 train/test sets"
  python "$PROJECT_DIR/utils/data_prep/train_test_split.py" "$DATASET" "$TILED_OUTPUT_DIR" --train_size=0.7
done

# Upload data to S3
aws s3 sync "$TILED_OUTPUT_DIR/train" "${S3_BUCKET}/train"
aws s3 sync "$TILED_OUTPUT_DIR/eval" "${S3_BUCKET}/eval"

cd - || exit 1
