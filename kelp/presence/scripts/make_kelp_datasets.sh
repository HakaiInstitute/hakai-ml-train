THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_DIR=$(realpath "$THIS_DIR/../../..")
TILED_OUTPUT_DIR="/mnt/Scratch/Taylor/ml/kelp_presence_data"
RAW_IMAGES_DIR="/mnt/geospatial/Working/Taylor/2021/KelpML/raw_datasets/presence"
S3_BUCKET=s3://hakai-deep-learning-datasets/kelp
GCP_BUCKET=gs://hakai_kelp

DATASETS=(
AdamsFringe_kelp_U0665
AdamsHarbour_kelp_U0682
Boas_kelp_U0683
Breaker_kelp_U0663
ChokedSouth_kelp_U0667
Edna_kelp_U0722
Golden_kelp_U0656
Golden_kelp_U0796
ManleyWomanley_kelp_U0652
ManleyWomanley_kelp_U0775
McMullinsCenter_kelp_U0728
McMullinsNereo_kelp_U0729
Meay_kelp_U0659
Meay_kelp_U0778
NorthBeach_kelp_U0669
NorthBeach_kelp_U0770
Odlum_kelp_U0681
Simmonds_kelp_MS_U0794
Simmonds_kelp_U0650
Starfish_kelp_U0666
Stirling_kelp_U0661
Stryker_kelp_U0655
Stryker_kelp_U0797
SurfPass_kelp_U0668
Triquet_kelp_U0653
Triquet_kelp_U0776
WestBeach_kelp_U0772
)

# Execute all following instructions in the uav-classif/deeplabv3/kelp directory
# Uncomment to rebuild all datasets
rm -rf "$TILED_OUTPUT_DIR/train"
rm -rf "$TILED_OUTPUT_DIR/eval"

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
  gdal_edit.py "./image.tif" -a_nodata 0  # Assumes that background is black

  # Remove extra bands and BGRI -> RGBI
  echo "Removing extra bands and reordering"
  gdal_translate \
    -scale \
    -b 3 -b 2 -b 1 \
    -ot 'Float32' \
    "./$DATASET/image.tif" "./$DATASET/image_4band.tif"

  # Normalize color and background data
  echo "Normalizing image colors"
  echo "Percentile min-max scaling"
  low=$(python "$PROJECT_DIR/utils/data_prep/img_stats.py" "percentile_low" "./$DATASET/image_4band.tif")
  echo "LOW: $low"
  high=$(python "$PROJECT_DIR/utils/data_prep/img_stats.py" "percentile_high" "./$DATASET/image_4band.tif")
  echo "HIGH: $high"
  gdal_calc.py \
    -A "./$DATASET/image_4band.tif" \
    --allBands A \
    --NoDataValue=0 \
    --calc="(A - ${low}) / (${high} - ${low})" \
    --outfile="./$DATASET/image_float.tif" \
    --type=Float32 \
    --overwrite
  rm "./$DATASET/image_4band.tif"

  # Mean normalize
  echo "Mean-Std Scaling"
  python "$PROJECT_DIR/utils/data_prep/normalize.py" mean_std_scale "./$DATASET/image_float.tif" "./$DATASET/image_rgbi.tif"
  rm "./$DATASET/image_float.tif"

#  # Scale to Byte
#  gdal_translate \
#    -scale 0 2 0 255 \
#    -b 1 -b 2 -b 3 -b 4 \
#    -ot 'Byte' \
#    "./$DATASET/image_float.tif" "./$DATASET/image_rgbi.tif"
#  rm "./$DATASET/image_float.tif"

#  # Convert all CRS to EPSG:4326 WGS84
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
  ul=$(gdalinfo "$DATASET/image_rgbi.tif" | grep "Upper Left")
  lr=$(gdalinfo "$DATASET/image_rgbi.tif" | grep "Lower Right")
  x_min=$(echo "$ul" | grep -P '(?<=\()([-]?\d{1,3}\.\d+)(?=.*)' -o)
  x_max=$(echo "$lr" | grep -P '(?<=\()([-]?\d{1,3}\.\d+)(?=.*)' -o)
  y_min=$(echo "$lr" | grep -P '(?<=,\s{2})([-]?\d{1,3}\.\d+)(?=.*)' -o)
  y_max=$(echo "$ul" | grep -P '(?<=,\s{2})([-]?\d{1,3}\.\d+)(?=.*)' -o)

  # Get the image res
  res=$(gdalinfo "$DATASET/image_rgbi.tif" | grep "Pixel Size")
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
#    rm "./$DATASET/label.tif"

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
    "./$DATASET/image_rgbi.tif" "./$DATASET/${DATASET}.tif"
  rm "./$DATASET/image_rgbi.tif"

  # Set (values > 3 or values <= 0) to 0
  echo "Cleaning label values"
  gdal_calc.py \
    -A"./$DATASET/label_res.tif" \
    --overwrite \
    --calc="where(logical_and(nan_to_num(A)>0, nan_to_num(A)<=3), nan_to_num(A), 0)" \
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

  echo "Removing unmatched images and labels."
  python "$PROJECT_DIR/utils/data_prep/filter_datasets.py" "delete_extra_labels" "$DATASET"
  python "$PROJECT_DIR/utils/data_prep/filter_datasets.py" "delete_extra_imgs" "$DATASET"

  # Split to train/test set
  echo "Splitting to 70/30 train/test sets"
  python "$PROJECT_DIR/utils/data_prep/train_test_split.py" "$DATASET" "$TILED_OUTPUT_DIR" --train_size=0.7
done

# Upload data to cloud
gsutil -m cp -r "$TILED_OUTPUT_DIR/train" "${GCP_BUCKET}/train"
gsutil -m cp -r "$TILED_OUTPUT_DIR/eval" "${GCP_BUCKET}/eval"

aws s3 sync "$TILED_OUTPUT_DIR/train" "${S3_BUCKET}/train"
aws s3 sync "$TILED_OUTPUT_DIR/eval" "${S3_BUCKET}/eval"

cd - || exit 1
