THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_DIR=$(realpath "$THIS_DIR/../..")
TILED_OUTPUT_DIR="/mnt/Scratch/Taylor/ml/mussels_presence_data"
RAW_IMAGES_DIR="/mnt/geospatial/Working/Taylor/2021/MusselsML/raw_datasets"
S3_BUCKET=s3://hakai-deep-learning-datasets/mussels

DATASETS=(
  mussels_0631
  mussels_0624
  mussels_0630
  mussels_0755
  mussels_0752
  mussels_0354
  mussels_0380
  mussels_0351
  mussels_0754
  mussel_0539
  mussel_0629
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

# Copy all the label tifs files to local drive with multiprocessing
echo "${DATASETS[@]}" | tr -d '\n' | xargs -d ' ' -i -P8 sh -c "cp -u -v $RAW_IMAGES_DIR/{}/label.* ./{}/"

# Convert dataset to the cropped format
for DATASET in "${DATASETS[@]}"; do
  printf "\nProcessing dataset: %s\n" "$DATASET" | cat
  echo "---------------------------------------------------------------"

  # Remove any weird noData values
  echo "Un-setting noData values"
  gdal_edit.py "./$DATASET/label.tif" -unsetnodata
  gdal_edit.py "./$DATASET/image.tif" -unsetnodata
  gdal_edit.py "./image.tif" -a_nodata 0 # Assumes that background is black

  # Remove extra bands and BGRI -> RGBI
  echo "Removing extra bands and reordering"
  gdal_translate \
    -b 1 -b 2 -b 3 \
    -ot 'Byte' \
    "./$DATASET/image.tif" "./$DATASET/image_rgb.tif"

  # Get image extent
  ul=$(gdalinfo "$DATASET/image_rgb.tif" | grep "Upper Left")
  lr=$(gdalinfo "$DATASET/image_rgb.tif" | grep "Lower Right")
  x_min=$(echo "$ul" | grep -P '(?<=\()([-]?\d{1,3}\.\d+)(?=.*)' -o)
  x_max=$(echo "$lr" | grep -P '(?<=\()([-]?\d{1,3}\.\d+)(?=.*)' -o)
  y_min=$(echo "$lr" | grep -P '(?<=,\s{2})([-]?\d{1,3}\.\d+)(?=.*)' -o)
  y_max=$(echo "$ul" | grep -P '(?<=,\s{2})([-]?\d{1,3}\.\d+)(?=.*)' -o)

  # Get the image res
  res=$(gdalinfo "$DATASET/image_rgb.tif" | grep "Pixel Size")
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
    "./$DATASET/image_rgb.tif" "./$DATASET/${DATASET}.tif"
  rm "./$DATASET/image_rgb.tif"

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
  python "$PROJECT_DIR/data_prep/filter_datasets.py" "bg_only_labels" "$DATASET" --filter_prob=0.9
  sleep 1

  echo "Deleting tile pairs with blank image data"
  python "$PROJECT_DIR/data_prep/filter_datasets.py" "blank_imgs" "$DATASET"
  sleep 1

  echo "Deleting tile pairs less than half the required size"
  python "$PROJECT_DIR/data_prep/filter_datasets.py" "skinny_labels" "$DATASET" --min_height=256 --min_width=256
  sleep 1

  # Pad images that aren't 512 x 512 shaped
  echo "Padding incorrectly shaped images."
  python "$PROJECT_DIR/data_prep/preprocess_chips.py" "expand_chips" "$DATASET" --size=512
  sleep 1

  echo "Removing unmatched images and labels."
  python "$PROJECT_DIR/data_prep/filter_datasets.py" "delete_extra_labels" "$DATASET"
  python "$PROJECT_DIR/data_prep/filter_datasets.py" "delete_extra_imgs" "$DATASET"
  sleep 1

  # Split to train/test set
  echo "Splitting to 70/30 train/test sets"
  python "$PROJECT_DIR/data_prep/train_test_split.py" "$DATASET" "$TILED_OUTPUT_DIR" --train_size=0.7
  sleep 1
done

# Upload data to cloud
aws s3 sync "$TILED_OUTPUT_DIR/train" "${S3_BUCKET}/train"
aws s3 sync "$TILED_OUTPUT_DIR/eval" "${S3_BUCKET}/eval"

cd - || exit 1
