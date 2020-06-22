#!/bin/bash

# Weight version for qualitative test
WEIGHTS_VER='200620'

# Get the path to this script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Download the weights
aws s3 sync "s3://hakai-deep-learning-datasets/kelp_species/weights/deeplabv3_kelp_species_$WEIGHTS_VER.ckpt" \
  "$DIR/train_output/weights/"

# Build the docker image
DOCKER_BUILDKIT=1 docker build --file ./Dockerfile --compress --tag tayden/deeplabv3-kelp-species ..

# Triquet 2019 test
docker run -it --rm \
  -v "$DIR/train_input":/opt/ml/input \
  -v "$DIR/train_output":/opt/ml/output \
  --user "$(id -u):$(id -g)" \
  --ipc host \
  --gpus all \
  --name kelp-species-pred \
  tayden/deeplabv3-kelp-species:latest pred \
  "/opt/ml/input/data/segmentation/triquet_small.tif" \
  "/opt/ml/output/segmentation/triquet_small_kelp_species_$WEIGHTS_VER.tif" \
  "/opt/ml/output/weights/deeplabv3_kelp_species_$WEIGHTS_VER.ckpt"

## Nanwakolas test
#docker run -it --rm \
#  -v "$DIR/train_input":/opt/ml/input \
#  -v "$DIR/train_output":/opt/ml/output \
#  --user "$(id -u):$(id -g)" \
#  --ipc host \
#  --gpus all \
#  --name kelp-species-pred \
#  tayden/deeplabv3-kelp-species:latest pred \
#  "/opt/ml/input/data/segmentation/nanwakolas_km_k02.tif" \
#  "/opt/ml/output/segmentation/nanwakolas_km_k02_kelp_species_$WEIGHTS_VER.tif" \
#  "/opt/ml/output/weights/deeplabv3_kelp_species_$WEIGHTS_VER.ckpt"
