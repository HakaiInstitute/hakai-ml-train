#!/bin/bash

# Get the path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Build the docker image
DOCKER_BUILDKIT=1 docker build --file ./Dockerfile --compress --tag tayden/deeplabv3-kelp-species ..

# Run the docker image and bind data
docker run -it --rm \
-v "$DIR/train_input":/opt/ml/input \
-v "$DIR/train_output":/opt/ml/output \
--user "$(id -u):$(id -g)" \
--ipc host \
--gpus all \
--name kelp-species-pred \
tayden/deeplabv3-kelp-species:latest pred "/opt/ml/input/data/segmentation/mcnaughton_small.tif" \
  "/opt/ml/output/segmentation/mcnaughton_small_kelp.tif" "/opt/ml/output/weights/deeplabv3_kelp_species_200611.ckpt"