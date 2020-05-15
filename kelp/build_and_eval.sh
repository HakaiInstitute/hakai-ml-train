#!/bin/bash

# Get the path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Build the docker image
docker build --file ./Dockerfile --compress --tag tayden/deeplabv3-kelp ..

# Sync datasets
aws s3 sync --exclude="*" --include="*/[0-99].png" s3://hakai-deep-learning-datasets/kelp/train ./train_input/data/train
aws s3 sync s3://hakai-deep-learning-datasets/kelp/eval ./train_input/data/eval

docker run -it --rm \
-v "$DIR/train_input":/opt/ml/input \
-v "$DIR/train_output":/opt/ml/output \
--user "$(id -u):$(id -g)" \
--ipc host \
--gpus all \
--name kelp-eval \
tayden/deeplabv3-kelp eval "/opt/ml/input/data/train" \
  "/opt/ml/input/data/eval" "/opt/ml/output/weights/deeplabv3_kelp_200506.pt"

# Wait for process so AWS exits when it's done
docker wait kelp-eval