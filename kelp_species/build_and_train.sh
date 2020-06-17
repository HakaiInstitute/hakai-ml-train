#!/bin/bash

# Get the path to this script
NAME=OneCycleLR_AdamW_FTL_TL
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PORT=6006

# Build the docker image
DOCKER_BUILDKIT=1 docker build --file ./Dockerfile --tag tayden/deeplabv3-kelp-species ..

# Sync datasets
# For testing, add: --exclude="*" --include="**/[0-9].png"
aws s3 sync s3://hakai-deep-learning-datasets/kelp_species/train ./train_input/data/train
aws s3 sync s3://hakai-deep-learning-datasets/kelp_species/eval ./train_input/data/eval

# Make output dirs
mkdir -p "./train_output/checkpoints"

# Run the docker image
docker run -dit --rm \
  -p 0.0.0.0:$PORT:$PORT \
  -v "$DIR/train_input":/opt/ml/input \
  -v "$DIR/train_output":/opt/ml/output \
  --user "$(id -u):$(id -g)" \
  --ipc host \
  --gpus all \
  --name kelp-species-train \
  tayden/deeplabv3-kelp-species train "/opt/ml/input/data/train" "/opt/ml/input/data/eval" "/opt/ml/output/checkpoints" \
  --name=$NAME --epochs=100 --lr=0.01 --weight_decay=0.001 \
  --gradient_clip_val=0.5 --batch_size=8 --precision=16 --amp_level="O1"

# Can start tensorboard in running container as follows:
docker exec -dit kelp-species-train tensorboard --logdir=/opt/ml/output/checkpoints --host=0.0.0.0 --port=$PORT
# Navigate to localhost:6006 to see train stats

# Wait for process so AWS exits when it's done
docker wait kelp-species-train

# Sync results to S3
ARCHIVE="$(date +'%Y-%m-%d-%H%M')_$NAME.tar.gz"
cd ./train_output/checkpoints/$NAME || exit 1
tar -czvf "$ARCHIVE" ./*
aws s3 cp "$ARCHIVE" s3://hakai-deep-learning-datasets/kelp_species/output/
