#!/bin/bash

# Get the path to this script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PORT=6006

# Build the docker image
docker build --file ./Dockerfile --compress --tag tayden/deeplabv3-kelp ..

# Sync datasets
aws s3 sync s3://hakai-deep-learning-datasets/kelp/train ./train_input/data/train
aws s3 sync s3://hakai-deep-learning-datasets/kelp/eval ./train_input/data/eval

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
  --name kelp-train \
  tayden/deeplabv3-kelp train "/opt/ml/input/data/train" "/opt/ml/input/data/eval" "/opt/ml/output/checkpoints" \
  --accumulate_grad_batches=16 --precision=16 --amp_level='02' --auto_lr_find

# Can start tensorboard in running container as follows:
docker exec -dit kelp-train tensorboard --logdir=/opt/ml/output/checkpoints/runs --host=0.0.0.0 --port=$PORT
# Navigate to localhost:6006 to see train stats

# Wait for process so AWS exits when it's done
docker wait kelp-train

# Sync results to S3
ARCHIVE="$(date +'%Y-%m-%d-%H%M').tar.gz"
cd ./train_output/checkpoints/ || exit 1
tar -czvf "../$ARCHIVE" ./*.ckpt
cd ../
aws s3 cp "$ARCHIVE" s3://hakai-deep-learning-datasets/kelp/output/
