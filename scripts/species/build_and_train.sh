#!/bin/bash

# Get the path to this script
NAME=L_RASPP_MobileNetV3
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PORT=6006

# Build the docker image
#DOCKER_BUILDKIT=1 docker build --file ../../Dockerfile --tag tayden/deeplabv3-kelp ../../
DOCKER_BUILDKIT=1 docker build --file ../../Dockerfile --tag tayden/lraspp-mobilenetv3-kelp ../../

# Sync datasets
aws s3 sync s3://hakai-deep-learning-datasets/kelp_species/train "$DIR/train_input/data/train"
aws s3 sync s3://hakai-deep-learning-datasets/kelp_species/eval "$DIR/train_input/data/eval"

# Get initial weights
#aws s3 sync --exclude="*" --include="best-val_miou=0.9393-epoch=97-step=34789.pt" \
#  "s3://hakai-deep-learning-datasets/kelp/weights/2021-05-19-0316_May2020_Data" "$DIR/train_input/data/"
aws s3 sync --exclude="*" --include="best-val_miou=0.9218-epoch=196-step=69934.pt" \
  "s3://hakai-deep-learning-datasets/kelp/weights/2021-05-21-0106_LRASPP_MobileNetV3" "$DIR/train_input/data/"

# Make output dirs
mkdir -p "$DIR/train_output/checkpoints/$NAME"

# Run the docker image
## DeepLab V3
#docker run -dit --rm \
#  -p 0.0.0.0:$PORT:$PORT \
#  -v "$DIR/train_input":/opt/ml/input \
#  -v "$DIR/train_output":/opt/ml/output \
#  --user "$(id -u):$(id -g)" \
#  --ipc host \
#  --gpus all \
#  --name kelp-species-train \
#  tayden/deeplabv3-kelp train /opt/ml/input/data /opt/ml/output/checkpoints \
#  --name=$NAME --num_classes=3 \
#  --lr=0.001 --backbone_lr=0.0001 --weight_decay=0.001 --gradient_clip_val=0.5 \
#  --pa_weights="/opt/ml/input/data/best-val_miou=0.9393-epoch=97-step=34789.pt" \
#  --auto_select_gpus --gpus=-1 --benchmark --sync_batchnorm \
#  --max_epochs=100 --batch_size=8 --amp_level=O2 --precision=16 --accelerator=ddp --log_every_n_steps=10  # AWS
##  --max_epochs=10 --batch_size=2 --unfreeze_backbone_epoch=100 --log_every_n_steps=5 --overfit_batches=2 --no_train_backbone_bn  # TESTING

# L-RASPP MobileNet v3
docker run -dit --rm \
  -p 0.0.0.0:$PORT:$PORT \
  -v "$DIR/train_input":/opt/ml/input \
  -v "$DIR/train_output":/opt/ml/output \
  --user "$(id -u):$(id -g)" \
  --ipc host \
  --gpus all \
  --name kelp-species-train \
  tayden/lraspp-mobilenetv3-kelp train /opt/ml/input/data /opt/ml/output/checkpoints \
  --name=$NAME --num_classes=3 \
  --lr=0.0001 --weight_decay=0.001 --gradient_clip_val=0.5 \
  --pa_weights="/opt/ml/input/data/best-val_miou=0.9218-epoch=196-step=69934.pt" \
  --auto_select_gpus --gpus=-1 --benchmark --sync_batchnorm \
  --max_epochs=100 --batch_size=8 --amp_level=O2 --precision=16 --accelerator=ddp --log_every_n_steps=10  # AWS
#  --max_epochs=1 --batch_size=2 --log_every_n_steps=5  --overfit_batches=2  # TESTING

# Can start tensorboard in running container as follows:
docker exec -dit kelp-species-train tensorboard --logdir=/opt/ml/output/checkpoints --host=0.0.0.0 --port=$PORT
# Navigate to localhost:6006 to see train stats

# Wait for process so AWS exits when it's done
docker wait kelp-species-train

# Sync results to S3
ARCHIVE="$(date +'%Y-%m-%d-%H%M')_$NAME.tar.gz"
tar -czvf "$DIR/train_output/$ARCHIVE" -C "$DIR/train_output/checkpoints/$NAME" .
aws s3 cp "$DIR/train_output/$ARCHIVE" s3://hakai-deep-learning-datasets/kelp_species/output/
