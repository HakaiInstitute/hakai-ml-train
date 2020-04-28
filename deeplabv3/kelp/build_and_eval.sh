# Get the path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Build the docker image
docker build --file ../Dockerfile --compress --tag tayden/deeplabv3-kelp ../..

weights="/home/tadenoud/PycharmProjects/uav-classif/deeplabv3/kelp/train_output/model/200421/deeplabv3_final.pt"

# Sync datasets
aws s3 sync --exclude "*" --include "*/[0-9].png" s3://hakai-deep-learning-datasets/kelp/train ./train_input/data/train
aws s3 sync --exclude "*" --include "*/[0-9].png" s3://hakai-deep-learning-datasets/kelp/eval ./train_input/data/eval

# Make output dirs
mkdir -p "./train_output/checkpoints"
mkdir -p "./train_output/model"
mkdir -p "./train_output/segmentation"

# Example build and run command
docker build --file ../Dockerfile --tag deeplabv3/kelp ../..

docker run -it --rm \
-v "$DIR/train_input":/opt/ml/input \
-v "$DIR/train_output":/opt/ml/output \
--mount type=bind,source="$weights",target=/opt/ml/input/weights.pt,readonly \
--user "$(id -u):$(id -g)" \
--ipc host \
--gpus all \
--name kelp-eval \
deeplabv3/kelp eval

# Wait for process so AWS exits when it's done
docker wait kelp-eval