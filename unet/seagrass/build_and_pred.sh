# Get the path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Example build and run command
docker build --file ../Dockerfile --compress --tag unet/seagrass ../..

docker run -it --rm \
-v "$DIR/train_input":/opt/ml/input \
-v "$DIR/train_output/logs":/opt/ml/output \
-v "$DIR/train_output/checkpoints":/opt/ml/checkpoints \
-v "$DIR/train_output/model_weights":/opt/ml/model \
-v "$DIR/train_output/segmentation":/opt/ml/segmentation \
--user "$(id -u):$(id -g)" \
--ipc host \
--gpus all \
--name seagrass-pred \
unet/seagrass pred