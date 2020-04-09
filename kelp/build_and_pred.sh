# Get the path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Example build and run command
docker build -t deeplabv3/kelp-train ..

docker run -it --rm \
-v "$DIR/train_input":/opt/ml/input \
-v "$DIR/train_output":/opt/ml/output \
-v "$DIR/checkpoints":/opt/ml/checkpoints \
-v "$DIR/model_weights":/opt/ml/model \
--user "$(id -u):$(id -g)" \
--ipc host \
--gpus all \
--name kelp-pred \
deeplabv3/kelp-train pred