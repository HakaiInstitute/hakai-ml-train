# Get the path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Sync datasets
aws s3 sync s3://hakai-deep-learning-datasets/kelp/train ./train_input/data/train
aws s3 sync s3://hakai-deep-learning-datasets/kelp/eval ./train_input/data/eval

# Example build and run command
docker build -t deeplabv3/kelp-train ..

docker run -it --rm \
-v "$DIR/train_input/config":/opt/ml/input/config \
-v "$DIR/train_input/data":/opt/ml/input/data \
-v "$DIR/train_output/logs":/opt/ml/output \
-v "$DIR/train_output/checkpoints":/opt/ml/checkpoints \
-v "$DIR/train_output/model_weights":/opt/ml/model \
--user "$(id -u):$(id -g)" \
--ipc host \
--gpus all \
--name kelp-eval \
deeplabv3/kelp-train eval

# Wait for process so AWS exits when it's done
docker wait kelp-eval