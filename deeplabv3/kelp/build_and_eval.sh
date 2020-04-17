# Get the path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Sync datasets
aws s3 sync s3://hakai-deep-learning-datasets/kelp/train ./train_input/data/train
aws s3 sync s3://hakai-deep-learning-datasets/kelp/eval ./train_input/data/eval

# Example build and run command
docker build --file ../Dockerfile --tag deeplabv3/kelp ../..

docker run -it --rm \
-v "$DIR/train_input":/opt/ml/input \
-v "$DIR/train_output":/opt/ml/output \
-v "$DIR/checkpoints":/opt/ml/checkpoints \
-v "$DIR/model_weights":/opt/ml/model \
--user "$(id -u):$(id -g)" \
--ipc host \
--gpus all \
--name kelp-eval \
deeplabv3/kelp eval

# Wait for process so AWS exits when it's done
docker wait kelp-eval