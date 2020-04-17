# Get the path to this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PORT=6006

# Sync datasets
aws s3 sync s3://hakai-deep-learning-datasets/kelp/train ./train_input/data/train
aws s3 sync s3://hakai-deep-learning-datasets/kelp/eval ./train_input/data/eval

# Example build and run command
docker build --file ../Dockerfile --compress --tag deeplabv3/kelp ../..

docker run -dit --rm \
-p 0.0.0.0:$PORT:$PORT \
-v "$DIR/train_input":/opt/ml/input \
-v "$DIR/train_output/logs":/opt/ml/output \
-v "$DIR/train_output/checkpoints":/opt/ml/checkpoints \
-v "$DIR/train_output/model_weights":/opt/ml/model \
-v "$DIR/train_output/segmentation":/opt/ml/segmentation \
--user "$(id -u):$(id -g)" \
--ipc host \
--gpus all \
--name kelp-train \
deeplabv3/kelp train

# Can start tensorboard in running container as follows:
docker exec -dit kelp-train tensorboard --logdir=/opt/ml/checkpoints/runs --host=0.0.0.0 --port=$PORT
# Navigate to localhost:6006 to see train stats

# Wait for process so AWS exits when it's done
docker wait kelp-train

# Sync results to S3
ARCHIVE="./train_output/$(date +'%Y-%m-%d-%H%M').tar.gz"
tar -czvf "$ARCHIVE" ./train_output/model_weights/
aws s3 cp "$ARCHIVE" s3://hakai-deep-learning-datasets/kelp/output/