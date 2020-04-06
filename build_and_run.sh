aws s3 sync s3://hakai-deep-learning-datasets/kelp/train ./train_input/data/train
aws s3 sync s3://hakai-deep-learning-datasets/kelp/eval ./train_input/data/eval

# Example build and run command
docker build -t deeplabv3/kelp-train .

docker run -dit --rm \
-p 0.0.0.0:6006:6006 \
-v "$PWD/train_input/config":/opt/ml/input/config \
-v "$PWD/train_input/data":/opt/ml/input/data \
-v "$PWD/train_output/logs":/opt/ml/output \
-v "$PWD/train_output/checkpoints":/opt/ml/checkpoints \
-v "$PWD/train_output/model_weights":/opt/ml/model \
--user "$(id -u):$(id -g)" \
--ipc host \
--gpus all \
--name kelp-train \
deeplabv3/kelp-train train

# Can start tensorboard in running container as follows:
docker exec -it kelp-train tensorboard --logdir=/opt/ml/checkpoints/runs --host=0.0.0.0 --port=6006
# Navigate to localhost:6006 to see train stats

# Wait for process so AWS exits when it's done
docker wait kelp-train