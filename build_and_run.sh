docker container rm kelp-train --force

aws s3 sync s3://hakai-deep-learning-datasets/kelp/train ./train_input/data/train
aws s3 sync s3://hakai-deep-learning-datasets/kelp/eval ./train_input/data/eval

# Example build and run command
docker build -t deeplabv3/kelp-train .

docker run -d -p 0.0.0.0:6006:6006 \
-v "$PWD/train_input/config":/opt/ml/input/config:Z \
-v "$PWD/train_input/data":/opt/ml/input/data:Z \
-v "$PWD/train_output/logs":/opt/ml/output:Z \
-v "$PWD/train_output/checkpoints":/opt/ml/checkpoints:Z \
-v "$PWD/train_output/model_weights":/opt/ml/model:Z \
--ipc host \
--gpus all \
--name kelp-train \
deeplabv3/kelp-train train

# Can start tensorboard in running container as follows:
docker exec -it kelp-train tensorboard --logdir=/opt/ml/checkpoints/runs --host=0.0.0.0 --port=6006

# Navigate to localhost:6006 to see train stats
