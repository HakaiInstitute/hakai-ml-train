ARG VERSION=1.4-cuda10.1-cudnn7-runtime
FROM pytorch/pytorch:$VERSION

RUN pip install tensorboard tqdm

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/code:${PATH}"

# Set up the program in the image
COPY ./utils /opt/code/utils
COPY ./models /opt/code/models
COPY ./scripts/train_deeplabv3.py /opt/code/train_deeplabv3.py
WORKDIR /opt/code

# Note: During Docker run, data and output directories are bound via volumes

# Run the train script
ENTRYPOINT ["python", "train_deeplabv3.py"]

# Example build and run command
# docker build -t deeplabv3/kelp-train . && docker run -p 0.0.0.0:6006:6006 \
# -v /home/tadenoud/PycharmProjects/uav-classif/train_input:/opt/ml/input \
# -v /home/tadenoud/PycharmProjects/uav-classif/train_output/logs:/opt/ml/output \
# -v /home/tadenoud/PycharmProjects/uav-classif/train_output/checkpoints:/opt/ml/checkpoints \
# -v /home/tadenoud/PycharmProjects/uav-classif/train_output/model_weights:/opt/ml/model \
# --ipc host \
# --gpus all \
# --name kelp-train \
# deeplabv3/kelp-train

# Can start tensorboard in running container as follows:
# docker exec -it kelp-train tensorboard --logdir=/opt/ml/checkpoints/runs --host=0.0.0.0 --port=6006

# Navigate to localhost:6006 to see train stats