ARG VERSION=1.4-cuda10.1-cudnn7-runtime
FROM pytorch/pytorch:$VERSION

RUN pip install tensorboard tqdm

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/code:${PATH}"

# Set up the program in the image
COPY utils /opt/code/utils
COPY models /opt/code/models
COPY scripts/train_deeplabv3.py /opt/code/train_deeplabv3.py
WORKDIR /opt/code

# Note: During Docker run, data and output directories are bound via volumes

# Run the train script
ENTRYPOINT ["python", "train_deeplabv3.py"]
