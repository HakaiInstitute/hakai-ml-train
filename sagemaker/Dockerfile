ARG VERSION=1.4-cuda10.1-cudnn7-runtime
FROM pytorch/pytorch:$VERSION

RUN pip install tensorboard tqdm

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/code:${PATH}"

# Set up the program in the image
COPY ./code /opt/code
WORKDIR /opt/code

ENTRYPOINT ["python", "sagemaker_train_deeplabv3.py"]