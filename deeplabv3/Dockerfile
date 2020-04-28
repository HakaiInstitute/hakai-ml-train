ARG VERSION=1.4-cuda10.1-cudnn7-runtime
FROM pytorch/pytorch:$VERSION

RUN pip install tensorboard tqdm rasterio fire

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/code:${PATH}"
ENV DOCKER=TRUE

# Set up the program in the image
COPY deeplabv3/deeplabv3.py /opt/code/deeplabv3.py
COPY models /opt/code/models
COPY utils /opt/code/utils
WORKDIR /opt/code

# Note: During Docker run, data and output directories are bound via volumes

# Run the train script
ENTRYPOINT ["python", "deeplabv3.py"]
